# Helion → MLIR Lowering Policy

This repository currently hosts an experimental, Python-based lowering path that
translates Helion kernels into a textual MLIR "stage-0" module. The goal is to
capture the loop structure, data movement, and FX provenance needed for further
passes while iterating toward richer MLIR integrations. The project now ships a
bespoke `helion` dialect plus a standalone `helion-opt` driver so the emitted IR
parses without depending on `-allow-unregistered-dialect`.

## Tile Size Handling

The MLIR generator supports both **concrete** and **symbolic** tile sizes based on
Helion's `static_shapes` setting:

| `static_shapes` | Tile Size Type | MLIR Representation |
|-----------------|----------------|---------------------|
| `True` | Concrete `int` | Emitted as constants: `arith.constant 128 : index` |
| `False` | `SymInt` | Emitted as function arguments: `%tile_m_size: index` |

Both modes use the **affine dialect** (`affine.parallel`, `affine.for`, `affine.min`, 
`affine.apply`). Symbolic tile sizes are passed as affine symbols, enabling proper
affine analysis while maintaining dynamic behavior:

- Trip counts: `affine.apply affine_map<()[s0, s1] -> (s0 ceildiv s1)>()[%dim, %tile_size]`
- Tile sizes: `affine.min affine_map<(d0)[s0, s1] -> (s1, s0 - d0 * s1)>(%iv)[%dim, %tile_size]`

Example function signatures:

```mlir
// static_shapes=True (concrete tile sizes)
func.func @matmul(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32>

// static_shapes=False (symbolic tile sizes as function arguments)
func.func @matmul(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, 
                  %tile_m_size: index, %tile_n_size: index, %tile_k_size: index) -> tensor<?x?xf32>
```

## FX Graph Structure

Helion compiles kernels into an FX-based Device IR containing multiple graphs:

| Graph Type | Purpose | `block_ids` |
|------------|---------|-------------|
| `ForLoopGraphInfo` | Innermost loop body (reduction loops) | `[2]` (e.g., tile_k) |
| `RootGraphInfo` | Outer parallel structure and control flow | `None` |

### FX Node Types

FX nodes represent operations in the Helion kernel. Each node has:
- `op`: Operation type (`call_function`, `placeholder`, `output`)
- `name`: SSA-style name used in MLIR (`fx_node` attribute)
- `target`: The Python function being called
- `args`: Arguments (often other FX nodes)

### Device IR Structure

Helion transforms the Python kernel into two FX graphs. Here is how the original `matmul` kernel maps to the Device IR:

**Original Python Kernel:**
```python
@helion.kernel
def matmul(x: torch.Tensor, y: torch.Tensor, out: torch.Tensor) -> None:
    # Root Graph Logic
    for tile_m, tile_n in hl.tile([x.size(0), y.size(1)]):
        # acc initialization
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        
        # Inner Loop (ForLoopGraphInfo)
        for tile_k in hl.tile(x.size(1)):
            x_tile = x[tile_m, tile_k]  # load
            y_tile = y[tile_k, tile_n]  # load_1
            acc = torch.addmm(acc, x_tile, y_tile)  # acc
            
        # Store result
        out[tile_m, tile_n] = acc  # store
```

**1. Inner Loop Graph (`ForLoopGraphInfo`)**:
Represents the body of the `tile_k` loop.
```
Graph 0: ForLoopGraphInfo
opcode         name            target                                     args                                             kwargs
-------------  --------------  -----------------------------------------  -----------------------------------------------  --------
placeholder    arg0_1          arg0_1                                     ()                                               {}
call_function  _new_var        <function _new_var ...>                    (arg0_1,)                                        {}
call_function  x               <function _host_tensor ...>                ('x',)                                           {}
call_function  sym_size_int    aten.sym_size.int                          (arg0_1, 0)                                      {}
call_function  block_size_2    <function _get_symnode ...>                ('block_size_2',)                                {}
call_function  load            <function load ...>                        (x, [sym_size_int, block_size_2], None, None)    {}
call_function  y               <function _host_tensor ...>                ('y',)                                           {}
call_function  sym_size_int_1  aten.sym_size.int                          (arg0_1, 1)                                      {}
call_function  load_1          <function load ...>                        (y, [block_size_2, sym_size_int_1], None, None)  {}
call_function  acc             aten.addmm.default                         (_new_var, load, load_1)                         {}
output         output          output                                     ([acc],)                                         {}
```
- `x`/`y` (`_host_tensor`): References to the input tensors `x` and `y`.
- `sym_size_int` / `sym_size_int_1`: Extracts `tile_m` and `tile_n` indices from the loop context (`arg0_1`).
- `load`/`load_1`: Corresponds to `x[tile_m, tile_k]` and `y[tile_k, tile_n]`.
- `acc` (`aten.addmm`): Corresponds to `torch.addmm(acc, x_tile, y_tile)`.
- `output`: Returns the updated `acc` for the next iteration.

**2. Root Graph (`RootGraphInfo`)**:
Represents the outer loops `tile_m, tile_n` and the reduction accumulation logic.
```
Graph 1: RootGraphInfo
opcode         name          target                                     args                                                      kwargs
-------------  ------------  -----------------------------------------  --------------------------------------------------------  --------
call_function  block_size_0  <function _get_symnode ...>                ('block_size_0',)                                         {}
call_function  block_size_1  <function _get_symnode ...>                ('block_size_1',)                                         {}
call_function  acc           <function full ...>                        ([block_size_0, block_size_1], 0.0, torch.float32, None)  {}
call_function  x_size1       <function _get_symnode ...>                ('x_size1',)                                              {}
call_function  _for_loop     <function _for_loop ...>                   (0, [0], [x_size1], [acc])                                {}
call_function  getitem       <built-in function getitem>                (_for_loop, 0)                                            {}
call_function  _phi          <function _phi ...>                        (acc, getitem)                                            {}
call_function  out           <function _host_tensor ...>                ('out',)
call_function  store         <function store ...>                       (out, [block_size_0, block_size_1], _phi, None)           {}
output         output        output                                     (None,)                                                   {}
```
- `acc` (`full`): Corresponds to `hl.zeros(...)`.
- `_for_loop`: Executes the inner loop over `tile_k`.
- `_phi`: Represents the final value of `acc` after the loop finishes (The Reduction).
- `out` (`_host_tensor`): Reference to the `out` argument.
- `store`: Corresponds to `out[tile_m, tile_n] = acc`.

**Key FX targets and their meaning:**

| FX Target | Description | Example Node Name |
|-----------|-------------|-------------------|
| `helion.language.memory_ops.load` | Tile load from tensor | `load`, `load_1` |
| `helion.language.memory_ops.store` | Tile store to tensor | `store` |
| `helion.language._tracing_ops._host_tensor` | Reference to kernel argument tensor | `x`, `y`, `out` |
| `helion.language._tracing_ops._phi` | Loop-carried value (reduction) | `_phi` |
| `aten.addmm.default` | Matrix multiply-add | `acc` |
| `helion.language._tracing_ops._for_loop` | Reduction loop marker | `_for_loop` |

## Loop & Tensor Mapping

| Helion construct | FX representation | MLIR lowering | Notes |
| ---------------- | ----------------- | ------------- | ----- |
| `hl.tile(...)` outer blocks | `_for_loop` in RootGraph | `affine.parallel` | Trip counts via `affine.apply` with ceildiv |
| Nested `hl.tile` / reduction | ForLoopGraphInfo with `block_ids` | `affine.for` with `iter_args` | Each block_id becomes a loop |
| Loop-carried values | `_phi` FX nodes | `helion.phi` | Captures initial and final values |
| Tensor arguments | `_host_tensor('name')` | Function args: `%name: tensor<...>` | Preserves original parameter names |

## Tensor Loads & Stores

| Helion op | FX Target | MLIR Op | FX → MLIR Mapping |
| --------- | --------- | ------- | ----------------- |
| `x[tile_m, tile_k]` | `memory_ops.load` | `helion.load_tile_dynamic` | `load.args[0]` → source tensor SSA |
| `out[tile_m, tile_n] = val` | `memory_ops.store` | `helion.store_tile_dynamic` | `store.args[0]` → target tensor SSA |
| `hl.zeros([...])` | `full` | `helion.zero_tile` | Creates accumulator tile |
| `torch.addmm(acc, lhs, rhs)` | `aten.addmm.default` | `helion.call_torch` | `fn_name = "aten.addmm"` |

### Load Node Structure

Each `load` FX node contains:
```
load.args[0] = source tensor node (e.g., "x" from _host_tensor('x'))
load.args[1] = tile indices as string (e.g., "[tile_m, tile_k]")
load.name    = unique name (e.g., "load", "load_1")
```

The MLIR emitter extracts the source tensor name to:
1. Look up the corresponding kernel argument SSA (`%x`, `%y`)
2. Infer tile dimensions based on tensor position (LHS: `[M,K]`, RHS: `[K,N]`)
3. Preserve the FX node name via `fx_node` attribute

Each operation retains the originating FX node name (`fx_node = "load"`, `"store"`, …) so downstream passes can reconcile the MLIR with the original Helion FX graph.

## Limitations

> [!IMPORTANT]
> The following limitations apply to the current MLIR generation:

### Tensor Indexing
- **Only simple indexing syntax is supported**: `x[tile_m, tile_k]` where `tile_m` and `tile_k` are defined by `hl.tile()`.
- **Not supported**: Complex indexing with `tile.start()`, `tile.index()`, or slice expressions. Support for these will be added in future versions.

### Shape Handling
- **Only symbolic shapes are supported**. We do not handle truly dynamic shapes at runtime.
- **No tile boundary checking**: The generated MLIR does not use `affine.min` for tile boundary handling. All tiles use the full block size.
- Symbolic shape parameters (M, N, K) and block sizes appear as module attributes.

### Loop Structure
- Loop induction variables use block_id-based naming: `%iv_block0`, `%iv_block1`, etc.
- Block IDs correspond to `hl.tile()` calls in order of appearance.

## TODO: Generalize Matmul-Specific Logic

> [!NOTE]
> Many of these items have been addressed in the generalization refactoring. The code now uses block_id-based indexing and loop-aware inference instead of hardcoded matmul patterns.

### Completed Generalizations

- [x] **Output shape inference**: Now uses outer loop extents instead of hardcoded `[lhs.size(0), rhs.size(1)]`
- [x] **Loop IV mapping**: Uses block_id-based IVs (`%iv_block0`, etc.) with dynamic lookup instead of hardcoded `tile_m/n/k`
- [x] **Tile dimension inference**: `_infer_tile_dims_for_load()` now uses `LoadInfo.tile_block_ids` from FX graph
- [x] **Store tile emission**: `_emit_store_tile()` uses outer loop info instead of hardcoded tile_m/tile_n
- [x] **Computation dispatch**: `_emit_computation()` dispatches to addmm/bmm/baddbmm based on FX graph analysis
- [x] **Tensor arg access**: `get_tensor_arg_ssa/type(index)` replaces hardcoded LHS/RHS methods

### Remaining Limitations (for complex kernels like attention)
- [ ] **3D tensor support**: Current code assumes 2D tensors; 3D batched operations need explicit handling
- [ ] **Multiple loop-carried values**: Attention has `(m_i, l_i, acc)` but code assumes single accumulator
- [ ] **Slice indexing**: `slice(None, None, None)` in load args not yet handled
- [ ] **Multi-output reduction loops**: `ForLoopGraphInfo` returning multiple values

### Legacy Fallbacks Still Present

The following use legacy naming as fallbacks but work generically:

- `_loop_name_to_dim_letter()`: Maps tile_m→m etc., used for module attribute naming
- `_symnode_to_tile_dim()`: Fallback when block_id not available from FX metadata
- `m_extent/n_extent/k_extent` properties: Kept for backward compatibility

### Future Improvements

1. **Parse slice expressions**: Support `x[tile_b, tile_m, :]` syntax in load operations
2. **Multi-value iter_args**: Handle multiple loop-carried values for flash attention
3. **3D tensor types**: Emit `tensor<?x?x?xf32>` for batched operations

## Status & Next Steps

- ✔️ **Loop topology**: reconstructed from Helion's `DeviceIR` (supports multiple roots / nested `_for_loop` blocks).
- ✔️ **Dynamic bounds**: tile sizes are handled via `affine.min` (concrete) or `arith.minsi` (symbolic) for partial tile handling.
- ✔️ **Symbolic tile sizes**: supports both concrete (`static_shapes=True`) and symbolic (`static_shapes=False`) tile sizes.
- ✔️ **FX provenance**: node names are threaded through attributes for debugging and later lowering.
- ✔️ **Dialect integration**: the bespoke `helion.*` ops are defined in C++ and registered via `helion-opt`.
- ⏳ **Math lowering**: arithmetic ops (`aten.addmm`, reductions, softmax pieces) are still opaque.
- ⏳ **Type/memory modeling**: `tensor_meta` strings are advisory; the eventual dialect will formalize buffer semantics.

## Building `helion-opt`

The dialect and driver live under `mlir/`. Configure and build the project using
the pre-installed MLIR toolchain exposed at `/mnt/fast/llvm-mlir`:

```bash
cmake -S . -B build
cmake --build build --target helion-opt
```

This produces `build/mlir/helion-opt`, which registers the Helion dialect along
with the affine/arith/func/tensor dialects the prototype currently emits. The
Python helper (`src/helion_fx_mlir/helion_mlir.py`) defaults to that binary when
present, falling back to upstream `mlir-opt` if needed.

Use the helper scripts (`examples/matmul.py`, `examples/attn.py`) to
inspect the stage-0 MLIR. When iterating on the dialect, re-run `helion-opt` (or
`mlir-opt -allow-unregistered-dialect` if the dialect is unavailable) to confirm
the generated IR parses cleanly. This README will evolve as we lock in the final
lowering rules.
