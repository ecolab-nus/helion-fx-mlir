# Helion → MLIR Lowering Policy

This repository currently hosts an experimental, Python-based lowering path that
translates Helion kernels into a textual MLIR “stage-0” module. The goal is to
capture the loop structure, data movement, and FX provenance needed for further
passes while iterating toward richer MLIR integrations. The project now ships a
bespoke `helion` dialect plus a standalone `helion-opt` driver so the emitted IR
parses without depending on `-allow-unregistered-dialect`.

## Loop & Tensor Mapping

| Helion construct | Lowered form | Notes |
| ---------------- | ------------ | ----- |
| `hl.tile(...)` outer blocks | `affine.parallel` | Trip counts derived from `tensor.dim` + `arith.ceildivsi`, supports an arbitrary number of iterators (e.g. `tile_b`, `tile_m`, `tile_n`). |
| Nested `hl.tile` / `for` blocks | `affine.for` | Each additional block id becomes its own loop with dynamic bounds. |
| Loop-carried values (`_phi` FX nodes) | `helion.phi` | Captures the carried tensor and FX node name for later legalization. |

## Tensor Loads & Stores

| Helion op | Lowered placeholder | Details |
| --------- | ------------------- | ------- |
| `helion.language.memory_ops.load` | `helion.load_tile_dynamic` | Accepts SSA tile sizes (`index` operands), records static `sizes = [...]` and carries a string `tensor_meta` + `fx_node` for provenance. |
| `helion.language.memory_ops.store` | `helion.store_tile_dynamic` | Writes back using the same dynamic size operands and `fx_node` metadata. |
| `hl.zeros`, `hl.full` | `helion.zero_tile`, `helion.alloc_like` | Simplified allocators that model buffer creation without committing to a dialect yet. |
| `torch.addmm`, `torch.baddbmm`, etc. | `helion.call_torch` | Keeps the math as an opaque op with `fn_name = "aten.*"`; future work will legalize these into Torch or Linalg dialects. |

Each operation retains the originating FX node name (`fx_node = "load"`, `"store"`, …) so downstream passes can reconcile the MLIR with the original Helion FX graph.

## Status & Next Steps

- ✔️ **Loop topology**: reconstructed from Helion’s `DeviceIR` (supports multiple roots / nested `_for_loop` blocks).
- ✔️ **Dynamic bounds**: all tile sizes/bounds are SSA-driven, enabling partial tile handling.
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

Use the helper scripts (`examples/helion_matmul.py`, `examples/attn.py`) to
inspect the stage-0 MLIR. When iterating on the dialect, re-run `helion-opt` (or
`mlir-opt -allow-unregistered-dialect` if the dialect is unavailable) to confirm
the generated IR parses cleanly. This README will evolve as we lock in the final
lowering rules.
