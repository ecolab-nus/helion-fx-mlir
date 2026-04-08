# Block-Loop Bound Lowering README

This document explains exactly what was implemented to support `mamba_chunk_scan` and similar kernels with expression-based block-loop bounds.

It covers:
- what changed in the codebase,
- where loop-bound information comes from,
- how loop bounds are built and lowered,
- how loop dialect (`affine.for` vs `scf.for`) is selected,
- what validation was run.

## Problem Statement

`examples/scripts/mamba_chunk_scan.py` uses an inner tiled loop:

```python
for tile_k in hl.tile((tile_m.id + 1) * block_m, block_size=block_k):
```

The loop upper bound is not a simple static symbol. It is an expression derived from outer loop state (`tile_m.id`) and block symbols (`block_m`).

The previous lowering assumed block-loop trip counts were always available from metadata (`ctx.reduction_trip_counts`) and/or that loop extents could be resolved directly from a single symbol lookup, which is insufficient for this pattern.

## What Was Changed

### 1) Tolerant extent resolution in `LoweringContext`

File: `src/helion_mlir/lowering_context.py`

Implemented:
- `_resolve_block_extent(info) -> int | None`
- `get_loop_extent(block_id) -> int | None`
- `get_loop_extent_or_hint(block_id, default=1) -> int`

Behavior:
- No hard failure if `info.size._sympy_()` is not a direct key in `shape_env.var_to_val`.
- Resolution tries:
  1. `info.size._sympy_()`
  2. fallback `info.var._sympy_()`
  3. unresolved => `None` (not exception)
- Call sites that need a value can use hint fallback (`info.size_hint()` under `self.env`) instead of crashing.

### 2) Relaxed setup-time requirements in module generation

File: `src/helion_mlir/helion_mlir.py`

Implemented:
- No requirement that every block has an eagerly-resolved `ctx.loop_extents` entry.
- Replaced direct `ctx.loop_extents[...]` uses with `ctx.get_loop_extent_or_hint(...)` where needed.
- Static block sizes are emitted as:

```mlir
%tile_x = arith.constant <value> : index
```

instead of requiring symbolic `loom.sym` in all cases.

### 3) FX-bound-first loop lowering with hybrid dialect selection

File: `src/helion_mlir/ir_visitor.py`

Implemented:
- `visit_for_loop` now uses dual-path trip-count derivation:
  - primary: derive from FX `_for_loop` bounds,
  - fallback: precomputed metadata (`ctx.reduction_trip_counts`).

- Added `_derive_trip_count_from_for_loop_args(...)`:
  - reads `_for_loop` bounds from `node.args[1]` (lower bounds) and `node.args[2]` (upper bounds),
  - optional step validation from `node.args[4]` if present,
  - enforces current phase constraints:
    - unit step only,
    - lower bound must be `0`.

- Added expression emission for index arithmetic via:
  - `_as_index_ssa(...)`
  - `visit_python_index_arith(...)`
  - supports at least `add`, `sub`, `mul`, `floordiv`, literals/constants, symbol nodes.

- Trip count is built as:

```mlir
trip_count = ceildiv(upper_bound, tile_size)
```

emitted as:

```mlir
%trip_count = affine.apply affine_map<(d0)[s0] -> (d0 ceildiv s0)>(%ub)[%tile_size]
```

- Dialect selection:
  - emit `affine.for` when bound expression is in the simple affine-compatible subset used here (`_get_symnode`, `aten.sym_size.int`, constants),
  - emit `scf.for` for more complex/dynamic expression bounds.

- Iter-arg semantics remain unchanged:
  - `_phi`, `getitem`, and loop-carried merge behavior still map to loop `iter_args`/yield results as before.

### 4) State-scoping fix for nested loop graph visitation

File: `src/helion_mlir/ir_visitor.py`

`ForLoopGraphInfo` and parent graph can reuse node names. To prevent inner graph SSA/type mappings leaking into outer scope:
- added save/restore scoping around `visit_graph(for_graph)` for:
  - `ctx.node_values`,
  - `ctx.node_types`,
  - `ctx.loop_result_values`,
  - `range_index_block_ids`.

This fixed undeclared SSA failures caused by cross-scope name collisions.

### 5) Subview layout typing fixes required to pass verifier

File: `src/helion_mlir/ir_visitor.py`

After enabling dynamic expression bounds and `scf.for`, verifier checks exposed layout typing gaps for rank-reduced/dynamic-offset `memref.subview`.

Implemented:
- retained-source-dimension stride tracking,
- strided result type emission with `offset: ?` for dynamic offsets,
- support for static and dynamic source shapes (dynamic stride components as `?`),
- scalar rank-reduced dynamic-offset case:

```mlir
memref<f16, strided<[], offset: ?>>
```

This was necessary for `mlir-opt` correctness on `mamba_chunk_scan`.

### 6) Targeted tests

File: `tests/test_block_loop_bounds.py`

Added tests for:
- expression-based bound path emits `scf.for` and passes `mlir-opt`,
- simple bound path keeps `affine.for`,
- non-zero lower bound is rejected with clear diagnostic text.

## Where Loop-Bound Information Comes From

Loop-bound lowering now uses the following sources in order:

1. FX `_for_loop` node (primary source)
- `node.args[1]`: lower bounds list
- `node.args[2]`: upper bounds list
- optional `node.args[4]`: step list/value

2. Loop-to-block mapping
- `ForLoopGraphInfo.block_ids` (canonicalized through alias map) to identify which block symbol controls this loop.

3. Block-size metadata
- `ctx.env.block_sizes[block_id]` for tile size info.
- If symbolic: `ctx.block_size_ssa[block_id]`.

4. Shape environment and origin metadata
- `ctx.bound_kernel.env.shape_env.var_to_val` for resolved concrete symbols.
- `ctx.bound_kernel.host_function.expr_to_origin` + `BlockSizeOrigin` to relate symbols to block IDs.

5. Fallback metadata trip count (legacy path)
- `ctx.reduction_trip_counts[block_id]` if FX-bound derivation is unavailable.

## Loop-Bound Construction Algorithm

Given a `_for_loop` node:

1. Resolve canonical loop block ID from `ForLoopGraphInfo.block_ids`.
2. Try FX-bound derivation:
   - parse `lb`, `ub`,
   - validate `step == 1` if explicit step exists,
   - require `lb == 0`,
   - lower `ub` to index SSA (recursively through arithmetic/symbol nodes),
   - resolve `tile_size` from block metadata/SSA,
   - compute `trip_count = ceildiv(ub, tile_size)`.
3. Decide loop dialect:
   - simple affine-compatible `ub` => `affine.for`,
   - otherwise => `scf.for`.
4. If FX path unavailable, use precomputed `ctx.reduction_trip_counts`.
5. Emit loop with unchanged `iter_args` and yield/phi semantics.

## Validation Performed

Commands run:

```bash
source .venv/bin/activate
python examples/scripts/mamba_chunk_scan.py
python examples/scripts/matmul.py
python examples/scripts/attn.py
python examples/scripts/matmul_split_k.py
pytest -q
```

Observed result:
- all listed scripts pass `mlir-opt` validation,
- tests pass (`3 passed`).

## Notes and Current Constraints

- Block-level `_for_loop` currently supports:
  - unit step only,
  - zero lower bound only.
- Outer grid loops remain on the existing lowering path.
- Hybrid policy is intentional:
  - keep `affine.for` where straightforward/beneficial,
  - use `scf.for` where needed for correctness with dynamic expression bounds.
