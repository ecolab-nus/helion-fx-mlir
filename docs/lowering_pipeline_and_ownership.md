# Lowering Pipeline And Ownership

`generate_mlir()` now coordinates a staged lowering pipeline instead of owning the entire lowering flow directly.

## Stages

1. `build_kernel_analysis()`
   - Collects immutable kernel facts from `bound_kernel`
   - Owns graph classification, reachable-graph discovery, canonical block aliasing, host tensor typing, and module attributes

2. `LoweringSession`
   - Owns mutable state during lowering
   - Tracks SSA values, node types, host tensor bindings, loop scopes, and loop-result projections

3. `ModuleEmitter`
   - Owns module/function scaffolding
   - Emits block-size symbols and reduction trip counts before graph lowering begins

4. `IRVisitor`
   - Walks FX graphs and lowers individual operations
   - Uses handler registration modules in `src/helion_mlir/handlers/` to keep dispatch ownership grouped by domain

## Where To Add New Code

- New whole-kernel analysis logic: `analysis.py`
- New mutable lowering state or scope behavior: `session.py`
- Symbolic/block/type resolution rules: `resolvers.py`
- Module/function-level MLIR scaffolding: `emitter.py`
- New operation lowering entrypoint: add it to the relevant file in `handlers/`
- torch-mlir adapter plumbing: `src/helion_mlir/torch_mlir/`

## Maintenance Rules

- Keep `generate_mlir()` orchestration-only.
- Prefer `LoweringSession` APIs over direct dict mutation when adding new stateful behavior.
- Keep reusable parsing/resolution logic out of op handlers.
- Treat `torch_mlir_helper.py` as a compatibility facade; prefer the `torch_mlir/` package for new imports.

## Testing Guidance

- Use `examples/scripts/*` kernels as characterization fixtures for end-to-end lowering.
- Add focused unit tests for resolver behavior, handler registration, and subview/type helpers when changing internal infrastructure.
- Preserve current `_mask_to` pass-through behavior unless intentionally implementing masking semantics.
