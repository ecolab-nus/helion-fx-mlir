# Helion → MLIR Lowering

This repository hosts an instruction-driven lowering path that translates
Helion kernels into MLIR by walking Device IR FX graphs node-by-node. Each
Device IR operation is mapped to a corresponding MLIR operation.

For detailed architecture, FX graph structure, and lowering internals, see [Software Architecture](docs/software_architecture.md).

## MLIR Generation

The main entry point is `generate_mlir()`:

```python
from helion_fx_mlir import generate_mlir, validate_with_helion_opt

# Generate MLIR from a bound Helion kernel
mlir_text = generate_mlir(bound_kernel, kernel_name="matmul")

# Validate with mlir-opt
result = validate_with_helion_opt(mlir_text)
```

## Op Mapping Overview

The system maps operations from two sources:
1.  **Helion Nodes**: Control flow and memory operations (loops, load, store) are lowered directly by this project.
2.  **ATen Operations**: Compute operations (e.g., `addmm`, `relu`) are lowered via **torch-mlir** integration.

| Operation Type | Examples | Generated Dialect |
|----------------|----------|-------------------|
| **Control Flow** | loops, phi nodes | `affine`, `helion` |
| **Memory** | load, store, alloc | `tensor`, `helion` |
| **Compute** | addmm, mul, div | `linalg`, `arith` |

For a detailed mapping reference, see the [Architecture Documentation](docs/software_architecture.md#op-mapping-reference-generators).

## Building `helion-opt`

The Helion dialect and driver live under `mlir/`. Build with:

```bash
cmake -S . -B build
cmake --build build --target helion-opt
```

This produces `build/mlir/helion-opt`. Note that ops like `loom.get_symbol` require 
using `mlir-opt -allow-unregistered-dialect` until they are registered in the C++ dialect.

## Running Examples

```bash
python examples/matmul.py
```

This prints the Device IR and generated MLIR, then validates with `mlir-opt`.

## TODO: Temporary ATen Op Patches

The following ATen operations are currently handled with temporary `helion.aten_op` placeholders in `ir_visitor.py`. These should eventually be replaced with proper linalg lowerings via torch-mlir:

| ATen Op | Status | Notes |
|---------|--------|-------|
| `aten.full.default` | ✅ Done | Uses `tensor.empty` + `linalg.fill` |
| `aten.bmm.default` | ⚠️ Placeholder | Batch matrix multiply |
| `aten.baddbmm.default` | ⚠️ Placeholder | Batch add batch matrix multiply |
| `aten.amax.default` | ⚠️ Placeholder | Max along dimension |
| `aten.maximum.default` | ⚠️ Placeholder | Element-wise max |
| `aten.mul.Tensor` | ⚠️ Placeholder | Element-wise/scalar multiply |
| `aten.sub.Tensor` | ⚠️ Placeholder | Element-wise subtract |
| `aten.add.Tensor` | ⚠️ Placeholder | Element-wise add |
| `aten.exp2.default` | ⚠️ Placeholder | Element-wise 2^x |
| `aten.sum.dim_IntList` | ⚠️ Placeholder | Sum along dimensions |
| `aten.div.Tensor` | ⚠️ Placeholder | Element-wise divide |

### Helion-Specific Operations

| Operation | Status | Notes |
|-----------|--------|-------|
| `_mask_to` | ✅ Shortcircuit | Passes through input tensor; boundary checks ignored for now |
| `subscript` | ✅ `helion.subscript` | Emits placeholder op; proper slice handling TODO |

### Long-term Solution
These temporary patches should be replaced by either:
1. Proper torch-mlir integration once it supports dynamic FX node shapes
2. Custom linalg lowerings registered via `@register_lowering` decorator
3. Full implementation of `_mask_to` boundary checking logic
4. Proper slice/index operand handling in `helion.subscript`
