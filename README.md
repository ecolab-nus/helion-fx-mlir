# Helion → MLIR Lowering

This repository provides an instruction-driven lowering path that translates
[Helion](https://github.com/pytorch-labs/helion) kernels into MLIR. It walks Device IR FX graphs node-by-node,
mapping each operation to corresponding MLIR dialects.

For detailed architecture, FX graph structure, and lowering internals, see [Software Architecture](docs/software_architecture.md).

## Quick Start

```python
from helion_fx_mlir import generate_mlir, validate_with_helion_opt

# Generate MLIR from a bound Helion kernel
mlir_text = generate_mlir(bound_kernel, kernel_name="matmul")

# Validate with helion-opt
result = validate_with_helion_opt(mlir_text, extra_args=["-allow-unregistered-dialect"])
```

## Op Mapping Overview

The system maps operations from two sources:

| Operation Type | Examples | Generated Dialect |
|----------------|----------|-------------------|
| **Control Flow** | `_for_loop`, `_phi` | `affine`, `helion` |
| **Memory** | `load`, `store`, `subscript` | `tensor` (`extract_slice`, `insert_slice`, `expand_shape`) |
| **Tensor Creation** | `full`, `zeros` | `tensor.empty` + `linalg.fill` |
| **Compute** | `addmm`, `bmm`, `exp2`, `amax`, ... | `torch.aten.*` (via torch-mlir) |
| **Symbols** | `_get_symnode` | `loom.get_symbol` |

### Helion-Specific Operations

| Device IR Node | Generated MLIR |
|----------------|----------------|
| `_for_loop` | `affine.for` with `iter_args` |
| `_phi` | `helion.phi` |
| `load` | `tensor.extract_slice` |
| `store` | `tensor.insert_slice` |
| `subscript` | `tensor.extract_slice` / `tensor.expand_shape` |
| `full` / `zeros` | `tensor.empty` + `linalg.fill` |
| `_host_tensor` | SSA lookup (function parameters) |
| `_mask_to` | Pass-through (TODO: boundary checks) |

### ATen Operations

All ATen operations (`aten.*`) are lowered through **torch-mlir** integration, producing `torch.aten.*` dialect ops:

```
aten.addmm   → torch.aten.addmm
aten.bmm     → torch.aten.bmm
aten.exp2    → torch.aten.exp2
aten.amax    → torch.aten.amax
aten.sum     → torch.aten.sum.dim_IntList
...
```

See [`torch_mlir_helper.py`](src/helion_fx_mlir/torch_mlir_helper.py) for the FxImporter-based integration.

## Package Structure

```
src/helion_fx_mlir/
├── __init__.py              # Public API exports
├── helion_mlir.py           # Entry point: generate_mlir(), validate_with_helion_opt()
├── ir_visitor.py            # IRVisitor: walks FX graphs, dispatches to visit_* methods
├── lowering_context.py      # LoweringContext: state (loops, args, SSA mappings)
├── mlir_builder.py          # MLIRBuilder: text emission, SSA naming, indentation
└── torch_mlir_helper.py     # torch-mlir integration for ATen ops
```

## Building `helion-opt`

The Helion dialect and driver live under `mlir/`. Build with:

```bash
cmake -S . -B build
cmake --build build --target helion-opt
```

This produces `build/mlir/helion-opt`. Some ops (e.g., `loom.get_symbol`, `torch.aten.*`) require
`-allow-unregistered-dialect` until they are registered in the C++ dialect.

## Running Examples

### Matrix Multiplication

```bash
python examples/matmul.py
```

Prints Device IR, generated MLIR, and validates with `helion-opt`.

### Flash Attention

```bash
python examples/attn.py
```

Demonstrates a more complex kernel with 3D tensors, batch matrix operations, and reduction loops.

## Requirements

- Python 3.11+
- PyTorch with Helion (`helion` package)
- torch-mlir (for ATen operation lowering)
- LLVM/MLIR (for building `helion-opt`)

See `requirements.txt` for Python dependencies.

## Current Limitations

- **Masking**: `_mask_to` passes through tensors without boundary checks
- **Dynamic Shapes**: Full dynamic shape support is work-in-progress
- **torch Dialect**: `helion-opt` does not register torch dialect; use `-allow-unregistered-dialect`

## License

MIT License. See [LICENSE](LICENSE).
