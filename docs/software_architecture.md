# Helion FX-to-MLIR Software Architecture

This document describes the software architecture of the `helion_fx_mlir` package, which converts Helion Device IR (FX graphs) to MLIR text representation.

## Overview

The `helion_fx_mlir` package uses an **instruction-driven** approach that walks Device IR FX graphs node-by-node. It leverages a **registry-based lowering system** to dispatch operations to specific handlers, allowing for modular extensibility and integration with **torch-mlir** for lowering ATen operations to the `linalg` dialect.

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        BoundKernel (Helion)                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────────┐  │
│  │ fake_args   │  │    env      │  │      host_function          │  │
│  │ (Tensors)   │  │(BlockSizes) │  │  └─> device_ir (DeviceIR)   │  │
│  └─────────────┘  └─────────────┘  │      └─> graphs [GraphInfo] │  │
│                                    │          └─> graph (FX)     │  │
│                                    └─────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         generate_mlir()                             │
│                        (helion_mlir.py)                             │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
            ┌───────────┐   ┌──────────────┐   ┌──────────────┐
            │ IRVisitor │   │LoweringContext│  │  MLIRBuilder │
            │  (Walk)   │   │   (State)     │  │  (Emission)  │
            └───────────┘   └──────────────┘   └──────────────┘
                    │               │               │
                    ▼               ▼               ▼
            ┌───────────┐   ┌──────────────┐   ┌──────────────┐
            │Registry & │   │ TorchMLIR    │   │  Linalg      │
            │Lowerings  │   │ NodeImporter │   │  Dialect     │
            └───────────┘   └──────────────┘   └──────────────┘
                    │               │               │
                    └───────────────┼───────────────┘
                                    ▼
                            ┌───────────────┐
                            │  MLIR Text    │
                            │   Output      │
                            └───────────────┘
```

---

## Package Structure

```
src/helion_fx_mlir/
├── __init__.py              # Public API exports
├── helion_mlir.py           # Main entry point: generate_mlir()
├── ir_visitor.py            # IRVisitor: walks FX graphs and dispatches to registry
├── mlir_builder.py          # MLIR text emission utilities
├── lowering_context.py      # Lowering state management
├── op_registry.py           # Registry for lowering implementations
├── torch_mlir_helper.py     # Wrapper for torch-mlir FxImporter
└── lowerings/               # Modular lowering implementations
    ├── __init__.py          # Auto-registration of lowerings
    ├── base.py              # Base classes for lowerings
    ├── aten_ops.py          # ATen op lowerings (uses linalg)
    ├── memory_ops.py        # helion.load/store
    ├── control_flow.py      # affine.for/parallel
    └── ...
```

---

## Core Modules

### 1. `helion_mlir.py` - Main Entry Point

**Purpose**: Entry point for MLIR generation.

**Key Functions**:

| Function | Description |
|----------|-------------|
| `generate_mlir(bound_kernel, kernel_name)` | Generate MLIR from a bound kernel |
| `validate_with_helion_opt(mlir_text, ...)` | Validate MLIR using helion-opt |

### 2. `ir_visitor.py` - Graph Walker

**Purpose**: Walks Device IR FX graphs node-by-node, dispatching to handlers via the `LoweringRegistry`.

#### Class: `IRVisitor`

**Responsibilities**:
- Visit all nodes in Device IR graphs in order.
- Dispatch to `LoweringRegistry` if a lowering is registered for the node's target.
- Fall back to internal handler methods if no registry entry exists.
- Track SSA values in `node_values` and sync with `LoweringContext.fx_value_map`.

### 3. `op_registry.py` & `lowering_context.py`

**Purpose**: Manage lowering implementations and state.

*   **`LoweringRegistry`**: A singleton registry mapping FX node targets (e.g., `aten.addmm.default`) to `MLIRLowering` classes.
*   **`LoweringContext`**: Holds global state (builder, kernel args, loop info) and the `fx_value_map` which maps FX node names to MLIR SSA values.

### 4. `torch_mlir_helper.py`

**Purpose**: Integration with `torch-mlir`.

#### Class: `TorchMLIRNodeImporter`
Wraps `torch-mlir`'s `FxImporter` to allow leveraging its lowering logic. It can import a single FX node into a standalone MLIR module, which can then be lowered or inspected to generate corresponding Helion MLIR.

---

## Data Flow

### Complete Lowering Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│ 1. INPUT: BoundKernel                                               │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 2. CREATE CONTEXT: LoweringContext                                  │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 3. IRVisitor: Walk Graph                                            │
│    For each FX node:                                                │
│      a. Check LoweringRegistry.has(node.target)                     │
│      b. IF REGISTERED:                                              │
│         - Sync node_values -> ctx.fx_value_map                      │
│         - Call Lowering.emit(ctx, node)                             │
│           (e.g., AddmmLowering emits linalg.matmul)                 │
│      c. ELSE (Fallback):                                            │
│         - Dispatch to visit_* methods (e.g., visit_load)            │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 4. EMISSION                                                         │
│    - Builder emits MLIR text (helion.*, linalg.*, affine.*)         │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Op Mapping Reference

| Device IR | Handler Source | MLIR Output | Description |
|-----------|----------------|-------------|-------------|
| `aten.addmm` | `lowerings/aten_ops.py` | `linalg.matmul` | Matrix multiplication |
| `load` | `lowerings/memory_ops.py` | `helion.load` | Tile load |
| `store` | `lowerings/memory_ops.py` | `helion.store` | Tile store |
| `_for_loop` | `lowerings/control_flow.py` | `affine.for` | Reduction loop |
| `_phi` | `lowerings/control_flow.py` | `helion.phi` | Value merge |
| `full` | `visit_full` (Legacy) | `helion.full` | Tensor init |

*Note: The architecture is transitioning to fully registry-based handlers.*

---

## Extending the System

### Adding a New Op Handler

Use the `@register_lowering` decorator to register a handler for an FX target:

```python
from .op_registry import register_lowering
from .lowerings.base import MLIRLowering
import torch

@register_lowering(torch.ops.aten.add.Tensor)
class AddLowering(MLIRLowering):
    def emit(self, ctx, node):
        # Implementation to emit linalg.add or helion.add
        ...
```
