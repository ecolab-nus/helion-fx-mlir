# Helion FX-to-MLIR Software Architecture

This document describes the software architecture of the `helion_fx_mlir` package, which provides infrastructure for converting Helion kernels (via their Device IR and FX graphs) to MLIR text representation.

## Table of Contents

1. [Overview](#overview)
2. [Package Structure](#package-structure)
3. [Core Modules](#core-modules)
4. [Lowering System](#lowering-system)
5. [Data Flow](#data-flow)
6. [Extending the System](#extending-the-system)

---

## Overview

The `helion_fx_mlir` package implements a modular lowering system that converts Helion's intermediate representation (Device IR containing FX graphs) into MLIR text. The architecture follows these design principles:

- **Separation of Concerns**: Each module handles a specific aspect of the lowering process
- **Registry Pattern**: FX node targets are mapped to lowering implementations via a central registry
- **Extensibility**: New operation lowerings can be added without modifying existing code
- **Context-Driven**: All lowering state is managed through a context object

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
│                    generate_plan_stage0_mlir()                      │
│                         (helion_mlir.py)                            │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
            ┌───────────┐   ┌──────────────┐   ┌──────────────┐
            │MLIRBuilder│   │LoweringContext│  │LoweringRegistry│
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
├── helion_mlir.py           # Main entry point and orchestration
├── mlir_builder.py          # MLIR text emission utilities
├── lowering_context.py      # Lowering state management
├── op_registry.py           # FX target -> lowering mapping
└── lowerings/               # Lowering implementations
    ├── __init__.py          # Auto-registration of lowerings
    ├── base.py              # Abstract base class
    ├── memory_ops.py        # load/store operations
    ├── control_flow.py      # phi, for_loop, if, while
    ├── aten_ops.py          # PyTorch ATen operations
    └── tracing_ops.py       # Internal tracing operations
```

---

## Core Modules

### 1. `helion_mlir.py` - Main Entry Point

**Purpose**: Orchestrates the entire lowering process from a bound Helion kernel to MLIR text.

**Key Functions**:

| Function | Description |
|----------|-------------|
| `generate_plan_stage0_mlir(bound_kernel, kernel_name)` | Main entry point that generates MLIR from a bound kernel |
| `validate_with_helion_opt(mlir_text, opt_path, extra_args)` | Validates emitted MLIR using helion-opt or mlir-opt |

**Internal Helper Functions**:

| Function | Description |
|----------|-------------|
| `_emit_dimension_queries(ctx)` | Emits `tensor.dim` operations |
| `_emit_tensor_annotations(ctx)` | Emits `helion.annotate_tensor` operations |
| `_emit_loop_bounds(ctx)` | Computes and emits loop bound SSA values |
| `_emit_parallel_loop_structure(ctx, bound_kernel)` | Emits `affine.parallel` and nested content |
| `_emit_reduction_loops(ctx)` | Emits `affine.for` reduction loops |
| `_emit_lhs_load(ctx, ...)` | Emits left-hand side tile load |
| `_emit_rhs_load(ctx, ...)` | Emits right-hand side tile load |
| `_emit_addmm(ctx, lhs, rhs)` | Emits matrix multiply via `helion.call_torch` |
| `_emit_store_tile(ctx)` | Emits final tile store operation |

**Interaction**: Creates `LoweringContext`, uses `MLIRBuilder` for emission, and can invoke `LoweringRegistry` for individual FX nodes.

---

### 2. `mlir_builder.py` - MLIR Text Emission

**Purpose**: Manages MLIR text generation, including indentation, SSA value naming, and structured emission of common MLIR constructs.

#### Class: `MLIRBuilder`

**Responsibilities**:
- Track indentation levels for nested structures
- Generate unique SSA value names
- Emit MLIR operations, functions, and modules

**Core Methods**:

| Method | Description |
|--------|-------------|
| `emit(text)` | Emit a line with current indentation |
| `emit_comment(comment)` | Emit a comment line |
| `push()` / `pop()` | Increase/decrease indentation |
| `fresh(hint)` | Generate unique SSA name like `%hint0` |
| `build()` | Return complete MLIR text |

**Structure Emission Methods**:

| Method | Description |
|--------|-------------|
| `emit_module_start()` / `emit_module_end()` | Module delimiters |
| `emit_func_start(name, args, result_type)` | Function header |
| `emit_func_end()` | Function closing |
| `emit_return(values, types)` | Return statement |

**Operation Emission Methods**:

| Method | Description |
|--------|-------------|
| `emit_op(op_name, operands, attrs, ...)` | Generic operation |
| `emit_index_constant(value)` | `arith.constant N : index` |
| `emit_tensor_dim(tensor, dim, type)` | `tensor.dim` operation |

**Affine Dialect Methods**:

| Method | Description |
|--------|-------------|
| `emit_affine_parallel_start(ivs, lbs, ubs, steps)` | Start parallel loop |
| `emit_affine_parallel_end()` | End with `affine.yield` |
| `emit_affine_for_start(iv, lb, ub, iter_args, ...)` | Start for loop |
| `emit_affine_for_end(yield_values, yield_types)` | End with yield |
| `emit_affine_apply(map_str, dims, symbols)` | Affine map application |
| `emit_affine_min(map_str, dims, symbols)` | Affine minimum |

#### Utility Functions

| Function | Description |
|----------|-------------|
| `is_concrete_size(size)` | Check if size is concrete int (not symbolic) |
| `torch_dtype_to_mlir_element_type(dtype)` | Convert torch dtype to MLIR type string |
| `format_tensor_type(shape, element_type)` | Format `tensor<...>` type |
| `format_shape_attr(shape)` | Format shape as `[dim0, dim1, ...]` |
| `format_string_attr(value)` | Format string as `"value"` |
| `format_attr_dict(attrs)` | Format `{key = value, ...}` |
| `format_indices_attr(indices)` | Format indices array |
| `format_dynamic_tensor_meta(dim0, dim1, dtype)` | Format tensor metadata |

---

### 3. `lowering_context.py` - State Management

**Purpose**: Holds all state needed during the lowering process, providing a single context object passed to all lowering functions.

#### Class: `KernelArgInfo`

**Purpose**: Stores information about a kernel function argument, extracted from the Helion kernel signature.

**Fields**:

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Original parameter name from kernel signature |
| `index` | `int` | Position in the argument list |
| `is_tensor` | `bool` | Whether this is a tensor argument |
| `dtype` | `torch.dtype \| None` | Tensor dtype if is_tensor |
| `shape` | `list[int \| None] \| None` | Tensor shape (None for dynamic dims) |
| `mlir_type` | `str \| None` | MLIR type string |
| `ssa_name` | `str \| None` | SSA value name (e.g., "%x") |

#### Class: `LoopInfo`

**Purpose**: Stores information about a single loop dimension.

**Fields**:

| Field | Type | Description |
|-------|------|-------------|
| `block_id` | `int` | Helion block ID for this dimension |
| `name` | `str` | Loop name (e.g., "tile_m", "tile_k") |
| `tile_size` | `int \| None` | Concrete tile size, None if symbolic |
| `trip_count` | `int \| None` | Static trip count, None if dynamic |
| `total_extent` | `int` | Total dimension extent |
| `is_symbolic` | `bool` | Whether tile size is symbolic |
| `tile_const` | `str \| None` | SSA value for tile size constant |
| `trip_count_ssa` | `str \| None` | SSA value for trip count |
| `iv_name` | `str \| None` | Induction variable SSA name |

#### Class: `LoweringContext`

**Purpose**: Central context object containing all lowering state.

**Fields**:

| Field | Type | Description |
|-------|------|-------------|
| `builder` | `MLIRBuilder` | MLIR text builder |
| `bound_kernel` | `BoundKernel` | Source Helion kernel |
| `device_ir` | `DeviceIR` | Device IR from kernel |
| `current_graph` | `GraphInfo \| None` | Currently processing graph |
| `element_type` | `str` | MLIR element type (e.g., "f32") |
| `tensor_type` | `str` | MLIR tensor type string |
| `dim_m`, `dim_n`, `dim_k` | `str \| None` | Dimension SSA values |
| `out_value` | `str \| None` | Output tensor SSA value |
| `acc_seed` | `str \| None` | Initial accumulator SSA value |
| `outer_loops` | `list[LoopInfo]` | Parallel loop information |
| `reduction_loops` | `list[LoopInfo]` | Reduction loop information |
| `symbolic_arg_ssa` | `dict[str, str]` | Symbolic arg name -> SSA |
| `outer_tile_sizes` | `dict[str, str]` | Computed tile sizes |
| `dims_map` | `dict[str, str]` | Dimension name -> SSA |
| `fx_value_map` | `dict[str, str]` | FX node name -> MLIR SSA |
| `block_sizes` | `dict[int, BlockSizeInfo]` | Block ID -> size info |
| `current_acc` | `str \| None` | Current accumulator value |
| `fx_names` | `dict[str, str]` | Extracted FX node names |
| `root_fx_info` | `dict[str, str]` | Root graph FX info |
| `tile_shape_attr` | `str` | Tile shape attribute string |
| `kernel_args` | `list[KernelArgInfo]` | Kernel function arguments |

**Factory Method**:

```python
@classmethod
def from_bound_kernel(cls, bound_kernel, kernel_name) -> LoweringContext
```

Creates a fully initialized context from a bound Helion kernel.

**Key Methods**:

| Method | Description |
|--------|-------------|
| `get_symbolic_tile_args()` | Get list of symbolic tile size arguments |
| `get_loop_map()` | Get name -> LoopInfo mapping |
| `setup_dims_map()` | Initialize dimension name mapping |
| `get_tensor_args()` | Get only tensor arguments from kernel_args |
| `get_func_signature_args()` | Get function signature as (ssa_name, type) tuples |
| `get_lhs_tensor_ssa()` | Get SSA name of first tensor argument |
| `get_rhs_tensor_ssa()` | Get SSA name of second tensor argument |
| `get_lhs_tensor_type()` | Get MLIR type of first tensor argument |
| `get_rhs_tensor_type()` | Get MLIR type of second tensor argument |
| `get_tensor_arg_by_name(name)` | Get kernel argument by name |
| `get_tensor_arg_ssa(index)` | Get tensor argument SSA name by index |

#### Helper Functions

| Function | Description |
|----------|-------------|
| `_extract_kernel_args(bound_kernel)` | Extract argument info from kernel signature |
| `first_debug_name(names, fallback)` | Get sanitized debug name |
| `resolve_extent(name, lhs, rhs)` | Resolve dimension extent |
| `collect_reduction_block_ids(device_ir)` | Collect reduction block IDs |

---

### 4. `op_registry.py` - Lowering Registration

**Purpose**: Provides a registry that maps FX node targets to their corresponding lowering implementations.

#### Class: `LoweringRegistry`

**Design**: Singleton pattern with class-level storage.

**Class Attributes**:

| Attribute | Type | Description |
|-----------|------|-------------|
| `_registry` | `dict[object, type[MLIRLowering]]` | Target -> lowering class |
| `_instances` | `dict[object, MLIRLowering]` | Target -> cached instance |

**Methods**:

| Method | Description |
|--------|-------------|
| `register(target)` | Decorator to register a lowering class |
| `register_multiple(*targets)` | Register for multiple targets |
| `get(target)` | Get lowering class for target |
| `get_instance(target)` | Get cached lowering instance |
| `has(target)` | Check if target is registered |
| `clear()` | Clear all registrations (for testing) |
| `list_registered()` | List all registered targets |
| `emit_node(ctx, node)` | Emit MLIR for an FX node |

**Usage Example**:

```python
from helion_fx_mlir import register_lowering, MLIRLowering

@register_lowering(my_target_function)
class MyLowering(MLIRLowering):
    def emit(self, ctx, node):
        ...
```

#### Convenience Functions

| Function | Description |
|----------|-------------|
| `register_lowering(target)` | Alias for `LoweringRegistry.register` |
| `register_lowering_multiple(*targets)` | Alias for `register_multiple` |

---

## Lowering System

### Base Class: `MLIRLowering`

**Location**: `lowerings/base.py`

**Purpose**: Abstract base class that all lowering implementations must inherit.

```python
class MLIRLowering(ABC):
    @abstractmethod
    def emit(self, ctx: LoweringContext, node: torch.fx.Node) -> str | None:
        """Emit MLIR for the given FX node.
        
        Returns:
            SSA value name of result, or None for void operations.
        """
        ...
```

**Helper Methods**:

| Method | Description |
|--------|-------------|
| `get_operand_ssa(ctx, arg)` | Get SSA value for FX node argument |
| `get_node_name(node)` | Get sanitized node name |
| `get_tensor_from_node(node)` | Get fake tensor from node metadata |

**Utility Classes**:

| Class | Description |
|-------|-------------|
| `PassthroughLowering` | No-op lowering (returns None) |
| `CommentLowering` | Emits a comment for documentation |

---

### Memory Operations: `lowerings/memory_ops.py`

**Registered Lowerings**:

| Target | Class | MLIR Operation |
|--------|-------|----------------|
| `memory_ops.load` | `LoadTileLowering` | `helion.load_tile_dynamic` |
| `memory_ops.store` | `StoreTileLowering` | `helion.store_tile_dynamic` |

**LoadTileLowering Details**:

```
Input:  FX node for hl.load(tensor, indices, ...)
Output: %result = "helion.load_tile_dynamic"(%tensor, %size0, %size1){attrs}
            : (tensor_type, index, index) -> tensor_type
```

Key methods:
- `_get_tensor_ssa()` - Determine which argument tensor
- `_get_tile_sizes()` - Compute tile dimensions
- `_get_tile_indices()` - Determine index variables

---

### Control Flow: `lowerings/control_flow.py`

**Registered Lowerings**:

| Target | Class | MLIR Operation |
|--------|-------|----------------|
| `_tracing_ops._phi` | `PhiLowering` | `helion.phi` |
| `_tracing_ops._for_loop` | `ForLoopLowering` | Comment (structure by main driver) |
| `_tracing_ops._if` | `IfLowering` | Comment |
| `_tracing_ops._while_loop` | `WhileLoopLowering` | Comment |

**Helper Classes** (not registered, used by main driver):

| Class | Description |
|-------|-------------|
| `ReductionLoopEmitter` | Emit `affine.for` with `iter_args` |
| `ParallelLoopEmitter` | Emit `affine.parallel` |

**Note**: Control flow structure (loops) is emitted by the main driver in `helion_mlir.py`, not by individual lowerings, because it requires graph-level information.

---

### ATen Operations: `lowerings/aten_ops.py`

**Base Class**: `AtenOpLowering`

All ATen operations lower to `helion.call_torch`:

```
%result = "helion.call_torch"(%arg0, %arg1, ...){fn_name = "aten.op_name"}
    : (type0, type1, ...) -> result_type
```

**Registered Operations**:

| ATen Target | Class | fn_name |
|-------------|-------|---------|
| `aten.addmm.default` | `AddmmLowering` | `"aten.addmm"` |
| `aten.mm.default` | `MmLowering` | `"aten.mm"` |
| `aten.add.Tensor` | `AddTensorLowering` | `"aten.add"` |
| `aten.mul.Tensor` | `MulTensorLowering` | `"aten.mul"` |
| `aten.sub.Tensor` | `SubTensorLowering` | `"aten.sub"` |
| `aten.div.Tensor` | `DivTensorLowering` | `"aten.div"` |
| `aten.exp.default` | `ExpLowering` | `"aten.exp"` |
| `aten.log.default` | `LogLowering` | `"aten.log"` |
| `aten.sqrt.default` | `SqrtLowering` | `"aten.sqrt"` |
| `aten.relu.default` | `ReluLowering` | `"aten.relu"` |
| `aten.sigmoid.default` | `SigmoidLowering` | `"aten.sigmoid"` |
| `aten.tanh.default` | `TanhLowering` | `"aten.tanh"` |
| `aten.sum.default` | `SumLowering` | `"aten.sum"` |
| `aten.sum.dim_IntList` | `SumDimLowering` | `"aten.sum"` |
| `aten.max.default` | `MaxLowering` | `"aten.max"` |
| `aten.softmax.int` | `SoftmaxLowering` | `"aten.softmax"` |

---

### Tracing Operations: `lowerings/tracing_ops.py`

**Registered Lowerings**:

| Target | Class | Description |
|--------|-------|-------------|
| `_host_tensor` | `HostTensorLowering` | Maps to function argument |
| `_get_symnode` | `GetSymnodeLowering` | Symbolic index value |
| `_new_var` | `NewVarLowering` | Value copy/rename |
| `_constant_tensor` | `ConstantTensorLowering` | Scalar constant |
| `_and` | `AndLowering` | Logical AND (`arith.andi`) |
| `_or` | `OrLowering` | Logical OR (`arith.ori`) |
| `_not` | `NotLowering` | Logical NOT (`arith.xori`) |
| `_mask_to` | `MaskToLowering` | Masked value operation |
| `_inductor_lowering_extra` | `PassthroughLowering` | No-op |

---

## Data Flow

### Complete Lowering Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│ 1. INPUT: BoundKernel                                               │
│    - fake_args: [Tensor(128x128), Tensor(128x256), ...]            │
│    - env.block_sizes: {0: BlockSizeInfo(tile_m), ...}              │
│    - host_function.device_ir.graphs: [RootGraphInfo, ForLoop...]   │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 2. CREATE CONTEXT: LoweringContext.from_bound_kernel()              │
│    - Initialize MLIRBuilder                                         │
│    - Extract block sizes, grid block IDs                           │
│    - Build LoopInfo for outer/reduction loops                      │
│    - Determine element_type, tensor_type                           │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 3. EMIT PROLOGUE                                                    │
│    builder.emit_module_start()                                      │
│    builder.emit_func_start(name, args, result_type)                │
│    _emit_dimension_queries() → %dim_m, %dim_k, %dim_n              │
│    _emit_tensor_annotations() → helion.annotate_tensor             │
│    _emit_loop_bounds() → tile constants, trip counts               │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 4. EMIT LOOP STRUCTURE                                              │
│    affine.parallel (%iv0, %iv1) = (0, 0) to (%ub0, %ub1) {         │
│      _emit_outer_tile_sizes() → actual tile sizes                  │
│      _emit_reduction_loops():                                       │
│        affine.for %k = 0 to %k_trips iter_args(%acc = %init) {     │
│          _emit_lhs_load() → helion.load_tile_dynamic               │
│          _emit_rhs_load() → helion.load_tile_dynamic               │
│          _emit_addmm() → helion.call_torch                         │
│          affine.yield %acc_next                                    │
│        }                                                            │
│      _emit_phi_if_present() → helion.phi                           │
│      _emit_store_tile() → helion.store_tile_dynamic                │
│      affine.yield                                                   │
│    }                                                                │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 5. EMIT EPILOGUE                                                    │
│    builder.emit("return %out : tensor<...>")                       │
│    builder.emit_func_end()                                          │
│    builder.emit_module_end()                                        │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 6. OUTPUT: builder.build() → MLIR text string                       │
└─────────────────────────────────────────────────────────────────────┘
```

### FX Node Lowering Flow (when using registry directly)

```
FX Node with target T
        │
        ▼
LoweringRegistry.get_instance(T)
        │
        ├─── Found? ──► lowering.emit(ctx, node)
        │                       │
        │                       ▼
        │               Builder operations
        │               ctx.fx_value_map[node.name] = result
        │                       │
        │                       ▼
        │               Return SSA value or None
        │
        └─── Not found? ──► Return None (unhandled)
```

---

## Extending the System

### Adding a New ATen Operation

1. **Create the lowering class**:

```python
# In lowerings/aten_ops.py

@register_lowering(aten.new_op.default)
class NewOpLowering(AtenOpLowering):
    def __init__(self):
        super().__init__("aten.new_op")
```

2. **For custom behavior, override emit()**:

```python
@register_lowering(aten.custom_op.default)
class CustomOpLowering(MLIRLowering):
    def emit(self, ctx, node):
        builder = ctx.builder
        
        # Get operands
        input_ssa = ctx.fx_value_map.get(node.args[0].name, "%input")
        
        # Emit custom MLIR
        result = builder.fresh("custom")
        builder.emit(f'{result} = "my.custom_op"({input_ssa}) : (...) -> ...')
        
        # Store result
        ctx.fx_value_map[node.name] = result
        return result
```

### Adding a New Operation Category

1. **Create new file** `lowerings/my_ops.py`:

```python
from ..op_registry import register_lowering
from .base import MLIRLowering

@register_lowering(my_module.my_function)
class MyFunctionLowering(MLIRLowering):
    def emit(self, ctx, node):
        ...
```

2. **Import in** `lowerings/__init__.py`:

```python
from . import my_ops  # Add this line
```

### Adding New MLIR Dialect Support

1. **Add methods to MLIRBuilder**:

```python
# In mlir_builder.py

def emit_my_dialect_op(self, ...):
    result = self.fresh("my_result")
    self.emit(f'{result} = "my_dialect.op"(...) : ...')
    return result
```

2. **Use in lowerings**:

```python
def emit(self, ctx, node):
    return ctx.builder.emit_my_dialect_op(...)
```

---

## Module Dependency Graph

```
                    ┌─────────────────┐
                    │   __init__.py   │
                    │  (Public API)   │
                    └────────┬────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
         ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│  helion_mlir.py │ │ mlir_builder.py │ │ op_registry.py  │
│  (Main Entry)   │ │   (Emission)    │ │   (Registry)    │
└────────┬────────┘ └────────┬────────┘ └────────┬────────┘
         │                   │                   │
         │                   │                   │
         ▼                   ▼                   │
┌─────────────────────────────────────┐          │
│       lowering_context.py           │          │
│         (State Management)          │          │
└─────────────────────────────────────┘          │
                                                 │
                                                 ▼
                              ┌─────────────────────────────────┐
                              │        lowerings/               │
                              │  ┌──────────────────────────┐  │
                              │  │       base.py            │  │
                              │  │   (MLIRLowering ABC)     │  │
                              │  └──────────────────────────┘  │
                              │              ▲                  │
                              │              │                  │
                              │  ┌───────────┴───────────┐     │
                              │  │           │           │     │
                              │  ▼           ▼           ▼     │
                              │ memory_   control_   aten_     │
                              │ ops.py   flow.py    ops.py     │
                              │                                │
                              │         tracing_ops.py         │
                              └─────────────────────────────────┘
```

---

## Summary

The `helion_fx_mlir` package provides a clean, extensible architecture for converting Helion kernels to MLIR:

1. **`MLIRBuilder`** handles all text emission concerns
2. **`LoweringContext`** manages state throughout the process
3. **`LoweringRegistry`** provides dynamic dispatch to lowering implementations
4. **Individual lowerings** in `lowerings/` handle specific FX node types

This separation allows:
- Easy addition of new operation lowerings
- Clear understanding of the lowering process
- Testability of individual components
- Future extension to new MLIR dialects
