"""Utilities for Helion-to-MLIR lowering.

This package provides modular infrastructure for converting Helion kernels
(via their Device IR and FX graphs) to MLIR text representation.

Main entry points:
- generate_plan_stage0_mlir: Generate MLIR from a bound Helion kernel
- validate_with_helion_opt: Validate emitted MLIR with helion-opt or mlir-opt

Modular architecture:
- MLIRBuilder: Text emission and SSA naming
- LoweringContext: State management during lowering
- LoweringRegistry: Maps FX targets to lowering implementations
- lowerings: Individual lowering implementations by op category
"""

# Main entry points
from .helion_mlir import generate_plan_stage0_mlir, validate_with_helion_opt

# Core infrastructure (for extending with new lowerings)
from .mlir_builder import (
    MLIRBuilder,
    is_concrete_size,
    torch_dtype_to_mlir_element_type,
    format_tensor_type,
    format_shape_attr,
    format_string_attr,
    format_attr_dict,
)

from .lowering_context import (
    LoweringContext,
    LoopInfo,
    KernelArgInfo,
    first_debug_name,
    resolve_extent,
    collect_reduction_block_ids,
)

from .op_registry import (
    LoweringRegistry,
    register_lowering,
    register_lowering_multiple,
)

# Base lowering class for custom implementations
from .lowerings.base import (
    MLIRLowering,
    PassthroughLowering,
    CommentLowering,
)

__all__ = [
    # Main entry points
    "generate_plan_stage0_mlir",
    "validate_with_helion_opt",
    # Builder and utilities
    "MLIRBuilder",
    "is_concrete_size",
    "torch_dtype_to_mlir_element_type",
    "format_tensor_type",
    "format_shape_attr",
    "format_string_attr",
    "format_attr_dict",
    # Lowering context
    "LoweringContext",
    "LoopInfo",
    "KernelArgInfo",
    "first_debug_name",
    "resolve_extent",
    "collect_reduction_block_ids",
    # Registry
    "LoweringRegistry",
    "register_lowering",
    "register_lowering_multiple",
    # Base lowering classes
    "MLIRLowering",
    "PassthroughLowering",
    "CommentLowering",
]
