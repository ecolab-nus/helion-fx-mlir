"""MLIR lowerings for PyTorch ATen operations.

This module provides lowerings for ATen operations to helion.call_torch,
which represents a placeholder for torch operations that will be lowered
in a later compilation stage.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from torch.ops import aten

from ..op_registry import register_lowering, register_lowering_multiple
from ..mlir_builder import format_attr_dict, format_string_attr
from .base import MLIRLowering

if TYPE_CHECKING:
    import torch.fx
    from ..lowering_context import LoweringContext


class AtenOpLowering(MLIRLowering):
    """Base lowering for ATen operations -> helion.call_torch.
    
    This class handles the generic case of lowering ATen operations to
    the helion.call_torch placeholder operation, which preserves the
    operation semantics while deferring actual code generation.
    """
    
    def __init__(self, op_name: str | None = None):
        """Initialize with an optional explicit op name.
        
        Args:
            op_name: Override name for the operation. If None, derived from target.
        """
        self._op_name = op_name
    
    def emit(self, ctx: "LoweringContext", node: "torch.fx.Node") -> str | None:
        """Emit helion.call_torch for an ATen operation.
        
        The operation signature is:
            %result = "helion.call_torch"(%arg0, %arg1, ...){fn_name = "aten.op"}
                : (type0, type1, ...) -> result_type
        """
        builder = ctx.builder
        
        # Get the operation name
        op_name = self._get_op_name(node)
        
        # Collect operand SSA values
        operands = []
        operand_types = []
        
        for arg in node.args:
            ssa = self._get_operand_ssa(ctx, arg)
            if ssa is not None:
                operands.append(ssa)
                operand_types.append(self._get_operand_type(ctx, arg))
        
        # Build attributes
        attrs = {
            "fn_name": format_string_attr(op_name),
        }
        
        # Add FX node name for traceability
        if node.name:
            attrs["fx_node"] = format_string_attr(node.name)
        
        # Determine result type
        result_type = self._get_result_type(ctx, node)
        
        # Emit the operation
        result = builder.fresh("call")
        attrs_str = format_attr_dict(attrs)
        operands_str = ", ".join(operands)
        types_str = ", ".join(operand_types)
        
        builder.emit(
            f'{result} = "helion.call_torch"({operands_str}){attrs_str} '
            f": ({types_str}) -> {result_type}"
        )
        
        # Store the result
        ctx.fx_value_map[node.name] = result
        
        return result
    
    def _get_op_name(self, node: "torch.fx.Node") -> str:
        """Get the operation name for the call_torch attribute."""
        if self._op_name:
            return self._op_name
        
        target = node.target
        if hasattr(target, "__name__"):
            return f"aten.{target.__name__}"
        elif hasattr(target, "name"):
            return f"aten.{target.name()}"
        else:
            return f"aten.{str(target)}"
    
    def _get_operand_ssa(self, ctx: "LoweringContext", arg: Any) -> str | None:
        """Get SSA value for an operand."""
        import torch.fx
        import torch
        
        if isinstance(arg, torch.fx.Node):
            if arg.name in ctx.fx_value_map:
                return ctx.fx_value_map[arg.name]
            # Check for common argument patterns
            if arg.op == "placeholder":
                # Map placeholder names to function arguments
                if arg.name.startswith("arg"):
                    return f"%{arg.name}"
            return f"%{arg.name}"
        elif isinstance(arg, (int, float)):
            # For numeric constants, emit as constant ops
            return ctx.builder.emit_index_constant(int(arg))
        elif isinstance(arg, torch.Tensor):
            # Shouldn't happen often, but handle gracefully
            return None
        elif arg is None:
            return None
        else:
            return None
    
    def _get_operand_type(self, ctx: "LoweringContext", arg: Any) -> str:
        """Get the MLIR type for an operand."""
        import torch.fx
        import torch
        
        if isinstance(arg, torch.fx.Node):
            # Try to get type from metadata
            val = arg.meta.get("val")
            if isinstance(val, torch.Tensor):
                from ..mlir_builder import torch_dtype_to_mlir_element_type, format_tensor_type
                element_type = torch_dtype_to_mlir_element_type(val.dtype)
                shape = [None] * val.ndim  # Use dynamic dimensions
                return format_tensor_type(shape, element_type)
        
        # Default to the context's tensor type
        return ctx.tensor_type
    
    def _get_result_type(self, ctx: "LoweringContext", node: "torch.fx.Node") -> str:
        """Get the MLIR result type for the operation."""
        import torch
        
        val = node.meta.get("val")
        if isinstance(val, torch.Tensor):
            from ..mlir_builder import torch_dtype_to_mlir_element_type, format_tensor_type
            element_type = torch_dtype_to_mlir_element_type(val.dtype)
            shape = [None] * val.ndim
            return format_tensor_type(shape, element_type)
        
        return ctx.tensor_type


# Register commonly used ATen operations
# Each is registered to use the generic AtenOpLowering with appropriate op name

@register_lowering(aten.addmm.default)
class AddmmLowering(AtenOpLowering):
    """Lowering for aten.addmm (matrix multiply with add)."""
    
    def __init__(self):
        super().__init__("aten.addmm")


@register_lowering(aten.mm.default)
class MmLowering(AtenOpLowering):
    """Lowering for aten.mm (matrix multiply)."""
    
    def __init__(self):
        super().__init__("aten.mm")


@register_lowering(aten.add.Tensor)
class AddTensorLowering(AtenOpLowering):
    """Lowering for aten.add.Tensor."""
    
    def __init__(self):
        super().__init__("aten.add")


@register_lowering(aten.mul.Tensor)
class MulTensorLowering(AtenOpLowering):
    """Lowering for aten.mul.Tensor."""
    
    def __init__(self):
        super().__init__("aten.mul")


@register_lowering(aten.sub.Tensor)
class SubTensorLowering(AtenOpLowering):
    """Lowering for aten.sub.Tensor."""
    
    def __init__(self):
        super().__init__("aten.sub")


@register_lowering(aten.div.Tensor)
class DivTensorLowering(AtenOpLowering):
    """Lowering for aten.div.Tensor."""
    
    def __init__(self):
        super().__init__("aten.div")


@register_lowering(aten.exp.default)
class ExpLowering(AtenOpLowering):
    """Lowering for aten.exp."""
    
    def __init__(self):
        super().__init__("aten.exp")


@register_lowering(aten.log.default)
class LogLowering(AtenOpLowering):
    """Lowering for aten.log."""
    
    def __init__(self):
        super().__init__("aten.log")


@register_lowering(aten.sqrt.default)
class SqrtLowering(AtenOpLowering):
    """Lowering for aten.sqrt."""
    
    def __init__(self):
        super().__init__("aten.sqrt")


@register_lowering(aten.relu.default)
class ReluLowering(AtenOpLowering):
    """Lowering for aten.relu."""
    
    def __init__(self):
        super().__init__("aten.relu")


@register_lowering(aten.sigmoid.default)
class SigmoidLowering(AtenOpLowering):
    """Lowering for aten.sigmoid."""
    
    def __init__(self):
        super().__init__("aten.sigmoid")


@register_lowering(aten.tanh.default)
class TanhLowering(AtenOpLowering):
    """Lowering for aten.tanh."""
    
    def __init__(self):
        super().__init__("aten.tanh")


@register_lowering(aten.sum.default)
class SumLowering(AtenOpLowering):
    """Lowering for aten.sum."""
    
    def __init__(self):
        super().__init__("aten.sum")


@register_lowering(aten.sum.dim_IntList)
class SumDimLowering(AtenOpLowering):
    """Lowering for aten.sum.dim_IntList."""
    
    def __init__(self):
        super().__init__("aten.sum")


@register_lowering(aten.max.default)
class MaxLowering(AtenOpLowering):
    """Lowering for aten.max."""
    
    def __init__(self):
        super().__init__("aten.max")


@register_lowering(aten.softmax.int)
class SoftmaxLowering(AtenOpLowering):
    """Lowering for aten.softmax.int."""
    
    def __init__(self):
        super().__init__("aten.softmax")


# Fallback for unregistered ATen ops
class GenericAtenLowering(AtenOpLowering):
    """Generic lowering for any ATen operation not specifically registered.
    
    This can be used as a fallback handler for operations that don't have
    a specific lowering implementation.
    """
    
    def __init__(self):
        super().__init__(None)  # Will derive name from node.target
