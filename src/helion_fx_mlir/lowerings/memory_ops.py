"""MLIR lowerings for Helion memory operations.

This module provides lowerings for:
- helion.language.memory_ops.load -> helion.load_tile_dynamic
- helion.language.memory_ops.store -> helion.store_tile_dynamic
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import helion.language.memory_ops as hl_memory_ops

from ..op_registry import register_lowering
from ..mlir_builder import format_attr_dict, format_string_attr, format_indices_attr
from .base import MLIRLowering

if TYPE_CHECKING:
    import torch.fx
    from ..lowering_context import LoweringContext


@register_lowering(hl_memory_ops.load)
class LoadTileLowering(MLIRLowering):
    """Lowering for helion.language.memory_ops.load -> helion.load_tile_dynamic.
    
    This lowering handles tensor tile loads in Helion kernels, converting
    them to the helion.load_tile_dynamic MLIR operation with dynamic
    tile size information.
    """
    
    def emit(self, ctx: "LoweringContext", node: "torch.fx.Node") -> str | None:
        """Emit helion.load_tile_dynamic for a load operation.
        
        The operation signature is:
            %result = "helion.load_tile_dynamic"(%tensor, %size0, %size1){attrs}
                : (tensor_type, index, index) -> tensor_type
        """
        builder = ctx.builder
        
        # Get the tensor being loaded from
        tensor_arg = node.args[0] if node.args else None
        if tensor_arg is None:
            builder.emit_comment(f"// Warning: load node {node.name} has no tensor argument")
            return None
        
        # Determine which tensor argument this is (arg0 or arg1)
        tensor_ssa = self._get_tensor_ssa(ctx, tensor_arg)
        
        # Get tile sizes - these should come from the context
        size0, size1 = self._get_tile_sizes(ctx, node)
        
        # Get indices for the tile attribute
        indices = self._get_tile_indices(ctx, node)
        
        # Build attributes
        attrs = {
            "tile": format_indices_attr(indices),
            "sizes": ctx.tile_shape_attr,
            "tensor_meta": self._get_tensor_meta(ctx, size0, size1),
        }
        
        # Add FX node name attribute if available
        fx_node_name = node.name
        if fx_node_name:
            attrs["fx_node"] = format_string_attr(fx_node_name)
        
        # Emit the operation
        result = builder.fresh("load")
        attrs_str = format_attr_dict(attrs)
        builder.emit(
            f'{result} = "helion.load_tile_dynamic"({tensor_ssa}, {size0}, {size1}){attrs_str} '
            f": ({ctx.tensor_type}, index, index) -> {ctx.tensor_type}"
        )
        
        # Store the result in the FX value map
        ctx.fx_value_map[node.name] = result
        
        return result
    
    def _get_tensor_ssa(self, ctx: "LoweringContext", tensor_arg: object) -> str:
        """Get the SSA value for the tensor argument."""
        import torch.fx
        
        if isinstance(tensor_arg, torch.fx.Node):
            # Check if it's mapped to a function argument
            if tensor_arg.name in ctx.fx_value_map:
                return ctx.fx_value_map[tensor_arg.name]
            # Default to arg0 or arg1 based on naming convention
            if "x" in tensor_arg.name.lower() or tensor_arg.name == "arg0":
                return "%arg0"
            elif "y" in tensor_arg.name.lower() or tensor_arg.name == "arg1":
                return "%arg1"
        return "%arg0"  # Default
    
    def _get_tile_sizes(
        self, ctx: "LoweringContext", node: "torch.fx.Node"
    ) -> tuple[str, str]:
        """Determine tile sizes for this load operation."""
        # Use context's outer_tile_sizes if available
        # This is a simplified implementation - actual implementation
        # would need to analyze the load indices to determine which
        # dimensions are being loaded
        
        loop_map = ctx.get_loop_map()
        
        # Check if we're loading LHS (tile_m x tile_k) or RHS (tile_k x tile_n)
        # by inspecting the FX node's name or usage pattern
        node_name = node.name.lower()
        
        if "lhs" in node_name or "x" in node_name:
            size0 = self._choose_tile_size(ctx, loop_map, "tile_m")
            size1 = self._choose_tile_size(ctx, loop_map, "tile_k")
        else:
            size0 = self._choose_tile_size(ctx, loop_map, "tile_k")
            size1 = self._choose_tile_size(ctx, loop_map, "tile_n")
        
        return size0, size1
    
    def _choose_tile_size(
        self,
        ctx: "LoweringContext",
        loop_map: dict,
        key: str,
    ) -> str:
        """Choose the appropriate tile size SSA value."""
        if key in ctx.outer_tile_sizes:
            return ctx.outer_tile_sizes[key]
        
        loop = loop_map.get(key)
        if loop and loop.tile_const:
            return loop.tile_const
        
        # Fallback to emitting a constant
        if loop and loop.tile_size is not None:
            return ctx.builder.emit_index_constant(loop.tile_size)
        
        return ctx.builder.emit_index_constant(0)
    
    def _get_tile_indices(
        self, ctx: "LoweringContext", node: "torch.fx.Node"
    ) -> list[str]:
        """Get the tile indices for this load."""
        # For matmul pattern, indices depend on whether this is LHS or RHS
        # LHS: [outer_iv_m, reduction_iv]
        # RHS: [reduction_iv, outer_iv_n]
        
        outer_ivs = [f"%{loop.name}_iv" for loop in ctx.outer_loops]
        outer_iv_m = outer_ivs[0] if outer_ivs else "%tile_m_iv"
        outer_iv_n = outer_ivs[1] if len(outer_ivs) > 1 else "%tile_n_iv"
        
        reduction_iv = None
        if ctx.reduction_loops:
            reduction_iv = f"%{ctx.reduction_loops[0].name}_iv"
        else:
            reduction_iv = "%tile_k_iv"
        
        node_name = node.name.lower()
        if "lhs" in node_name or "x" in node_name:
            return [outer_iv_m, reduction_iv]
        else:
            return [reduction_iv, outer_iv_n]
    
    def _get_tensor_meta(self, ctx: "LoweringContext", size0: str, size1: str) -> str:
        """Format tensor metadata for the load."""
        from ..mlir_builder import format_dynamic_tensor_meta
        return format_dynamic_tensor_meta(size0, size1, ctx.element_type)


@register_lowering(hl_memory_ops.store)
class StoreTileLowering(MLIRLowering):
    """Lowering for helion.language.memory_ops.store -> helion.store_tile_dynamic.
    
    This lowering handles tensor tile stores in Helion kernels, converting
    them to the helion.store_tile_dynamic MLIR operation.
    """
    
    def emit(self, ctx: "LoweringContext", node: "torch.fx.Node") -> str | None:
        """Emit helion.store_tile_dynamic for a store operation.
        
        The operation signature is:
            "helion.store_tile_dynamic"(%tensor, %value, %size0, %size1){attrs}
                : (tensor_type, tensor_type, index, index) -> ()
        """
        builder = ctx.builder
        
        # Get the tensor being stored to
        tensor_arg = node.args[0] if node.args else None
        if tensor_arg is None:
            builder.emit_comment(f"// Warning: store node {node.name} has no tensor argument")
            return None
        
        tensor_ssa = self._get_tensor_ssa(ctx, tensor_arg)
        
        # Get the value being stored
        value_arg = node.args[2] if len(node.args) > 2 else None
        value_ssa = self._get_value_ssa(ctx, value_arg)
        
        # Get tile sizes
        size0, size1 = self._get_tile_sizes(ctx)
        
        # Get indices for the tile attribute
        indices = self._get_tile_indices(ctx)
        
        # Build attributes
        attrs = {
            "tile": format_indices_attr(indices),
            "sizes": ctx.tile_shape_attr,
            "tensor_meta": self._get_tensor_meta(ctx, size0, size1),
        }
        
        # Add FX node name attribute if available
        fx_node_name = node.name
        if fx_node_name:
            attrs["fx_node"] = format_string_attr(fx_node_name)
        
        # Emit the operation (no result)
        attrs_str = format_attr_dict(attrs)
        builder.emit(
            f'"helion.store_tile_dynamic"({tensor_ssa}, {value_ssa}, {size0}, {size1}){attrs_str} '
            f": ({ctx.tensor_type}, {ctx.tensor_type}, index, index) -> ()"
        )
        
        return None  # Store has no result
    
    def _get_tensor_ssa(self, ctx: "LoweringContext", tensor_arg: object) -> str:
        """Get the SSA value for the output tensor."""
        import torch.fx
        
        if isinstance(tensor_arg, torch.fx.Node):
            if tensor_arg.name in ctx.fx_value_map:
                return ctx.fx_value_map[tensor_arg.name]
        
        # Default to the output tensor from context
        return ctx.out_value or "%out0"
    
    def _get_value_ssa(self, ctx: "LoweringContext", value_arg: object) -> str:
        """Get the SSA value being stored."""
        import torch.fx
        
        if isinstance(value_arg, torch.fx.Node):
            if value_arg.name in ctx.fx_value_map:
                return ctx.fx_value_map[value_arg.name]
        
        # Default to current accumulator
        return ctx.current_acc or "%acc0"
    
    def _get_tile_sizes(self, ctx: "LoweringContext") -> tuple[str, str]:
        """Determine tile sizes for the store operation."""
        loop_map = ctx.get_loop_map()
        
        size0 = self._choose_tile_size(ctx, loop_map, "tile_m")
        size1 = self._choose_tile_size(ctx, loop_map, "tile_n")
        
        return size0, size1
    
    def _choose_tile_size(
        self,
        ctx: "LoweringContext",
        loop_map: dict,
        key: str,
    ) -> str:
        """Choose the appropriate tile size SSA value."""
        if key in ctx.outer_tile_sizes:
            return ctx.outer_tile_sizes[key]
        
        loop = loop_map.get(key)
        if loop and loop.tile_const:
            return loop.tile_const
        
        if loop and loop.tile_size is not None:
            return ctx.builder.emit_index_constant(loop.tile_size)
        
        return ctx.builder.emit_index_constant(0)
    
    def _get_tile_indices(self, ctx: "LoweringContext") -> list[str]:
        """Get the tile indices for the store."""
        outer_ivs = [f"%{loop.name}_iv" for loop in ctx.outer_loops]
        outer_iv_m = outer_ivs[0] if outer_ivs else "%tile_m_iv"
        outer_iv_n = outer_ivs[1] if len(outer_ivs) > 1 else "%tile_n_iv"
        return [outer_iv_m, outer_iv_n]
    
    def _get_tensor_meta(self, ctx: "LoweringContext", size0: str, size1: str) -> str:
        """Format tensor metadata for the store."""
        from ..mlir_builder import format_dynamic_tensor_meta
        return format_dynamic_tensor_meta(size0, size1, ctx.element_type)
