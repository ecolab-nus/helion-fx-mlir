"""MLIR lowerings for Helion control flow operations.

This module provides lowerings for:
- helion.language._tracing_ops._phi -> helion.phi
- helion.language._tracing_ops._for_loop -> affine.for structure
- helion.language._tracing_ops._if -> scf.if
- helion.language._tracing_ops._while_loop -> scf.while
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import helion.language._tracing_ops as hl_tracing_ops

from ..op_registry import register_lowering
from ..mlir_builder import format_attr_dict, format_string_attr
from .base import MLIRLowering

if TYPE_CHECKING:
    import torch.fx
    from ..lowering_context import LoweringContext


@register_lowering(hl_tracing_ops._phi)
class PhiLowering(MLIRLowering):
    """Lowering for helion.language._tracing_ops._phi -> helion.phi.
    
    Phi nodes combine values from different control flow paths, typically
    used to merge values from inside and outside of loops.
    """
    
    def emit(self, ctx: "LoweringContext", node: "torch.fx.Node") -> str | None:
        """Emit helion.phi for a phi node.
        
        The operation signature is:
            %result = "helion.phi"(%lhs, %rhs){attrs}
                : (tensor_type, tensor_type) -> tensor_type
        """
        builder = ctx.builder
        
        # Get the two input values
        if len(node.args) < 2:
            builder.emit_comment(f"// Warning: phi node {node.name} has insufficient arguments")
            return None
        
        lhs_arg = node.args[0]
        rhs_arg = node.args[1]
        
        lhs_ssa = self._get_value_ssa(ctx, lhs_arg)
        rhs_ssa = self._get_value_ssa(ctx, rhs_arg)
        
        # Build attributes
        attrs = {}
        fx_node_name = node.name
        if fx_node_name:
            attrs["fx_node"] = format_string_attr(fx_node_name)
        
        # Emit the operation
        result = builder.fresh("phi")
        attrs_str = format_attr_dict(attrs)
        builder.emit(
            f'{result} = "helion.phi"({lhs_ssa}, {rhs_ssa}){attrs_str} '
            f": ({ctx.tensor_type}, {ctx.tensor_type}) -> {ctx.tensor_type}"
        )
        
        # Store the result
        ctx.fx_value_map[node.name] = result
        
        return result
    
    def _get_value_ssa(self, ctx: "LoweringContext", arg: object) -> str:
        """Get SSA value for a phi input."""
        import torch.fx
        
        if isinstance(arg, torch.fx.Node):
            if arg.name in ctx.fx_value_map:
                return ctx.fx_value_map[arg.name]
            # Try to infer from common patterns
            name_lower = arg.name.lower()
            if "acc" in name_lower or "init" in name_lower:
                return ctx.acc_seed or "%acc_init0"
        
        # Fallbacks
        if ctx.current_acc:
            return ctx.current_acc
        return "%v0"


@register_lowering(hl_tracing_ops._for_loop)
class ForLoopLowering(MLIRLowering):
    """Lowering for helion.language._tracing_ops._for_loop -> affine.for structure.
    
    For loops in Helion are traced as _for_loop ops with a graph ID that
    references the loop body. This lowering handles the structural emission
    of the loop but delegates body emission to the main lowering driver.
    
    Note: The actual loop body emission is handled by the main generation
    function, as it requires walking the referenced graph. This lowering
    primarily provides metadata and structural information.
    """
    
    def emit(self, ctx: "LoweringContext", node: "torch.fx.Node") -> str | None:
        """Emit a comment for _for_loop as the structure is handled by the main driver.
        
        The actual affine.for emission is done by the main generation function
        because it needs access to the referenced graph and loop bounds.
        """
        builder = ctx.builder
        
        # Extract graph ID and bounds from arguments
        graph_id = node.args[0] if node.args else None
        begin = node.args[1] if len(node.args) > 1 else None
        end = node.args[2] if len(node.args) > 2 else None
        
        # Emit a comment indicating the for loop
        builder.emit_comment(f"for_loop: {node.name} (graph_id={graph_id}, begin={begin}, end={end})")
        
        # The actual loop structure is emitted by the main driver
        # Store the node info for reference
        ctx.fx_value_map[node.name] = f"%for_result_{node.name}"
        
        return None


@register_lowering(hl_tracing_ops._if)
class IfLowering(MLIRLowering):
    """Lowering for helion.language._tracing_ops._if -> scf.if structure.
    
    Similar to ForLoopLowering, the actual structure is handled by the main
    driver, but this provides metadata handling.
    """
    
    def emit(self, ctx: "LoweringContext", node: "torch.fx.Node") -> str | None:
        """Emit a comment for _if as the structure is handled by the main driver."""
        builder = ctx.builder
        
        # Extract test and graph ID
        test = node.args[0] if node.args else None
        graph_id = node.args[1] if len(node.args) > 1 else None
        
        builder.emit_comment(f"if: {node.name} (graph_id={graph_id})")
        
        return None


@register_lowering(hl_tracing_ops._while_loop)
class WhileLoopLowering(MLIRLowering):
    """Lowering for helion.language._tracing_ops._while_loop -> scf.while structure.
    
    Similar to ForLoopLowering, provides metadata handling while the main
    driver handles structural emission.
    """
    
    def emit(self, ctx: "LoweringContext", node: "torch.fx.Node") -> str | None:
        """Emit a comment for _while_loop as the structure is handled by the main driver."""
        builder = ctx.builder
        
        cond_graph_id = node.args[0] if node.args else None
        body_graph_id = node.args[1] if len(node.args) > 1 else None
        
        builder.emit_comment(f"while_loop: {node.name} (cond={cond_graph_id}, body={body_graph_id})")
        
        return None


# Additional control flow helpers

class ReductionLoopEmitter:
    """Helper class for emitting reduction loop structures.
    
    This is not a registered lowering but a helper used by the main
    generation function to emit affine.for loops with iter_args for
    reduction patterns.
    """
    
    @staticmethod
    def emit_reduction_loop_start(
        ctx: "LoweringContext",
        loop_info: "LoopInfo",
        init_acc: str,
    ) -> tuple[str, str]:
        """Emit the start of a reduction loop.
        
        Args:
            ctx: The lowering context.
            loop_info: Information about the loop.
            init_acc: Initial accumulator SSA value.
            
        Returns:
            Tuple of (loop_result_ssa, iter_arg_ssa).
        """
        from ..lowering_context import LoopInfo
        
        builder = ctx.builder
        
        iv_name = f"%{loop_info.name}_iv"
        loop_info.iv_name = iv_name
        
        # Determine the upper bound
        if loop_info.trip_count_ssa:
            upper_bound = loop_info.trip_count_ssa
        elif loop_info.trip_count is not None:
            upper_bound = str(loop_info.trip_count)
        else:
            upper_bound = "1"
        
        # Emit the affine.for with iter_args
        loop_result = builder.fresh(f"{loop_info.name}_acc")
        iter_arg = "%acc_iter"
        
        builder.emit(
            f"{loop_result} = affine.for {iv_name} = 0 to {upper_bound} "
            f"iter_args({iter_arg} = {init_acc}) -> ({ctx.tensor_type}) {{"
        )
        builder.push()
        
        return loop_result, iter_arg
    
    @staticmethod
    def emit_reduction_loop_end(
        ctx: "LoweringContext",
        yield_value: str,
    ) -> None:
        """Emit the end of a reduction loop.
        
        Args:
            ctx: The lowering context.
            yield_value: SSA value to yield.
        """
        builder = ctx.builder
        builder.emit(f"affine.yield {yield_value} : {ctx.tensor_type}")
        builder.pop()
        builder.emit("}")


class ParallelLoopEmitter:
    """Helper class for emitting parallel loop structures.
    
    This is not a registered lowering but a helper used by the main
    generation function to emit affine.parallel loops for grid-parallel
    iteration.
    """
    
    @staticmethod
    def emit_parallel_loop_start(
        ctx: "LoweringContext",
        loops: list["LoopInfo"],
    ) -> list[str]:
        """Emit the start of a parallel loop.
        
        Args:
            ctx: The lowering context.
            loops: List of loop info for each parallel dimension.
            
        Returns:
            List of induction variable SSA names.
        """
        from ..lowering_context import LoopInfo
        
        builder = ctx.builder
        
        iv_names = []
        lower_bounds = []
        upper_bounds = []
        steps = []
        
        for loop in loops:
            iv_name = f"%{loop.name}_iv"
            loop.iv_name = iv_name
            iv_names.append(iv_name)
            
            lower_bounds.append("0")
            
            if loop.trip_count_ssa:
                upper_bounds.append(loop.trip_count_ssa)
            elif loop.trip_count is not None:
                upper_bounds.append(str(loop.trip_count))
            else:
                upper_bounds.append("1")
            
            steps.append("1")
        
        builder.emit_affine_parallel_start(iv_names, lower_bounds, upper_bounds, steps)
        
        return iv_names
    
    @staticmethod
    def emit_parallel_loop_end(ctx: "LoweringContext") -> None:
        """Emit the end of a parallel loop."""
        ctx.builder.emit_affine_parallel_end()
