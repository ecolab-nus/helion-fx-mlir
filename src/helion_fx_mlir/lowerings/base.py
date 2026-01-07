"""Base class for MLIR lowering implementations.

This module defines the abstract base class that all FX-to-MLIR lowering
implementations must inherit from.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch.fx
    from ..lowering_context import LoweringContext


class MLIRLowering(ABC):
    """Abstract base class for MLIR lowering implementations.
    
    Each lowering implementation handles the conversion of a specific
    FX node target to MLIR operations. Implementations should be
    registered using the LoweringRegistry decorator.
    
    Example:
        @LoweringRegistry.register(memory_ops.load)
        class LoadLowering(MLIRLowering):
            def emit(self, ctx: LoweringContext, node: torch.fx.Node) -> str | None:
                # Emit helion.load_tile_dynamic
                ...
                return result_ssa_name
    """
    
    @abstractmethod
    def emit(self, ctx: "LoweringContext", node: "torch.fx.Node") -> str | None:
        """Emit MLIR for the given FX node.
        
        Args:
            ctx: The lowering context containing builder and state.
            node: The FX node to lower.
            
        Returns:
            The SSA value name of the result, or None if the operation
            has no result (e.g., store operations).
        """
        ...
    
    def get_operand_ssa(self, ctx: "LoweringContext", arg: object) -> str | None:
        """Get the SSA value for an FX node argument.
        
        This helper method handles various types of FX node arguments
        and returns the corresponding MLIR SSA value.
        
        Args:
            ctx: The lowering context.
            arg: The FX node argument (could be a Node, constant, etc.)
            
        Returns:
            The SSA value name, or None if not found.
        """
        import torch.fx
        
        if isinstance(arg, torch.fx.Node):
            # Look up the SSA value from the context's value map
            return ctx.fx_value_map.get(arg.name)
        elif isinstance(arg, (int, float)):
            # Emit a constant for numeric values
            return ctx.builder.emit_index_constant(int(arg))
        elif isinstance(arg, str):
            # String arguments are typically attribute values, not SSA values
            return None
        else:
            return None
    
    def get_node_name(self, node: "torch.fx.Node") -> str:
        """Get a sanitized name from an FX node for use in MLIR.
        
        Args:
            node: The FX node.
            
        Returns:
            A sanitized name string.
        """
        name = node.name
        # Replace characters that aren't valid in MLIR identifiers
        return name.replace(".", "_").replace("-", "_")
    
    def get_tensor_from_node(self, node: "torch.fx.Node") -> "torch.Tensor | None":
        """Get the fake tensor from an FX node's metadata.
        
        Args:
            node: The FX node.
            
        Returns:
            The fake tensor from the node's "val" metadata, or None.
        """
        import torch
        val = node.meta.get("val")
        if isinstance(val, torch.Tensor):
            return val
        return None


class PassthroughLowering(MLIRLowering):
    """A lowering that simply passes through, emitting nothing.
    
    This is useful for FX nodes that don't need explicit MLIR emission,
    such as placeholder nodes or nodes whose values are already available.
    """
    
    def emit(self, ctx: "LoweringContext", node: "torch.fx.Node") -> str | None:
        # No emission needed
        return None


class CommentLowering(MLIRLowering):
    """A lowering that emits a comment for debugging/documentation.
    
    This is useful for FX nodes that we want to track but don't
    directly lower to MLIR operations.
    """
    
    def __init__(self, comment_prefix: str = "FX node"):
        self.comment_prefix = comment_prefix
    
    def emit(self, ctx: "LoweringContext", node: "torch.fx.Node") -> str | None:
        target_name = getattr(node.target, "__name__", str(node.target))
        ctx.builder.emit_comment(f"{self.comment_prefix}: {node.name} ({target_name})")
        return None
