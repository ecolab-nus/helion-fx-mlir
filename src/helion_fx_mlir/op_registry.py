"""Registry for mapping FX node targets to MLIR lowering implementations.

This module provides a registry pattern similar to Helion's own codegen
decorator system, allowing lowering implementations to be registered
for specific FX node targets.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, TypeVar

if TYPE_CHECKING:
    from .lowerings.base import MLIRLowering
    from .lowering_context import LoweringContext
    import torch.fx

_T = TypeVar("_T", bound="type[MLIRLowering]")


class LoweringRegistry:
    """Registry mapping FX node targets to MLIR lowering classes.
    
    This class provides a singleton registry that maps FX node targets
    (functions, methods, etc.) to their corresponding MLIR lowering
    implementations.
    
    Usage:
        @LoweringRegistry.register(target_function)
        class MyLowering(MLIRLowering):
            def emit(self, ctx: LoweringContext, node: torch.fx.Node) -> str | None:
                ...
    """
    
    _registry: dict[object, type["MLIRLowering"]] = {}
    _instances: dict[object, "MLIRLowering"] = {}
    
    @classmethod
    def register(cls, target: object) -> Callable[[_T], _T]:
        """Decorator to register a lowering class for an FX target.
        
        Args:
            target: The FX node target (function, method, etc.) to register for.
            
        Returns:
            A decorator that registers the lowering class.
            
        Example:
            @LoweringRegistry.register(memory_ops.load)
            class LoadLowering(MLIRLowering):
                ...
        """
        def decorator(lowering_cls: _T) -> _T:
            cls._registry[target] = lowering_cls
            return lowering_cls
        return decorator
    
    @classmethod
    def register_multiple(cls, *targets: object) -> Callable[[_T], _T]:
        """Decorator to register a lowering class for multiple FX targets.
        
        Args:
            *targets: Multiple FX node targets to register for.
            
        Returns:
            A decorator that registers the lowering class for all targets.
        """
        def decorator(lowering_cls: _T) -> _T:
            for target in targets:
                cls._registry[target] = lowering_cls
            return lowering_cls
        return decorator
    
    @classmethod
    def get(cls, target: object) -> type["MLIRLowering"] | None:
        """Get the lowering class for an FX target.
        
        Args:
            target: The FX node target to look up.
            
        Returns:
            The registered lowering class, or None if not found.
        """
        return cls._registry.get(target)
    
    @classmethod
    def get_instance(cls, target: object) -> "MLIRLowering | None":
        """Get a cached instance of the lowering class for an FX target.
        
        This method caches instances to avoid repeated instantiation.
        
        Args:
            target: The FX node target to look up.
            
        Returns:
            An instance of the registered lowering class, or None if not found.
        """
        if target in cls._instances:
            return cls._instances[target]
        
        lowering_cls = cls.get(target)
        if lowering_cls is None:
            return None
        
        instance = lowering_cls()
        cls._instances[target] = instance
        return instance
    
    @classmethod
    def has(cls, target: object) -> bool:
        """Check if a lowering is registered for an FX target.
        
        Args:
            target: The FX node target to check.
            
        Returns:
            True if a lowering is registered, False otherwise.
        """
        return target in cls._registry
    
    @classmethod
    def clear(cls) -> None:
        """Clear all registered lowerings and cached instances.
        
        This is primarily useful for testing.
        """
        cls._registry.clear()
        cls._instances.clear()
    
    @classmethod
    def list_registered(cls) -> list[object]:
        """List all registered FX targets.
        
        Returns:
            A list of all registered FX node targets.
        """
        return list(cls._registry.keys())
    
    @classmethod
    def emit_node(
        cls,
        ctx: "LoweringContext",
        node: "torch.fx.Node",
    ) -> str | None:
        """Emit MLIR for an FX node using the registered lowering.
        
        This is a convenience method that looks up the lowering and calls
        its emit method.
        
        Args:
            ctx: The lowering context.
            node: The FX node to emit.
            
        Returns:
            The result SSA value name, or None if no result.
            
        Raises:
            KeyError: If no lowering is registered for the node's target.
        """
        if node.op != "call_function":
            return None
        
        lowering = cls.get_instance(node.target)
        if lowering is None:
            # Not registered - could be handled by a default/fallback
            return None
        
        return lowering.emit(ctx, node)


def register_lowering(target: object) -> Callable[[_T], _T]:
    """Module-level convenience decorator for registering lowerings.
    
    This is equivalent to @LoweringRegistry.register(target).
    
    Example:
        @register_lowering(memory_ops.load)
        class LoadLowering(MLIRLowering):
            ...
    """
    return LoweringRegistry.register(target)


def register_lowering_multiple(*targets: object) -> Callable[[_T], _T]:
    """Module-level convenience decorator for registering lowerings for multiple targets.
    
    This is equivalent to @LoweringRegistry.register_multiple(*targets).
    """
    return LoweringRegistry.register_multiple(*targets)
