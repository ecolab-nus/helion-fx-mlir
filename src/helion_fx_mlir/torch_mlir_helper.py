"""Torch-MLIR integration helper using FxImporter infrastructure.

This module provides utilities to convert ATen operations from Helion Device IR
into MLIR using torch-mlir's FxImporter. It supports generating either:
- Raw torch dialect MLIR
- Linalg-on-tensors MLIR (via automatic lowering)

Key Functions:
- import_aten_node: Import a single FX node using torch-mlir
- TorchMLIRNodeImporter: Class for importing FX nodes to MLIR text
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.fx as fx
from torch._ops import OpOverload

if TYPE_CHECKING:
    from .lowering_context import LoweringContext


def get_aten_op_info(target: Any) -> tuple[str, str]:
    """Extract ATen operation name and overload from target.
    
    Args:
        target: FX node target (typically torch._ops.OpOverload)
        
    Returns:
        Tuple of (op_name, overload) e.g., ("addmm", "default")
    """
    if isinstance(target, OpOverload):
        return target.__name__, target._overloadname
    
    # Fallback: parse from string representation
    target_str = str(target)
    if "aten." in target_str:
        parts = target_str.replace("aten::", "aten.").split(".")
        if len(parts) >= 2:
            op_name = parts[1]
            overload = parts[2] if len(parts) > 2 else "default"
            return op_name, overload
    
    return target_str, "default"


class TorchMLIRNodeImporter:
    """Imports FX nodes to MLIR using torch-mlir's FxImporter.
    
    This class wraps torch-mlir's infrastructure to convert individual 
    ATen operations to MLIR text that can be embedded in helion MLIR.
    """
    
    def __init__(self, output_type: str = "linalg-on-tensors"):
        """Initialize the importer.
        
        Args:
            output_type: Target MLIR dialect - "raw" for torch dialect,
                        "linalg-on-tensors" for linalg, etc.
        """
        self.output_type = output_type
        self._context = None
        self._importer = None
        
    def _ensure_initialized(self):
        """Lazily initialize torch-mlir context and importer."""
        if self._context is not None:
            return
            
        try:
            from torch_mlir import ir
            from torch_mlir.dialects import torch as torch_d
            from torch_mlir.extras.fx_importer import FxImporter
            
            self._context = ir.Context()
            torch_d.register_dialect(self._context)
            self._importer = FxImporter(context=self._context)
        except ImportError as e:
            raise RuntimeError(
                f"torch-mlir not available: {e}. "
                "Please install torch-mlir to use this functionality."
            )
    
    def import_graph(
        self,
        graph: fx.Graph,
        func_name: str = "aten_op",
    ) -> str:
        """Import an FX graph to MLIR.
        
        Args:
            graph: FX Graph to import
            func_name: Name for the generated MLIR function
            
        Returns:
            MLIR text representation
        """
        self._ensure_initialized()
        
        from torch_mlir.compiler_utils import OutputType, lower_mlir_module
        
        # Import the graph
        self._importer.import_stateless_graph(graph, func_name=func_name)
        
        # Get the module and lower if needed
        module = self._importer.module
        
        if self.output_type != "raw":
            module = lower_mlir_module(
                False,  # verbose
                OutputType.get(self.output_type),
                module
            )
        
        return str(module)
    
    def import_node(
        self,
        node: fx.Node,
        input_tensors: list[torch.Tensor],
    ) -> str:
        """Import a single FX node to MLIR by creating a minimal graph.
        
        Args:
            node: The FX node to import (should be an ATen op)
            input_tensors: List of tensor shapes/dtypes for inputs
            
        Returns:
            MLIR text for the operation
        """
        self._ensure_initialized()
        
        # Create a minimal FX graph containing just this node
        # This requires wrapping it in a proper graph structure
        graph = fx.Graph()
        
        # Create placeholder nodes for inputs
        placeholder_nodes = []
        for i, tensor in enumerate(input_tensors):
            placeholder = graph.placeholder(f"input_{i}")
            # Set metadata to help torch-mlir infer types
            placeholder.meta["val"] = tensor
            placeholder_nodes.append(placeholder)
        
        # Create the operation node
        op_node = graph.call_function(node.target, tuple(placeholder_nodes))
        
        # Try to infer output type from node metadata
        if "val" in node.meta:
            op_node.meta["val"] = node.meta["val"]
        
        # Create output
        graph.output(op_node)
        
        # Import and return
        return self.import_graph(graph, func_name="aten_op")


def create_fake_tensors_for_node(node: fx.Node) -> list[torch.Tensor]:
    """Create fake tensors for a node's inputs based on metadata.
    
    Args:
        node: FX node whose inputs need fake tensors
        
    Returns:
        List of fake tensors matching input shapes/dtypes
    """
    fake_tensors = []
    
    for arg in node.args:
        if isinstance(arg, fx.Node):
            # Try to get tensor metadata from the node
            if "val" in arg.meta and hasattr(arg.meta["val"], "shape"):
                val = arg.meta["val"]
                fake = torch.empty(val.shape, dtype=val.dtype, device="meta")
                fake_tensors.append(fake)
            else:
                # Default to dynamic 2D float32 tensor
                fake = torch.empty([1, 1], dtype=torch.float32, device="meta")
                fake_tensors.append(fake)
    
    return fake_tensors


def import_aten_node_to_mlir(
    node: fx.Node,
    output_type: str = "linalg-on-tensors",
) -> Optional[str]:
    """Import an ATen FX node to MLIR using torch-mlir.
    
    This is the main entry point for converting ATen operations.
    
    Args:
        node: FX node containing an ATen operation
        output_type: Target MLIR dialect ("raw", "linalg-on-tensors", "tosa", "stablehlo")
        
    Returns:
        MLIR text for the operation, or None if import fails
    """
    try:
        importer = TorchMLIRNodeImporter(output_type=output_type)
        fake_tensors = create_fake_tensors_for_node(node)
        return importer.import_node(node, fake_tensors)
    except Exception as e:
        # Log the error but don't fail
        import warnings
        warnings.warn(f"Failed to import ATen node {node.name}: {e}")
        return None


# Compatibility aliases
LinalgEmitter = TorchMLIRNodeImporter
