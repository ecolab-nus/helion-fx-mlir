"""Lowering implementations for FX-to-MLIR conversion.

This package contains the lowering implementations for converting
Helion FX graphs to MLIR. Each module handles a category of operations:

- base: Base class and utilities for lowerings
- memory_ops: Load/store operations (helion.language.memory_ops)
- control_flow: Control flow operations (_phi, _for_loop, etc.)
- aten_ops: PyTorch ATen operations
- tracing_ops: Internal tracing operations (_host_tensor, etc.)

All lowerings are automatically registered when this package is imported.
"""

# Import all lowering modules to trigger registration
from . import base
from . import memory_ops
from . import control_flow
from . import aten_ops
from . import tracing_ops

# Re-export base classes and utilities
from .base import (
    MLIRLowering,
    PassthroughLowering,
    CommentLowering,
)

from .control_flow import (
    ReductionLoopEmitter,
    ParallelLoopEmitter,
)

__all__ = [
    "MLIRLowering",
    "PassthroughLowering",
    "CommentLowering",
    "ReductionLoopEmitter",
    "ParallelLoopEmitter",
]
