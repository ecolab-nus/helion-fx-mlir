from __future__ import annotations

from .compute import register_compute_handlers
from .control_flow import register_control_flow_handlers
from .memory import register_memory_handlers
from .symbols import register_symbol_handlers
from .tensor_ops import register_tensor_handlers
from .tile import register_tile_handlers

__all__ = [
    "register_compute_handlers",
    "register_control_flow_handlers",
    "register_memory_handlers",
    "register_symbol_handlers",
    "register_tensor_handlers",
    "register_tile_handlers",
]
