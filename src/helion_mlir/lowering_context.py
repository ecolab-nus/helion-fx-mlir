"""Compatibility exports for lowering state.

The lowering pipeline now uses:
- `build_kernel_analysis()` for immutable kernel facts
- `LoweringSession` for mutable lowering state

`LoweringContext` is kept as a compatibility shim for older imports.
"""

from __future__ import annotations

from .session import LoweringContext, LoweringSession, collect_reduction_block_ids

__all__ = [
    "LoweringContext",
    "LoweringSession",
    "collect_reduction_block_ids",
]
