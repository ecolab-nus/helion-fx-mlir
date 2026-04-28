from __future__ import annotations


class HelionMLIRError(RuntimeError):
    """Base class for lowering-related runtime errors."""


class UnsupportedTargetError(HelionMLIRError):
    """Raised when a call_function target cannot be lowered."""


class MissingGraphError(HelionMLIRError):
    """Raised when expected graph metadata is missing."""
