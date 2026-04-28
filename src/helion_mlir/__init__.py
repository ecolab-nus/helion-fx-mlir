"""Utilities for Helion-to-MLIR lowering.

Public API is intentionally narrow. Internal symbols remain available via
compatibility exports but emit a deprecation warning when accessed.
"""

from __future__ import annotations

from warnings import warn

from .debug_utils import validate_with_mlir_opt
from .helion_mlir import generate_mlir

__all__ = [
    "generate_mlir",
    "validate_with_mlir_opt",
]

_COMPAT_EXPORTS: dict[str, tuple[str, str]] = {
    "IRVisitor": (".ir_visitor", "IRVisitor"),
    "MLIROutputHelper": (".mlir_utils", "MLIROutputHelper"),
    "torch_dtype_to_mlir_element_type": (".mlir_utils", "torch_dtype_to_mlir_element_type"),
    "format_tensor_type": (".mlir_utils", "format_tensor_type"),
    "format_shape_attr": (".mlir_utils", "format_shape_attr"),
    "format_string_attr": (".mlir_utils", "format_string_attr"),
    "format_attr_dict": (".mlir_utils", "format_attr_dict"),
    "LoweringContext": (".lowering_context", "LoweringContext"),
    "collect_reduction_block_ids": (".lowering_context", "collect_reduction_block_ids"),
    "TorchMLIRNodeImporter": (".torch_mlir_helper", "TorchMLIRNodeImporter"),
    "import_aten_node_to_mlir": (".torch_mlir_helper", "import_aten_node_to_mlir"),
    "print_debug_info": (".debug_utils", "print_debug_info"),
    "print_device_ir": (".debug_utils", "print_device_ir"),
    "print_nodes_with_symbols": (".debug_utils", "print_nodes_with_symbols"),
    "print_compile_env": (".debug_utils", "print_compile_env"),
    "run_dce_cleanup": (".debug_utils", "run_dce_cleanup"),
}


def __getattr__(name: str):
    target = _COMPAT_EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)

    module_name, attr_name = target
    warn(
        f"helion_mlir.{name} is deprecated as a top-level export and may be removed in a future release. "
        f"Import from {module_name} instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    module = __import__(f"helion_mlir{module_name}", fromlist=[attr_name])
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
