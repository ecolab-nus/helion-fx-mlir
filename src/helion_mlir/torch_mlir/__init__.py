"""Internal torch-mlir adapter surface.

New code should prefer importing from this package instead of reaching into
`torch_mlir_helper.py` directly. The legacy helper module remains as a
compatibility facade for existing callers.
"""

from .importer import TorchMLIRNodeImporter, create_fake_tensors_for_node, get_cached_context
from .inline import inline_torch_mlir_output
from .rewrites import rewrite_batch_matmul_f16_accumulation, rewrite_linalg_generic_scalars

__all__ = [
    "TorchMLIRNodeImporter",
    "create_fake_tensors_for_node",
    "get_cached_context",
    "inline_torch_mlir_output",
    "rewrite_batch_matmul_f16_accumulation",
    "rewrite_linalg_generic_scalars",
]
