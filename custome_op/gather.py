"""Custom gather op registered with Helion's decorator API.

gather(tile, src) collects `src` across every iteration of `tile`,
returning a tensor of shape [N, *src.shape] where N = ceil(extent / block_size).

This is a minimal spike for MLIR lowering verification only.
Triton codegen and ref-mode are intentionally omitted.
"""
from __future__ import annotations

import torch
from torch.fx import has_side_effect

from helion import exc
from helion.language import _decorators


@has_side_effect
@_decorators.api(allow_host_tensor=False, tiles_as_sizes=True)
def gather(tile: object, src: torch.Tensor) -> torch.Tensor:
    """Collect ``src`` across every iteration of ``tile``.

    Args:
        tile: A Tile proxy (e.g. ``tile_k`` from ``for tile_k in hl.tile(k)``).
              With ``tiles_as_sizes=True``, Helion automatically converts this
              to the tile's block_size SymInt before tracing, so the FX graph
              node carries the block_size symnode as its first argument.
              The MLIR lowering uses this symnode to recover the block_id and
              compute the trip count N = ceil(extent / block_size).
        src:  A device-local tensor (shape ``S``) computed within that tile
              iteration (e.g. a partial matmul accumulator).

    Returns:
        A device tensor of shape ``[N, *S]`` stacking all tile contributions.
    """
    raise exc.NotInsideKernel


@_decorators.register_fake(gather)
def _(tile: torch.SymInt, src: torch.Tensor) -> torch.Tensor:
    # NOTE: The `tile` (block_size SymInt) is used as a stand-in for the leading
    # dimension (the trip count N = ceil(extent / block_size)) during FX tracing.
    #
    # While logically incorrect (the result rank is correct but the leading dim 
    # size is off), this ensures that:
    #   (a) The FakeTensor carries a dynamic first dim.
    #   (b) Downstream ATen consumers (e.g. operators mapping to linalg.generic)
    #       correctly infer rank-3 / rank-4 output types.
    #
    # The Helion MLIR Frontend's IRVisitor.visit_gather() and visit_aten_compute()
    # are responsible for overriding this dimension with the actual trip-count SSA
    # value during lowering to ensure MLIR rank and shape consistency.
    return src.new_empty((tile, *src.shape))
