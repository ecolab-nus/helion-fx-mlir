from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

import helion
import helion.language as hl


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from helion_mlir import generate_mlir, validate_with_mlir_opt  # noqa: E402
from examples.scripts.matmul import matmul  # noqa: E402
from examples.scripts.mamba_chunk_scan import helion_mamba2_chunk_scan_kernel  # noqa: E402


@helion.kernel(static_shapes=False)
def _nonzero_lb_matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    k2, n = y.size()
    assert k == k2
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=x.dtype)
        for tile_k in hl.tile(1, k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = acc
    return out


def test_expression_bound_uses_scf_for() -> None:
    batch = 1
    seqlen = 128
    nheads = 4
    headdim = 16
    chunk_size = 64
    ngroups = 1
    dstate = 8
    nchunks = (seqlen + chunk_size - 1) // chunk_size

    cb = torch.randn([batch, nchunks, ngroups, chunk_size, chunk_size], dtype=torch.float16)
    x = torch.randn([batch, seqlen, nheads, headdim], dtype=torch.float16)
    dt = torch.randn([batch, nheads, nchunks, chunk_size], dtype=torch.float16)
    dA_cumsum = torch.randn([batch, nheads, nchunks, chunk_size], dtype=torch.float16)
    C = torch.randn([batch, seqlen, ngroups, dstate], dtype=torch.float16)
    prev_states = torch.randn([batch, nchunks, nheads, headdim, dstate], dtype=torch.float16)
    D = torch.randn([nheads], dtype=torch.float16)

    bound_kernel = helion_mamba2_chunk_scan_kernel.bind((cb, x, dt, dA_cumsum, C, prev_states, D))
    mlir_text = generate_mlir(bound_kernel)

    assert "scf.for" in mlir_text
    assert "affine.apply" not in mlir_text
    result = validate_with_mlir_opt(mlir_text)
    assert result.returncode == 0, result.stderr


def test_simple_bound_uses_scf_for_without_affine_apply() -> None:
    x = torch.randn([128, 64], dtype=torch.float16)
    y = torch.randn([64, 128], dtype=torch.float16)

    bound_kernel = matmul.bind((x, y))
    mlir_text = generate_mlir(bound_kernel)

    assert "scf.for" in mlir_text
    assert "affine.apply" not in mlir_text
    result = validate_with_mlir_opt(mlir_text)
    assert result.returncode == 0, result.stderr


def test_nonzero_lower_bound_block_loop_is_rejected() -> None:
    x = torch.randn([128, 64], dtype=torch.float16)
    y = torch.randn([64, 128], dtype=torch.float16)
    bound_kernel = _nonzero_lb_matmul.bind((x, y))

    with pytest.raises(ValueError, match="zero-lower-bound block loops"):
        generate_mlir(bound_kernel)
