"""
Helion Split-K Matmul with Barrier Example
===========================================
This example demonstrates a two-stage split-K matmul using hl.barrier()
for deterministic cross-block reduction (as opposed to hl.atomic_add).

The helion-mlir lowering emits two ``affine.parallel`` regions separated by
a barrier, with block sizes shared across grids for the same source dimensions.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

import helion
import helion.language as hl
from helion._testing import DEVICE
from helion._testing import run_example
from helion.autotuner import PowerOfTwoFragment

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from helion_mlir import print_debug_info, generate_mlir, validate_with_mlir_opt

@helion.kernel(static_shapes=False, dot_precision="ieee")
def split_k_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Two-stage split-K matmul using hl.barrier().  The barrier approach
    gives deterministic results as opposed to the atomic_add approach.

    Stage 1:
      - Split K into `split_k` contiguous chunks.
      - Each chunk computes a partial [tile_m, tile_n] product into its own slice of `tmp`.

    Barrier:
      - Grid-wide barrier to ensure all partials are written before reduction.

    Stage 2:
      - Reduce partials across the split dimension and write `out`.

    Shapes:
      a: [M, K]
      b: [K, N]
      tmp: [M, N, split_k]
      out: [M, N]

    Notes:
      - Static shapes keep codegen simpler.
      - `split_k` is fixed for clarity; autotuning could choose it instead.
    """
    m, k = a.shape
    _, n = b.shape
    split_k = hl.register_tunable("split_k", PowerOfTwoFragment(16, 512, 64))
    block_k = helion.next_power_of_2(helion.cdiv(k, split_k))
    tmp = torch.zeros((m, n, split_k), device=a.device, dtype=a.dtype)
    out = torch.empty((m, n), device=a.device, dtype=a.dtype)

    for tile_m, tile_n, tile_k_outer in hl.tile(
        [m, n, k], block_size=[None, None, block_k]
    ):
        acc = hl.zeros([tile_m, tile_n], device=a.device, dtype=a.dtype)
        for tile_k_inner in hl.tile(tile_k_outer.begin, tile_k_outer.end):
            acc = torch.addmm(acc, a[tile_m, tile_k_inner], b[tile_k_inner, tile_n])
        # this could be a hl.atomic_add to avoid the barrier, but that would be non-determinstic
        tmp[tile_m, tile_n, tile_k_outer.id] = acc

    hl.barrier()

    for tile_m, tile_n in hl.tile([m, n]):
        out[tile_m, tile_n] = torch.sum(tmp[tile_m, tile_n, :], dim=-1)

    return out


def main() -> None:
    m, k, n = 256, 4096, 256
    a = torch.randn([m, k], device="cpu", dtype=torch.float32)
    b = torch.randn([k, n], device="cpu", dtype=torch.float32)
    bound_kernel = split_k_matmul.bind((a, b))

    print_debug_info(bound_kernel)

    # print("=== Config Spec ===")
    # print(bound_kernel.config_spec)
    # print()

    # config = helion.Config(
    #     block_sizes=[64, 64, 64, 64, 64],
    #     num_warps=4,
    #     num_stages=2,
    #     pid_type="persistent_blocked",
    #     split_k=64,
    # )
    # print("=== Triton Code (Helion backend) ===")
    # print(bound_kernel.to_triton_code(config))

    try:
        mlir_text = generate_mlir(bound_kernel)
        print("=== MLIR Dump ===")
        print(mlir_text)

        result = validate_with_mlir_opt(mlir_text)
        if result.returncode != 0:
            print(result.stderr, file=sys.stderr)
            print("mlir-opt validation failed.")
        else:
            print("mlir-opt validation succeeded.\n")
    except Exception as e:
        print(f"\nMLIR generation failed:")
        print(f"  {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
