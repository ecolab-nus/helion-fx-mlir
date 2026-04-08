"""
Helion Matmul Kernel Example
============================
This example demonstrates a Helion kernel implementation of matrix multiplication
with support for a customizable epilogue function. It includes autotuning,
correctness checks against PyTorch baselines, and integration with tritonbench.
"""

# %%
from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

import sys
from pathlib import Path

import torch
from torch import Tensor

import helion
from helion._testing import DEVICE
from helion._testing import run_example
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Callable

# Make sure the repo-local src/ package is importable when the example is
# executed directly.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from helion_mlir import generate_mlir, validate_with_mlir_opt, print_debug_info


# %%
@helion.kernel(
    static_shapes=False,
    # Disable autotung over unrolling/range_num_stages
    # tl.dot is pipelined with num_stages
    autotune_config_overrides={
        "range_unroll_factors": [0, 0],
        "range_num_stages": [0, 0],
    },
)
def batch_matmul(
    x: Tensor,
    y: Tensor,
    epilogue: Callable[[Tensor, tuple[Tensor, ...]], Tensor] = lambda acc, tile: acc,
) -> Tensor:
    """
    Performs batch matrix multiplication of x and y with an optional epilogue function.
    Args:
        x (Tensor): Left matrix of shape [b, m, k].
        y (Tensor): Right matrix of shape [b, k, n].
        epilogue (Callable, optional): Function applied to the accumulator and tile indices
            after the matmul. Defaults to identity (no change).
    Returns:
        Tensor: Resulting matrix of shape [b, m, n].
    """
    b, m, k = x.size()
    b2, k2, n = y.size()
    assert k == k2, f"size mismatch {k} != {k2}"
    assert b == b2, f"size mismatch {b} != {b2}"
    out = torch.empty(
        [b, m, n], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device
    )
    for tile_b, tile_m, tile_n in hl.tile([b, m, n]):
        acc = hl.zeros([tile_b, tile_m, tile_n], dtype=torch.float16)
        for tile_k in hl.tile(k):
            acc = torch.baddbmm(acc, x[tile_b, tile_m, tile_k], y[tile_b, tile_k, tile_n])
        out[tile_b, tile_m, tile_n] = epilogue(acc, (tile_b, tile_m, tile_n))
    return out


# %%
def autotune(b: int, m: int, k: int, n: int) -> None:
    """
    Runs autotuning on the matmul kernel with a ReLU epilogue and saves the best config.
    Args:
        m (int): Number of rows in matrix x.
        k (int): Number of columns in matrix x and rows in matrix y.
        n (int): Number of columns in matrix y.
    """
    x = torch.randn([b, m, k], device=DEVICE, dtype=torch.float16)
    y = torch.randn([b, k, n], device=DEVICE, dtype=torch.float16)
    bias = torch.randn([n], device=DEVICE, dtype=torch.float16)
    args = (x, y, lambda acc, tile: torch.relu(acc + bias[tile[1]]))
    best_config = matmul.autotune(args, force=True)
    print(f"Best config: {best_config}")
    best_config.save("best_config.json")


# %%
def check(b: int, m: int, k: int, n: int) -> None:
    """
    Checks the correctness of the matmul kernel against PyTorch baselines.
    Tests:
    - Plain matmul without bias.
    - Matmul with bias added in the epilogue.
    - Matmul with a more complex epilogue applying ReLU after bias addition.
    Args:
        m (int): Number of rows in matrix x.
        k (int): Number of columns in matrix x and rows in matrix y.
        n (int): Number of columns in matrix y.
    """
    x = torch.randn([b, m, k], device=DEVICE, dtype=torch.float16)
    y = torch.randn([b, k, n], device=DEVICE, dtype=torch.float16)
    bias = torch.randn([n], device=DEVICE, dtype=torch.float16)
    bias_scalar = torch.randn([1], device=DEVICE, dtype=torch.float16)
    # Test without bias
    run_example(batch_matmul, torch.matmul, (x, y))


def main() -> None:
    """
    Main function to run autotuning (commented out) and correctness checks.
    """
    b, m, k, n =8, 4096, 512, 4096
    x = torch.randn([b, m, k], dtype=torch.float16)
    y = torch.randn([b, k, n], dtype=torch.float16)
    bound_kernel = batch_matmul.bind((x, y))

    print_debug_info(bound_kernel)



    mlir_text = generate_mlir(
        bound_kernel
    )
    print("=== MLIR Dump ===")
    print(mlir_text)

    # Validate with mlir-opt
    result = validate_with_mlir_opt(mlir_text)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise SystemExit("mlir-opt validation failed (see stderr above).")
    print("mlir-opt validation succeeded.\n")

    # autotune(1024, 1024, 1024)
    # check(1024, 1024, 1024)


# %%
if __name__ == "__main__":
    main()
