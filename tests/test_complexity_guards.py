from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _line_count(relpath: str) -> int:
    return len((REPO_ROOT / relpath).read_text(encoding="utf-8").splitlines())


def test_ir_visitor_size_budget() -> None:
    assert _line_count("src/helion_mlir/ir_visitor.py") <= 3100


def test_torch_mlir_helper_size_budget() -> None:
    assert _line_count("src/helion_mlir/torch_mlir_helper.py") <= 1300
