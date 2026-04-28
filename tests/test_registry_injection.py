from __future__ import annotations

from helion_mlir.registry import build_handler_registry, load_custom_ops


def test_build_handler_registry_not_empty() -> None:
    direct, predicates = build_handler_registry()
    assert direct
    assert predicates


def test_load_custom_ops_has_known_keys() -> None:
    ops = load_custom_ops()
    assert "gather" in ops
    assert "broadcast" in ops
