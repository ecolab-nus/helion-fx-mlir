from __future__ import annotations

import helion_mlir


def test_public_api_is_narrow() -> None:
    assert hasattr(helion_mlir, "generate_mlir")
    assert hasattr(helion_mlir, "validate_with_mlir_opt")
    assert helion_mlir.__all__ == ["generate_mlir", "validate_with_mlir_opt"]
