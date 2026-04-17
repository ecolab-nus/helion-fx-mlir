from __future__ import annotations


def _is_dot(target: object) -> bool:
    return hasattr(target, "__name__") and target.__name__ == "dot"


def _is_aten_compute(target: object) -> bool:
    return hasattr(target, "__module__") and ("aten" in str(target) or "prims." in str(target))


def register_compute_handlers(direct: dict[object, str], predicates: list[tuple[object, str]]) -> None:
    predicates.append((_is_dot, "visit_dot"))
    predicates.append((_is_aten_compute, "visit_aten_compute"))
