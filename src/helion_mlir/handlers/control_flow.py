from __future__ import annotations

import operator

import helion.language._tracing_ops as hl_tracing_ops


def register_control_flow_handlers(direct: dict[object, str], predicates: list[tuple[object, str]]) -> None:
    direct[hl_tracing_ops._for_loop] = "visit_for_loop"
    direct[hl_tracing_ops._phi] = "visit_phi"
    direct[hl_tracing_ops._if] = "visit_if"
    direct[operator.getitem] = "visit_getitem"
