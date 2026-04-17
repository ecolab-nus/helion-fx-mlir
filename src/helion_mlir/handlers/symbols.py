from __future__ import annotations

from torch.ops import aten

import helion.language._tracing_ops as hl_tracing_ops


def register_symbol_handlers(direct: dict[object, str], predicates: list[tuple[object, str]]) -> None:
    direct[hl_tracing_ops._get_symnode] = "visit_get_symnode"
    direct[hl_tracing_ops._new_var] = "visit_new_var"
    direct[hl_tracing_ops._host_tensor] = "visit_host_tensor"
    direct[aten.sym_size.int] = "visit_sym_size"
