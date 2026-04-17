from __future__ import annotations

from torch.ops import aten

import helion.language._tracing_ops as hl_tracing_ops
import helion.language.creation_ops as hl_creation_ops
import helion.language.view_ops as hl_view_ops


def register_tensor_handlers(direct: dict[object, str], predicates: list[tuple[object, str]]) -> None:
    direct[hl_creation_ops.full] = "visit_full"
    direct[aten.full.default] = "visit_aten_full"
    direct[hl_tracing_ops._mask_to] = "visit_mask_to"
    direct[hl_view_ops.subscript] = "visit_subscript"
