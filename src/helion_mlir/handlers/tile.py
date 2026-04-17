from __future__ import annotations

import helion.language.tile_ops as hl_tile_ops


def register_tile_handlers(direct: dict[object, str], predicates: list[tuple[object, str]]) -> None:
    direct[hl_tile_ops.tile_index] = "visit_tile_index"
    direct[hl_tile_ops.tile_id] = "visit_tile_id"
    direct[hl_tile_ops.tile_begin] = "visit_tile_begin"
    direct[hl_tile_ops.tile_end] = "visit_tile_end"
