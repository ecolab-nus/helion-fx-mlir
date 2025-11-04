from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.fx as fx
from helion._compiler.device_ir import DeviceIR, ForLoopGraphInfo
from helion.language import _tracing_ops
from helion.language.memory_ops import load
from helion.runtime.kernel import BoundKernel, Kernel

from .schema import validate_spec

DTYPE_ALIASES: dict[torch.dtype, str] = {
    torch.float16: "f16",
    torch.bfloat16: "bf16",
    torch.float32: "f32",
    torch.float64: "f64",
    torch.int32: "i32",
    torch.int64: "i64",
}


@dataclass
class BlockMeta:
    block_id: int
    tile_name: str
    sym: str
    size: int | None

    @property
    def iter_symbol(self) -> str:
        if self.tile_name.startswith("tile_"):
            return self.tile_name[5:]
        return f"iter_{self.tile_name}"

    def size_literal(self) -> int | str:
        if self.size is not None:
            return int(self.size)
        return self.sym


def _dtype_alias(dtype: torch.dtype) -> str:
    try:
        return DTYPE_ALIASES[dtype]
    except KeyError:
        raise ValueError(f"Unsupported dtype for JSON export: {dtype}") from None


def _tensor_type(tensor: torch.Tensor) -> dict[str, Any]:
    shape: list[int | str] = []
    for dim in tensor.shape:
        if isinstance(dim, torch.SymInt):
            shape.append("?")
        else:
            shape.append("?")
    return {
        "tensor": {
            "shape": shape,
            "elem": _dtype_alias(tensor.dtype),
        }
    }


def _collect_blocks(bound_kernel) -> list[BlockMeta]:
    blocks: list[BlockMeta] = []
    for block in bound_kernel.env.block_sizes:
        debug_names = sorted(block.debug_names) if block.debug_names else []
        tile_name = debug_names[0] if debug_names else f"tile_{block.block_id}"
        size_value: int | None
        try:
            size_value = int(block.size)
        except (TypeError, ValueError):
            size_value = None
        blocks.append(
            BlockMeta(
                block_id=block.block_id,
                tile_name=tile_name,
                sym=str(block.var),
                size=size_value,
            )
        )
    return blocks


def _sym_to_tile(blocks: Iterable[BlockMeta]) -> dict[str, str]:
    return {block.sym: block.tile_name for block in blocks}


def _size_expr(meta_val: Any, sym_lookup: dict[str, str]) -> int | str:
    if isinstance(meta_val, torch.SymInt):
        sym = str(meta_val)
        tile = sym_lookup.get(sym)
        if tile is not None:
            return f"size({tile})"
        return sym
    if isinstance(meta_val, (int, float)):
        return int(meta_val)
    if meta_val is None:
        return "?"
    sym = str(meta_val)
    tile = sym_lookup.get(sym)
    if tile is not None:
        return f"size({tile})"
    return sym


def _size_to_tile(size_expr: int | str) -> str:
    if isinstance(size_expr, str) and size_expr.startswith("size(") and size_expr.endswith(")"):
        return size_expr[5:-1]
    if isinstance(size_expr, str):
        return size_expr
    return str(size_expr)


def _extract_addmm_op(graph, sym_lookup: dict[str, str]) -> dict[str, Any]:
    addmm_node = None
    g_loads: list[tuple[str, list[int | str]]] = []

    for node in graph.nodes:
        if node.op == "call_function" and node.target == torch.ops.aten.addmm.default:
            addmm_node = node
        elif node.op == "call_function" and node.target is load:
            base_node = node.args[0]
            if isinstance(base_node, fx.Node) and base_node.target is _tracing_ops._host_tensor:
                tensor_name = base_node.args[0]
            else:
                tensor_name = str(base_node)
            indices = []
            for idx in node.args[1]:
                if isinstance(idx, fx.Node):
                    indices.append(_size_expr(idx.meta.get("val"), sym_lookup))
                else:
                    indices.append(_size_expr(idx, sym_lookup))
            g_loads.append((tensor_name, indices))

    if addmm_node is None or len(g_loads) < 2:
        raise ValueError("Unable to locate addmm pattern in Helion device IR")

    loads_by_name = {name: indices for name, indices in g_loads}
    x_indices = loads_by_name.get("x")
    y_indices = loads_by_name.get("y")
    if x_indices is None or y_indices is None:
        raise ValueError("Expected loads for tensors 'x' and 'y' in addmm loop")

    x_offsets = [_size_to_tile(entry) for entry in x_indices]
    y_offsets = [_size_to_tile(entry) for entry in y_indices]

    return {
        "op": "torch.addmm",
        "result": "acc",
        "args": [
            "acc",
            {
                "slice": {
                    "base": "x",
                    "offsets": x_offsets,
                    "sizes": x_indices,
                }
            },
            {
                "slice": {
                    "base": "y",
                    "offsets": y_offsets,
                    "sizes": y_indices,
                }
            },
        ],
        "attrs": {
            "alpha": 1.0,
            "beta": 1.0,
        },
    }


def _export_from_bound(bound: BoundKernel, *, module_name: str) -> dict[str, Any]:
    kernel = bound.kernel
    device_ir: DeviceIR = bound.host_function.device_ir
    blocks = _collect_blocks(bound)
    sym_lookup = _sym_to_tile(blocks)

    arg_entries = []
    for (param_name, _), value in zip(kernel.signature.parameters.items(), bound.fake_args, strict=False):
        if isinstance(value, torch.Tensor):
            arg_entries.append(
                {
                    "id": param_name,
                    "type": _tensor_type(value),
                }
            )

    if not arg_entries:
        raise ValueError("No tensor arguments were discovered; cannot export kernel")

    out_tensor = None
    for graph_id in device_ir.root_ids:
        graph_info = device_ir.graphs[graph_id]
        for node in graph_info.graph.nodes:
            if (
                node.op == "call_function"
                and node.target is _tracing_ops._host_tensor
                and node.args
                and node.args[0] == "out"
            ):
                meta_val = node.meta.get("val")
                if isinstance(meta_val, torch.Tensor):
                    out_tensor = meta_val
                break
        if out_tensor is not None:
            break

    if out_tensor is None:
        first_tensor = next(
            (value for value in bound.fake_args if isinstance(value, torch.Tensor)),
            None,
        )
        if first_tensor is None:
            raise ValueError("Unable to infer return tensor; provide example arguments")
        out_tensor = torch.empty_like(first_tensor)

    returns = [
        {
            "type": _tensor_type(out_tensor),
        }
    ]

    body: list[dict[str, Any]] = []

    out_dtype = _dtype_alias(out_tensor.dtype)
    output_shape = []
    for entry in arg_entries[:2]:
        arg_id = entry["id"]
        axis = len(output_shape)
        output_shape.append(f"dim({arg_id},{axis})")

    body.append(
        {
            "op": "hl.alloc",
            "shape": output_shape,
            "dtype": out_dtype,
            "result": "out",
        }
    )

    grid_block_groups = device_ir.grid_block_ids or []
    outer_block_ids: list[int] = grid_block_groups[0] if grid_block_groups else []
    outer_tiles = [blocks[block_id] for block_id in outer_block_ids]

    if outer_tiles:
        body.append(
            {
                "op": "hl.tile.begin",
                "iters": [block.iter_symbol for block in outer_tiles],
                "sizes": [block.size_literal() for block in outer_tiles],
                "result": [block.tile_name for block in outer_tiles],
            }
        )

    body.append(
        {
            "op": "hl.zeros",
            "shape": [f"size({block.tile_name})" for block in outer_tiles],
            "dtype": "f32",
            "result": "acc",
        }
    )

    inner_graph = next(
        (graph for graph in device_ir.graphs if isinstance(graph, ForLoopGraphInfo)),
        None,
    )
    if inner_graph is None:
        raise ValueError("Expected a nested tile loop in the device IR")

    inner_blocks = [blocks[block_id] for block_id in inner_graph.block_ids]
    body.append(
        {
            "op": "hl.tile.begin",
            "iters": [block.iter_symbol for block in inner_blocks],
            "sizes": [block.size_literal() for block in inner_blocks],
            "result": [block.tile_name for block in inner_blocks],
        }
    )

    body.append(_extract_addmm_op(inner_graph.graph, sym_lookup))
    body.append({"op": "hl.tile.end"})

    body.append(
        {
            "op": "hl.store_slice",
            "dst": "out",
            "offsets": [block.tile_name for block in outer_tiles],
            "src": "acc",
        }
    )

    if outer_tiles:
        body.append({"op": "hl.tile.end"})

    body.append({"op": "hl.return", "values": ["out"]})

    payload = {
        "version": 1,
        "module": {
            "name": module_name,
            "funcs": [
                {
                    "name": kernel.name,
                    "args": arg_entries,
                    "rets": returns,
                    "body": body,
                }
            ],
        },
    }
    validate_spec(payload)
    return payload


def _pick_cached_bound_kernel(cache: Mapping[object, BoundKernel]) -> BoundKernel | None:
    for bound in cache.values():
        return bound
    return None


def export_kernel(
    kernel: Kernel,
    example_args: Sequence[object] | None = None,
    *,
    module_name: str = "helion_module",
) -> dict[str, Any]:
    """
    Lower a Helion kernel to the JSON interchange format defined in project.plan.md.

    Args:
        kernel: Helion kernel object (@helion.kernel decorated function).
        example_args: Optional arguments used to bind the kernel (provides shapes/dtypes).
        module_name: Name of the top-level module in the emitted JSON document.

    Returns:
        dict: JSON-ready structure describing the kernel.
    """

    bound: BoundKernel | None = None
    if example_args is not None:
        bound = kernel.bind(tuple(example_args))
    else:
        cache = getattr(kernel, "_bound_kernels", {})
        bound = _pick_cached_bound_kernel(cache)
        if bound is None:
            raise ValueError(
                "No cached Helion binding found. Provide example_args so the kernel can be analyzed."
            )

    return _export_from_bound(bound, module_name=module_name)


def export_kernel_json(
    kernel: Kernel,
    example_args: Sequence[object] | None = None,
    *,
    module_name: str = "helion_module",
    indent: int = 2,
) -> str:
    """Helper that returns a JSON string."""
    spec = export_kernel(kernel, example_args, module_name=module_name)
    return json.dumps(spec, indent=indent, sort_keys=True)


def dump_kernel_json(
    kernel: Kernel,
    dump_path: str | Path,
    *,
    example_args: Sequence[object] | None = None,
    module_name: str = "helion_module",
    indent: int = 2,
) -> Path:
    """
    Convenience wrapper that writes the JSON payload to disk.

    Args:
        kernel: Helion kernel to export.
        dump_path: Destination file path.
        example_args: Optional binding arguments; if omitted, a cached binding is reused.
        module_name: Name for the JSON module section.
        indent: Pretty-print indent level.

    Returns:
        Path to the written file.
    """

    path = Path(dump_path)
    payload = export_kernel_json(
        kernel,
        example_args,
        module_name=module_name,
        indent=indent,
    )
    path.write_text(payload + "\n", encoding="utf-8")
    return path
