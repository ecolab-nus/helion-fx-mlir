from __future__ import annotations

"""
Translate Helion Device IR graphs into the normalized JSON interchange format.

The exporter walks the FX graphs produced by Helion's compiler and reconstructs a
structured representation containing explicit tile loops, tensor slices, and the
original torch dialect operations.  The output is designed to be consumed by the
C++ json2mlir lowering path, so we preserve enough metadata (tile extents, SSA
names, loop-carried values) for later reconstruction of MLIR modules.
"""

import json
from dataclasses import dataclass
import operator
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.fx as fx
from helion._compiler.device_ir import DeviceIR, ForLoopGraphInfo
from helion.language import _tracing_ops, creation_ops
from helion.language.memory_ops import load, store
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
    """Describe a tiling block emitted by Helion (size, symbol, debug name)."""

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
    """Map a torch dtype to the compact string literal used in the schema."""

    try:
        return DTYPE_ALIASES[dtype]
    except KeyError:
        raise ValueError(f"Unsupported dtype for JSON export: {dtype}") from None


def _tensor_type(tensor: torch.Tensor) -> dict[str, Any]:
    """Build the JSON type object for a tensor argument or result."""

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
    """
    Gather block metadata from the bound kernel environment.

    Helion attaches block sizes and debug names to the bound kernel; we retain the
    information so outer tile loops can re-use the same identifiers and step sizes.
    """

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
    """Create a mapping from symbolic SymInt names to their associated tile names."""

    return {block.sym: block.tile_name for block in blocks}


def _size_expr(meta_val: Any, sym_lookup: dict[str, str]) -> int | str:
    """
    Convert a SymInt or literal dimension into the textual expression expected by the schema.

    We preserve references to loop-carried tile sizes so later phases can recover the
    relationship between iteration variables and slice extents.
    """

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
    """Extract the tile identifier from a 'size(tile_x)' expression or pass literals through."""

    if isinstance(size_expr, str) and size_expr.startswith("size(") and size_expr.endswith(")"):
        return size_expr[5:-1]
    if isinstance(size_expr, str):
        return size_expr
    return str(size_expr)


def _normalize_torch_op(target: Any) -> str:
    """Derive the canonical torch.* name for an ATen overload or Python callable."""

    if hasattr(target, "overloadpacket"):
        name = target.overloadpacket.__name__
    elif hasattr(target, "__name__"):
        name = target.__name__
    else:
        name = str(target)
    return f"torch.{name}"


def _collect_default_attrs(node: fx.Node) -> dict[str, Any]:
    """
    Gather keyword-only arguments for an ATen overload.

    The torch schema exposes default values; we resend them explicitly so downstream
    consumers do not need to re-materialize the PyTorch operator registry simply to
    recover attribute defaults.
    """

    target = node.target
    schema = getattr(target, "_schema", None)
    if schema is None:
        return {}
    attrs: dict[str, Any] = {}
    for arg in schema.arguments:
        if not arg.kwarg_only:
            continue
        value = node.kwargs.get(arg.name, arg.default_value)
        if value is None:
            continue
        if isinstance(value, (int, float)):
            attrs[arg.name] = float(value) if isinstance(value, float) else value
        else:
            attrs[arg.name] = value
    return attrs


class _ValueEncoder:
    """
    Lazily translate FX Nodes into JSON-ready literals.

    The encoder memoizes previously translated nodes so multiple references to the same
    SSA value resolve to the same JSON identifier.  When encountering load nodes we emit
    slice descriptors so downstream consumers can reason about tile windows.
    """

    def __init__(self, sym_lookup: Mapping[str, str], initial: Mapping[fx.Node, Any] | None = None) -> None:
        self._sym_lookup = dict(sym_lookup)
        self._values: dict[fx.Node, Any] = dict(initial or {})

    def seed(self, node: fx.Node, value: Any) -> None:
        """Pre-populate the encoder with a known value for the given node."""

        self._values[node] = value

    def value(self, obj: Any) -> Any:
        """Return the JSON-friendly encoding for the FX Node or literal."""

        if isinstance(obj, fx.Node):
            if obj in self._values:
                return self._values[obj]
            encoded = self._encode_node(obj)
            self._values[obj] = encoded
            return encoded
        if isinstance(obj, torch.SymInt):
            return _size_expr(obj, self._sym_lookup)
        if isinstance(obj, (int, float)):
            return int(obj) if isinstance(obj, int) else float(obj)
        return obj

    def _encode_node(self, node: fx.Node) -> Any:
        """Internal helper that dispatches on FX op kinds."""

        if node.op == "placeholder":
            raise KeyError("Encountered unseeded placeholder")
        if node.op == "call_function":
            target = node.target
            if target is _tracing_ops._new_var:
                return self.value(node.args[0])
            if target is _tracing_ops._host_tensor:
                return str(node.args[0])
            if target is _tracing_ops._get_symnode:
                sym = str(node.meta.get("val"))
                return self._sym_lookup.get(sym, sym)
            if target == torch.ops.aten.sym_size.int:
                return _size_expr(node.meta.get("val"), self._sym_lookup)
            if target is load:
                return self._encode_load(node)
        return node.name

    def _encode_load(self, node: fx.Node) -> dict[str, Any]:
        """Translate a Helion load() call into a JSON slice description."""

        base = self.value(node.args[0])
        offsets = [self._index_expr(idx) for idx in node.args[1]]
        fake_tensor = node.meta.get("val")
        if fake_tensor is None:
            raise ValueError("Missing fake tensor metadata for load op")
        sizes = [_size_expr(dim, self._sym_lookup) for dim in fake_tensor.shape]
        return {
            "slice": {
                "base": base,
                "offsets": offsets,
                "sizes": sizes,
            }
        }

    def _index_expr(self, value: Any) -> int | str:
        """Normalize index expressions, collapsing 'size(tile)' to the tile symbol."""

        expr = self.value(value)
        if isinstance(expr, str):
            return _size_to_tile(expr)
        return expr


class LoopBodyTranslator:
    """
    Walk the loop body graph and emit JSON operations.

    The translator mirrors SSA semantics: placeholders are seeded from the outer graph,
    torch ops become `torch.*` JSON entries, and the final `output` node records the
    loop-carried values so the root translator can wire up phi-like behaviour.
    """

    def __init__(
        self,
        graph_info: ForLoopGraphInfo,
        sym_lookup: Mapping[str, str],
        outer_names: Mapping[fx.Node, str],
    ) -> None:
        self._graph_info = graph_info
        self._encoder = _ValueEncoder(sym_lookup)
        self._ops: list[dict[str, Any]] = []
        for placeholder, outer_node in zip(
            [node for node in graph_info.graph.nodes if node.op == "placeholder"],
            graph_info.node_args,
            strict=True,
        ):
            seed_name = outer_names.get(outer_node, outer_node.name)
            self._encoder.seed(placeholder, seed_name)
        self.carry: list[str] = []

    def translate(self) -> list[dict[str, Any]]:
        """Return the ordered list of JSON operations represented by the loop body."""

        for node in self._graph_info.graph.nodes:
            if node.op == "placeholder":
                continue
            if node.op == "output":
                outputs = node.args[0]
                self.carry = [self._as_name(val) for val in outputs]
                continue
            if node.op != "call_function":
                self._encoder.value(node)
                continue
            target = node.target
            if target == torch.ops.aten.sym_size.int:
                self._encoder.value(node)
                continue
            if target is _tracing_ops._new_var or target is load or target is _tracing_ops._host_tensor or target is _tracing_ops._get_symnode:
                self._encoder.value(node)
                continue
            if isinstance(target, torch._ops.OpOverload):
                op_dict = self._convert_torch_op(node)
                self._ops.append(op_dict)
                self._encoder.seed(node, op_dict.get("result", node.name))
                continue
            self._encoder.value(node)
        return self._ops

    def _as_name(self, value: Any) -> str:
        """Ensure loop-carried values are represented as string identifiers."""

        encoded = self._encoder.value(value)
        if isinstance(encoded, dict):
            raise ValueError("Loop carry cannot be a slice expression")
        if not isinstance(encoded, str):
            return str(encoded)
        return encoded

    def _convert_torch_op(self, node: fx.Node) -> dict[str, Any]:
        """
        Materialise a torch.* JSON op derived from the original FX node.

        The op name, argument list, and defaulted keyword parameters match the original
        ATen overload so the MLIR lowering can reconstruct torch dialect ops without
        relying on hard-coded names.
        """

        op_name = _normalize_torch_op(node.target)
        args = [self._encoder.value(arg) for arg in node.args]
        attrs = _collect_default_attrs(node)
        result_name = node.name if node.name else None
        op_dict: dict[str, Any] = {"op": op_name, "args": args, "attrs": attrs}
        if result_name:
            op_dict["result"] = result_name
        return op_dict


class RootGraphTranslator:
    """
    Translate the outer root graph responsible for allocations and nested loop calls.

    We emit the initial `hl.alloc`, zero initialisations, tile loops, and stores by
    interpreting helper intrinsics (`hl.zeros`, `hl.store_slice`, `hl.tile`) present in
    the captured FX graph.  Intermediate SSA names are recorded in `value_names` so nested
    loop bodies can reference them by name.
    """

    def __init__(
        self,
        graph_info,
        device_ir: DeviceIR,
        blocks: list[BlockMeta],
        sym_lookup: Mapping[str, str],
    ) -> None:
        self._graph_info = graph_info
        self._device_ir = device_ir
        self._blocks = blocks
        self._sym_lookup = sym_lookup
        self.value_names: dict[fx.Node, str] = {}
        self._ops: list[dict[str, Any]] = []

    def translate(self) -> list[dict[str, Any]]:
        """Produce the flattened list of JSON ops for the root graph."""

        for node in self._graph_info.graph.nodes:
            if node.op == "placeholder" or node.op == "output":
                continue
            if node.op != "call_function":
                continue
            target = node.target
            if target is _tracing_ops._host_tensor:
                self.value_names[node] = str(node.args[0])
                continue
            if target is _tracing_ops._phi:
                merged = node.args[1]
                self.value_names[node] = self.value_names.get(merged, merged.name if isinstance(merged, fx.Node) else str(merged))
                continue
            if target is load:
                continue
            if target == torch.ops.aten.sym_size.int or target is _tracing_ops._get_symnode:
                continue
            if target is creation_ops.full:
                zeros_op = self._convert_zeros(node)
                self._ops.append(zeros_op)
                self.value_names[node] = zeros_op["result"]
                continue
            if target is _tracing_ops._for_loop:
                loop_ops = self._convert_for_loop(node)
                self._ops.extend(loop_ops)
                continue
            if target is operator.getitem:
                source, index = node.args
                carry = self.value_names.get(source, [])
                if isinstance(carry, list) and index < len(carry):
                    self.value_names[node] = carry[index]
                continue
            if target is store:
                store_op = self._convert_store(node)
                self._ops.append(store_op)
                continue
        return self._ops

    def _convert_zeros(self, node: fx.Node) -> dict[str, Any]:
        """Lower Helion's creation_ops.full() into an hl.zeros JSON operation."""

        shape_exprs = []
        for dim in node.args[0]:
            if isinstance(dim, fx.Node):
                expr = _size_expr(dim.meta.get("val"), self._sym_lookup)
            else:
                expr = _size_expr(dim, self._sym_lookup)
            shape_exprs.append(expr)
        dtype = _dtype_alias(node.args[2])
        return {
            "op": "hl.zeros",
            "shape": shape_exprs,
            "dtype": dtype,
            "result": node.name,
        }

    def _convert_for_loop(self, node: fx.Node) -> list[dict[str, Any]]:
        """Lower a Helion tiled loop invocation into `hl.tile` blocks plus its body."""

        loop_id = node.args[0]
        loop_info = self._device_ir.graphs[loop_id]
        if not isinstance(loop_info, ForLoopGraphInfo):
            raise ValueError("Expected ForLoopGraphInfo for loop body")
        tiles = [self._blocks[block_id] for block_id in loop_info.block_ids]
        ops: list[dict[str, Any]] = []
        if tiles:
            ops.append(
                {
                    "op": "hl.tile.begin",
                    "iters": [tile.iter_symbol for tile in tiles],
                    "sizes": [tile.size_literal() for tile in tiles],
                    "result": [tile.tile_name for tile in tiles],
                }
            )
        translator = LoopBodyTranslator(loop_info, self._sym_lookup, self.value_names)
        ops.extend(translator.translate())
        if tiles:
            ops.append({"op": "hl.tile.end"})
        self.value_names[node] = translator.carry
        return ops

    def _convert_store(self, node: fx.Node) -> dict[str, Any]:
        """Lower the `store` intrinsic into an hl.store_slice JSON operation."""

        dst_node = node.args[0]
        dst = self.value_names.get(dst_node, str(dst_node.args[0]) if isinstance(dst_node, fx.Node) else str(dst_node))
        offsets = []
        for offset in node.args[1]:
            if isinstance(offset, fx.Node):
                expr = _size_expr(offset.meta.get("val"), self._sym_lookup)
                offsets.append(_size_to_tile(expr))
            else:
                offsets.append(offset)
        sizes = [f"size({entry})" if isinstance(entry, str) else entry for entry in offsets]
        src_node = node.args[2]
        src = self.value_names.get(src_node, src_node.name if isinstance(src_node, fx.Node) else str(src_node))
        return {
            "op": "hl.store_slice",
            "dst": dst,
            "offsets": offsets,
            "sizes": sizes,
            "src": src,
        }


def _export_from_bound(bound: BoundKernel, *, module_name: str) -> dict[str, Any]:
    """
    Convert a bound Helion kernel into the JSON interchange representation.

    The function binds arguments, captures output metadata, orchestrates block/tile
    collection, and delegates to the graph translators defined above.  The final payload
    is validated against the JSON schema before being returned to callers.
    """

    kernel = bound.kernel
    device_ir: DeviceIR = bound.host_function.device_ir
    blocks = _collect_blocks(bound)
    sym_lookup = _sym_to_tile(blocks)

    tensor_args: list[tuple[str, torch.Tensor]] = []
    arg_entries = []
    for (param_name, _), value in zip(kernel.signature.parameters.items(), bound.fake_args, strict=False):
        if isinstance(value, torch.Tensor):
            tensor_args.append((param_name, value))
            arg_entries.append({"id": param_name, "type": _tensor_type(value)})

    if not tensor_args:
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

    dim_exprs: dict[Any, list[str]] = {}
    for arg_id, tensor in tensor_args:
        for axis, dim in enumerate(tensor.shape):
            expr = f"dim({arg_id},{axis})"
            dim_exprs.setdefault(dim, []).append(expr)

    out_dtype = _dtype_alias(out_tensor.dtype)
    shape_usage: dict[Any, int] = {}
    output_shape: list[int | str] = []
    for dim in out_tensor.shape:
        candidates = dim_exprs.get(dim)
        if candidates:
            idx = shape_usage.get(dim, 0)
            expr = candidates[idx] if idx < len(candidates) else candidates[-1]
            shape_usage[dim] = idx + 1
            output_shape.append(expr)
        else:
            if isinstance(dim, torch.SymInt):
                output_shape.append("?")
            elif isinstance(dim, (int, float)):
                output_shape.append(int(dim))
            else:
                output_shape.append("?")

    body.append(
        {
            "op": "hl.alloc",
            "shape": output_shape,
            "dtype": out_dtype,
            "result": "out",
        }
    )

    tile_stack: list[list[BlockMeta]] = []
    for block_group in device_ir.grid_block_ids or []:
        tiles = [blocks[block_id] for block_id in block_group]
        if not tiles:
            continue
        body.append(
            {
                "op": "hl.tile.begin",
                "iters": [tile.iter_symbol for tile in tiles],
                "sizes": [tile.size_literal() for tile in tiles],
                "result": [tile.tile_name for tile in tiles],
            }
        )
        tile_stack.append(tiles)

    root_ops: list[dict[str, Any]] = []
    for root_id in device_ir.root_ids:
        root_graph = device_ir.graphs[root_id]
        translator = RootGraphTranslator(root_graph, device_ir, blocks, sym_lookup)
        root_ops.extend(translator.translate())
    body.extend(root_ops)

    for tiles in reversed(tile_stack):
        if tiles:
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
