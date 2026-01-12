"""MLIR emission helpers for the staged Helion lowering prototype.

The goal is to make the textual IR evolve alongside project.plan.md. The
stage-0 milestone now threads real metadata from a bound Helion kernel into the
generated MLIR so contributors can inspect loop-carried values, loads, stores,
and placeholder torch calls before the full DeviceIR pipeline is wired in.

This module uses a modular lowering architecture:
- MLIRBuilder: Handles MLIR text emission and SSA naming
- LoweringContext: Holds state during lowering
- LoweringRegistry: Maps FX targets to lowering implementations
- lowerings/: Individual lowering implementations per op category
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Iterable, Sequence, TYPE_CHECKING

import torch
import helion.language.memory_ops as hl_memory_ops
import helion.language._tracing_ops as hl_tracing_ops
from torch.ops import aten

# Import the modular lowering infrastructure
from .mlir_builder import (
    MLIRBuilder,
    is_concrete_size,
    torch_dtype_to_mlir_element_type,
    format_tensor_type,
    format_shape_attr,
    format_indices_attr,
    format_string_attr,
    format_attr_dict,
    format_dynamic_tensor_meta,
    as_optional_int,
)
from .lowering_context import (
    LoweringContext,
    LoopInfo,
    KernelArgInfo,
    LoadInfo,
    first_debug_name,
    resolve_extent,
    collect_reduction_block_ids,
)
from .op_registry import LoweringRegistry

# Import lowerings to trigger registration
from . import lowerings  # noqa: F401

if TYPE_CHECKING:
    from helion._compiler.device_ir import DeviceIR
    from helion._compiler.device_ir import GraphInfo
    from helion._compiler.compile_environment import BlockSizeInfo
    from helion.runtime.kernel import BoundKernel
    from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[2]
HELION_OPT_CANDIDATES = [
    REPO_ROOT / "build" / "mlir" / "helion-opt",
    REPO_ROOT / "build" / "bin" / "helion-opt",
    Path("/mnt/fast/llvm-mlir/bin/helion-opt"),
    Path("/mnt/fast/llvm-mlir/bin/mlir-opt"),
]


def generate_plan_stage0_mlir(
    bound_kernel: "BoundKernel",
    *,
    kernel_name: str = "helion_matmul_plan_stage0",
) -> str:
    """Generate the stage-0 MLIR skeleton using real Helion metadata.
    
    This function uses the modular lowering architecture to convert a bound
    Helion kernel to MLIR text. It handles:
    - Module and function structure
    - Parallel and reduction loop emission
    - FX node lowering via the registry
    
    Tile sizes are handled as follows:
    - Concrete int values: emitted as MLIR constants
    - Symbolic values (SymInt, AutoSize, None): emitted as function arguments
    """
    # Create lowering context from bound kernel
    ctx = LoweringContext.from_bound_kernel(bound_kernel, kernel_name)
    builder = ctx.builder
    
    # Infer output shape from outer loop extents (generalized from matmul [M, N])
    # The outer loops define the output dimensions
    full_shape = _infer_output_shape(ctx)
    full_shape_attr = format_shape_attr(full_shape)
    
    # Emit module start with tile size attributes
    module_attrs = ctx.get_module_attributes()
    builder.emit_module_start(module_attrs)
    
    # Build function signature from kernel arguments only (no tile size args)
    func_args = ctx.get_func_signature_args()
    symbolic_tile_args = ctx.get_symbolic_tile_args()
    
    # Determine result type from inferred output shape
    result_type = format_tensor_type(full_shape, ctx.element_type)
    
    # Emit function start
    builder.emit_func_start(kernel_name, func_args, result_type)
    
    # Emit get_module_attribute ops for each symbolic tile size
    # Use block_id-based naming for consistency
    for sym_arg in symbolic_tile_args:
        loop_name = sym_arg["name"]
        block_id = sym_arg.get("block_id")
        # Use block_id if available, otherwise fall back to loop name
        if block_id is not None:
            attr_name = f"loom.block_{block_id}"
            ssa_hint = f"block_{block_id}"
        else:
            # Fallback to legacy naming for compatibility
            dim_letter = _loop_name_to_dim_letter(loop_name)
            attr_name = f"loom.block_{dim_letter}"
            ssa_hint = f"block_{dim_letter}"
        ssa = builder.emit_get_module_attribute(attr_name, ssa_hint)
        ctx.symbolic_arg_ssa[loop_name] = ssa
    
    # Emit output allocation (using first tensor arg as template)
    first_tensor = ctx.get_tensor_arg_ssa(0)
    first_tensor_type = ctx.get_tensor_arg_type(0)
    
    ctx.out_value = builder.fresh("out")
    output_type = format_tensor_type(full_shape, ctx.element_type)
    alloc_attrs = format_attr_dict({"shape": full_shape_attr})
    builder.emit(
        f'{ctx.out_value} = "helion.alloc_like"({first_tensor}){alloc_attrs} : ({first_tensor_type}) -> {output_type}'
    )
    
    # Store output shape for later use
    ctx.output_shape = full_shape
    
    # Emit accumulator initialization
    ctx.acc_seed = builder.fresh("acc_init")
    zero_attrs = format_attr_dict({"shape": ctx.tile_shape_attr, "dtype": ctx.element_type})
    builder.emit(
        f'{ctx.acc_seed} = "helion.zero_tile"(){zero_attrs} : () -> {ctx.tensor_type}'
    )
    
    # Emit dimension queries
    _emit_dimension_queries(ctx)
    

    # Set up dimension mapping
    ctx.setup_dims_map()
    
    # Emit loop bounds computation and metadata
    _emit_loop_bounds(ctx)
    
    # Emit outer parallel loop
    _emit_parallel_loop_structure(ctx, bound_kernel)
    
    # Emit function end and return
    output_type = format_tensor_type(ctx.output_shape, ctx.element_type)
    builder.emit(f"return {ctx.out_value} : {output_type}")
    builder.emit_func_end()
    builder.emit_module_end()
    
    return builder.build()


def _infer_output_shape(ctx: LoweringContext) -> list[int | None]:
    """Infer output shape from outer loop extents.
    
    The outer (parallel) loops define the output tensor dimensions.
    For matmul: [tile_m, tile_n] -> [M, N]
    For attention: [tile_b, tile_m] + head_dim
    
    This generalizes the previous hardcoded [lhs.size(0), rhs.size(1)] pattern.
    """
    shape = []
    for loop in ctx.outer_loops:
        extent = loop.total_extent
        if isinstance(extent, int):
            shape.append(extent)
        else:
            shape.append(None)  # Dynamic dimension
    
    # Ensure we have at least 2 dimensions for typical 2D output tensors
    while len(shape) < 2:
        shape.append(None)
    
    return shape


def _loop_name_to_dim_letter(name: str) -> str:
    """Map loop name to dimension letter for legacy compatibility.
    
    This is used as a fallback when block_id is not available.
    Prefer using block_id directly when possible.
    """
    if name in {"tile_m", "m"}:
        return "m"
    elif name in {"tile_n", "n"}:
        return "n"
    elif name in {"tile_k", "k"}:
        return "k"
    elif name in {"tile_b", "b"}:
        return "b"
    # Extract trailing letter or number from name
    if name.startswith("tile_"):
        return name[5:].lower()
    return name.lower()


def _emit_dimension_queries(ctx: LoweringContext) -> None:
    """Store dimension values from loop extents.
    
    This generalizes the previous hardcoded dim_m/dim_k/dim_n extraction
    to work with any loop names by populating loop_extents dict.
    """
    # The loop extents are already populated in LoweringContext._build_loop_info_generic()
    # Just ensure backward compatibility by setting legacy attributes if the loops exist
    loop_map = ctx.get_loop_map()
    
    # Legacy compatibility: set dim_m, dim_k, dim_n if corresponding loops exist
    for loop_name, legacy_attr in [("tile_m", "dim_m"), ("tile_k", "dim_k"), ("tile_n", "dim_n")]:
        loop = loop_map.get(loop_name)
        if loop:
            setattr(ctx, legacy_attr, int(loop.total_extent))
        else:
            setattr(ctx, legacy_attr, 0)


def _emit_loop_bounds(ctx: LoweringContext) -> None:
    """Emit loop bounds computation for outer and reduction loops."""
    builder = ctx.builder
    
    # Process outer loops
    for loop in ctx.outer_loops:
        loop_dim = ctx.dims_map.get(loop.name)
        
        if loop.is_symbolic:
            # Use the symbolic argument as the tile size
            tile_ssa = ctx.symbolic_arg_ssa.get(loop.name)
            if tile_ssa is None:
                raise ValueError(f"Missing symbolic argument for loop {loop.name}")
            loop.tile_const = tile_ssa
            
            if loop_dim is not None:
                # Compute trip count with ceildiv
                if isinstance(loop_dim, int):
                    # Inline loop_dim
                    trip_count_ssa = builder.emit_affine_apply(
                        f"()[s0] -> ({loop_dim} ceildiv s0)",
                        [],
                        [tile_ssa],
                    )
                else:
                    trip_count_ssa = builder.emit_affine_apply(
                        "()[s0, s1] -> (s0 ceildiv s1)",
                        [],
                        [loop_dim, tile_ssa],
                    )
                loop.trip_count_ssa = trip_count_ssa
            else:
                loop.trip_count_ssa = tile_ssa
        elif loop_dim is not None:
            # Emit tile size constant - still useful for other things or inline?
            # User wants to avoid arith.constant for SHAPES. 
            # Tile sizes are config parameters, usually constant is fine, but can also be inlined.
            # Let's keep tile size as constant SSA for reusable clarity, or inline it if requested?
            # The prompt said "shapes of inputs ... use directly the value hard-coded inline".
            # It didn't explicitly forbid tile size constants, but consistently inlining everything is better.
            
            # However, `loop.tile_const` is expected to be an SSA value by other parts (e.g. `_emit_dynamic_tile_size`).
            # Let's create the constant for tile size as before, but inline the DIMENSION.
            
            tile_const = builder.fresh(f"{loop.name}_tile")
            builder.emit(f"{tile_const} = arith.constant {loop.tile_size} : index")
            loop.tile_const = tile_const
            
            # Compute trip count with ceildiv
            if isinstance(loop_dim, int):
                # Inline loop_dim
                trip_count_ssa = builder.emit_affine_apply(
                    f"()[s0] -> ({loop_dim} ceildiv s0)",
                    [],
                    [tile_const],
                )
            else:
                trip_count_ssa = builder.emit_affine_apply(
                    "()[s0, s1] -> (s0 ceildiv s1)",
                    [],
                    [loop_dim, tile_const],
                )
            loop.trip_count_ssa = trip_count_ssa
        else:
            loop.trip_count_ssa = str(loop.trip_count)


def _emit_parallel_loop_structure(ctx: LoweringContext, bound_kernel: "BoundKernel") -> None:
    """Emit the outer parallel loop and its contents."""
    builder = ctx.builder
    
    # Build IV names using block_id pattern for clear block-to-loop correspondence
    iv_names = [f"%iv_block{loop.block_id}" for loop in ctx.outer_loops]
    for i, loop in enumerate(ctx.outer_loops):
        loop.iv_name = iv_names[i]
    
    zero_bounds = ["0"] * len(ctx.outer_loops)
    upper_bounds = [loop.trip_count_ssa or str(loop.trip_count) for loop in ctx.outer_loops]
    steps = ["1"] * len(ctx.outer_loops)
    
    # Emit parallel loop start
    builder.emit_affine_parallel_start(iv_names, zero_bounds, upper_bounds, steps)
    
    # Compute dynamic tile sizes for outer loops
    _emit_outer_tile_sizes(ctx)
    
    # Extract FX names from the device IR graphs
    _extract_fx_metadata(ctx, bound_kernel)
    
    # Emit reduction loops and body
    ctx.current_acc = ctx.acc_seed
    _emit_reduction_loops(ctx)
    
    # Emit phi node if present
    _emit_phi_if_present(ctx)
    
    # Emit store operation
    _emit_store_tile(ctx)
    
    # Emit parallel loop end
    builder.emit_affine_parallel_end()


def _emit_outer_tile_sizes(ctx: LoweringContext) -> None:
    """Set up tile sizes for outer loops using direct block sizes.
    
    Note: We don't use affine.min for tile boundary handling. This assumes
    symbolic shapes where the tile size is always the full block size.
    """
    for loop in ctx.outer_loops:
        # Simply use the tile size from the loop (either constant or symbolic SSA)
        if loop.tile_const is not None:
            ctx.outer_tile_sizes[loop.name] = loop.tile_const


def _extract_fx_metadata(ctx: LoweringContext, bound_kernel: "BoundKernel") -> None:
    """Extract FX node metadata from device IR graphs."""
    device_ir = bound_kernel.host_function.device_ir
    
    for_graph = _first_graph_with_block_ids(device_ir)
    if for_graph is not None:
        load_infos, other_names = _extract_load_infos(for_graph, ctx.kernel_args, ctx.block_sizes)
        ctx.load_infos = load_infos
        ctx.fx_names = other_names
    else:
        ctx.load_infos = []
        ctx.fx_names = {}
    
    root_graph = _first_root_graph(device_ir)
    ctx.root_fx_info = _extract_root_fx_info(root_graph)


def _emit_reduction_loops(ctx: LoweringContext) -> None:
    """Emit reduction loop(s) with loads and computation."""
    builder = ctx.builder
    loop_map = ctx.get_loop_map()
    
    outer_ivs = [loop.iv_name for loop in ctx.outer_loops]
    
    for loop in ctx.reduction_loops:
        
        # Set up loop IV using block_id pattern
        reduction_iv = f"%iv_block{loop.block_id}"
        loop.iv_name = reduction_iv
        loop_result = builder.fresh(f"{loop.name}_acc")
        
        # Compute trip count
        loop_dim = ctx.dims_map.get(loop.name)
        trip_bound = _compute_reduction_trip_bound(ctx, loop, loop_dim)
        
        # Emit affine.for with iter_args
        builder.emit(
            f"{loop_result} = affine.for {reduction_iv} = 0 to {trip_bound} "
            f"iter_args(%acc_iter = {ctx.current_acc}) -> ({ctx.tensor_type}) {{"
        )
        builder.push()
        
        # Compute tile sizes for this iteration
        reduction_tile_size = _compute_reduction_tile_size(ctx, loop, loop_dim)
        
        # Build IV map for loads using block_id-based pattern (generalized)
        # Map loop names AND block_ids to corresponding IVs
        loop_ivs = {}
        
        # Add all outer loop IVs by name and block_id
        for i, outer_loop in enumerate(ctx.outer_loops):
            loop_ivs[outer_loop.name] = outer_loop.iv_name
            loop_ivs[f"block_{outer_loop.block_id}"] = outer_loop.iv_name
            # Legacy compatibility: map first two outer loops to tile_m/tile_n
            if i == 0:
                loop_ivs["tile_m"] = outer_loop.iv_name
            elif i == 1:
                loop_ivs["tile_n"] = outer_loop.iv_name
        
        # Add all reduction loop IVs by name and block_id
        for red_loop in ctx.reduction_loops:
            loop_ivs[red_loop.name] = red_loop.iv_name
            loop_ivs[f"block_{red_loop.block_id}"] = red_loop.iv_name
            # Legacy compatibility
            loop_ivs["tile_k"] = red_loop.iv_name
        
        # Emit all loads dynamically
        emitted_loads: list[str] = []
        for load_info in ctx.load_infos:
            load_ssa = _emit_load(ctx, load_info, loop_ivs, reduction_tile_size)
            emitted_loads.append(load_ssa)
            load_info.ssa_name = load_ssa
        
        # Emit computation using registry-based dispatch or fallback to call_torch
        acc_next = _emit_computation(ctx, emitted_loads)
        
        # Emit yield
        builder.emit(f"affine.yield {acc_next} : {ctx.tensor_type}")
        builder.pop()
        builder.emit("}")
        
        ctx.current_acc = loop_result


def _compute_reduction_trip_bound(
    ctx: LoweringContext, loop: LoopInfo, loop_dim: str | int | None
) -> str:
    """Compute the trip bound for a reduction loop."""
    builder = ctx.builder
    
    if loop.is_symbolic:
        tile_ssa = ctx.symbolic_arg_ssa.get(loop.name)
        if tile_ssa is None:
            raise ValueError(f"Missing symbolic argument for reduction loop {loop.name}")
        loop.tile_const = tile_ssa
        
        if loop_dim is not None:
            if isinstance(loop_dim, int):
                trip_count_ssa = builder.emit_affine_apply(
                    f"()[s0] -> ({loop_dim} ceildiv s0)",
                    [],
                    [tile_ssa],
                )
            else:
                trip_count_ssa = builder.emit_affine_apply(
                    "()[s0, s1] -> (s0 ceildiv s1)",
                    [],
                    [loop_dim, tile_ssa],
                )
            loop.trip_count_ssa = trip_count_ssa
            return trip_count_ssa
        else:
            loop.trip_count_ssa = tile_ssa
            return tile_ssa
    elif loop_dim is not None:
        tile_const = builder.fresh(f"{loop.name}_tile")
        builder.emit(f"{tile_const} = arith.constant {loop.tile_size} : index")
        loop.tile_const = tile_const
        
        if isinstance(loop_dim, int):
            trip_count_ssa = builder.emit_affine_apply(
                f"()[s0] -> ({loop_dim} ceildiv s0)",
                [],
                [tile_const],
            )
        else:
            trip_count_ssa = builder.emit_affine_apply(
                "()[s0, s1] -> (s0 ceildiv s1)",
                [],
                [loop_dim, tile_const],
            )
        loop.trip_count_ssa = trip_count_ssa
        return trip_count_ssa
    else:
        loop.trip_count_ssa = str(loop.trip_count)
        return str(loop.trip_count)


def _compute_reduction_tile_size(
    ctx: LoweringContext, loop: LoopInfo, loop_dim: str | int | None
) -> str:
    """Get the tile size for the reduction loop.
    
    Note: We use the direct tile size without affine.min boundary checking.
    This assumes symbolic shapes where the tile size is always the full block size.
    """
    # Simply return the tile size constant or SSA value
    return loop.tile_const or str(loop.tile_size)


def _emit_load(
    ctx: LoweringContext,
    load_info: LoadInfo,
    loop_ivs: dict[str, str],
    reduction_tile_size: str,
) -> str:
    """Emit a tile load operation for any input tensor.
    
    This generalizes the previous _emit_lhs_load and _emit_rhs_load functions
    to handle any number of input tensors by extracting info from LoadInfo.
    
    Args:
        ctx: Lowering context
        load_info: Information about this load extracted from FX graph
        loop_ivs: Map from loop/block name to IV SSA value. Supports both legacy
                  names (e.g., "tile_m") and block_id pattern (e.g., "block_0")
        reduction_tile_size: SSA value for the reduction tile size (tile_k)
    
    Returns:
        SSA name of the loaded tile
    """
    builder = ctx.builder
    loop_map = ctx.get_loop_map()
    
    # Get tensor info from kernel args
    tensor_arg = _get_tensor_arg_for_load(ctx, load_info)
    tensor_ssa = tensor_arg.ssa_name if tensor_arg else f"%{load_info.source_tensor_name}"
    tensor_type = tensor_arg.mlir_type if tensor_arg else ctx.tensor_type
    
    # Get tile dimensions and block IDs from LoadInfo
    # Prefer resolved block_ids when available
    tile_block_ids = load_info.tile_block_ids
    tile_dim_names = load_info.tile_dim_names
    
    # If we don't have block_ids, fall back to inferring from tensor position
    if not tile_block_ids:
        tile_dims = _infer_tile_dims_for_load(ctx, load_info, tensor_arg)
        tile_dim_names = tile_dims
        tile_block_ids = [None] * len(tile_dims)
    
    # Build tile sizes and indices using block_id-based IVs when available
    tile_sizes: list[str] = []
    indices: list[str] = []
    
    for i, (dim_name, block_id) in enumerate(zip(tile_dim_names, tile_block_ids)):
        # Determine the IV to use
        if block_id is not None:
            # Use block_id-based IV lookup (authoritative)
            iv = loop_ivs.get(f"block_{block_id}", f"%iv_block{block_id}")
        else:
            # Fall back to legacy name-based lookup
            iv = loop_ivs.get(dim_name, f"%{dim_name}_iv")
        indices.append(iv)
        
        # Determine tile size
        if block_id is not None:
            # Get tile size from block_sizes via block_id
            loop_info = None
            for loop in ctx.outer_loops + ctx.reduction_loops:
                if loop.block_id == block_id:
                    loop_info = loop
                    break
            if loop_info and loop_info.tile_const:
                tile_sizes.append(loop_info.tile_const)
            else:
                tile_size = _choose_tile_size(builder, ctx.outer_tile_sizes, loop_map, dim_name)
                tile_sizes.append(tile_size)
        elif dim_name == "tile_k":
            tile_sizes.append(reduction_tile_size)
        else:
            tile_size = _choose_tile_size(builder, ctx.outer_tile_sizes, loop_map, dim_name)
            tile_sizes.append(tile_size)
    
    # Ensure we have exactly 2 dimensions for 2D tensors
    while len(tile_sizes) < 2:
        tile_sizes.append(reduction_tile_size)
        indices.append(loop_ivs.get("tile_k", "%iv_block2"))
    
    # Create fresh SSA name for the load result
    load_result = builder.fresh(load_info.source_tensor_name)
    
    # Format attributes
    indices_attr = format_indices_attr(indices[:2])
    tensor_meta = format_dynamic_tensor_meta(tile_sizes[0], tile_sizes[1], ctx.element_type)
    
    attrs = format_attr_dict({
        "tile": indices_attr,
        "sizes": ctx.tile_shape_attr,
        "tensor_meta": tensor_meta,
        "fx_node": format_string_attr(load_info.fx_node_name) if load_info.fx_node_name else None,
    })
    
    builder.emit(
        f'{load_result} = "helion.load"({tensor_ssa}, {tile_sizes[0]}, {tile_sizes[1]}){attrs} '
        f": ({tensor_type}, index, index) -> {ctx.tensor_type}"
    )
    
    return load_result


def _get_tensor_arg_for_load(
    ctx: LoweringContext, load_info: LoadInfo
) -> KernelArgInfo | None:
    """Get the kernel argument corresponding to a load's source tensor."""
    # First try by index if available
    if load_info.source_tensor_arg_idx is not None:
        tensor_args = ctx.get_tensor_args()
        if load_info.source_tensor_arg_idx < len(tensor_args):
            return tensor_args[load_info.source_tensor_arg_idx]
    
    # Fall back to matching by name
    return ctx.get_tensor_arg_by_name(load_info.source_tensor_name)


def _infer_tile_dims_for_load(
    ctx: LoweringContext,
    load_info: LoadInfo,
    tensor_arg: KernelArgInfo | None,
) -> list[str]:
    """Infer the tile dimension names for a load operation.
    
    Generalized to use:
    1. Explicit tile_dim_names from LoadInfo (parsed from FX graph) - preferred
    2. Block ID based inference from outer/reduction loops
    3. Fallback to first outer loops as default
    
    Previous matmul-specific logic (LHS[M,K], RHS[K,N]) is kept as legacy fallback.
    """
    # If the LoadInfo already has explicit tile_dim_names from FX, use them
    if load_info.tile_dim_names:
        return load_info.tile_dim_names
    
    # If we have block_ids from LoadInfo, convert them to loop names
    if load_info.tile_block_ids:
        dim_names = []
        all_loops = ctx.outer_loops + ctx.reduction_loops
        block_to_loop = {loop.block_id: loop for loop in all_loops}
        
        for block_id in load_info.tile_block_ids:
            if block_id is not None and block_id in block_to_loop:
                dim_names.append(block_to_loop[block_id].name)
            else:
                # Fallback for unknown block_id
                dim_names.append(f"block_{block_id}" if block_id is not None else "tile_m")
        
        if dim_names:
            return dim_names
    
    # Fallback: use outer loops + reduction loops in order
    # This covers generic cases where we don't have explicit mappings
    all_loop_names = [loop.name for loop in ctx.outer_loops + ctx.reduction_loops]
    if len(all_loop_names) >= 2:
        return all_loop_names[:2]
    elif all_loop_names:
        return all_loop_names
    
    # Ultimate fallback
    return ["tile_m", "tile_k"]



def _emit_computation(ctx: LoweringContext, emitted_loads: list[str]) -> str:
    """Emit computation op based on FX graph analysis.
    
    This generalizes the previous hardcoded _emit_addmm by:
    1. Looking up the computation target from FX names
    2. Dispatching to appropriate handler based on the op
    3. Falling back to addmm pattern for backward compatibility
    
    Args:
        ctx: Lowering context
        emitted_loads: List of SSA names for loaded tiles
    
    Returns:
        SSA name of the computation result
    """
    if not emitted_loads:
        return "%acc_iter"
    
    if len(emitted_loads) == 1:
        # Single load case - just pass through
        return emitted_loads[0]
    
    # Check what computation op is in the FX graph
    # fx_names contains op type -> fx_node_name mapping
    computation_op = None
    for op_key in ctx.fx_names:
        if op_key in {"addmm", "mm", "bmm", "baddbmm", "matmul"}:
            computation_op = op_key
            break
    
    builder = ctx.builder
    
    if computation_op == "addmm" or computation_op is None:
        # Default: use addmm pattern (accumulator + lhs @ rhs)
        return _emit_addmm(ctx, emitted_loads[0], emitted_loads[1])
    elif computation_op == "bmm":
        # Batched matrix multiply
        return _emit_bmm(ctx, emitted_loads[0], emitted_loads[1])
    elif computation_op == "baddbmm":
        # Batched addmm
        return _emit_baddbmm(ctx, emitted_loads[0], emitted_loads[1])
    else:
        # Generic fallback using call_torch
        return _emit_generic_computation(ctx, computation_op, emitted_loads)


def _emit_addmm(ctx: LoweringContext, lhs_tile: str, rhs_tile: str) -> str:
    """Emit aten.addmm via helion.call_torch."""
    builder = ctx.builder
    
    acc_next = builder.fresh("acc")
    addmm_fx_attr = ctx.fx_names.get("addmm")
    
    call_attrs = format_attr_dict({
        "fn_name": format_string_attr("aten.addmm"),
        "fx_node": format_string_attr(addmm_fx_attr) if addmm_fx_attr else None,
    })
    
    builder.emit(
        f'{acc_next} = "helion.call_torch"(%acc_iter, {lhs_tile}, {rhs_tile}){call_attrs} '
        f": ({ctx.tensor_type}, {ctx.tensor_type}, {ctx.tensor_type}) -> {ctx.tensor_type}"
    )
    
    return acc_next


def _emit_bmm(ctx: LoweringContext, lhs_tile: str, rhs_tile: str) -> str:
    """Emit aten.bmm via helion.call_torch."""
    builder = ctx.builder
    
    acc_next = builder.fresh("acc")
    bmm_fx_attr = ctx.fx_names.get("bmm")
    
    call_attrs = format_attr_dict({
        "fn_name": format_string_attr("aten.bmm"),
        "fx_node": format_string_attr(bmm_fx_attr) if bmm_fx_attr else None,
    })
    
    builder.emit(
        f'{acc_next} = "helion.call_torch"({lhs_tile}, {rhs_tile}){call_attrs} '
        f": ({ctx.tensor_type}, {ctx.tensor_type}) -> {ctx.tensor_type}"
    )
    
    return acc_next


def _emit_baddbmm(ctx: LoweringContext, lhs_tile: str, rhs_tile: str) -> str:
    """Emit aten.baddbmm via helion.call_torch."""
    builder = ctx.builder
    
    acc_next = builder.fresh("acc")
    fx_attr = ctx.fx_names.get("baddbmm")
    
    call_attrs = format_attr_dict({
        "fn_name": format_string_attr("aten.baddbmm"),
        "fx_node": format_string_attr(fx_attr) if fx_attr else None,
    })
    
    builder.emit(
        f'{acc_next} = "helion.call_torch"(%acc_iter, {lhs_tile}, {rhs_tile}){call_attrs} '
        f": ({ctx.tensor_type}, {ctx.tensor_type}, {ctx.tensor_type}) -> {ctx.tensor_type}"
    )
    
    return acc_next


def _emit_generic_computation(
    ctx: LoweringContext, op_name: str, operands: list[str]
) -> str:
    """Emit a generic computation via helion.call_torch."""
    builder = ctx.builder
    
    acc_next = builder.fresh("acc")
    fx_attr = ctx.fx_names.get(op_name)
    
    # Build operand list including accumulator
    all_operands = ["%acc_iter"] + operands
    operand_str = ", ".join(all_operands)
    type_str = ", ".join([ctx.tensor_type] * len(all_operands))
    
    call_attrs = format_attr_dict({
        "fn_name": format_string_attr(f"aten.{op_name}"),
        "fx_node": format_string_attr(fx_attr) if fx_attr else None,
    })
    
    builder.emit(
        f'{acc_next} = "helion.call_torch"({operand_str}){call_attrs} '
        f": ({type_str}) -> {ctx.tensor_type}"
    )
    
    return acc_next


def _emit_phi_if_present(ctx: LoweringContext) -> None:
    """Emit phi node if present in root FX info."""
    builder = ctx.builder
    
    phi_fx_name = ctx.root_fx_info.get("phi")
    if phi_fx_name is not None:
        phi_result = builder.fresh("phi")
        phi_attrs = format_attr_dict({"fx_node": format_string_attr(phi_fx_name)})
        builder.emit(
            f'{phi_result} = "helion.phi"({ctx.acc_seed}, {ctx.current_acc}){phi_attrs} '
            f": ({ctx.tensor_type}, {ctx.tensor_type}) -> {ctx.tensor_type}"
        )
        ctx.current_acc = phi_result


def _emit_store_tile(ctx: LoweringContext) -> None:
    """Emit store tile operation.
    
    Generalized to use outer loop indices and tile sizes instead of
    hardcoded tile_m/tile_n assumptions.
    """
    builder = ctx.builder
    loop_map = ctx.get_loop_map()
    
    # Get outer loop IVs and tile sizes dynamically
    outer_ivs = [loop.iv_name for loop in ctx.outer_loops]
    
    # Get tile sizes for each outer loop dimension
    store_tile_sizes = []
    for loop in ctx.outer_loops:
        # Try to get from outer_tile_sizes first, then from loop info
        tile_size = ctx.outer_tile_sizes.get(loop.name)
        if tile_size is None:
            tile_size = _choose_tile_size(builder, ctx.outer_tile_sizes, loop_map, loop.name)
        store_tile_sizes.append(tile_size)
    
    # Ensure we have at least 2 dimensions for 2D output
    while len(outer_ivs) < 2:
        outer_ivs.append("%unused_iv")
    while len(store_tile_sizes) < 2:
        # Use first tile size as fallback
        fallback = store_tile_sizes[0] if store_tile_sizes else builder.emit_index_constant(1)
        store_tile_sizes.append(fallback)
    
    store_meta = format_dynamic_tensor_meta(store_tile_sizes[0], store_tile_sizes[1], ctx.element_type)
    
    store_attrs = format_attr_dict({
        "tile": format_indices_attr(outer_ivs[:2]),
        "sizes": ctx.tile_shape_attr,
        "tensor_meta": store_meta,
        "fx_node": format_string_attr(ctx.root_fx_info.get("store"))
            if ctx.root_fx_info.get("store") else None,
    })
    
    # Use output shape from context
    output_type = format_tensor_type(ctx.output_shape, ctx.element_type)
    
    builder.emit(
        f'"helion.store"({ctx.out_value}, {ctx.current_acc}, {store_tile_sizes[0]}, {store_tile_sizes[1]}){store_attrs} '
        f": ({output_type}, {ctx.tensor_type}, index, index) -> ()"
    )


# -----------------------------------------------------------------------------
# Helper functions for tile size computation
# -----------------------------------------------------------------------------




def _choose_tile_size(
    builder: MLIRBuilder,
    dynamic_sizes: dict[str, str],
    loop_map: dict[str, LoopInfo],
    key: str,
) -> str:
    """Choose the appropriate tile size SSA value."""
    if key in dynamic_sizes:
        return dynamic_sizes[key]
    
    fallback_loop = loop_map.get(key)
    if fallback_loop:
        if fallback_loop.tile_const:
            return fallback_loop.tile_const
        if fallback_loop.tile_size is not None:
            const = builder.emit_index_constant(fallback_loop.tile_size)
            fallback_loop.tile_const = const
            return const
    
    return builder.emit_index_constant(0)


# -----------------------------------------------------------------------------
# FX graph inspection helpers
# -----------------------------------------------------------------------------


def _first_graph_with_block_ids(device_ir: "DeviceIR") -> "GraphInfo | None":
    """Find the first graph with block_ids (usually a ForLoopGraphInfo)."""
    for graph_info in device_ir.graphs:
        if getattr(graph_info, "block_ids", None):
            return graph_info
    return None


def _first_root_graph(device_ir: "DeviceIR") -> "GraphInfo | None":
    """Find the first root graph (without block_ids)."""
    for graph_info in device_ir.graphs:
        if getattr(graph_info, "block_ids", None):
            continue
        return graph_info
    return None


def _extract_load_infos(
    graph_info: "GraphInfo | None",
    kernel_args: list[KernelArgInfo],
    block_sizes: dict[int, Any],
) -> tuple[list[LoadInfo], dict[str, str]]:
    """Extract load operation information and other FX names from a graph.
    
    Args:
        graph_info: The FX graph to extract from
        kernel_args: Kernel argument information
        block_sizes: Mapping from block_id to BlockSizeInfo (from env.block_sizes)
    
    Returns:
        A tuple of (list of LoadInfo, dict of other FX names like addmm).
    """
    if graph_info is None:
        return [], {}
    
    load_infos: list[LoadInfo] = []
    other_names: dict[str, str] = {}
    
    # Build a map from tensor name to kernel arg index
    tensor_name_to_idx: dict[str, int] = {}
    for arg in kernel_args:
        if arg.is_tensor:
            tensor_name_to_idx[arg.name] = arg.index
    
    # Build a map from sympy symbol to block_id using BlockSizeInfo
    # This is the authoritative mapping from CompileEnvironment
    import sympy
    symbol_to_block_id: dict[sympy.Symbol, int] = {}
    for block_id, info in block_sizes.items():
        symbol = info.symbol()
        symbol_to_block_id[symbol] = block_id
    
    for node in graph_info.graph.nodes:
        if node.op == "call_function":
            if node.target is hl_memory_ops.load:
                # Extract source tensor from first argument
                tensor_arg = node.args[0] if node.args else None
                source_name = _get_source_tensor_name(tensor_arg)
                
                # Match to kernel argument
                arg_idx = tensor_name_to_idx.get(source_name)
                
                # Extract tile dimension names and block IDs from args[1] if present
                tile_dim_names, tile_block_ids = _parse_tile_indices_with_block_ids(
                    node.args[1] if len(node.args) > 1 else None,
                    symbol_to_block_id,
                )
                
                load_infos.append(LoadInfo(
                    fx_node_name=node.name,
                    source_tensor_name=source_name,
                    source_tensor_arg_idx=arg_idx,
                    tile_dim_names=tile_dim_names,
                    tile_block_ids=tile_block_ids,
                ))
            elif node.target is aten.addmm.default:
                other_names["addmm"] = node.name
    
    return load_infos, other_names


def _parse_tile_indices_with_block_ids(
    indices_arg: object,
    symbol_to_block_id: dict,
) -> tuple[list[str], list[int | None]]:
    """Parse tile index expression to extract dimension names AND block IDs.
    
    Uses the authoritative symbol_to_block_id mapping from CompileEnvironment's
    BlockSizeInfo to resolve FX node symbols to block IDs.
    
    Args:
        indices_arg: The FX node args representing tile indices
        symbol_to_block_id: Mapping from sympy.Symbol to block_id
    
    Returns:
        Tuple of (list of dimension names, list of block IDs)
    """
    import torch.fx
    import torch
    
    if indices_arg is None:
        return [], []
    
    dim_names: list[str] = []
    block_ids: list[int | None] = []
    
    # If it's a list of FX nodes, extract their names and resolve block IDs
    if isinstance(indices_arg, (list, tuple)):
        for item in indices_arg:
            if isinstance(item, torch.fx.Node):
                # Get the legacy dimension name from node name
                dim_name = _symnode_to_tile_dim(item.name)
                dim_names.append(dim_name)
                
                # Try to resolve block_id from the node's meta value
                block_id = None
                if hasattr(item, 'meta') and 'val' in item.meta:
                    val = item.meta['val']
                    if hasattr(val, '_sympy_'):
                        sympy_val = val._sympy_()
                        block_id = symbol_to_block_id.get(sympy_val)
                block_ids.append(block_id)
            elif hasattr(item, 'name'):
                dim_names.append(_symnode_to_tile_dim(item.name))
                block_ids.append(None)
        return dim_names, block_ids
    
    # Fallback for string format (legacy)
    if isinstance(indices_arg, str):
        import re
        match = re.search(r'\[([^\]]+)\]', indices_arg)
        if match:
            inner = match.group(1)
            parts = [p.strip() for p in inner.split(',')]
            for p in parts:
                if p:
                    dim_names.append(_symnode_to_tile_dim(p))
                    block_ids.append(None)  # Can't resolve from string
    
    return dim_names, block_ids


def _parse_tile_indices(indices_arg: object) -> list[str]:
    """Legacy: Parse tile index expression to extract dimension names only."""
    dim_names, _ = _parse_tile_indices_with_block_ids(indices_arg, {})
    return dim_names


def _symnode_to_tile_dim(name: str) -> str:
    """Convert a SymNode name to a tile dimension name.
    
    E.g., "sym_size_int" -> "tile_m", "block_size_2" -> "tile_k"
    Or preserve known names like "tile_m", "tile_n", "tile_k".
    
    Note: This is a fallback heuristic. Prefer using block_id resolution
    from CompileEnvironment when available.
    """
    # Already a tile name
    if name.startswith("tile_"):
        return name
    
    # Common SymNode patterns from Helion
    # sym_size_int typically refers to first dim (M), sym_size_int_1 to second (N)
    if "block_size" in name or "sym_size" in name:
        # Extract the index if present
        import re
        match = re.search(r'(\d+)$', name)
        if match:
            idx = int(match.group(1))
            # Map block_size indices to tile names
            # block_size_0 -> tile_m, block_size_1 -> tile_n, block_size_2 -> tile_k
            dim_map = {0: "tile_m", 1: "tile_n", 2: "tile_k", 3: "tile_b"}
            return dim_map.get(idx, f"tile_{idx}")
        else:
            # No index, assume first dimension
            return "tile_m"
    
    # Fallback: just prefix with tile_
    return f"tile_{name}"


def _get_source_tensor_name(tensor_arg: object) -> str:
    """Extract the source tensor name from an FX node argument.
    
    The tensor argument in a load node is typically a node representing
    the _host_tensor('name') call, so we extract the name from it.
    """
    import torch.fx
    
    if tensor_arg is None:
        return "unknown"
    
    if isinstance(tensor_arg, torch.fx.Node):
        # The node name itself is often the tensor name (e.g., "x", "y")
        return tensor_arg.name
    
    return str(tensor_arg)


def _extract_root_fx_info(graph_info: "GraphInfo | None") -> dict[str, str]:
    """Extract FX node info from root graph (store, phi)."""
    info: dict[str, str] = {}
    if graph_info is None:
        return info
    
    for node in graph_info.graph.nodes:
        if node.op == "call_function":
            if node.target is hl_memory_ops.store:
                info["store"] = node.name
            elif node.target is hl_tracing_ops._phi:
                info["phi"] = node.name
    
    return info


# -----------------------------------------------------------------------------
# Validation utility
# -----------------------------------------------------------------------------


def validate_with_helion_opt(
    mlir_text: str,
    *,
    opt_path: str | Path | None = None,
    extra_args: Iterable[str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run `helion-opt` (falling back to `mlir-opt`) to confirm the emitted IR parses."""
    tool_candidates: Iterable[Path] = HELION_OPT_CANDIDATES if opt_path is None else [Path(opt_path)]
    
    tool: Path | None = None
    for candidate in tool_candidates:
        if candidate.exists():
            tool = candidate
            break
    
    if tool is None:
        raise FileNotFoundError(
            "Unable to locate `helion-opt` or `mlir-opt`. "
            "Pass `mlir_opt_path` explicitly once the project is built."
        )
    
    args = [str(tool)]
    if tool.name == "mlir-opt":
        args.append("-allow-unregistered-dialect")
    if extra_args:
        args.extend(extra_args)
    
    return subprocess.run(
        args,
        input=mlir_text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
