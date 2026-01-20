from typing import Any

def print_device_ir(bound_kernel: Any) -> None:
    """Prints the Device IR graphs, filtering out rolled reductions."""
    print("=== Device IR ===")
    
    # Filter out graphs that are part of rolled reductions
    rolled_ids = set()
    # Check for rolled_reductions attribute safely, though it should exist on valid device_ir
    if hasattr(bound_kernel.host_function.device_ir, "rolled_reductions"):
        rolled_ids = {
            info.new_graph_id 
            for info in bound_kernel.host_function.device_ir.rolled_reductions 
            if info.new_graph_id is not None
        }

    for i, g in enumerate(bound_kernel.host_function.device_ir.graphs):
        if i in rolled_ids:
            continue
        print(f"Graph {i}: {type(g).__name__}")
        g.graph.print_tabular()
    print("\n")

def print_nodes_with_symbols(bound_kernel: Any) -> None:
    """Prints nodes that have symbolic values associated with them."""
    print("=== Nodes with symbols ===")
    for i, g in enumerate(bound_kernel.host_function.device_ir.graphs):
        for node in g.graph.nodes:
            if "val" in node.meta:
                print(f"Node {node.name} : {node.meta['val']}")
                    
    print("\n")

def print_compile_env(bound_kernel: Any) -> None:
    """Prints the CompileEnvironment information."""
    print("=== Compile Environment ===")
    env = bound_kernel.env
    print(f"Block Sizes ({len(env.block_sizes)}):")
    for bs in env.block_sizes:
        print(f"  Block {bs.block_id}: Size={bs.size}, Var={bs.var}, Reduction={bs.reduction}, Source={bs.block_size_source}")
    print(f"Shape Env ({len(env.shape_env.var_to_val)}):")
    for var, val in env.shape_env.var_to_val.items():
        print(f"  Var {var}: {val}")
    print("\n")

def print_debug_info(bound_kernel: Any) -> None:
    """Prints Device IR, nodes with symbols, and compile environment."""
    print_device_ir(bound_kernel)
    print_nodes_with_symbols(bound_kernel)
    print_compile_env(bound_kernel)
