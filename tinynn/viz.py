"""
Visualization utilities for drawing computation graphs.
"""

from io import StringIO
import networkx as nx  # type: ignore
from networkx.drawing.nx_pydot import write_dot  # type: ignore
import matplotlib.pyplot as plt
from typing import Any


def trace(root):
    """
    Builds a set of all nodes and edges in a graph.

    Args:
        root: The root node of the computation graph.

    Returns:
        Tuple of (nodes, edges) where nodes is a set of nodes in the graph,
        and edges is a set of tuples (parent, child).
    """
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges


def create_nx_graph(root):
    """
    Creates a networkx graph from a computation graph.

    Args:
        root: The root node of the computation graph.

    Returns:
        A networkx DiGraph object.
    """
    G = nx.DiGraph()

    nodes, edges = trace(root)

    # Add nodes
    for n in nodes:
        uid = str(id(n))
        label = f"data: {n.data:.4f}\ngrad: {n.grad:.4f}"
        G.add_node(uid, label=label, shape="box")

        if n._op:
            # Add operation node
            op_uid = uid + n._op
            G.add_node(op_uid, label=n._op, shape="ellipse")
            G.add_edge(op_uid, uid)

            # Connect inputs to operation
            for child in n._prev:
                G.add_edge(str(id(child)), op_uid)

    return G


def draw_dot(root, output_file=None, format="png"):
    """
    Visualizes a computation graph using networkx and pydot.

    Args:
        root: The root node of the computation graph.
        output_file: Path to save the output file (without extension).
        format: The output format (png, pdf, etc.)

    Returns:
        The networkx DiGraph object.
    """
    G = create_nx_graph(root)

    if output_file:
        write_dot(G, f"{output_file}.dot")

    return G


def visualize(root: Any, name: str = "computation_graph", show: bool = True) -> str:
    """
    Visualizes a computation graph and saves it to a file.

    Args:
        root: The root node of the computation graph.
        name: The output file name (without extension).
        show: Whether to display the plot.
    """
    G = draw_dot(root, output_file=name)

    plt.figure(figsize=(12, 8))

    # Capture the output of the dot command
    out = StringIO()
    write_dot(G, out)
    return out.getvalue()
