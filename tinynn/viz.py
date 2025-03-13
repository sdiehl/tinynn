"""
Visualization utilities for drawing computation graphs.
"""

import sys
import networkx as nx  # type: ignore
from networkx.drawing.nx_pydot import write_dot, graphviz_layout  # type: ignore
import matplotlib.pyplot as plt


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


def visualize(root, name="computation_graph", show=True):
    """
    Visualizes a computation graph and saves it to a file.

    Args:
        root: The root node of the computation graph.
        name: The output file name (without extension).
        show: Whether to display the plot.
    """
    G = draw_dot(root, output_file=name)

    plt.figure(figsize=(12, 8))

    # Get node labels and shapes
    node_labels = nx.get_node_attributes(G, "label")
    node_shapes = nx.get_node_attributes(G, "shape")

    write_dot(G, sys.stdout)

    return

    # Use graphviz_layout for better tree layout
    pos = graphviz_layout(G, prog="dot")

    # Draw nodes with different shapes
    box_nodes = [n for n, s in node_shapes.items() if s == "box"]
    ellipse_nodes = [n for n, s in node_shapes.items() if s == "ellipse"]

    # Draw the graph
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=box_nodes,
        node_color="lightblue",
        node_size=2000,
        node_shape="s",
    )
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=ellipse_nodes,
        node_color="lightgreen",
        node_size=1500,
        node_shape="o",
    )
    nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=20)
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8)

    plt.axis("off")

    if name:
        plt.savefig(f"{name}.png", format="png", dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return G
