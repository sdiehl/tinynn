"""
Visualization utilities for drawing computation graphs.
"""
import os
import networkx as nx
import matplotlib.pyplot as plt
from micrograd_simple.engine import Value

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

def draw_dot(root, format='png', rankdir='LR', filename='computation_graph'):
    """
    Visualizes the computational graph using networkx and matplotlib.

    Args:
        root: The root node of the computation graph.
        format: The output format (png, pdf, etc.)
        rankdir: The direction of the graph layout (TB for top-bottom, LR for left-right)
        filename: The output filename without extension
    """
    # Build the graph
    nodes, edges = trace(root)

    # Create a networkx graph
    G = nx.DiGraph()

    # Add nodes
    for n in nodes:
        # Node color based on operation
        color = 'skyblue' if n._op else 'lightgreen'

        # Generate node label
        if n._op:
            label = f"{n._op}\ndata: {n.data:.4f}\ngrad: {n.grad:.4f}"
        else:
            label = f"{n.label if n.label else ''}\ndata: {n.data:.4f}\ngrad: {n.grad:.4f}"

        G.add_node(id(n), label=label, color=color)

    # Add edges
    for n1, n2 in edges:
        G.add_edge(id(n1), id(n2))

    # Create the figure
    plt.figure(figsize=(12, 8))

    # Use spring layout
    pos = nx.spring_layout(G, seed=42)

    # Draw nodes
    node_colors = [G.nodes[n]['color'] for n in G.nodes]
    labels = {n: G.nodes[n]['label'] for n in G.nodes}

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, alpha=0.8, node_size=2000)
    nx.draw_networkx_edges(G, pos, edge_color='gray', width=1.0, arrowsize=20)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10)

    plt.axis('off')
    plt.tight_layout()

    # Save the figure
    plt.savefig(f"{filename}.{format}", format=format, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Visualization saved to '{filename}.{format}'")
    return G

def visualize(root, name='computation_graph'):
    """
    Visualizes a computation graph and saves it to a file.

    Args:
        root: The root node of the computation graph.
        name: The output file name (without extension).
    """
    return draw_dot(root, filename=name)
