"""
Visualization utilities for drawing computation graphs.
"""

from graphviz import Digraph


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


def draw_dot(root, format="png", rankdir="LR"):
    """
    Visualizes a computation graph using graphviz.

    Args:
        root: The root node of the computation graph.
        format: The output format (png, pdf, etc.)
        rankdir: The direction of the graph layout (TB for top-bottom, LR for left-right)

    Returns:
        Graphviz Digraph object.
    """
    dot = Digraph(format=format, graph_attr={"rankdir": rankdir})

    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))

        # for any value in the graph, create a rectangular ('record') node for it
        dot.node(
            name=uid,
            label=f"{{ data: {n.data:.4f} | grad: {n.grad:.4f} }}",
            shape="record",
        )

        if n._op:
            # if this value is the result of some operation, create a node for the op
            # and connect the input values to it, and then connect it to the output value
            op_uid = uid + n._op
            dot.node(name=op_uid, label=n._op)
            dot.edge(op_uid, uid)

            for child in n._prev:
                dot.edge(str(id(child)), op_uid)

    return dot


def visualize(root, name="computation_graph"):
    """
    Visualizes a computation graph and saves it to a file.

    Args:
        root: The root node of the computation graph.
        name: The output file name (without extension).
    """
    dot = draw_dot(root)
    dot.render(name, cleanup=True)
