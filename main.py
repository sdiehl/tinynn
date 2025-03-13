from tinynn.engine import Value
from tinynn.nn import MLP

from tinynn.viz import visualize


def demo_basic_operations():
    """Demonstrate basic autograd operations"""
    print("=== Basic Operations Demo ===")

    # Create some values
    a = Value(2.0, label="a")
    b = Value(-3.0, label="b")
    c = Value(10.0, label="c")

    # Perform operations
    d = a * b
    d.label = "d"
    e = d + c
    e.label = "e"
    f = e.relu()
    f.label = "f"

    print(f"a = {a.data}, b = {b.data}, c = {c.data}")
    print(f"d = a * b = {d.data}")
    print(f"e = d + c = {e.data}")
    print(f"f = e.relu() = {f.data}")

    # Compute gradients
    f.backward()

    print("\nGradients after backward pass:")
    print(f"a.grad = {a.grad}")
    print(f"b.grad = {b.grad}")
    print(f"c.grad = {c.grad}")
    print(f"d.grad = {d.grad}")
    print(f"e.grad = {e.grad}")
    print(f"f.grad = {f.grad}")

    # Visualize the computation graph
    print("\nGenerating computation graph visualization...")
    visualize(f, "output/basic_operations_graph")


def demo_simple_neural_network():
    """Demonstrate a very simple neural network without external data"""
    print("\n=== Simple Neural Network Demo ===")

    # Create a small multilayer perceptron
    x = [Value(1.0), Value(2.0), Value(3.0)]

    # Neural network with 3 inputs, hidden layer of 4 neurons, and 1 output
    model = MLP(3, [4, 1])

    # Forward pass
    # Since the output layer has only one neuron, the result will be a single Value
    y_pred = model(x)

    # Handle both single output and list output cases
    y_pred_value = y_pred[0] if isinstance(y_pred, list) else y_pred

    print(f"Input: [{', '.join(str(i.data) for i in x)}]")
    print(f"Prediction: {y_pred_value.data}")

    # Define a target and compute loss
    y_true = 0.5
    loss = (y_pred_value - y_true) ** 2

    print(f"Target: {y_true}")
    print(f"Loss: {loss.data}")

    # Backward pass
    loss.backward()

    # Update weights using gradient descent (manually)
    learning_rate = 0.1
    for p in model.parameters():
        p.data -= learning_rate * p.grad

    # Second forward pass after weight update
    y_pred = model(x)
    y_pred_value = y_pred[0] if isinstance(y_pred, list) else y_pred
    loss = (y_pred_value - y_true) ** 2

    print("\nAfter one gradient descent step:")
    print(f"New prediction: {y_pred_value.data}")
    print(f"New loss: {loss.data}")

    # Visualize the computation graph
    print("\nGenerating computation graph visualization...")
    visualize(loss, "output/neural_network_graph")


def demo_nn_architecture():
    from tinynn.viz_nn import visualize_mlp, visualize_network

    # Visualize a specific network structure
    visualize_network([2, 32, 32, 16, 1], save_path="output/network_architecture.png")

    # Visualize an existing MLP model
    model = MLP(nin=2, nouts=[8, 8, 1])
    visualize_mlp(model, save_path="output/model_architecture_2.png")


def main():
    # Run the basic operations demo
    demo_basic_operations()

    # Run the simple neural network demo
    demo_simple_neural_network()

    # Run the neural network architecture demo
    demo_nn_architecture()


if __name__ == "__main__":
    main()
