from sklearn.datasets import make_moons
import numpy as np
import matplotlib.pyplot as plt

from tinynn.engine import Value
from tinynn.nn import MLP
from tinynn.optim import SGD
from tinynn.trainer import Trainer


def generate_data(n_samples=100, noise=0.1):
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
    return X, y


def plot_decision_boundary(model, X, y, title="Decision Boundary"):
    # Set up a grid of points to evaluate the model
    h = 0.25
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Create mesh of input points
    Xmesh = np.c_[xx.ravel(), yy.ravel()]

    # Create Value objects for each point and evaluate model
    inputs = [list(map(Value, xrow)) for xrow in Xmesh]
    scores = list(map(model, inputs))

    # Convert to binary decision (above or below 0)
    Z = np.array([s[0].data > 0 if isinstance(s, list) else s.data > 0 for s in scores])
    Z = Z.reshape(xx.shape)

    # Plot the decision boundary and data points
    plt.figure(figsize=(10, 8))

    # Color regions using spectral colormap
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)

    # Data points
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral, edgecolors="k")

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(title)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.savefig("output/decision_boundary.png")
    plt.close()


def main():
    """Run the neural network example."""
    print("=== Neural Network Example ===")

    # Generate a dataset with very low noise for better separation
    X, y = generate_data(n_samples=100, noise=0.01)

    # Create a neural network 2 inputs -> 32 -> 32 -> 16 -> 1 output
    model = MLP(nin=2, nouts=[32, 32, 16, 1])

    # Create an optimizer with appropriate learning rate
    optimizer = SGD(model.parameters(), lr=0.005)

    # Create trainer
    trainer = Trainer(model, optimizer)

    # Train the model
    trainer.train(X, y, n_epochs=500, batch_size=10, verbose=True, early_stopping=True)

    # Plot training progress
    trainer.plot_training_progress()

    # Plot decision boundary
    plot_decision_boundary(model, X, y, title="Neural Network Decision Boundary")
    print("Decision boundary plot saved to 'decision_boundary.png'")


if __name__ == "__main__":
    main()
