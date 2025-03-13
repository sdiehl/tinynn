import random
from sklearn.datasets import make_moons
import numpy as np
import matplotlib.pyplot as plt

from tinynn.engine import Value
from tinynn.nn import MLP
from tinynn.optim import SGD


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

    # Training parameters
    n_epochs = 500  # More epochs for better convergence
    batch_size = 10  # Smaller batch size for better generalization

    # Lists to track progress
    losses = []
    accuracies = []

    print("Training neural network...")

    # Training loop
    for epoch in range(n_epochs):
        # Track metrics for this epoch
        total_loss = 0.0
        correct = 0

        # Shuffle the data
        indices = list(range(len(X)))
        random.shuffle(indices)

        # Mini-batch training
        for start_idx in range(0, len(X), batch_size):
            end_idx = min(start_idx + batch_size, len(X))
            batch_indices = indices[start_idx:end_idx]

            # Zero gradients
            optimizer.zero_grad()

            # Accumulate loss and accuracy over the batch
            batch_loss = Value(0.0)

            for idx in batch_indices:
                # Convert numpy features to Value objects
                x_vals = [Value(X[idx][0]), Value(X[idx][1])]

                # Forward pass
                pred = model(x_vals)
                pred_value = pred[0] if isinstance(pred, list) else pred

                # Binary cross-entropy loss with tanh activation
                target = 1.0 if y[idx] == 1 else -1.0  # Use -1/1 targets for tanh

                # Use MSE loss which works better for this dataset
                loss = (pred_value - target) * (pred_value - target)

                # Accumulate loss
                batch_loss = batch_loss + loss

                # Check accuracy
                pred_class = 1 if pred_value.data > 0 else 0
                if pred_class == y[idx]:
                    correct += 1

            # Scale loss by batch size
            batch_loss = batch_loss * (1.0 / len(batch_indices))

            # Backward pass
            batch_loss.backward()

            # Update parameters
            optimizer.step()

            # Track total loss
            total_loss += batch_loss.data

        # Record metrics
        avg_loss = total_loss / max(1, (len(X) // batch_size))
        accuracy = correct / len(X)
        losses.append(avg_loss)
        accuracies.append(accuracy)

        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch+1}/{n_epochs}: Loss={avg_loss:.4f}, Accuracy={accuracy:.4f}"
            )

        # Early stopping if we reach perfect accuracy
        if accuracy == 1.0 and avg_loss < 0.01:
            print(f"Early stopping at epoch {epoch+1} with 100% accuracy!")
            break

    # Plot training progress
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(accuracies)
    plt.title("Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.tight_layout()
    plt.savefig("output/training_progress.png")
    plt.close()

    print("Training complete!")
    print(f"Final accuracy: {accuracies[-1]:.4f}")

    # Plot decision boundary
    plot_decision_boundary(model, X, y, title="Neural Network Decision Boundary")
    print("Decision boundary plot saved to 'decision_boundary.png'")


if __name__ == "__main__":
    main()
