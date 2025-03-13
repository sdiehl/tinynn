"""
Neural network example using MicroAutograd.

This example demonstrates training a simple neural network on a toy dataset
using the MicroAutograd library.
"""
import sys
import os
import random
import math
from sklearn.datasets import make_moons, make_blobs
import numpy as np
import matplotlib.pyplot as plt

# Add the parent directory to the path so we can import our library
sys.path.append(os.path.abspath('.'))

from micrograd.engine import Value
from micrograd.nn import MLP
from micrograd.optim import SGD

def generate_spiral_data(n_points=100, n_classes=2, noise=0.1):
    X, y = make_moons(n_samples=100, noise=0.1)
    y = y*2
    return X, y


def plot_decision_boundary(model, X, y, title='Decision Boundary'):
    """
    Plot the decision boundary of a binary classifier.

    Args:
        model: The trained model that takes 2D inputs and outputs a scalar
        X: Input features (2D points)
        y: Labels (class indices)
        title: Plot title
    """
    # Set up a grid of points to evaluate the model
    h = 0.05  # Step size in the mesh
    margin = 0.5  # Margin around the data
    x_min, x_max = X[:, 0].min() - margin, X[:, 0].max() + margin
    y_min, y_max = X[:, 1].min() - margin, X[:, 1].max() + margin
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Create mesh of input points
    mesh_points = np.c_[xx.ravel(), yy.ravel()]

    # Evaluate model on all mesh points
    Z = np.zeros(mesh_points.shape[0])
    batch_size = 100  # Process in batches to avoid memory issues

    for i in range(0, len(mesh_points), batch_size):
        batch = mesh_points[i:i+batch_size]
        for j, point in enumerate(batch):
            # Convert to Value objects
            x_val = Value(point[0])
            y_val = Value(point[1])
            # Forward pass and store output
            pred = model([x_val, y_val])
            Z[i+j] = pred.data

    # Reshape back to grid
    Z = Z.reshape(xx.shape)

    # Plot the decision boundary and data points
    plt.figure(figsize=(10, 8))

    # Decision boundary contour
    contour = plt.contour(xx, yy, Z, levels=[0.5], colors='k', linestyles='-')

    # Color regions
    plt.contourf(xx, yy, Z, levels=[0, 0.5, 1], alpha=0.3,
                 colors=['#FFAAAA', '#AAAAFF'])

    # Data points
    plt.scatter(X[y==0, 0], X[y==0, 1], c='red', edgecolors='k', s=50, label='Class 0')
    plt.scatter(X[y==1, 0], X[y==1, 1], c='blue', edgecolors='k', s=50, label='Class 1')

    plt.title(title)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend()
    plt.savefig('decision_boundary.png')
    plt.close()

def main():
    """Run the neural network example."""
    print("=== Neural Network Example ===")

    # Generate a simple binary classification dataset (spiral)
    X, y = generate_spiral_data(n_points=100, n_classes=2, noise=0.1)

    # Create a neural network: 2 inputs -> 16 hidden -> 8 hidden -> 1 output
    # Multiple layers help with nonlinear decision boundaries
    model = MLP(nin=2, nouts=[16, 16, 1])

    # Create an optimizer with appropriate learning rate
    optimizer = SGD(model.parameters(), lr=0.1)

    # Training parameters
    n_epochs = 100
    batch_size = 32

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

                # Compute loss: binary cross-entropy (more appropriate for classification)
                target = 1.0 if y[idx] == 1 else -1.0  # Use -1/1 targets for better tanh performance
                # Use a margin loss: maximize correct class score by at least margin
                margin = 1.0
                loss = (margin - pred.data * target)**2

                # Accumulate loss
                batch_loss = batch_loss + loss

                # Check accuracy (with tanh activation, output is close to -1 or 1)
                pred_class = 1 if pred.data > 0 else 0
                if pred_class == y[idx]:
                    correct += 1

            # Average loss over the batch
            batch_loss = batch_loss * (1.0 / len(batch_indices))

            # Backward pass
            batch_loss.backward()

            # Update parameters
            optimizer.step()

            # Track total loss
            total_loss += batch_loss.data

        # Record metrics
        avg_loss = total_loss / (len(X) / batch_size)
        accuracy = correct / len(X)
        losses.append(avg_loss)
        accuracies.append(accuracy)

        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}: Loss={avg_loss:.4f}, Accuracy={accuracy:.4f}")

    # Plot training progress
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(accuracies)
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.tight_layout()
    plt.savefig('training_progress.png')
    plt.close()

    print("Training complete!")
    print(f"Final accuracy: {accuracies[-1]:.4f}")

    # Plot decision boundary
    plot_decision_boundary(model, X, y, title='Neural Network Decision Boundary')
    print("Decision boundary plot saved to 'decision_boundary.png'")

    # Visualize the computation graph for a single prediction
    print("Generating computation graph visualization...")
    x_sample = [Value(X[0][0], label='x1'), Value(X[0][1], label='x2')]
    pred = model(x_sample)
    # draw_dot(pred, filename='neural_network_graph')
    # print("Visualization saved to 'neural_network_graph.png'")

if __name__ == "__main__":
    main()
