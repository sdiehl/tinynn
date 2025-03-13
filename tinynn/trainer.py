import random
import matplotlib.pyplot as plt
from tinynn.engine import Value


class Trainer:
    def __init__(self, model, optimizer, loss_fn=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn or self._mse_loss

        # Training history
        self.losses = []
        self.accuracies = []

    def _mse_loss(self, pred, target):
        """Mean squared error loss function."""
        return (pred - target) * (pred - target)

    def train(
        self, X, y, n_epochs=100, batch_size=32, verbose=True, early_stopping=True
    ):
        """
        Train the model on the given dataset.

        Args:
            X: Features (list or numpy array)
            y: Target values (list or numpy array)
            n_epochs: Number of training epochs
            batch_size: Mini-batch size
            verbose: Whether to print progress
            early_stopping: Whether to stop early on perfect accuracy

        Returns:
            Dictionary containing training history
        """
        print("Training neural network...")

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
                self.optimizer.zero_grad()

                # Accumulate loss and accuracy over the batch
                batch_loss = Value(0.0)

                for idx in batch_indices:
                    # Convert numpy features to Value objects
                    x_vals = [Value(X[idx][0]), Value(X[idx][1])]

                    # Forward pass
                    pred = self.model(x_vals)
                    pred_value = pred[0] if isinstance(pred, list) else pred

                    # Binary cross-entropy loss with tanh activation
                    target = 1.0 if y[idx] == 1 else -1.0  # Use -1/1 targets for tanh

                    # Calculate loss
                    loss = self.loss_fn(pred_value, target)

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
                self.optimizer.step()

                # Track total loss
                total_loss += batch_loss.data

            # Record metrics
            avg_loss = total_loss / max(1, (len(X) // batch_size))
            accuracy = correct / len(X)
            self.losses.append(avg_loss)
            self.accuracies.append(accuracy)

            # Print progress every 10 epochs
            if verbose and (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch+1}/{n_epochs}: Loss={avg_loss:.4f}, Accuracy={accuracy:.4f}"
                )

            # Early stopping if we reach perfect accuracy
            if early_stopping and accuracy == 1.0 and avg_loss < 0.01:
                print(f"Early stopping at epoch {epoch+1} with 100% accuracy!")
                break

        print("Training complete!")
        print(f"Final accuracy: {self.accuracies[-1]:.4f}")

        return {"losses": self.losses, "accuracies": self.accuracies}

    def plot_training_progress(self, save_path="output/training_progress.png"):
        """Plot and save the training progress."""
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(self.losses)
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")

        plt.subplot(1, 2, 2)
        plt.plot(self.accuracies)
        plt.title("Training Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
