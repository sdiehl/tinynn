# TinyNN

A minimalist neural network implementation in Python. Showing automatic differentiation and training. This is for teaching purposes, it's pure Python and very slow, but easy to understand.

## Installation

```bash
git clone https://github.com/sdiehl/tinynn.git
uv sync
```

## Examples

Run the automatic differentiation example:

```bash
uv run main.py
```

Run the training example:

```bash
uv run train.py
```

## Usage

Basic example of creating and training a neural network:

```python
from tinynn.nn import MLP
from tinynn.optim import SGD
from tinynn.trainer import Trainer

# Create model (2 inputs -> 32 hidden -> 32 hidden -> 16 hidden -> 1 output)
model = MLP(nin=2, nouts=[32, 32, 16, 1])

# Initialize optimizer
optimizer = SGD(model.parameters(), lr=0.005)

# Create trainer
trainer = Trainer(model, optimizer)

# Train model
history = trainer.train(
    X_train, 
    y_train,
    n_epochs=500,
    batch_size=10,
    verbose=True
)

# Make predictions
predictions = model(X_test)
```

The trainer will automatically track and print training progress, including loss and accuracy metrics. You can also visualize the training progress:

```python
trainer.plot_training_progress()
```

## License

MIT