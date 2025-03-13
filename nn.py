"""
Neural network module built on top of the autograd engine.
"""
import random
from micrograd_simple.engine import Value

class Module:
    """Base class for all neural network modules."""
    
    def zero_grad(self):
        """Sets gradients of all parameters to zero."""
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        """Returns a list of all parameters in the module."""
        return []

class Neuron(Module):
    """
    A single neuron with multiple inputs and one output.
    """
    
    def __init__(self, nin, nonlin=True):
        """
        Initialize a neuron with 'nin' inputs.
        
        Args:
            nin: Number of inputs
            nonlin: Whether to apply non-linearity (ReLU)
        """
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0)
        self.nonlin = nonlin

    def __call__(self, x):
        """Forward pass: compute the output of the neuron for the given input."""
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        return act.relu() if self.nonlin else act

    def parameters(self):
        """Return a list of parameters (weights and bias)."""
        return self.w + [self.b]

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"

class Layer(Module):
    """
    A layer of neurons, where each neuron has the same number of inputs.
    """
    
    def __init__(self, nin, nout, **kwargs):
        """
        Initialize a layer with 'nin' inputs and 'nout' neurons.
        
        Args:
            nin: Number of inputs to each neuron
            nout: Number of neurons (outputs)
            **kwargs: Additional arguments to pass to Neuron constructor
        """
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        """Forward pass: compute outputs of all neurons for the given input."""
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        """Return a list of parameters from all neurons."""
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP(Module):
    """
    Multi-layer perceptron (fully connected feed-forward neural network).
    """
    
    def __init__(self, nin, nouts):
        """
        Initialize a multi-layer perceptron.
        
        Args:
            nin: Number of inputs
            nouts: List of number of neurons in each layer
        """
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]

    def __call__(self, x):
        """Forward pass: compute the output of the network for the given input."""
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        """Return a list of parameters from all layers."""
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"