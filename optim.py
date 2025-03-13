"""
Optimization algorithms for neural networks.
"""

class SGD:
    """
    Stochastic Gradient Descent optimizer.
    """
    
    def __init__(self, parameters, lr=0.01):
        """
        Initialize the SGD optimizer.
        
        Args:
            parameters: List of parameters to optimize
            lr: Learning rate
        """
        self.parameters = parameters
        self.lr = lr
    
    def zero_grad(self):
        """Set all parameter gradients to zero."""
        for p in self.parameters:
            p.grad = 0
    
    def step(self):
        """
        Perform one optimization step using gradient descent.
        
        Updates each parameter as: p = p - lr * p.grad
        """
        for p in self.parameters:
            p.data -= self.lr * p.grad