class SGD:
    """
    Stochastic Gradient Descent optimizer.
    """

    def __init__(self, parameters, lr=0.01):
        self.parameters = parameters
        self.lr = lr

    def zero_grad(self):
        """Set all parameter gradients to zero."""
        for p in self.parameters:
            p.grad = 0

    def step(self):
        """
        Updates each parameter as: p = p - lr * p.grad
        """
        for p in self.parameters:
            p.data -= self.lr * p.grad
