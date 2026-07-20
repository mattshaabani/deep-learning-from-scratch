"""
src/phase1_mlp/activations.py

Activation functions and their derivatives, implemented from scratch.

Every activation function needs TWO implementations for backprop:
    forward(x)  — the function itself
    backward(x) — its derivative, needed to propagate gradients

Usage:
    from src.phase1_mlp.activations import get_activation
    relu = get_activation("relu")
    output = relu.forward(x)
    grad   = relu.backward(x)
"""

import numpy as np


class Activation:
    """Base class — every activation implements forward and backward."""
    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, x: np.ndarray) -> np.ndarray:
        """Derivative of the activation, evaluated at the PRE-activation input x."""
        raise NotImplementedError


class Sigmoid(Activation):
    """
    Sigmoid: sigma(x) = 1 / (1 + e^-x)
    Derivative: sigma'(x) = sigma(x) * (1 - sigma(x))

    Numerically stable implementation avoids overflow for large negative x
    by branching on the sign of x (standard trick).
    """
    def forward(self, x: np.ndarray) -> np.ndarray:
        # Clip to avoid overflow in exp() for very negative/positive x
        x_clipped = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x_clipped))

    def backward(self, x: np.ndarray) -> np.ndarray:
        s = self.forward(x)
        return s * (1 - s)


class ReLU(Activation):
    """
    ReLU: f(x) = max(0, x)
    Derivative: f'(x) = 1 if x > 0 else 0

    Why ReLU is popular: no vanishing gradient for x>0 (derivative is
    constant 1, unlike sigmoid/tanh which saturate), and it's cheap to compute.
    Downside: "dying ReLU" — units stuck at x<=0 forever have zero gradient.
    """
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def backward(self, x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(np.float64)


class Tanh(Activation):
    """
    Tanh: f(x) = (e^x - e^-x) / (e^x + e^-x)
    Derivative: f'(x) = 1 - tanh(x)^2

    Zero-centered output (unlike sigmoid's [0,1] range) often helps
    optimization, but still saturates for large |x| like sigmoid.
    """
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    def backward(self, x: np.ndarray) -> np.ndarray:
        return 1 - np.tanh(x) ** 2


class Linear(Activation):
    """Identity activation — f(x) = x, f'(x) = 1. Used for regression outputs."""
    def forward(self, x: np.ndarray) -> np.ndarray:
        return x

    def backward(self, x: np.ndarray) -> np.ndarray:
        return np.ones_like(x)


ACTIVATION_REGISTRY = {
    "sigmoid": Sigmoid,
    "relu":    ReLU,
    "tanh":    Tanh,
    "linear":  Linear,
}


def get_activation(name: str) -> Activation:
    """Factory function — get an activation by name."""
    if name not in ACTIVATION_REGISTRY:
        raise ValueError(f"Unknown activation '{name}'. Choose from {list(ACTIVATION_REGISTRY.keys())}")
    return ACTIVATION_REGISTRY[name]()