"""
src/phase1_mlp/layers.py

Dense (fully-connected) layer implemented from scratch, with
forward and backward passes, and pluggable weight initialization.

Usage:
    from src.phase1_mlp.layers import DenseLayer
    layer = DenseLayer(input_dim=30, output_dim=16, activation="relu")
    output = layer.forward(X)
    grad_input = layer.backward(grad_output)
"""

import numpy as np
from src.phase1_mlp.activations import get_activation


class DenseLayer:
    """
    A single fully-connected layer: z = X.W + b, a = activation(z)

    Forward pass caches X and z (pre-activation) since both are
    needed during the backward pass to compute gradients correctly.
    """

    def __init__(
        self,
        input_dim:  int,
        output_dim: int,
        activation: str = "relu",
        weight_init: str = "he",
    ):
        self.input_dim   = input_dim
        self.output_dim  = output_dim
        self.activation  = get_activation(activation)

        self.W = self._initialize_weights(input_dim, output_dim, weight_init)
        self.b = np.zeros((1, output_dim))

        # Caches for backward pass
        self._X_cache = None
        self._z_cache = None

        # Gradients — populated during backward()
        self.dW = None
        self.db = None

    def _initialize_weights(self, input_dim: int, output_dim: int, method: str) -> np.ndarray:
        """
        Weight initialization schemes.

        Why initialization matters:
            zeros:  every neuron computes the SAME gradient -> network
                    never breaks symmetry, effectively stays a single neuron.
                    (We include this specifically to demonstrate the failure.)
            random: small random values break symmetry, but wrong SCALE
                    causes vanishing (too small) or exploding (too large)
                    activations in deep networks.
            xavier: scales by sqrt(1/input_dim) - derived to keep variance
                    of activations roughly constant across layers, assuming
                    tanh/sigmoid activations.
            he:     scales by sqrt(2/input_dim) - the "2" accounts for ReLU
                    zeroing out half of all pre-activations on average,
                    so we need extra variance to compensate. Standard for ReLU nets.
        """
        if method == "zeros":
            return np.zeros((input_dim, output_dim))

        elif method == "random":
            return np.random.randn(input_dim, output_dim) * 0.01

        elif method == "xavier":
            limit = np.sqrt(1.0 / input_dim)
            return np.random.randn(input_dim, output_dim) * limit

        elif method == "he":
            limit = np.sqrt(2.0 / input_dim)
            return np.random.randn(input_dim, output_dim) * limit

        else:
            raise ValueError(f"Unknown weight_init '{method}'")

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass: z = X.W + b, a = activation(z)

        Caches X and z since backward() needs both:
            X is needed for dL/dW = X^T . dL/dz
            z is needed for activation'(z) in the chain rule
        """
        self._X_cache = X
        z = X @ self.W + self.b
        self._z_cache = z
        return self.activation.forward(z)

    def backward(self, grad_output: np.ndarray, skip_activation_grad: bool = False) -> np.ndarray:
        """
        Backward pass — computes and stores dW, db, and returns
        the gradient to pass to the PREVIOUS layer.

        Args:
            grad_output: dL/da from the layer AFTER this one (or dL/dz
                        directly if skip_activation_grad=True, used for
                        the fused sigmoid+BCE gradient at the output layer).
            skip_activation_grad: if True, grad_output IS ALREADY dL/dz
                        (used when the loss function's fused gradient
                        already accounts for the activation derivative).

        Math:
            dL/dz = dL/da * activation'(z)          [unless skip_activation_grad]
            dL/dW = X^T . dL/dz
            dL/db = sum(dL/dz, axis=0)
            dL/dX = dL/dz . W^T   <- passed to previous layer
        """
        if skip_activation_grad:
            dz = grad_output
        else:
            dz = grad_output * self.activation.backward(self._z_cache)

        self.dW = self._X_cache.T @ dz
        self.db = np.sum(dz, axis=0, keepdims=True)

        dX = dz @ self.W.T
        return dX