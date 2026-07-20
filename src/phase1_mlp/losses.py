"""
src/phase1_mlp/losses.py

Loss functions and their derivatives, implemented from scratch.

Usage:
    from src.phase1_mlp.losses import BinaryCrossEntropy
    loss_fn = BinaryCrossEntropy()
    loss = loss_fn.forward(y_true, y_pred)
    grad = loss_fn.backward(y_true, y_pred)
"""

import numpy as np


class BinaryCrossEntropy:
    """
    Binary Cross-Entropy loss for binary classification.

    L = -1/N * sum[ y*log(y_hat) + (1-y)*log(1-y_hat) ]

    Naive gradient w.r.t. y_hat (the POST-sigmoid probability):
        dL/dy_hat = -(y/y_hat - (1-y)/(1-y_hat)) / N
                  = (y_hat - y) / (y_hat * (1 - y_hat)) / N

    This naive gradient is numerically UNSTABLE near y_hat=0 or y_hat=1
    (division by near-zero). In practice we combine this with the
    sigmoid's own derivative during backprop through the output layer,
    where the two cancel into the clean, stable form: (y_hat - y).

    We implement BOTH here — the naive version for transparency/testing,
    and expose a `fused_sigmoid_gradient` used in practice by the network.
    """

    EPSILON = 1e-12   # prevents log(0)

    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_pred_clipped = np.clip(y_pred, self.EPSILON, 1 - self.EPSILON)
        loss = -np.mean(
            y_true * np.log(y_pred_clipped) +
            (1 - y_true) * np.log(1 - y_pred_clipped)
        )
        return float(loss)

    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Naive gradient dL/dy_hat — used only for gradient-checking tests.
        NOT used directly in the network's backward pass (see fused version).
        """
        y_pred_clipped = np.clip(y_pred, self.EPSILON, 1 - self.EPSILON)
        n = y_true.shape[0]
        return (
            (y_pred_clipped - y_true) /
            (y_pred_clipped * (1 - y_pred_clipped)) / n
        )

    def fused_sigmoid_gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        The mathematically-simplified gradient of BCE-loss-composed-with-sigmoid
        with respect to the PRE-activation logit z.

        dL/dz = (y_hat - y) / N

        This is what the network actually uses during backprop through
        the final layer — numerically stable and elegant, derived by
        the sigmoid derivative canceling the BCE gradient's denominator.
        """
        n = y_true.shape[0]
        return (y_pred - y_true) / n