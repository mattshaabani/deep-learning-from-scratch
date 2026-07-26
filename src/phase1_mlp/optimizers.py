"""
src/phase1_mlp/optimizers.py

Optimizers implemented from scratch: SGD, SGD+Momentum, RMSProp, Adam.

Each optimizer maintains its own internal state (velocity, squared-gradient
averages, timestep) PER PARAMETER TENSOR, since different layers' weight
matrices need independent optimizer state.

Usage:
    from src.phase1_mlp.optimizers import get_optimizer
    optimizer = get_optimizer("adam", learning_rate=0.01)
    optimizer.update(param, grad, param_id="layer1_W")
"""

import numpy as np
from src.utils.config import settings


class Optimizer:
    """Base class — every optimizer implements update()."""
    def update(self, param: np.ndarray, grad: np.ndarray, param_id: str) -> np.ndarray:
        raise NotImplementedError


class SGD(Optimizer):
    """
    Vanilla Stochastic Gradient Descent.

    theta = theta - lr * d_theta

    No memory of past gradients — each update depends only on the
    CURRENT gradient. Simple, but can be slow and oscillate in
    ravines (steep in one direction, shallow in another).
    """
    def __init__(self, learning_rate: float):
        self.lr = learning_rate

    def update(self, param: np.ndarray, grad: np.ndarray, param_id: str) -> np.ndarray:
        return param - self.lr * grad


class Momentum(Optimizer):
    """
    SGD with Momentum.

    v = beta*v + (1-beta)*d_theta
    theta = theta - lr*v

    Intuition: a ball rolling downhill accumulates velocity in
    consistent gradient directions and dampens oscillation in
    inconsistent (noisy) directions -- like exponential smoothing
    applied to the gradient signal itself.
    """
    def __init__(self, learning_rate: float, beta: float = 0.9):
        self.lr   = learning_rate
        self.beta = beta
        self.velocity = {}   # per-parameter state, keyed by param_id

    def update(self, param: np.ndarray, grad: np.ndarray, param_id: str) -> np.ndarray:
        if param_id not in self.velocity:
            self.velocity[param_id] = np.zeros_like(param)

        v = self.velocity[param_id]
        v = self.beta * v + (1 - self.beta) * grad
        self.velocity[param_id] = v

        return param - self.lr * v


class RMSProp(Optimizer):
    """
    RMSProp -- adapts the learning rate PER PARAMETER based on the
    running average of squared gradients.

    s = beta*s + (1-beta)*d_theta^2
    theta = theta - lr * d_theta / (sqrt(s) + eps)

    Intuition: parameters with historically large gradients get
    their effective step size shrunk (dividing by a large sqrt(s)),
    while parameters with small gradients get relatively larger
    steps -- this helps navigate ravines where different parameters
    need very different step sizes.
    """
    def __init__(self, learning_rate: float, beta: float = 0.999, epsilon: float = 1e-8):
        self.lr      = learning_rate
        self.beta    = beta
        self.epsilon = epsilon
        self.sq_avg  = {}

    def update(self, param: np.ndarray, grad: np.ndarray, param_id: str) -> np.ndarray:
        if param_id not in self.sq_avg:
            self.sq_avg[param_id] = np.zeros_like(param)

        s = self.sq_avg[param_id]
        s = self.beta * s + (1 - self.beta) * (grad ** 2)
        self.sq_avg[param_id] = s

        return param - self.lr * grad / (np.sqrt(s) + self.epsilon)


class Adam(Optimizer):
    """
    Adam -- combines Momentum (first moment) and RMSProp (second moment),
    with bias correction for early timesteps.

    m = beta1*m + (1-beta1)*d_theta          <- momentum term
    s = beta2*s + (1-beta2)*d_theta^2        <- RMSProp term
    m_hat = m / (1 - beta1^t)                <- bias correction
    s_hat = s / (1 - beta2^t)                <- bias correction
    theta = theta - lr * m_hat / (sqrt(s_hat) + eps)

    Why bias correction matters: m and s are initialized to zero,
    which biases their early estimates TOWARD zero (especially with
    beta close to 1). Dividing by (1 - beta^t) compensates for this,
    and the correction naturally fades as t grows since beta^t -> 0.
    """
    def __init__(
        self,
        learning_rate: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ):
        self.lr      = learning_rate
        self.beta1   = beta1
        self.beta2   = beta2
        self.epsilon = epsilon
        self.m = {}   # first moment (momentum) per parameter
        self.s = {}   # second moment (squared grad) per parameter
        self.t = {}   # timestep counter per parameter

    def update(self, param: np.ndarray, grad: np.ndarray, param_id: str) -> np.ndarray:
        if param_id not in self.m:
            self.m[param_id] = np.zeros_like(param)
            self.s[param_id] = np.zeros_like(param)
            self.t[param_id] = 0

        self.t[param_id] += 1
        t = self.t[param_id]

        m = self.beta1 * self.m[param_id] + (1 - self.beta1) * grad
        s = self.beta2 * self.s[param_id] + (1 - self.beta2) * (grad ** 2)

        self.m[param_id] = m
        self.s[param_id] = s

        m_hat = m / (1 - self.beta1 ** t)
        s_hat = s / (1 - self.beta2 ** t)

        return param - self.lr * m_hat / (np.sqrt(s_hat) + self.epsilon)


OPTIMIZER_REGISTRY = {
    "sgd":      SGD,
    "momentum": Momentum,
    "rmsprop":  RMSProp,
    "adam":     Adam,
}


def get_optimizer(name: str, learning_rate: float = None, **kwargs) -> Optimizer:
    """
    Factory function -- get an optimizer by name, pulling defaults
    from config where not explicitly overridden.
    """
    lr = learning_rate or settings.training.learning_rate

    if name not in OPTIMIZER_REGISTRY:
        raise ValueError(f"Unknown optimizer '{name}'. Choose from {list(OPTIMIZER_REGISTRY.keys())}")

    if name == "sgd":
        return SGD(learning_rate=lr)
    elif name == "momentum":
        beta = kwargs.get("beta", settings.training.momentum_beta)
        return Momentum(learning_rate=lr, beta=beta)
    elif name == "rmsprop":
        beta    = kwargs.get("beta", settings.training.rmsprop_beta)
        epsilon = kwargs.get("epsilon", settings.training.adam_epsilon)
        return RMSProp(learning_rate=lr, beta=beta, epsilon=epsilon)
    elif name == "adam":
        beta1   = kwargs.get("beta1", settings.training.adam_beta1)
        beta2   = kwargs.get("beta2", settings.training.adam_beta2)
        epsilon = kwargs.get("epsilon", settings.training.adam_epsilon)
        return Adam(learning_rate=lr, beta1=beta1, beta2=beta2, epsilon=epsilon)