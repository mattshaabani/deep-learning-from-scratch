import numpy as np
from src.phase1_mlp.activations import get_activation

def numerical_gradient(f, x, eps=1e-5):
    """
    Approximate df/dx using the finite difference method:
        f'(x) ~= (f(x+eps) - f(x-eps)) / (2*eps)

    This is THE standard way to verify hand-derived backprop math
    is implemented correctly, independent of the analytical derivative.
    """
    return (f(x + eps) - f(x - eps)) / (2 * eps)

np.random.seed(42)
x = np.random.randn(5) * 2   # random test points, some negative some positive

for name in ["sigmoid", "relu", "tanh", "linear"]:
    activation = get_activation(name)

    analytical_grad = activation.backward(x)
    numerical_grad   = numerical_gradient(activation.forward, x)

    max_diff = np.max(np.abs(analytical_grad - numerical_grad))

    print(f"{name:10s} | analytical: {analytical_grad.round(4)} | numerical: {numerical_grad.round(4)} | max_diff: {max_diff:.2e}")
    assert max_diff < 1e-4, f"{name} gradient check FAILED"

print("\nAll activation gradients verified correct via numerical gradient checking!")