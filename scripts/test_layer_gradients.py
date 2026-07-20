import numpy as np
from src.phase1_mlp.layers import DenseLayer
from src.phase1_mlp.losses import BinaryCrossEntropy

np.random.seed(42)

# Small toy problem for exact gradient checking
X = np.random.randn(8, 4)          # 8 samples, 4 features
y = np.random.randint(0, 2, (8, 1)).astype(float)

layer = DenseLayer(input_dim=4, output_dim=1, activation="sigmoid", weight_init="xavier")
loss_fn = BinaryCrossEntropy()

def compute_loss():
    y_pred = layer.forward(X)
    return loss_fn.forward(y, y_pred)

# Forward + backward pass using our implementation
y_pred = layer.forward(X)
loss = loss_fn.forward(y, y_pred)
grad_output = loss_fn.backward(y, y_pred)   # naive dL/dy_hat
layer.backward(grad_output)                  # computes layer.dW, layer.db

analytical_dW = layer.dW.copy()
analytical_db = layer.db.copy()

print(f"Initial loss: {loss:.6f}")
print(f"Analytical dW:\n{analytical_dW.round(5)}")
print(f"Analytical db: {analytical_db.round(5)}")

# Numerical gradient checking for W
eps = 1e-5
numerical_dW = np.zeros_like(layer.W)

for i in range(layer.W.shape[0]):
    for j in range(layer.W.shape[1]):
        original = layer.W[i, j]

        layer.W[i, j] = original + eps
        loss_plus = compute_loss()

        layer.W[i, j] = original - eps
        loss_minus = compute_loss()

        layer.W[i, j] = original  # restore
        numerical_dW[i, j] = (loss_plus - loss_minus) / (2 * eps)

max_diff_W = np.max(np.abs(analytical_dW - numerical_dW))
print(f"\nNumerical dW:\n{numerical_dW.round(5)}")
print(f"\nMax difference (dW): {max_diff_W:.2e}")

assert max_diff_W < 1e-4, "GRADIENT CHECK FAILED for W"
print("\nGradient check PASSED — our from-scratch backprop matches numerical gradients!")