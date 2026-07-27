# Backpropagation: Full Derivation

This document derives, from first principles, every gradient equation
implemented in `src/phase1_mlp/`. Every result here was independently
verified against numerical gradient checking in `scripts/test_activations.py`
and `scripts/test_layer_gradients.py`, with max differences on the order
of 1e-11 to 1e-12 — confirming the derivations below are correctly implemented.

---

## 1. Setup and Notation

Consider a single dense layer:

    z = X . W + b
    a = f(z)

Where:
- `X` — input, shape (N, d_in): N samples, d_in input features
- `W` — weight matrix, shape (d_in, d_out)
- `b` — bias vector, shape (1, d_out)
- `z` — pre-activation ("logits"), shape (N, d_out)
- `f`  — activation function (ReLU, sigmoid, tanh, etc.)
- `a` — post-activation output, shape (N, d_out)

For a network with L layers, we chain these: a(0) = X, and
a(l) = f(a(l-1) . W(l) + b(l)) for l = 1...L.

---

## 2. The Chain Rule Backbone

Backpropagation is repeated application of the multivariable chain rule.
Given the loss L as a function of the network's final output, we want
dL/dW(l) and dL/db(l) for every layer, plus dL/da(l-1) to propagate
further backward if this isn't the first layer.

The key insight: if we already know dL/da(l) (the gradient of the loss
with respect to this layer's OUTPUT), we can derive everything else:

    dL/dz(l)     = dL/da(l) * f'(z(l))            [element-wise product]
    dL/dW(l)     = (a(l-1))^T . dL/dz(l)
    dL/db(l)     = sum over samples of dL/dz(l)
    dL/da(l-1)   = dL/dz(l) . (W(l))^T            [passed to the previous layer]

This is exactly what `DenseLayer.backward()` implements.

---

## 3. Derivation of dL/dW

Start from a single scalar output element:

    z_j = sum_i(x_i * W_ij) + b_j

By definition of partial derivative:

    dz_j/dW_ij = x_i

By the chain rule:

    dL/dW_ij = dL/dz_j * dz_j/dW_ij = dL/dz_j * x_i

Written for the full batch and vectorized across all i, j simultaneously,
this is exactly a matrix multiplication:

    dL/dW = X^T . dL/dz

Shape check: X^T is (d_in, N), dL/dz is (N, d_out), so the product
is (d_in, d_out) — matching W's shape exactly, as required.

---

## 4. Derivation of dL/db

Since z_j = sum_i(x_i * W_ij) + b_j, we have dz_j/db_j = 1. So:

    dL/db_j = dL/dz_j * 1 = dL/dz_j

Summed across all N samples in the batch (since the same bias is shared
across every sample):

    dL/db = sum over samples of dL/dz

---

## 5. Derivation of dL/dX (propagating to the previous layer)

Symmetric to the dL/dW derivation. Since z_j = sum_i(x_i * W_ij) + b_j:

    dz_j/dx_i = W_ij

So:

    dL/dx_i = sum_j(dL/dz_j * W_ij)

Vectorized:

    dL/dX = dL/dz . W^T

This is what gets passed as `grad_output` to the PREVIOUS layer's
`backward()` call — the recursive mechanism that makes backprop "propagate
backward" through arbitrarily many layers.

---

## 6. The Fused Sigmoid + Binary Cross-Entropy Gradient

This is the most elegant simplification in the whole implementation.

**Binary cross-entropy loss:**

    L = -1/N * sum[ y*log(y_hat) + (1-y)*log(1-y_hat) ]

**Naive gradient w.r.t. y_hat:**

    dL/dy_hat = -1/N * [ y/y_hat - (1-y)/(1-y_hat) ]
              = (y_hat - y) / [ N * y_hat * (1 - y_hat) ]

**Sigmoid derivative** (where y_hat = sigmoid(z)):

    d(y_hat)/dz = y_hat * (1 - y_hat)

**Chain rule to get dL/dz:**

    dL/dz = dL/dy_hat * d(y_hat)/dz
          = [(y_hat - y) / (N * y_hat * (1-y_hat))] * [y_hat * (1-y_hat)]
          = (y_hat - y) / N

The `y_hat * (1 - y_hat)` terms cancel EXACTLY. This is why
`fused_sigmoid_gradient()` in `losses.py` implements just `(y_hat - y) / N`
directly — it is not an approximation, it is the exact analytical
simplification, and it is more numerically stable than computing the two
factors separately (which risks division by near-zero when y_hat is very
close to 0 or 1).

This was verified in `scripts/test_layer_gradients.py`, where the analytical
dW computed via the fused gradient matched numerical finite-difference
gradients to within 1e-11.

---

## 7. Numerical Gradient Checking — Why It Proves Correctness

For any function f and point x, the finite-difference approximation is:

    f'(x) ~= [f(x + eps) - f(x - eps)] / (2*eps)

This is derived directly from the definition of the derivative as a limit,
using a symmetric (central) difference for better accuracy than a one-sided
difference. With eps = 1e-5, this approximation is accurate to roughly
eps^2 = 1e-10, which is why our gradient checks use 1e-4 as the pass/fail
threshold — comfortably above float64 numerical noise, but tight enough to
catch any real implementation bug.

Every activation function and the full `DenseLayer.backward()` were
verified this way before being trusted in any experiment.

---

## 8. Why This Matters Beyond Phase 1

Every optimizer in `optimizers.py` (SGD, Momentum, RMSProp, Adam) operates
on the SAME gradients derived above — dW and db computed via backprop.
The optimizers differ only in HOW they use these gradients (direct update,
momentum-smoothed, adaptively-scaled), not in how the gradients themselves
are computed. Understanding this derivation is therefore the foundation
for every subsequent phase: CNNs (Phase 2) backprop through convolutions
using the identical chain-rule principle, RNNs (Phase 3) backprop through
time using the same rules applied repeatedly across timesteps, and
Transformers (Phase 4) backprop through attention using the same
principle applied to matrix multiplications inside the attention mechanism.