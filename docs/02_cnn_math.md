# Convolutional Neural Networks: Math Derivations and Experimental Findings

This document derives the mathematics behind convolution, the im2col
technique, and the vanishing gradient problem at depth -- backed by
the numerical verification and controlled experiments performed in
`src/phase2_cnn/` and `notebooks/phase2_cnn_from_scratch.ipynb`.

---

## 1. What Convolution Actually Computes

For a single output position, a 2D convolution computes a sliding-window
dot product between a kernel and a local patch of the input:

    output[c_out, i, j] = sum over (c_in, ky, kx) of
        kernel[c_out, c_in, ky, kx] * input[c_in, i*stride + ky, j*stride + kx]
        + bias[c_out]

Where (i, j) indexes the output spatial position, and (ky, kx) indexes
positions within the kernel window.

**Output spatial size:**

    H_out = floor((H + 2*padding - kernel_size) / stride) + 1

This formula was verified directly: a stride=1, padding=1, kernel_size=3
convolution preserves spatial dimensions exactly (H_out = H), and a
stride=2, padding=0 convolution on an 8x8 input with a 3x3 kernel
correctly produces a 3x3 output, as confirmed in
`scripts/test_conv_from_scratch.py`.

---

## 2. The im2col Technique

Naively, computing convolution requires a 6-nested loop (batch, output
channel, output height, output width, kernel height, kernel width) --
extremely slow in Python. The im2col technique reframes convolution
as a SINGLE matrix multiplication:

**Step 1 -- im2col:** extract every kernel-sized patch the kernel will
slide over, and flatten each into a column of a matrix:

    cols: shape (C_in * kernel_size * kernel_size, N * H_out * W_out)

**Step 2 -- reshape the kernel:**

    kernel_reshaped: shape (C_out, C_in * kernel_size * kernel_size)

**Step 3 -- matrix multiply:**

    output = kernel_reshaped @ cols
    output: shape (C_out, N * H_out * W_out) -> reshape to (N, C_out, H_out, W_out)

This is not a simplified teaching approximation -- it is the same
im2col + GEMM (General Matrix Multiply) strategy used internally by
cuDNN and most production deep learning frameworks, because dense
matrix multiplication is dramatically faster than nested Python loops
and is what GPUs are optimized to execute.

**Verification:** our from-scratch `conv2d_forward()` was compared
directly against PyTorch's `nn.Conv2d`, using IDENTICAL weights and
inputs loaded into both. Results:

    stride=1, padding=1: max difference = 3.75e-07
    stride=2, padding=0: max difference = 1.78e-15

Both differences are consistent with ordinary floating-point
accumulation noise between two independently-implemented matrix
multiply pathways (our NumPy `@` operator vs PyTorch's backend), NOT
a implementation discrepancy. This confirms our understanding of
convolution matches PyTorch's computation exactly.

---

## 3. The Vanishing Gradient Problem, Formally

Consider a deep plain network with L layers, each applying a
convolution followed by ReLU. During backpropagation, the gradient
flowing back to layer l is a product of L-l Jacobian terms:

    dL/da^(l) = dL/da^(L) * (product from k=l+1 to L of da^(k)/da^(k-1))

Each Jacobian term da^(k)/da^(k-1) includes:
    - the ReLU derivative (0 or 1, zeroing roughly half of all
      gradient components on average)
    - the convolution weight matrix's own scale

If each of these terms has expected magnitude less than 1 (common with
ReLU's ~50% zeroing rate and typical weight initialization), the
PRODUCT of many such terms shrinks EXPONENTIALLY with depth:

    ||dL/da^(l)|| ~ (typical_term_magnitude)^(L-l)

For a term magnitude of even 0.7 and a depth gap of 20 layers:

    0.7^20 ~= 0.0008

This is why deep plain networks don't just train "a bit slower" --
the gradient signal can shrink by orders of magnitude before reaching
early layers, effectively freezing them at their random initialization.

**This was verified directly, not just theoretically.** In our depth
ablation study (`src/phase2_cnn/depth_ablation.py`), PlainCNN at
depth 12 showed EXACTLY ZERO gradient norm in all 12 tracked
convolutional layers on the final training batch. At depth 20, layers
0 through 15 (the first 80% of the network) showed exactly zero
gradient, with only the last 4 layers retaining any signal at all --
and that remaining signal ranged from 2e-8 to 8.9e-4, vanishingly
small compared to a healthy network's typical gradient norms of 0.1-1.5.

The corresponding accuracy collapsed to EXACTLY 23.5% at depths 12,
16, and 20 -- statistically indistinguishable from chance-level
guessing on a 4-class problem (25%), and identical across three
different depths, the signature of complete training failure rather
than gradual degradation.

---

## 4. Why Residual Connections Solve This

A residual block computes:

    output = F(x) + x

Instead of learning a direct mapping H(x), the block learns a
RESIDUAL F(x) = H(x) - x, added back to the input via an identity
shortcut.

**The gradient of this operation, by the chain rule:**

    d(output)/dx = dF(x)/dx + d(x)/dx = dF(x)/dx + 1

The `+1` term is the critical insight. No matter how small
dF(x)/dx becomes (even if it approaches zero, exactly as we
observed in PlainCNN), the gradient flowing through the identity
shortcut is EXACTLY 1 -- providing an unimpeded path for gradient
signal to reach every earlier layer, completely bypassing whatever
vanishing might occur in the transformation path F(x).

Stacking L residual blocks, the gradient at the input becomes:

    dL/dx^(0) = dL/dx^(L) * product over l of (1 + dF^(l)/dx^(l))

Because each factor is (1 + something), rather than a pure product of
potentially-small terms, the gradient magnitude is bounded below by
contributions from the "+1" terms alone -- it cannot vanish to
exactly zero the way a plain network's product-of-small-terms can.

**This was verified directly.** ResNetCNN at depth 20 (50 tracked
convolutional layers, including all shortcut projections) showed
NON-ZERO gradients at every single layer, from `layer_0_stem.0`
(norm 1.48) through `layer_50_blocks.19.conv2` (norm 0.17) --
never dropping to zero, never becoming vanishingly small, despite
this network being nearly 3x deeper than the collapsed PlainCNN
configurations. This produced a 3x accuracy advantage over PlainCNN
at the identical depth budget (70.1% vs 23.5% at depth 12).

---

## 5. Batch Normalization: Why It Stabilizes Training

BatchNorm normalizes activations within each mini-batch, per channel:

    x_hat = (x - batch_mean) / sqrt(batch_variance + epsilon)
    y = gamma * x_hat + beta

Where gamma and beta are LEARNED parameters (allowing the network to
undo the normalization if that's actually optimal), and epsilon is a
small constant preventing division by zero.

**Why this stabilizes training at higher learning rates:** large
gradient steps can push activations to extreme values, which then
compound through subsequent layers -- exactly the exploding/vanishing
gradient issue this whole document addresses. BatchNorm re-centers
and re-scales activations back to a controlled range (mean 0,
variance 1, before the learned gamma/beta) after EVERY layer,
regardless of how large the preceding weight update was. This breaks
the compounding effect that would otherwise let a large learning
rate cause runaway instability.

**This was verified experimentally.** At learning rate 0.01, PlainCNN
collapsed to chance-level accuracy (24.0%) while BatchNormCNN
remained stable (73.5%) -- a ~50 percentage point gap at the
IDENTICAL learning rate and architecture depth, differing only in
the presence of BatchNorm layers.

---

## 6. Summary: From Theory to Verified Evidence

Every claim in this document was tested, not just asserted:

| Claim | Verification |
|---|---|
| Our conv2d matches PyTorch's conv2d | Max diff 1.78e-15 to 3.75e-07 across stride/padding configs |
| Deep plain networks suffer vanishing gradients | Gradient norms measured EXACTLY ZERO in 12-16 layers at depth 12-20 |
| This causes catastrophic accuracy loss | Accuracy collapsed to exactly chance level (23.5%) at those same depths |
| Residual connections prevent this | ResNetCNN showed non-zero gradients at every layer, even at depth 20 |
| Residual connections preserve accuracy | 3x accuracy advantage over PlainCNN at matched depth (70.1% vs 23.5%) |
| BatchNorm stabilizes higher learning rates | ~50 point accuracy gap at lr=0.01, identical architecture otherwise |

This is the same rigor pattern established in Phase 1: implement from
scratch, verify numerically against a trusted reference, then run
controlled experiments that isolate a single variable at a time to
produce genuine, falsifiable evidence for each theoretical claim.