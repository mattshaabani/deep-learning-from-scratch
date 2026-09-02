# Transformers and Attention: Math Derivations and Experimental Findings

This document derives the mathematics of self-attention, positional
encoding, and the transformer block -- backed by the numerical
verification and controlled experiments in `src/phase4_transformer/`
and `notebooks/phase4_transformer_gpt.ipynb`.

---

## 1. Scaled Dot-Product Attention

    Attention(Q, K, V) = softmax(Q . K^T / sqrt(d_k)) . V

Where Q (queries), K (keys), V (values) are linear projections of
the input, each of dimension d_k per head. Intuitively, attention
computes a weighted average of V, where the weight given to each
value is determined by how well its corresponding key K matches the
query Q -- a "soft dictionary lookup" over all positions
simultaneously.

**Why divide by sqrt(d_k):** the dot product Q.K sums d_k independent
terms, so its variance grows proportionally to d_k. Without scaling,
larger d_k pushes softmax inputs into large-magnitude, saturated
regions where softmax's gradient approaches zero -- a THIRD distinct
appearance of the vanishing gradient theme running through this
entire capstone (Phase 2: through depth, Phase 3: through time,
here: through the attention mechanism's own internal scale).
Dividing by sqrt(d_k) keeps the variance of the dot products roughly
constant regardless of dimensionality, preventing this saturation.

---

## 2. Multi-Head Attention

    MultiHead(Q,K,V) = Concat(head_1, ..., head_h) . W_O
    head_i = Attention(Q.W_i^Q, K.W_i^K, V.W_i^V)

Instead of one attention computation over the full d_model
dimensions, the model splits into h heads, each operating on
d_model/h dimensions independently, then concatenates results and
projects back to d_model. This lets different heads specialize in
different types of relationships (e.g. syntactic vs positional vs
semantic patterns) without any single head needing to capture
everything.

**Verification:** our from-scratch `MultiHeadSelfAttention`, with
Q/K/V projections, head splitting, scaled dot-product attention, and
concatenation all implemented as explicit tensor operations, passed
`torch.autograd.gradcheck` AND matched PyTorch's own
`nn.MultiheadAttention` to **0.00e+00** difference (bit-for-bit
identical) when identical weights were loaded into both --
confirming our understanding of the full multi-head mechanism is
exactly correct, at the tightest possible tolerance achieved across
all four phases of this capstone.

---

## 3. Causal Masking

For autoregressive language modeling, position i must not be able to
attend to positions j > i (the model cannot see the future during
training or generation). This is implemented by adding negative
infinity to the attention scores at forbidden positions, before the
softmax:

    scores[i,j] = -infinity   for all j > i
    softmax(-infinity) = 0

**Verification:** summing attention weight assigned to every
forbidden future position, across all heads and all query positions,
gave EXACTLY 0.0 -- confirming zero information leakage from future
positions, the correctness guarantee required for valid autoregressive
training.

---

## 4. Why Attention Needs Positional Encoding: A Direct Proof

Self-attention's formula:

    output_i = sum_j( softmax(Q_i . K_j / sqrt(d_k)) . V_j )

contains no reference to the numerical position index i or j -- only
to the CONTENT of Q, K, V at each position. This makes attention
PERMUTATION-EQUIVARIANT: permuting the input positions produces an
identically-permuted output, with no way to distinguish "the token at
position 3" from "the token at position 7" beyond their content alone.

**This was proven directly, not just argued theoretically.** Shuffling
a 6-position sequence and comparing (attention applied to the shuffled
input) against (the original output, then shuffled the same way) gave
a maximum difference of 1.79e-07 -- essentially zero, confirming exact
equivariance. Adding sinusoidal positional encoding before attention
and repeating the same test gave a maximum difference of 2.00e-01 --
six orders of magnitude larger, confirming positional encoding
genuinely breaks the equivariance and gives the model access to
position information it otherwise completely lacks.

**Sinusoidal positional encoding formula:**

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

Each position gets a unique vector, added directly to the token
embedding. Low-index dimensions oscillate at high frequency (encoding
fine position differences), high-index dimensions oscillate at low
frequency (encoding coarse position) -- together giving every
position a distinguishable "fingerprint."

---

## 5. Residual Connections in the Transformer Block: A Third Confirmation

    x = x + MultiHeadAttention(LayerNorm(x))
    x = x + FeedForward(LayerNorm(x))

This is "pre-norm": LayerNorm is applied BEFORE each sublayer, not
after summing with the residual. This keeps the residual path
completely unmodified by normalization -- the identity gradient path
(derivative exactly 1) is never touched, the same principle proven
in Phase 2 (ResNet's `+x` shortcut) and Phase 3 (LSTM's `f_t * c_{t-1}`
gated shortcut).

**Verification:** stacking 12 transformer blocks and measuring
gradient norm at each block (reusing Phase 2's exact per-layer
gradient tracking technique) showed EVERY block with a healthy,
non-zero gradient, ranging from 195.9 to 296.6 -- no systematic decay
from input-side blocks to output-side blocks. Compare directly to
Phase 2's PlainCNN at the same depth (12 layers), which showed
EXACTLY ZERO gradient in every single layer. The same architectural
fix (additive identity shortcut) generalizes across three completely
different underlying operations: convolution, gated recurrence, and
attention.

---

## 6. A Real Initialization Bug: sqrt(d_model) Embedding Scaling

Standard GPT implementations scale token embeddings by sqrt(d_model)
before adding positional encoding:

    embedded = token_embedding(x) * sqrt(d_model)

This scaling assumes a SPECIFIC embedding initialization scale to
begin with. Applying it to PyTorch's default `nn.Embedding`
initialization (standard normal, std=1.0) without adjustment caused
a real, measured problem:

    Raw embedding std:     0.997  (PyTorch default)
    Scaled embedding std:  7.977  (after x sqrt(64)=8 scaling)
    Final logits std:      10.96, ranging from -36.3 to +72.1
    Loss at random init:   57.86
    Theoretical random-guess loss (ln(65)):  4.17

The loss at initialization was **13.9x higher** than the theoretical
random-guessing baseline -- a genuine bug, not normal early-training
noise. The fix: initialize embedding weights with
`std = d_model^-0.5`, so that after the sqrt(d_model) scaling, the
effective embedding magnitude returns to approximately std=1.0:

    d_model^-0.5 * sqrt(d_model) = d_model^-0.5 * d_model^0.5 = 1.0

After this fix, loss at initialization dropped to 6.23 -- much closer
to the theoretical 4.17 baseline, with the small remaining gap
attributable to the output projection and LayerNorm layers'
own default initializations, not a red flag.

---

## 7. Honest Findings: When Attention Wins and When It Doesn't

**Character-level language modeling (Shakespeare corpus):** at
matched parameter budget (GPT: 104,321 params, LSTM: 92,897 params)
and identical 10,000-step training, LSTM achieved a clearly lower
validation loss (1.6140 vs 1.8900) in roughly one-quarter the
wall-clock training time. This is NOT evidence of a flawed
transformer implementation (independently verified to machine
precision above) -- it reflects that next-character prediction on a
small, single-domain corpus is dominated by local, short-range
structure (spelling, common words), which plays directly to LSTM's
built-in recency inductive bias. Sample generations confirmed this
concretely: LSTM produced recognizable Shakespeare character names
and mostly-correct English words, while GPT's small-scale, short-
training generation degraded into unrecognizable fragments after the
first line.

**The copy task (long-range memory, gap=20):** here the ordering
reversed completely. GPT reached 97.1% accuracy, LSTM plateaued at
41.4%, and VanillaRNN never exceeded the random baseline (12.5%) at
all, across identical training budgets. This task requires ONLY
long-range memory with no local structure to exploit -- exactly the
condition under which attention's core structural advantage (O(1)
path length between any two positions, versus recurrence's O(sequence
length) path that must survive a long multiplicative or gated chain)
should dominate, and the experiment confirmed this decisively.

**The honest lesson:** architecture choice should depend on the
TASK's actual structure, not a blanket assumption that "transformers
are always better." Attention's advantage is specifically in
removing the sequential bottleneck for long-range dependencies --
where that bottleneck isn't the limiting factor (as in short,
locally-structured text with limited training data), a strong
inductive bias like LSTM's recurrence can still win.

---

## 8. Summary: The Complete Evidentiary Chain

| Claim | Verification |
|---|---|
| Our attention matches PyTorch's exactly | 0.00e+00 difference vs `nn.MultiheadAttention` |
| Causal masking correctly blocks the future | Exactly 0.0 attention weight leakage |
| Attention alone has no sense of position | Shuffle test: 1.79e-07 difference (equivariant) |
| Positional encoding breaks this | Shuffle test with PE: 2.00e-01 (6 orders larger) |
| Residual connections prevent depth-collapse in transformers too | 12-block stack: gradients 195.9-296.6, never zero |
| A real embedding-scaling bug existed and was fixed | Init loss: 57.86 (buggy) -> 6.23 (fixed) vs 4.17 theoretical |
| LSTM beats small GPT on local-structure tasks | 1.61 vs 1.89 val loss, matched params, 10K steps |
| Attention decisively beats LSTM on long-range tasks | Copy task: 97.1% vs 41.4% vs 12.5% (random) |

This is the same rigor pattern established across all four phases:
implement from scratch, verify numerically against a trusted
reference, then run controlled experiments -- including experiments
that produced unexpected or nuanced results -- to build genuine,
falsifiable evidence rather than simply asserting what the theory
predicts.