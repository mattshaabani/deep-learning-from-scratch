# Deep Learning From First Principles

A five-phase capstone implementing neural networks from scratch: the
math, the backward pass, and the architectural ideas that solve
vanishing gradients, built up from a NumPy MLP to a GPT fine-tuned
with Direct Preference Optimization.

Every component is implemented explicitly (no `nn.Conv2d`, `nn.LSTM`,
or `nn.MultiheadAttention` used as a black box in the core
implementations) and verified against PyTorch's own reference modules
or numerical gradient checking before being trusted in any experiment.

---

## The Five Phases

    Phase 1: Neural Network Fundamentals       (NumPy, from scratch)
    Phase 2: Convolutional Neural Networks      (PyTorch)
    Phase 3: RNNs and LSTMs                     (PyTorch)
    Phase 4: Attention and Transformers (GPT)   (PyTorch)
    Phase 5: DPO Alignment                      (PyTorch)

Each phase follows the same pattern: implement from scratch, verify
against a trusted reference (numerical gradient checking in Phase 1,
PyTorch's own modules in Phases 2-4), then run controlled experiments
designed to produce falsifiable evidence for the underlying theory --
including experiments that didn't confirm the initial hypothesis, and
were debugged rather than discarded.

---

## The Thread Running Through All Five Phases

Every phase, in a different architecture, runs into the same core
problem: **gradients vanish when they have to survive a long chain of
multiplicative operations**, and every phase finds a version of the
same fix: **an additive path with derivative near 1**.

    Phase 2 (depth):      ResNet's x + F(x) identity shortcut
    Phase 3 (time):       LSTM's f_t * c_{t-1} gated shortcut (conditional on f_t -> 1)
    Phase 4 (attention):  removes the sequential chain entirely --
                           O(1) path length between any two positions
    Phase 4 (depth again): the SAME residual shortcut from Phase 2,
                           now proven to work in a transformer stack too

This was not assumed -- it was measured directly in every phase, with
gradient norms captured layer-by-layer (Phase 2), timestep-by-timestep
(Phase 3), and block-by-block (Phase 4).

---

## Phase Summaries

### Phase 1: Neural Networks From Scratch (NumPy)
Backpropagation, four optimizers (SGD, Momentum, RMSProp, Adam),
L1/L2/dropout regularization, and 5-fold cross-validation, implemented
in pure NumPy and verified with numerical gradient checking
(differences of 1e-11 to 1e-12 against hand-derived analytical
gradients). Key finding: L2 regularization cut 5-fold CV validation
loss by ~34% and roughly halved its variance across folds.

### Phase 2: CNNs From Scratch (PyTorch)
Convolution implemented via im2col, verified against `nn.Conv2d` to
float64 precision. A depth ablation showed a plain 12-layer CNN
collapsing to exactly chance-level accuracy (23.5%, identical across
depths 12/16/20) with gradients measured at EXACTLY ZERO in every
early layer -- and a ResNet at the same depth never collapsing,
maintaining non-zero gradients at every layer and 3x the accuracy.

### Phase 3: RNNs and LSTMs From Scratch (PyTorch)
Gate equations implemented explicitly, LSTM cell verified against
`nn.LSTMCell` to 1e-17 precision. A naive gradient measurement showed
no vanishing signal (a genuine negative result, investigated rather
than hidden); isolating the recurrent signal revealed both VanillaRNN
and LSTM collapsing at random initialization, and a forget-gate-bias
experiment showed a six-order-of-magnitude gradient improvement from
biasing the forget gate toward 1. The copy task then showed the
complete picture: VanillaRNN never learns a 20-step gap (stuck at
random baseline), LSTM reaches ~41%, decisively confirming the
mechanism at the task level.

### Phase 4: Attention and GPT From Scratch (PyTorch)
Multi-head attention verified against `nn.MultiheadAttention` to
0.00e+00 difference (bit-for-bit identical). Direct proof that
attention alone is permutation-equivariant (shuffle test: 1.79e-07
difference) and that positional encoding breaks this (2.00e-01, six
orders of magnitude larger). A real embedding-initialization bug was
found and fixed (initial loss 57.86 -> 6.23). On Shakespeare language
modeling, LSTM beat a matched-parameter GPT (1.61 vs 1.89 val loss) --
but on the copy task, GPT reached 97.1% accuracy against LSTM's 41%,
cleanly showing attention's advantage is task-dependent, not universal.

### Phase 5: DPO Alignment From Scratch (PyTorch)
The DPO loss derived step by step from the KL-regularized RLHF
objective and the Bradley-Terry preference model, implemented and
sanity-checked (real text scores 154 nats higher than random nonsense;
finite gradient norm of 31.87 through the full loss). Trained on a
synthetic anti-repetition preference dataset: reward margin rose from
0.03 to 3.41 over 20 epochs, win rate from 62% to 89%. A beta ablation,
corrected after an initial single-sample measurement gave an
unsupported result, showed approximate KL divergence falling
monotonically from 92.7 (beta=0.05) to 2.5 (beta=2.0) across 20
averaged generations per setting -- confirming beta's role as the
reference-anchoring dial -- alongside a visible alignment tax: every
trained configuration showed reduced repetition but less coherent
generation than the base model.

---

## Project Structure

    deep-learning-from-scratch/
    |-- src/
    |   |-- common/              # shared data utilities, k-fold CV
    |   |-- phase1_mlp/          # NumPy MLP: layers, activations, optimizers
    |   |-- phase2_cnn/          # im2col conv, CNN architectures, depth ablation
    |   |-- phase3_rnn/          # RNN/LSTM cells, BPTT trainer, copy task
    |   |-- phase4_transformer/  # attention, positional encoding, GPT
    |   └-- phase5_alignment/    # preference data, DPO loss, DPO trainer
    |-- notebooks/                # one consolidated notebook per phase
    |-- docs/                     # math derivations, one per phase
    |-- scripts/                  # verification and experiment scripts
    |-- configs/                  # one YAML per phase
    └-- checkpoints/               # trained model weights (not tracked in git)

---

## Quick Start

    git clone https://github.com/MattShaabani/deep-learning-from-scratch.git
    cd deep-learning-from-scratch
    conda env create -f environment.yml
    conda activate dl-from-scratch
    pip install -e .

Each phase's notebook in `notebooks/` is self-contained and can be run
independently; scripts in `scripts/` reproduce the individual
experiments referenced in `docs/`.

---

## Documentation

    docs/01_backprop_derivation.md   - backprop, optimizers, numerical gradient checking
    docs/02_cnn_math.md               - convolution, im2col, residual connections
    docs/03_rnn_lstm_math.md           - BPTT, LSTM gates, the forget-gate-bias finding
    docs/04_attention_math.md           - attention, positional encoding, the embedding bug
    docs/05_dpo_derivation.md            - the full DPO derivation and alignment findings

---

## License

MIT