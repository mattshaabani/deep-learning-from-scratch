# RNNs and LSTMs: Gate Equations, Backpropagation Through Time, and the Forget Gate Discovery

This document derives the mathematics of recurrent networks and
documents an empirical investigation that produced a more nuanced
finding than the standard "LSTM solves vanishing gradients" narrative.
Every claim here is backed by code in `src/phase3_rnn/` and verified
experiments in `scripts/test_rnn_lstm_cells.py`,
`scripts/test_bptt_gradients.py`, `scripts/test_bptt_isolated_recurrence.py`,
and `scripts/test_forget_gate_bias.py`.

---

## 1. The Vanilla RNN Recurrence

    h_t = tanh(W_hh . h_{t-1} + W_xh . x_t + b_h)

Unlike a feedforward network where each layer has its own weights,
an RNN reuses the SAME weight matrices (W_hh, W_xh) at every timestep.
This weight sharing is what lets an RNN process sequences of
arbitrary length with a fixed number of parameters.

**Verification:** our `VanillaRNNCell`, built from explicit tensor
operations (not `nn.RNNCell`), passed `torch.autograd.gradcheck`
with default tolerances -- confirming its forward pass produces
gradients (via autograd) that match independent numerical
finite-difference approximations.

---

## 2. Backpropagation Through Time (BPTT)

BPTT is standard backpropagation applied to the "unrolled" computational
graph of an RNN, where each timestep is treated as if it were a
separate layer sharing the same weights.

The gradient of the loss with respect to an early hidden state h_t
must travel through every subsequent timestep:

    dL/dh_t = dL/dh_T * (product from k=t+1 to T of dh_k/dh_{k-1})

Each factor dh_k/dh_{k-1} involves the tanh derivative (bounded in
[0, 1], and typically much less than 1 away from h_k=0) multiplied
by the recurrent weight matrix W_hh. If the typical magnitude of
this product is less than 1, the gradient shrinks EXPONENTIALLY with
the number of timesteps between t and T -- structurally identical to
Phase 2's layer-depth vanishing gradient analysis, with "timestep"
substituted for "layer."

---

## 3. LSTM Gate Equations

    f_t = sigmoid(W_f . [h_{t-1}, x_t] + b_f)      forget gate
    i_t = sigmoid(W_i . [h_{t-1}, x_t] + b_i)      input gate
    c_tilde_t = tanh(W_c . [h_{t-1}, x_t] + b_c)   candidate cell state
    c_t = f_t * c_{t-1} + i_t * c_tilde_t           cell state update
    o_t = sigmoid(W_o . [h_{t-1}, x_t] + b_o)      output gate
    h_t = o_t * tanh(c_t)

**Verification:** our `LSTMCell`, with all four gates implemented as
separate explicit tensor operations, passed `torch.autograd.gradcheck`,
AND was compared directly against PyTorch's `nn.LSTMCell` with
identical weights loaded into both. Maximum difference in hidden
state: 1.56e-17. Maximum difference in cell state: 3.82e-17 --
essentially machine-epsilon exact, confirming our four hand-written
gate equations compute exactly what PyTorch's production
implementation computes.

---

## 4. The "Constant Error Carousel" Argument

The standard explanation for why LSTM solves vanishing gradients
centers on the cell state recurrence:

    c_t = f_t * c_{t-1} + i_t * c_tilde_t

This is ADDITIVE, unlike the vanilla RNN's purely multiplicative
tanh(W.h + ...) recurrence. Taking the derivative:

    dc_t/dc_{t-1} = f_t

If f_t is close to 1, gradient flows through c_{t-1} almost
unimpeded -- the SAME "identity shortcut" principle as Phase 2's
residual connections (dF(x)/dx + 1), here implemented via a LEARNED
gate value instead of a fixed architectural skip connection.

This argument is correct, but as our experiments below show, it is
CONDITIONAL: it only holds when f_t is actually close to 1, which
is not automatically true at random initialization.

---

## 5. Empirical Investigation: A Refined Finding

### 5.1 The naive experiment failed to show the expected effect

Training both VanillaRNN and LSTM with standard next-character-prediction
loss (computed at EVERY timestep) on 100- and 300-character sequences
showed nearly FLAT gradient norms across all timesteps for both models
(ratio of earliest-to-latest gradient magnitude: 1.22-1.35 at seq_length=100,
0.62-0.71 at seq_length=300) -- no meaningful vanishing signal, and no
clear difference between architectures.

**Root cause:** with loss computed at every timestep, each hidden
state h_t receives a DIRECT gradient contribution from its own
timestep's prediction, in addition to whatever gradient flows
backward through the recurrence from later timesteps. This direct
contribution swamps the much weaker recurrent signal we intended
to isolate.

### 5.2 Isolating the recurrent signal

Recomputing with loss applied ONLY to the FINAL timestep's prediction
forces the ONLY path for gradient to reach early timesteps to be
through the recurrence itself, with no confounding direct contribution.

At RANDOM initialization, over 100 timesteps:

    VanillaRNN earliest-timestep gradient: EXACTLY 0.0
    LSTM earliest-timestep gradient:       3.01e-11 (numerically indistinguishable from zero)

Both architectures collapsed. This contradicts the naive expectation
that LSTM automatically preserves gradients -- and reveals why: at
random initialization, the forget gate f_t = sigmoid(random weights)
averages around 0.5-0.73 (our implementation biases the forget gate
toward 1.0 at init, giving f_t approximately 0.73), NOT close enough
to 1 to survive 100 successive multiplications
(0.73^100 is astronomically small).

### 5.3 Confirming the mechanism directly

Manually setting the LSTM's forget gate bias to 5.0 (pushing
f_t = sigmoid(5.0) approximately 0.99) and rerunning the identical
isolated-recurrence test:

    LSTM, forget_gate_bias=5.0, earliest-timestep gradient: 5.26e-02

This is roughly SIX ORDERS OF MAGNITUDE larger than the random-init
result (3.01e-11 to 5.26e-02) -- achieved purely by pushing f_t
closer to 1, with no other change to the architecture, data, or
training procedure.

---

## 6. The Refined Conclusion

LSTM's gradient-preserving property is not an automatic consequence
of having gating mechanisms. It is CONDITIONAL on the forget gate
learning (or being initialized) to sit close to 1 for information
that should persist across long time horizons. A randomly-initialized
LSTM, before any training has occurred, can suffer gradient decay
just as severe as a vanilla RNN over long sequences.

This is why the "bias the forget gate toward 1 at initialization"
trick -- which our `LSTMCell.__init__` already implements via
`nn.init.constant_(self.forget_gate.bias, 1.0)` -- is standard
practice in production LSTM implementations (originally proposed by
Jozefowicz et al., 2015): it gives the gradient-preserving mechanism
a head start before training has had a chance to learn it, rather
than leaving f_t at its default random value.

**The key distinction this experiment surfaces:** LSTM's architecture
makes gradient preservation POSSIBLE across arbitrary time horizons
(unlike vanilla RNN, where the purely multiplicative tanh recurrence
makes long-range preservation mathematically implausible regardless
of what the weights learn) -- but LSTM does not GUARANTEE gradient
preservation automatically. The forget gate must actually learn (or
be initialized) to behave close to an identity mapping for the
relevant information.

---

## 7. Why This Matters Beyond Phase 3

This finding connects directly to Phase 2's residual connection
result. Both ResNet's `+x` identity shortcut and LSTM's `f_t * c_{t-1}`
gated shortcut solve vanishing gradients through the same
mathematical mechanism: providing an additive path with derivative
close to 1, bypassing the need for gradient to survive a long chain
of multiplicative transformations. The difference is that ResNet's
shortcut is FIXED (always exactly derivative 1), while LSTM's is
LEARNED (derivative f_t, which must be trained or initialized toward 1
to be effective) -- a subtle but important architectural distinction
that this experiment made concrete and measurable rather than purely
theoretical.