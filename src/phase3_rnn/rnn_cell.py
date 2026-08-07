"""
src/phase3_rnn/rnn_cell.py

Vanilla RNN cell, implemented with explicit tensor operations so the
recurrence equation is fully visible -- not hidden inside nn.RNN.

Recurrence:
    h_t = tanh(W_hh . h_{t-1} + W_xh . x_t + b_h)

Backward pass is handled by PyTorch autograd (verified via
torch.autograd.gradcheck), the same trust relationship established
with nn.Conv2d in Phase 2.

Usage:
    from src.phase3_rnn.rnn_cell import VanillaRNNCell
    cell = VanillaRNNCell(input_size=64, hidden_size=128)
    h_t = cell(x_t, h_prev)
"""

import torch
import torch.nn as nn


class VanillaRNNCell(nn.Module):
    """
    A single vanilla RNN cell -- one timestep of recurrence.

    Explicitly implements: h_t = tanh(W_hh . h_{t-1} + W_xh . x_t + b_h)

    This is mathematically what nn.RNNCell computes internally, but
    written out so the recurrence structure is fully transparent.
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size  = input_size
        self.hidden_size = hidden_size

        # Combined weight matrix for [x_t, h_{t-1}] -> pre-activation,
        # equivalent to separate W_xh and W_hh but implemented as one
        # matmul for efficiency (standard practice)
        self.W = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_t:    (batch, input_size) -- input at this timestep
            h_prev: (batch, hidden_size) -- hidden state from previous timestep

        Returns:
            h_t: (batch, hidden_size) -- new hidden state
        """
        combined = torch.cat([x_t, h_prev], dim=1)
        h_t = torch.tanh(self.W(combined))
        return h_t

    def init_hidden(self, batch_size: int, device=None) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_size, device=device)


class VanillaRNN(nn.Module):
    """
    Full vanilla RNN: embedding -> unrolled VanillaRNNCell over all
    timesteps -> output projection to vocabulary logits.

    Unrolling the cell over timesteps in a Python loop (rather than
    using nn.RNN's fused implementation) keeps every timestep's
    computation visible -- essential for our gradient-through-time
    measurement experiment.
    """

    def __init__(self, vocab_size: int, embedding_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding    = nn.Embedding(vocab_size, embedding_size)
        self.cell          = VanillaRNNCell(embedding_size, hidden_size)
        self.output_layer   = nn.Linear(hidden_size, vocab_size)

    def forward(self, x: torch.Tensor, h_0: torch.Tensor = None) -> tuple:
        """
        Args:
            x: (batch, seq_length) character indices
            h_0: optional initial hidden state

        Returns:
            logits: (batch, seq_length, vocab_size)
            h_final: (batch, hidden_size) -- final hidden state
            all_hidden_states: list of (batch, hidden_size), one per timestep
                               -- needed for gradient-through-time analysis
        """
        batch_size, seq_length = x.shape
        embedded = self.embedding(x)   # (batch, seq_length, embedding_size)

        h_t = h_0 if h_0 is not None else self.cell.init_hidden(batch_size, device=x.device)

        all_hidden_states = []
        for t in range(seq_length):
            h_t = self.cell(embedded[:, t, :], h_t)
            all_hidden_states.append(h_t)

        hidden_stack = torch.stack(all_hidden_states, dim=1)   # (batch, seq_length, hidden_size)
        logits = self.output_layer(hidden_stack)                 # (batch, seq_length, vocab_size)

        return logits, h_t, all_hidden_states