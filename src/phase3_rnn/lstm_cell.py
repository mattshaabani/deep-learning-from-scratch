"""
src/phase3_rnn/lstm_cell.py

LSTM cell with all four gates implemented explicitly as separate
tensor operations -- not hidden inside nn.LSTMCell.

Gate equations:
    f_t = sigmoid(W_f . [h_{t-1}, x_t] + b_f)      forget gate
    i_t = sigmoid(W_i . [h_{t-1}, x_t] + b_i)      input gate
    c_tilde_t = tanh(W_c . [h_{t-1}, x_t] + b_c)   candidate cell state
    c_t = f_t * c_{t-1} + i_t * c_tilde_t           cell state update
    o_t = sigmoid(W_o . [h_{t-1}, x_t] + b_o)      output gate
    h_t = o_t * tanh(c_t)

Why this solves vanishing gradients: the cell state update is
ADDITIVE (f_t * c_{t-1} PLUS i_t * c_tilde_t), not purely
multiplicative like the vanilla RNN's tanh(W.h + ...) recurrence.
When f_t is close to 1, gradient flows through c_{t-1} almost
unimpeded -- the same "identity shortcut" principle as Phase 2's
residual connections, here implemented via a LEARNED gate rather
than a fixed skip connection.

Usage:
    from src.phase3_rnn.lstm_cell import LSTMCell
    cell = LSTMCell(input_size=64, hidden_size=128)
    h_t, c_t = cell(x_t, h_prev, c_prev)
"""

import torch
import torch.nn as nn


class LSTMCell(nn.Module):
    """
    A single LSTM cell -- one timestep of gated recurrence.

    All four gates are implemented as separate nn.Linear layers over
    the concatenated [x_t, h_{t-1}] input, making every equation from
    the derivation above directly visible in the forward() method.
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size  = input_size
        self.hidden_size = hidden_size

        combined_size = input_size + hidden_size

        # Four separate gates -- each is its own Linear layer, so the
        # forward pass reads as a direct transcription of the gate equations
        self.forget_gate    = nn.Linear(combined_size, hidden_size)
        self.input_gate      = nn.Linear(combined_size, hidden_size)
        self.candidate_gate   = nn.Linear(combined_size, hidden_size)
        self.output_gate       = nn.Linear(combined_size, hidden_size)

        # Standard LSTM initialization trick: bias the forget gate
        # toward 1 initially, so the network defaults to "remembering"
        # rather than forgetting at the start of training -- helps
        # gradient flow from step one, before any learning has happened
        nn.init.constant_(self.forget_gate.bias, 1.0)

    def forward(
        self,
        x_t: torch.Tensor,
        h_prev: torch.Tensor,
        c_prev: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x_t:    (batch, input_size) -- input at this timestep
            h_prev: (batch, hidden_size) -- hidden state from previous timestep
            c_prev: (batch, hidden_size) -- cell state from previous timestep

        Returns:
            h_t: (batch, hidden_size) -- new hidden state
            c_t: (batch, hidden_size) -- new cell state
        """
        combined = torch.cat([x_t, h_prev], dim=1)

        f_t = torch.sigmoid(self.forget_gate(combined))
        i_t = torch.sigmoid(self.input_gate(combined))
        c_tilde_t = torch.tanh(self.candidate_gate(combined))
        o_t = torch.sigmoid(self.output_gate(combined))

        # THE additive recurrence -- the mechanism that solves vanishing gradients
        c_t = f_t * c_prev + i_t * c_tilde_t

        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t

    def init_hidden(self, batch_size: int, device=None) -> tuple[torch.Tensor, torch.Tensor]:
        h_0 = torch.zeros(batch_size, self.hidden_size, device=device)
        c_0 = torch.zeros(batch_size, self.hidden_size, device=device)
        return h_0, c_0


class LSTM(nn.Module):
    """
    Full LSTM: embedding -> unrolled LSTMCell over all timesteps ->
    output projection to vocabulary logits.

    Structurally identical to VanillaRNN from rnn_cell.py, differing
    only in which cell type is unrolled -- making the two directly
    comparable for our gradient-through-time experiment.
    """

    def __init__(self, vocab_size: int, embedding_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding    = nn.Embedding(vocab_size, embedding_size)
        self.cell          = LSTMCell(embedding_size, hidden_size)
        self.output_layer   = nn.Linear(hidden_size, vocab_size)

    def forward(self, x: torch.Tensor, h_0=None, c_0=None) -> tuple:
        """
        Args:
            x: (batch, seq_length) character indices

        Returns:
            logits: (batch, seq_length, vocab_size)
            (h_final, c_final): final hidden and cell states
            all_hidden_states: list of (batch, hidden_size), one per timestep
        """
        batch_size, seq_length = x.shape
        embedded = self.embedding(x)

        if h_0 is None or c_0 is None:
            h_t, c_t = self.cell.init_hidden(batch_size, device=x.device)
        else:
            h_t, c_t = h_0, c_0

        all_hidden_states = []
        for t in range(seq_length):
            h_t, c_t = self.cell(embedded[:, t, :], h_t, c_t)
            all_hidden_states.append(h_t)

        hidden_stack = torch.stack(all_hidden_states, dim=1)
        logits = self.output_layer(hidden_stack)

        return logits, (h_t, c_t), all_hidden_states