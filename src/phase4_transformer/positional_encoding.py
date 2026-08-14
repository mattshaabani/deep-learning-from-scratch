"""
src/phase4_transformer/positional_encoding.py

Sinusoidal positional encoding, plus a direct experimental proof that
self-attention alone is permutation-equivariant -- shuffling the
input produces an identically-shuffled output, with zero information
about original position, unless positional encoding is added.

Usage:
    from src.phase4_transformer.positional_encoding import SinusoidalPositionalEncoding
    pe = SinusoidalPositionalEncoding(d_model=128, max_seq_length=256)
    x_with_position = pe(x)
"""

import math
import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """
    Adds position information to token embeddings via fixed
    (non-learned) sine/cosine functions of varying frequency.

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Precomputed once at construction time for all positions up to
    max_seq_length, then simply added to token embeddings at the
    start of the forward pass -- no learned parameters here.
    """

    def __init__(self, d_model: int, max_seq_length: int = 512):
        super().__init__()

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)   # (max_seq_length, 1)

        # div_term = 10000^(2i/d_model), computed in log-space for numerical stability
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)   # even dimensions
        pe[:, 1::2] = torch.cos(position * div_term)   # odd dimensions

        # Register as a buffer (not a parameter) -- moves with .to(device)
        # but is NOT updated by the optimizer, since it's fixed by formula
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_seq_length, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model) token embeddings

        Returns:
            x + positional encoding, same shape
        """
        seq_len = x.shape[1]
        return x + self.pe[:, :seq_len, :]