"""
src/phase4_transformer/gpt_model.py

A full GPT-style decoder-only transformer: token embedding + positional
encoding -> stacked TransformerBlocks -> final LayerNorm -> vocabulary
projection.

Reuses Phase 3's CharDataset for a direct, fair comparison against
the LSTM language model on identical data and task.

Usage:
    from src.phase4_transformer.gpt_model import GPT
    model = GPT(vocab_size=65, d_model=128, num_heads=4, num_layers=4, d_ff=512)
    logits, attn_weights_list = model(x)
"""

import torch
import torch.nn as nn
from src.phase4_transformer.transformer_block import TransformerBlock
from src.phase4_transformer.positional_encoding import SinusoidalPositionalEncoding


class GPT(nn.Module):
    """
    Decoder-only transformer language model, GPT-style.

    Every block uses causal masking, so the model is autoregressive:
    prediction at position i can only depend on positions 0..i.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        d_ff: int = 512,
        max_seq_length: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        # Initialize with smaller std so that AFTER the sqrt(d_model) scaling
        # in forward(), effective embedding magnitude matches what the rest
        # of the network (LayerNorm-normalized activations) expects.
        # Without this, embeddings end up ~8x too large at init (verified:
        # raw std=0.997 -> scaled std=7.977 -> logits std=10.96, loss=57.86
        # instead of the expected ~4.17 for random-guess baseline).
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=d_model ** -0.5)
        self.positional_encoding = SinusoidalPositionalEncoding(d_model, max_seq_length)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size)

        # Weight tying: share the embedding and output projection weights.
        # Standard GPT practice -- reduces parameters and often improves
        # quality, since "predicting token X" and "the embedding for
        # token X" are related representations.
        self.output_projection.weight = self.token_embedding.weight

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list]:
        """
        Args:
            x: (batch, seq_len) token indices

        Returns:
            logits: (batch, seq_len, vocab_size)
            attn_weights_list: list of (batch, num_heads, seq_len, seq_len),
                              one per block -- for interpretability/visualization
        """
        embedded = self.token_embedding(x) * (self.d_model ** 0.5)   # standard GPT scaling
        h = self.positional_encoding(embedded)
        h = self.dropout(h)

        attn_weights_list = []
        for block in self.blocks:
            h, attn_weights = block(h, causal_mask=True)
            attn_weights_list.append(attn_weights)

        h = self.final_norm(h)
        logits = self.output_projection(h)

        return logits, attn_weights_list

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)