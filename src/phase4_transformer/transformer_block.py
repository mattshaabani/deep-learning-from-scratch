"""
src/phase4_transformer/transformer_block.py

A single transformer decoder block: pre-norm multi-head self-attention
+ pre-norm feedforward, both wrapped in residual connections.

    x = x + MultiHeadAttention(LayerNorm(x))
    x = x + FeedForward(LayerNorm(x))

This is the SAME residual-connection principle proven in Phase 2
(ResNet's "+x" identity shortcut) applied to attention instead of
convolution -- the gradient highway idea generalizes across
architectures.

Usage:
    from src.phase4_transformer.transformer_block import TransformerBlock
    block = TransformerBlock(d_model=128, num_heads=4, d_ff=512)
    output, attn_weights = block(x, causal_mask=True)
"""

import torch
import torch.nn as nn
from src.phase4_transformer.attention import MultiHeadSelfAttention


class FeedForward(nn.Module):
    """
    Position-wise feedforward network: FFN(x) = Linear2(ReLU(Linear1(x)))

    Applied independently to each position -- no cross-position
    interaction here, since attention already handled routing
    information between positions. This is where much of the
    transformer's raw parameter capacity lives.
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))


class TransformerBlock(nn.Module):
    """
    A single pre-norm transformer decoder block.

    Pre-norm (LayerNorm BEFORE each sublayer, not after) keeps the
    residual path completely unmodified by normalization -- the
    identity gradient path (derivative exactly 1, same as Phase 2's
    ResNet) is never touched, which is why pre-norm trains more
    stably at depth than the original 2017 post-norm formulation.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.attention = MultiHeadSelfAttention(d_model, num_heads, dropout=dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, d_model)

        Returns:
            output: (batch, seq_len, d_model)
            attn_weights: (batch, num_heads, seq_len, seq_len) -- for interpretability
        """
        # Sublayer 1: pre-norm attention, residual connection
        attn_out, attn_weights = self.attention(self.norm1(x), causal_mask=causal_mask)
        x = x + self.dropout(attn_out)

        # Sublayer 2: pre-norm feedforward, residual connection
        ff_out = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_out)

        return x, attn_weights