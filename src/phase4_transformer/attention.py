"""
src/phase4_transformer/attention.py

Multi-head self-attention implemented from explicit tensor operations
-- every equation (Q/K/V projections, scaled dot-product attention,
multi-head split/concat, causal masking) is fully visible, not hidden
inside nn.MultiheadAttention.

Verified against PyTorch's own nn.MultiheadAttention via gradcheck
and direct output comparison -- the same "trust bridge" pattern
established with nn.Conv2d (Phase 2) and nn.LSTMCell (Phase 3).

Usage:
    from src.phase4_transformer.attention import MultiHeadSelfAttention
    attn = MultiHeadSelfAttention(d_model=128, num_heads=4)
    output, attn_weights = attn(x, causal_mask=True)
"""

import math
import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention.

    Attention(Q,K,V) = softmax(Q.K^T / sqrt(d_k)) . V

    Why divide by sqrt(d_k):
        Dot products Q.K grow in variance proportional to d_k (sum of
        d_k independent terms). Without scaling, larger d_k pushes
        softmax inputs into saturated regions with near-zero gradient
        -- a THIRD distinct appearance of the vanishing gradient theme
        from this capstone, now inside the attention mechanism itself.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model  = d_model
        self.num_heads = num_heads
        self.d_k        = d_model // num_heads   # dimension per head

        # Combined projections for efficiency (equivalent to separate
        # W_Q, W_K, W_V matrices per head, but computed as one matmul)
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)   # output projection after concat

        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, seq_len, d_model) -> (batch, num_heads, seq_len, d_k)"""
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, num_heads, seq_len, d_k) -> (batch, seq_len, d_model)"""
        batch_size, num_heads, seq_len, d_k = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, num_heads * d_k)

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, d_model)
            causal_mask: if True, position i cannot attend to position j > i

        Returns:
            output: (batch, seq_len, d_model)
            attn_weights: (batch, num_heads, seq_len, seq_len) -- for
                          visualization/interpretability
        """
        batch_size, seq_len, _ = x.shape

        Q = self._split_heads(self.W_q(x))   # (batch, heads, seq_len, d_k)
        K = self._split_heads(self.W_k(x))
        V = self._split_heads(self.W_v(x))

        # Scaled dot-product attention scores
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)   # (batch, heads, seq_len, seq_len)

        if causal_mask:
            mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
                diagonal=1,
            )
            scores = scores.masked_fill(mask, float("-inf"))

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = attn_weights @ V   # (batch, heads, seq_len, d_k)
        context = self._combine_heads(context)

        output = self.W_o(context)

        return output, attn_weights