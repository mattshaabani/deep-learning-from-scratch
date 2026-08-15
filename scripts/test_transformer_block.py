import torch
import torch.nn as nn
from src.phase4_transformer.transformer_block import TransformerBlock

torch.manual_seed(42)

print("=== Single block forward pass ===")
block = TransformerBlock(d_model=32, num_heads=4, d_ff=128)
x = torch.randn(2, 10, 32)
output, attn_weights = block(x, causal_mask=True)
print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Attention weights shape: {attn_weights.shape}")

print("\n=== Stacking many blocks -- the Phase 2 callback experiment ===")
print("Reusing the per-layer gradient norm tracking technique from Phase 2's")
print("depth ablation, now applied to a STACK of transformer blocks with")
print("residual connections, to confirm gradients survive at depth.\n")

num_blocks = 12
blocks = nn.ModuleList([
    TransformerBlock(d_model=32, num_heads=4, d_ff=128) for _ in range(num_blocks)
])

x = torch.randn(2, 10, 32, requires_grad=True)
h = x
for block in blocks:
    h, _ = block(h, causal_mask=True)

loss = h.sum()
loss.backward()

print(f"Gradient norms per block (0 = closest to input, {num_blocks-1} = closest to loss):")
for i, block in enumerate(blocks):
    # Grab gradient norm of the attention output projection weight,
    # as a representative signal per block
    grad_norm = block.attention.W_o.weight.grad.norm(2).item()
    print(f"  block_{i}: {grad_norm:.6f}")

print("\nUnlike Phase 2's PlainCNN (which hit EXACTLY ZERO past depth 8),")
print("residual connections should keep every block's gradient non-zero")
print("and reasonably uniform in magnitude, even at 12 layers deep.")