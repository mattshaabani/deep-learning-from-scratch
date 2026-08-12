import torch
import torch.nn as nn
from src.phase4_transformer.attention import MultiHeadSelfAttention

torch.manual_seed(42)

print("=== Gradient checking MultiHeadSelfAttention ===")
attn = MultiHeadSelfAttention(d_model=8, num_heads=2).double()
x = torch.randn(2, 5, 8, dtype=torch.double, requires_grad=True)

def forward_fn(x):
    output, _ = attn(x, causal_mask=False)
    return output

result = torch.autograd.gradcheck(forward_fn, (x,), eps=1e-6, atol=1e-4)
print(f"gradcheck passed: {result}")

print("\n=== Comparing against PyTorch's nn.MultiheadAttention ===")
d_model, num_heads = 8, 2
our_attn = MultiHeadSelfAttention(d_model, num_heads).double()
torch_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True).double()

# Copy weights so both start identical.
# PyTorch packs Q,K,V into one combined in_proj_weight/bias, in that order.
with torch.no_grad():
    combined_weight = torch.cat([our_attn.W_q.weight, our_attn.W_k.weight, our_attn.W_v.weight], dim=0)
    combined_bias    = torch.cat([our_attn.W_q.bias, our_attn.W_k.bias, our_attn.W_v.bias], dim=0)
    torch_attn.in_proj_weight.copy_(combined_weight)
    torch_attn.in_proj_bias.copy_(combined_bias)
    torch_attn.out_proj.weight.copy_(our_attn.W_o.weight)
    torch_attn.out_proj.bias.copy_(our_attn.W_o.bias)

x_test = torch.randn(2, 5, d_model, dtype=torch.double)

our_output, our_weights = our_attn(x_test, causal_mask=False)
torch_output, torch_weights = torch_attn(x_test, x_test, x_test, need_weights=True, average_attn_weights=False)

max_diff = (our_output - torch_output).abs().max().item()
print(f"Max output difference: {max_diff:.2e}")
print(f"{'VERIFIED' if max_diff < 1e-6 else 'MISMATCH'}: matches PyTorch's nn.MultiheadAttention")

print("\n=== Verifying causal masking ===")
our_output_masked, our_weights_masked = our_attn(x_test, causal_mask=True)

# Check: attention weight from position i to position j>i should be EXACTLY 0
future_leakage = 0
seq_len = x_test.shape[1]
for i in range(seq_len):
    for j in range(i + 1, seq_len):
        future_leakage += our_weights_masked[:, :, i, j].abs().sum().item()

print(f"Total attention weight leaking into the future (should be exactly 0.0): {future_leakage}")
print(f"{'VERIFIED' if future_leakage == 0.0 else 'LEAK DETECTED'}: causal mask correctly blocks future positions")