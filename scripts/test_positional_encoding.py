import torch
from src.phase4_transformer.attention import MultiHeadSelfAttention
from src.phase4_transformer.positional_encoding import SinusoidalPositionalEncoding

torch.manual_seed(42)

print("=== Positional encoding sanity check ===")
d_model = 16
pe = SinusoidalPositionalEncoding(d_model=d_model, max_seq_length=50)

x = torch.zeros(1, 10, d_model)   # all-zero embeddings, so output IS the positional encoding
encoded = pe(x)
print(f"Encoded shape: {encoded.shape}")
print(f"Position 0 encoding (first 8 dims): {encoded[0, 0, :8].round(decimals=4).tolist()}")
print(f"Position 5 encoding (first 8 dims): {encoded[0, 5, :8].round(decimals=4).tolist()}")
print("(Each position gets a distinct fingerprint)")


print("\n\n=== THE KEY EXPERIMENT: Is attention alone permutation-equivariant? ===\n")

seq_len = 6
d_model_attn = 8
attn = MultiHeadSelfAttention(d_model=d_model_attn, num_heads=2)
attn.eval()   # disable dropout for a deterministic comparison

x_original = torch.randn(1, seq_len, d_model_attn)

# A random permutation of the sequence positions
perm = torch.randperm(seq_len)
x_shuffled = x_original[:, perm, :]

with torch.no_grad():
    output_original, _ = attn(x_original, causal_mask=False)
    output_shuffled, _ = attn(x_shuffled, causal_mask=False)

# If attention is permutation-equivariant: output_shuffled should
# equal output_original permuted the SAME way
output_original_permuted = output_original[:, perm, :]

max_diff = (output_shuffled - output_original_permuted).abs().max().item()
print(f"Permutation applied: {perm.tolist()}")
print(f"Max difference between (shuffled input -> output) and (original output, then shuffled): {max_diff:.2e}")
print(f"{'CONFIRMED' if max_diff < 1e-6 else 'NOT EQUIVARIANT'}: attention alone is permutation-equivariant")
print("-> shuffling the input just shuffles the output identically. Attention has")
print("   NO intrinsic sense of position -- it must be told explicitly.")

print("\n\n=== Now WITH positional encoding added ===\n")

pe_attn = SinusoidalPositionalEncoding(d_model=d_model_attn, max_seq_length=20)

x_original_pos = pe_attn(x_original)
x_shuffled_pos = pe_attn(x_original)[:, perm, :]   # NOTE: encode positions BEFORE shuffling is wrong on purpose --
# we want to compare "real sequence with correct positions" vs "shuffled sequence with (now incorrect) positions"
# to show positional encoding breaks the equivariance

x_shuffled_then_pe = pe_attn(x_shuffled)   # shuffle FIRST, then add positional encoding (each position gets encoded fresh)

with torch.no_grad():
    output_with_pe_original, _ = attn(x_original_pos, causal_mask=False)
    output_with_pe_shuffled, _ = attn(x_shuffled_then_pe, causal_mask=False)

output_with_pe_original_permuted = output_with_pe_original[:, perm, :]
max_diff_pe = (output_with_pe_shuffled - output_with_pe_original_permuted).abs().max().item()

print(f"Max difference WITH positional encoding: {max_diff_pe:.2e}")
print(f"{'STILL equivariant (unexpected!)' if max_diff_pe < 1e-6 else 'NO LONGER equivariant'}: ")
print("-> Once positional encoding is added, each position's vector is tied to WHERE")
print("   it sits in the sequence, so shuffling genuinely changes the computation --")
print("   the model can now distinguish position 0 from position 5.")