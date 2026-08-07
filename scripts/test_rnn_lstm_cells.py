import torch
from src.phase3_rnn.rnn_cell import VanillaRNNCell

torch.manual_seed(42)

print("=== Gradient checking VanillaRNNCell ===")

# gradcheck requires float64 for numerical precision
cell = VanillaRNNCell(input_size=4, hidden_size=3).double()

batch_size = 2
x_t    = torch.randn(batch_size, 4, dtype=torch.double, requires_grad=True)
h_prev = torch.randn(batch_size, 3, dtype=torch.double, requires_grad=True)

# torch.autograd.gradcheck compares PyTorch's autograd gradients
# against numerical finite-difference gradients -- the exact same
# verification principle as Phase 1's manual gradient checking,
# just using PyTorch's built-in tool instead of our own NumPy version
result = torch.autograd.gradcheck(cell, (x_t, h_prev), eps=1e-6, atol=1e-4)

print(f"gradcheck passed: {result}")
print("\nVanillaRNNCell forward/backward verified correct via PyTorch's numerical gradient checker.")

# Quick sanity check on the full unrolled network
print("\n=== Testing full VanillaRNN forward pass ===")
from src.phase3_rnn.rnn_cell import VanillaRNN

model = VanillaRNN(vocab_size=65, embedding_size=16, hidden_size=32)
x = torch.randint(0, 65, (4, 10))   # batch=4, seq_length=10

logits, h_final, all_hidden = model(x)
print(f"Input shape: {x.shape}")
print(f"Logits shape: {logits.shape}")
print(f"Final hidden shape: {h_final.shape}")
print(f"Number of hidden states captured: {len(all_hidden)}")