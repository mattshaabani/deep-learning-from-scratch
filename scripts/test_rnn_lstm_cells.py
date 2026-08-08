import torch
import torch.nn as nn
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

print("\n\n=== Gradient checking LSTMCell ===")
from src.phase3_rnn.lstm_cell import LSTMCell, LSTM

lstm_cell = LSTMCell(input_size=4, hidden_size=3).double()

x_t    = torch.randn(2, 4, dtype=torch.double, requires_grad=True)
h_prev = torch.randn(2, 3, dtype=torch.double, requires_grad=True)
c_prev = torch.randn(2, 3, dtype=torch.double, requires_grad=True)

result = torch.autograd.gradcheck(lstm_cell, (x_t, h_prev, c_prev), eps=1e-6, atol=1e-4)
print(f"gradcheck passed: {result}")

print("\n=== Comparing our LSTMCell against PyTorch's nn.LSTMCell ===")
torch.manual_seed(0)

our_cell = LSTMCell(input_size=4, hidden_size=3).double()
torch_cell = nn.LSTMCell(4, 3).double()

# Manually copy weights so both cells start identical.
# PyTorch's nn.LSTMCell packs all 4 gates into one big weight matrix
# in order [input, forget, candidate(cell), output] -- we need to
# match that exact ordering when splitting our separate gate weights.
with torch.no_grad():
    W_ih = torch.cat([our_cell.input_gate.weight[:, :4],
                       our_cell.forget_gate.weight[:, :4],
                       our_cell.candidate_gate.weight[:, :4],
                       our_cell.output_gate.weight[:, :4]], dim=0)
    W_hh = torch.cat([our_cell.input_gate.weight[:, 4:],
                       our_cell.forget_gate.weight[:, 4:],
                       our_cell.candidate_gate.weight[:, 4:],
                       our_cell.output_gate.weight[:, 4:]], dim=0)
    b_ih = torch.cat([our_cell.input_gate.bias,
                       our_cell.forget_gate.bias,
                       our_cell.candidate_gate.bias,
                       our_cell.output_gate.bias])

    torch_cell.weight_ih.copy_(W_ih)
    torch_cell.weight_hh.copy_(W_hh)
    torch_cell.bias_ih.copy_(b_ih)
    torch_cell.bias_hh.zero_()   # avoid double-counting bias (we only use one set)

x_test = torch.randn(2, 4, dtype=torch.double)
h_test = torch.randn(2, 3, dtype=torch.double)
c_test = torch.randn(2, 3, dtype=torch.double)

our_h, our_c = our_cell(x_test, h_test, c_test)
torch_h, torch_c = torch_cell(x_test, (h_test, c_test))

h_diff = (our_h - torch_h).abs().max().item()
c_diff = (our_c - torch_c).abs().max().item()
print(f"Max hidden state difference: {h_diff:.2e}")
print(f"Max cell state difference: {c_diff:.2e}")
print(f"{'VERIFIED' if h_diff < 1e-8 and c_diff < 1e-8 else 'MISMATCH'}: our LSTM cell matches PyTorch's nn.LSTMCell")

print("\n=== Testing full LSTM forward pass ===")
model = LSTM(vocab_size=65, embedding_size=16, hidden_size=32)
x = torch.randint(0, 65, (4, 10))
logits, (h_final, c_final), all_hidden = model(x)
print(f"Logits shape: {logits.shape}")
print(f"Final hidden shape: {h_final.shape}, final cell shape: {c_final.shape}")