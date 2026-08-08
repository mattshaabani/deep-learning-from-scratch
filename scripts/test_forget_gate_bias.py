import torch
import torch.nn as nn
from src.phase3_rnn.sequence_data import CharDataset
from src.phase3_rnn.lstm_cell import LSTM

torch.manual_seed(42)
dataset = CharDataset()
seq_length = 100
batch_size = 32

model = LSTM(vocab_size=dataset.vocab_size, embedding_size=32, hidden_size=64)

# Force forget gate bias MUCH higher, so f_t ~= sigmoid(5+) ~= 0.99+
with torch.no_grad():
    model.cell.forget_gate.bias.fill_(5.0)

x, y = dataset.get_batch(seq_length=seq_length, batch_size=batch_size, split="train")
embedded = model.embedding(x)
h_t, c_t = model.cell.init_hidden(batch_size)

all_hidden = []
for t in range(seq_length):
    h_t, c_t = model.cell(embedded[:, t, :], h_t, c_t)
    h_t.retain_grad()
    all_hidden.append(h_t)

final_logits = model.output_layer(all_hidden[-1])
loss = nn.functional.cross_entropy(final_logits, y[:, -1])
loss.backward()

norms = [h.grad.norm(2).item() for h in all_hidden]
print(f"LSTM with forget_gate_bias=5.0 (f_t ~= 0.99):")
print(f"  Earliest 5: {[f'{n:.2e}' for n in norms[:5]]}")
print(f"  Latest 5:   {[f'{n:.2e}' for n in norms[-5:]]}")
print(f"  min={min(norms):.2e}, max={max(norms):.2e}")