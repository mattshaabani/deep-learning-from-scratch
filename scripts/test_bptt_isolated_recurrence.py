import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from src.phase3_rnn.sequence_data import CharDataset
from src.phase3_rnn.rnn_cell import VanillaRNN
from src.phase3_rnn.lstm_cell import LSTM

torch.manual_seed(42)
np.random.seed(42)

dataset = CharDataset()
seq_length = 100
batch_size = 32

def measure_recurrence_only_gradients(model, is_lstm: bool):
    """
    Loss computed ONLY on the final timestep's prediction, so the
    ONLY path for gradient to reach early timesteps is through the
    recurrence itself -- isolating the vanishing gradient signal
    from any direct per-timestep loss contribution.
    """
    x, y = dataset.get_batch(seq_length=seq_length, batch_size=batch_size, split="train")

    embedded = model.embedding(x)
    batch_sz = x.shape[0]

    if is_lstm:
        h_t, c_t = model.cell.init_hidden(batch_sz)
    else:
        h_t = model.cell.init_hidden(batch_sz)

    all_hidden = []
    for t in range(seq_length):
        if is_lstm:
            h_t, c_t = model.cell(embedded[:, t, :], h_t, c_t)
        else:
            h_t = model.cell(embedded[:, t, :], h_t)
        h_t.retain_grad()
        all_hidden.append(h_t)

    # Loss ONLY on the final timestep -- forces gradient to travel
    # backward through the FULL recurrence chain to reach early h_t's
    final_logits = model.output_layer(all_hidden[-1])
    loss = nn.functional.cross_entropy(final_logits, y[:, -1])

    loss.backward()

    return [h.grad.norm(2).item() for h in all_hidden]


results = {}
for name, model_cls in [("VanillaRNN", VanillaRNN), ("LSTM", LSTM)]:
    model = model_cls(vocab_size=dataset.vocab_size, embedding_size=32, hidden_size=64)
    is_lstm = (name == "LSTM")
    norms = measure_recurrence_only_gradients(model, is_lstm)
    results[name] = norms
    print(f"\n{name} (recurrence-only gradient, UNTRAINED model):")
    print(f"  Earliest 5: {[f'{n:.2e}' for n in norms[:5]]}")
    print(f"  Latest 5:   {[f'{n:.2e}' for n in norms[-5:]]}")
    print(f"  min={min(norms):.2e}, max={max(norms):.2e}, ratio(min/max)={min(norms)/max(norms):.2e}")

fig, ax = plt.subplots(figsize=(10, 5))
for name, norms in results.items():
    ax.plot(range(len(norms)), norms, label=name, marker='.')
ax.set_xlabel("Timestep (0 = earliest)")
ax.set_ylabel("Gradient norm (recurrence-only)")
ax.set_yscale("log")
ax.set_title("Gradient Through Time — Loss on FINAL timestep ONLY (isolates recurrence)")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("data/bptt_isolated_recurrence.png", dpi=150)
plt.show()