import torch
import numpy as np
import matplotlib.pyplot as plt
from src.phase3_rnn.sequence_data import CharDataset
from src.phase3_rnn.rnn_cell import VanillaRNN
from src.phase3_rnn.lstm_cell import LSTM
from src.phase3_rnn.bptt_trainer import BPTTTrainer

torch.manual_seed(42)
np.random.seed(42)

dataset = CharDataset()
seq_length = 300   # much longer, to make compounding effects visible

results = {}
for name, model_cls in [("VanillaRNN", VanillaRNN), ("LSTM", LSTM)]:
    print(f"\n=== Training {name} (seq_length={seq_length}) ===")
    model = model_cls(vocab_size=dataset.vocab_size, embedding_size=32, hidden_size=64)
    trainer = BPTTTrainer(model, dataset, device="cpu")
    history = trainer.fit(epochs=3, seq_length=seq_length, steps_per_epoch=30, verbose=True)
    results[name] = history

fig, ax = plt.subplots(figsize=(10, 5))
for name, history in results.items():
    norms = history["grad_norms_last_step"]
    ax.plot(range(len(norms)), norms, label=name, alpha=0.8)

ax.set_xlabel("Timestep (0 = earliest/furthest back)")
ax.set_ylabel("Gradient L2 norm")
ax.set_yscale("log")
ax.set_title(f"Gradient Norm Through Time (seq_length={seq_length})")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("data/bptt_gradient_diagnostic.png", dpi=150)
plt.show()

for name, history in results.items():
    norms = history["grad_norms_last_step"]
    print(f"\n{name}: min={min(norms):.6f}, max={max(norms):.6f}, ratio(min/max)={min(norms)/max(norms):.6f}")