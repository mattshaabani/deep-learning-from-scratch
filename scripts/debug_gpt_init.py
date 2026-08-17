import torch
from src.phase4_transformer.gpt_model import GPT
from src.phase3_rnn.sequence_data import CharDataset

torch.manual_seed(42)
dataset = CharDataset()

model = GPT(vocab_size=dataset.vocab_size, d_model=64, num_heads=2, num_layers=2, d_ff=256, max_seq_length=256)

x, y = dataset.get_batch(seq_length=100, batch_size=32)

with torch.no_grad():
    embedded_raw = model.token_embedding(x)
    embedded_scaled = embedded_raw * (model.d_model ** 0.5)
    logits, _ = model(x)

print(f"Raw embedding stats:    mean={embedded_raw.mean():.4f}, std={embedded_raw.std():.4f}")
print(f"Scaled embedding stats: mean={embedded_scaled.mean():.4f}, std={embedded_scaled.std():.4f}")
print(f"Final logits stats:     mean={logits.mean():.4f}, std={logits.std():.4f}, max={logits.max():.4f}, min={logits.min():.4f}")

import torch.nn.functional as F
loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
print(f"\nLoss at random init: {loss.item():.4f}")
print(f"Theoretical random-guess loss (ln(vocab_size)): {torch.log(torch.tensor(float(dataset.vocab_size))).item():.4f}")