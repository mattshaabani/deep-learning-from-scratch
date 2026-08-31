import torch
import torch.nn as nn
from src.phase3_rnn.copy_task import CopyTaskDataset
from src.phase4_transformer.gpt_model import GPT

torch.manual_seed(42)


def train_gpt_on_copy_task(gap_length: int, epochs: int = 300, eval_every: int = 30):
    dataset = CopyTaskDataset(num_symbols=8, key_length=4, gap_length=gap_length)

    model = GPT(
        vocab_size=dataset.vocab_size,
        d_model=64, num_heads=2, num_layers=2, d_ff=256,
        max_seq_length=dataset.seq_length + 10,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    accuracy_curve = []

    for epoch in range(epochs):
        x, y = dataset.get_batch(batch_size=64)
        optimizer.zero_grad()

        logits, _ = model(x)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
            ignore_index=-100,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if epoch % eval_every == 0 or epoch == epochs - 1:
            x_test, y_test = dataset.get_batch(batch_size=200)
            with torch.no_grad():
                logits_test, _ = model(x_test)
                preds = logits_test.argmax(dim=-1)
                acc = dataset.compute_copy_accuracy(preds, y_test)
            accuracy_curve.append((epoch, acc))

    return accuracy_curve


print("=== Copy Task: GPT (Transformer), gap=20 ===\n")
curve = train_gpt_on_copy_task(gap_length=20, epochs=300, eval_every=30)
for epoch, acc in curve:
    print(f"  epoch {epoch:>4}: accuracy={acc:.4f}")

print("\n\n=== Recap: LSTM at gap=20 (from Phase 3) ===")
lstm_curve = [(0, 0.1425), (30, 0.2150), (60, 0.3825), (90, 0.3750), (120, 0.4013),
              (150, 0.4150), (180, 0.4325), (210, 0.4175), (240, 0.4425), (270, 0.4075), (299, 0.4137)]
for epoch, acc in lstm_curve:
    print(f"  epoch {epoch:>4}: accuracy={acc:.4f}")

print("\n\n=== VanillaRNN at gap=20 (from Phase 3, for reference) ===")
rnn_curve = [(0, 0.1312), (30, 0.1063), (60, 0.1063), (90, 0.1275), (120, 0.1200),
             (150, 0.1287), (180, 0.1000), (210, 0.1238), (240, 0.1437), (270, 0.1275), (299, 0.1000)]
for epoch, acc in rnn_curve:
    print(f"  epoch {epoch:>4}: accuracy={acc:.4f}")

print("\nRandom baseline: 0.1250")