import torch
import torch.nn as nn
import numpy as np
from src.phase3_rnn.copy_task import CopyTaskDataset
from src.phase3_rnn.rnn_cell import VanillaRNN
from src.phase3_rnn.lstm_cell import LSTM

torch.manual_seed(42)
np.random.seed(42)


def train_on_copy_task_with_curve(model_cls, is_lstm: bool, gap_length: int, epochs: int = 300, eval_every: int = 20):
    dataset = CopyTaskDataset(num_symbols=8, key_length=4, gap_length=gap_length)
    model = model_cls(vocab_size=dataset.vocab_size, embedding_size=16, hidden_size=64)

    if is_lstm:
        with torch.no_grad():
            model.cell.forget_gate.bias.fill_(2.0)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    accuracy_curve = []

    for epoch in range(epochs):
        x, y = dataset.get_batch(batch_size=64)
        optimizer.zero_grad()

        logits, _, _ = model(x)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
            ignore_index=-100,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        if epoch % eval_every == 0 or epoch == epochs - 1:
            x_test, y_test = dataset.get_batch(batch_size=200)
            with torch.no_grad():
                logits, _, _ = model(x_test)
                preds = logits.argmax(dim=-1)
                acc = dataset.compute_copy_accuracy(preds, y_test)
            accuracy_curve.append((epoch, acc))

    return accuracy_curve


print("=== Copy Task: Longer Training, Gap=20 ===\n")
for name, model_cls, is_lstm in [("VanillaRNN", VanillaRNN, False), ("LSTM", LSTM, True)]:
    curve = train_on_copy_task_with_curve(model_cls, is_lstm, gap_length=20, epochs=300, eval_every=30)
    print(f"\n{name}:")
    for epoch, acc in curve:
        print(f"  epoch {epoch:>4}: accuracy={acc:.4f}")