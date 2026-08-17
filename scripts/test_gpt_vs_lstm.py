import torch
import torch.nn as nn
from src.phase3_rnn.sequence_data import CharDataset
from src.phase3_rnn.lstm_cell import LSTM
from src.phase4_transformer.gpt_model import GPT

torch.manual_seed(42)
dataset = CharDataset()

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_gpt(model, epochs=15, seq_length=100, steps_per_epoch=50, batch_size=32, lr=0.0003):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        for _ in range(steps_per_epoch):
            x, y = dataset.get_batch(seq_length=seq_length, batch_size=batch_size, split="train")
            optimizer.zero_grad()
            logits, _ = model(x)
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), y.reshape(-1)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            x_val, y_val = dataset.get_batch(seq_length=seq_length, batch_size=batch_size, split="val")
            logits_val, _ = model(x_val)
            val_loss = nn.functional.cross_entropy(
                logits_val.reshape(-1, logits_val.size(-1)), y_val.reshape(-1)
            ).item()

        train_loss = sum(epoch_losses) / len(epoch_losses)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"  Epoch {epoch+1}/{epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

    return history


print("=== Matched-Capacity Comparison (~93K-104K params) ===\n")

print("Training small GPT (d_model=64, 2 heads, 2 layers)...")
torch.manual_seed(42)
small_gpt = GPT(vocab_size=dataset.vocab_size, d_model=64, num_heads=2, num_layers=2, d_ff=256, max_seq_length=256)
print(f"Params: {count_params(small_gpt):,}\n")
gpt_history = train_gpt(small_gpt, epochs=15, seq_length=100)

print("\nTraining LSTM (embedding=32, hidden=128) -- reusing Phase 3's BPTTTrainer...")
from src.phase3_rnn.bptt_trainer import BPTTTrainer
torch.manual_seed(42)
lstm_model = LSTM(vocab_size=dataset.vocab_size, embedding_size=32, hidden_size=128)
with torch.no_grad():
    lstm_model.cell.forget_gate.bias.fill_(2.0)   # apply our Phase 3 lesson
print(f"Params: {count_params(lstm_model):,}\n")
lstm_trainer = BPTTTrainer(lstm_model, dataset, device="cpu")
lstm_history = lstm_trainer.fit(epochs=15, seq_length=100, steps_per_epoch=50, verbose=True)

print("\n\n=== FINAL COMPARISON (matched capacity, identical data/epochs/steps) ===")
print(f"{'Model':<15}{'Params':>12}{'Final Train Loss':>20}{'Final Val Loss':>18}")
print(f"{'GPT (small)':<15}{count_params(small_gpt):>12,}{gpt_history['train_loss'][-1]:>20.4f}{gpt_history['val_loss'][-1]:>18.4f}")
print(f"{'LSTM':<15}{count_params(lstm_model):>12,}{lstm_history['train_loss'][-1]:>20.4f}{lstm_history['val_loss'][-1]:>18.4f}")