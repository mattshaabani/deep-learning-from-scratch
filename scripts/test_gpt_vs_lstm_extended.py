import time
import torch
import torch.nn as nn
from src.phase3_rnn.sequence_data import CharDataset
from src.phase3_rnn.lstm_cell import LSTM
from src.phase3_rnn.bptt_trainer import BPTTTrainer
from src.phase4_transformer.gpt_model import GPT

torch.manual_seed(42)
dataset = CharDataset()

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_gpt_extended(model, epochs, seq_length=100, steps_per_epoch=50, batch_size=32, lr=0.0003, eval_every=5):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"epoch": [], "train_loss": [], "val_loss": []}

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

        if epoch % eval_every == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                x_val, y_val = dataset.get_batch(seq_length=seq_length, batch_size=batch_size, split="val")
                logits_val, _ = model(x_val)
                val_loss = nn.functional.cross_entropy(
                    logits_val.reshape(-1, logits_val.size(-1)), y_val.reshape(-1)
                ).item()

            train_loss = sum(epoch_losses) / len(epoch_losses)
            history["epoch"].append(epoch)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            print(f"  Epoch {epoch+1}/{epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

    return history


EPOCHS = 200            # ~13x longer than before
STEPS_PER_EPOCH = 50

print(f"=== Extended Training: {EPOCHS} epochs x {STEPS_PER_EPOCH} steps = {EPOCHS*STEPS_PER_EPOCH} total steps ===\n")

print("Training small GPT (d_model=64, 2 heads, 2 layers)...")
torch.manual_seed(42)
small_gpt = GPT(vocab_size=dataset.vocab_size, d_model=64, num_heads=2, num_layers=2, d_ff=256, max_seq_length=256)
print(f"Params: {count_params(small_gpt):,}\n")

start = time.time()
gpt_history = train_gpt_extended(small_gpt, epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH, eval_every=10)
gpt_time = time.time() - start
print(f"GPT training time: {gpt_time/60:.1f} minutes\n")

print("\nTraining LSTM (embedding=32, hidden=128)...")
torch.manual_seed(42)
lstm_model = LSTM(vocab_size=dataset.vocab_size, embedding_size=32, hidden_size=128)
with torch.no_grad():
    lstm_model.cell.forget_gate.bias.fill_(2.0)
print(f"Params: {count_params(lstm_model):,}\n")

lstm_trainer = BPTTTrainer(lstm_model, dataset, device="cpu")
start = time.time()
lstm_history = lstm_trainer.fit(epochs=EPOCHS, seq_length=100, steps_per_epoch=STEPS_PER_EPOCH, verbose=False)
lstm_time = time.time() - start
print(f"LSTM training time: {lstm_time/60:.1f} minutes")

# LSTM's fit() logs every epoch -- let's also print sparse checkpoints for fair comparison
print("\nLSTM val_loss at matching checkpoints:")
for e in gpt_history["epoch"]:
    print(f"  Epoch {e+1}: val_loss={lstm_history['val_loss'][e]:.4f}")

print("\n\n=== FINAL COMPARISON (after full extended training) ===")
print(f"{'Model':<15}{'Params':>12}{'Final Train Loss':>20}{'Final Val Loss':>18}{'Train Time':>15}")
print(f"{'GPT (small)':<15}{count_params(small_gpt):>12,}{gpt_history['train_loss'][-1]:>20.4f}{gpt_history['val_loss'][-1]:>18.4f}{gpt_time/60:>14.1f}m")
print(f"{'LSTM':<15}{count_params(lstm_model):>12,}{lstm_history['train_loss'][-1]:>20.4f}{lstm_history['val_loss'][-1]:>18.4f}{lstm_time/60:>14.1f}m")

# Save models for later generation comparison
torch.save(small_gpt.state_dict(), "checkpoints/gpt_extended.pt")
torch.save(lstm_model.state_dict(), "checkpoints/lstm_extended.pt")
print("\nModels saved to checkpoints/")