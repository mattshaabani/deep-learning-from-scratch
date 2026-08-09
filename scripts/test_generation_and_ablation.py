import torch
from src.phase3_rnn.sequence_data import CharDataset
from src.phase3_rnn.lstm_cell import LSTM
from src.phase3_rnn.bptt_trainer import BPTTTrainer
from src.phase3_rnn.text_generator import generate_text

torch.manual_seed(42)

print("=== Quick LSTM training + generation demo ===")
dataset = CharDataset()

model = LSTM(vocab_size=dataset.vocab_size, embedding_size=32, hidden_size=128)
# Apply what we learned: bias the forget gate toward "remember"
with torch.no_grad():
    model.cell.forget_gate.bias.fill_(2.0)

trainer = BPTTTrainer(model, dataset, device="cpu")

print("\nGeneration BEFORE training (should be gibberish):")
print(generate_text(model, dataset, seed_text="ROMEO:", length=150, temperature=0.8))

history = trainer.fit(epochs=15, seq_length=100, steps_per_epoch=50, verbose=True)

print("\nGeneration AFTER training (should show some structure):")
print(generate_text(model, dataset, seed_text="ROMEO:", length=150, temperature=0.8))

print("\n\n=== Small smoke test of sequence ablation (2 lengths, 2 epochs) ===")
from src.phase3_rnn.sequence_ablation import run_sequence_ablation

results = run_sequence_ablation(seq_lengths=[20, 100], epochs=2, steps_per_epoch=10)
for run_name, h in results.items():
    print(f"{run_name}: final_val_loss={h['val_loss'][-1]:.4f}")

print("\nRun 'mlflow ui --port 5000 --backend-store-uri sqlite:///mlflow.db' to inspect all runs.")