import torch
import sys
sys.path.insert(0, '.')
from src.phase3_rnn.sequence_data import CharDataset
from src.phase4_transformer.gpt_model import GPT

torch.manual_seed(42)

dataset = CharDataset()
print(f"Vocab size: {dataset.vocab_size}")

print("\n=== GPT forward pass ===")
model = GPT(
    vocab_size=dataset.vocab_size,
    d_model=128,
    num_heads=4,
    num_layers=4,
    d_ff=512,
    max_seq_length=256,
)

small_model = GPT(
    vocab_size=dataset.vocab_size,
    d_model=64,
    num_heads=2,
    num_layers=2,
    d_ff=256,
    max_seq_length=256,
)
print(f"Smaller GPT parameter count: {small_model.count_parameters():,}")

x, y = dataset.get_batch(seq_length=100, batch_size=32)
logits, attn_weights_list = model(x)

print(f"Input shape: {x.shape}")
print(f"Logits shape: {logits.shape}")
print(f"Number of attention weight tensors (one per block): {len(attn_weights_list)}")
print(f"Each attention weights shape: {attn_weights_list[0].shape}")
print(f"\nGPT parameter count: {model.count_parameters():,}")

print("\n=== Comparing to Phase 3's LSTM parameter count ===")
from src.phase3_rnn.lstm_cell import LSTM
lstm_model = LSTM(vocab_size=dataset.vocab_size, embedding_size=32, hidden_size=128)
lstm_params = sum(p.numel() for p in lstm_model.parameters() if p.requires_grad)
print(f"LSTM (embedding=32, hidden=128) parameter count: {lstm_params:,}")
print(f"GPT (d_model=128, 4 heads, 4 layers) parameter count: {model.count_parameters():,}")