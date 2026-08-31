import torch
from src.phase3_rnn.sequence_data import CharDataset
from src.phase3_rnn.lstm_cell import LSTM
from src.phase3_rnn.text_generator import generate_text
from src.phase4_transformer.gpt_model import GPT

torch.manual_seed(123)   # different seed than training, for a fresh generation sample
dataset = CharDataset()

print("Loading trained models from checkpoints...")

lstm_model = LSTM(vocab_size=dataset.vocab_size, embedding_size=32, hidden_size=128)
lstm_model.load_state_dict(torch.load("checkpoints/lstm_extended.pt"))
lstm_model.eval()

gpt_model = GPT(vocab_size=dataset.vocab_size, d_model=64, num_heads=2, num_layers=2, d_ff=256, max_seq_length=256)
gpt_model.load_state_dict(torch.load("checkpoints/gpt_extended.pt"))
gpt_model.eval()

print("\n" + "="*70)
print("LSTM generation (trained 200 epochs, final val_loss=1.6140):")
print("="*70)
lstm_text = generate_text(lstm_model, dataset, seed_text="ROMEO:", length=300, temperature=0.8)
print(lstm_text)

print("\n" + "="*70)
print("GPT generation (trained 200 epochs, final val_loss=1.8900):")
print("="*70)

# GPT needs its own generation loop since it doesn't use the RNN-style
# hidden-state API that text_generator.py was built for -- it needs
# the full sequence context re-fed each step (or a KV-cache, which
# we haven't implemented). This is the straightforward, uncached version.
def generate_gpt_text(model, dataset, seed_text, length=300, temperature=0.8, max_context=256):
    model.eval()
    input_ids = dataset.encode(seed_text).unsqueeze(0)

    with torch.no_grad():
        for _ in range(length):
            context = input_ids[:, -max_context:]   # respect max_seq_length
            logits, _ = model(context)
            next_logits = logits[0, -1, :] / temperature
            probs = torch.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_id.unsqueeze(0)], dim=1)

    return dataset.decode(input_ids[0].tolist())

gpt_text = generate_gpt_text(gpt_model, dataset, seed_text="ROMEO:", length=300, temperature=0.8)
print(gpt_text)