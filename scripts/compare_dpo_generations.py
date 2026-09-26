import torch
from src.phase3_rnn.sequence_data import CharDataset
from src.phase4_transformer.gpt_model import GPT

torch.manual_seed(99)
dataset = CharDataset()

base_model = GPT(vocab_size=dataset.vocab_size, d_model=64, num_heads=2, num_layers=2, d_ff=256, max_seq_length=256)
base_model.load_state_dict(torch.load("checkpoints/gpt_extended.pt"))
base_model.eval()

aligned_model = GPT(vocab_size=dataset.vocab_size, d_model=64, num_heads=2, num_layers=2, d_ff=256, max_seq_length=256)
aligned_model.load_state_dict(torch.load("checkpoints/gpt_dpo_aligned.pt"))
aligned_model.eval()

def generate(model, prompt, length=200, temperature=0.8):
    ids = dataset.encode(prompt).unsqueeze(0)
    with torch.no_grad():
        for _ in range(length):
            logits, _ = model(ids)
            next_logits = logits[0, -1, :] / temperature
            probs = torch.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            ids = torch.cat([ids, next_id.unsqueeze(0)], dim=1)
    return dataset.decode(ids[0].tolist())

from src.phase5_alignment.preference_data import compute_repetition_rate

prompt = "ROMEO:"
print("=== BEFORE DPO (base model) ===")
base_text = generate(base_model, prompt)
print(base_text)
base_ids = dataset.encode(base_text).tolist()
print(f"\nRepetition rate: {compute_repetition_rate(base_ids):.4f}")

print("\n\n=== AFTER DPO (aligned model) ===")
aligned_text = generate(aligned_model, prompt)
print(aligned_text)
aligned_ids = dataset.encode(aligned_text).tolist()
print(f"\nRepetition rate: {compute_repetition_rate(aligned_ids):.4f}")