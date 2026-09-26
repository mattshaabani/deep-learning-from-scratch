import torch
import copy
from src.phase3_rnn.sequence_data import CharDataset
from src.phase4_transformer.gpt_model import GPT
from src.phase5_alignment.preference_data import generate_preference_pairs, compute_repetition_rate
from src.phase5_alignment.dpo_trainer import DPOTrainer

torch.manual_seed(42)
dataset = CharDataset()

print("Loading base model and generating a SHARED preference dataset...")
base_state = torch.load("checkpoints/gpt_extended.pt")

reference_model = GPT(vocab_size=dataset.vocab_size, d_model=64, num_heads=2, num_layers=2, d_ff=256, max_seq_length=256)
reference_model.load_state_dict(base_state)
pairs = generate_preference_pairs(reference_model, dataset, num_pairs=100, completions_per_prompt=4)
print(f"Generated {len(pairs)} shared preference pairs\n")

def generate_sample(model, prompt="ROMEO:", length=200, temperature=0.8):
    ids = dataset.encode(prompt).unsqueeze(0)
    with torch.no_grad():
        for _ in range(length):
            logits, _ = model(ids)
            next_logits = logits[0, -1, :] / temperature
            probs = torch.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            ids = torch.cat([ids, next_id.unsqueeze(0)], dim=1)
    return dataset.decode(ids[0].tolist())

results = {}

for beta in [0.05, 0.1, 0.5, 2.0]:
    print(f"\n=== Training with beta={beta} ===")
    torch.manual_seed(42)
    model = GPT(vocab_size=dataset.vocab_size, d_model=64, num_heads=2, num_layers=2, d_ff=256, max_seq_length=256)
    model.load_state_dict(base_state)

    trainer = DPOTrainer(model, beta=beta, learning_rate=0.00005)
    history = trainer.fit(pairs, epochs=20, verbose=False)

    torch.manual_seed(99)
    generated = generate_sample(model)
    rep_rate = compute_repetition_rate(dataset.encode(generated).tolist())

    results[beta] = {
        "final_margin": history["reward_margin"][-1],
        "final_kl": history["approx_kl"][-1],
        "final_win_rate": history["win_rate"][-1],
        "repetition_rate": rep_rate,
        "sample": generated,
    }

    print(f"  final_margin={history['reward_margin'][-1]:.4f} | "
          f"final_kl={history['approx_kl'][-1]:.4f} | "
          f"repetition_rate={rep_rate:.4f}")

print("\n\n=== BETA ABLATION SUMMARY ===")
print(f"{'Beta':<8}{'Reward Margin':>16}{'Approx KL':>14}{'Win Rate':>12}{'Repetition Rate':>18}")
for beta, r in results.items():
    print(f"{beta:<8}{r['final_margin']:>16.4f}{r['final_kl']:>14.4f}{r['final_win_rate']:>12.4f}{r['repetition_rate']:>18.4f}")

print("\n\n=== Sample generations per beta ===")
for beta, r in results.items():
    print(f"\nbeta={beta}:")
    print(r['sample'][:150])