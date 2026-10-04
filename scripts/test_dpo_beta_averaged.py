import torch
from src.phase3_rnn.sequence_data import CharDataset
from src.phase4_transformer.gpt_model import GPT
from src.phase5_alignment.preference_data import generate_preference_pairs, compute_repetition_rate
from src.phase5_alignment.dpo_trainer import DPOTrainer
import numpy as np

torch.manual_seed(42)
dataset = CharDataset()

print("Loading base model and generating a SHARED preference dataset...")
base_state = torch.load("checkpoints/gpt_extended.pt")

reference_model = GPT(vocab_size=dataset.vocab_size, d_model=64, num_heads=2, num_layers=2, d_ff=256, max_seq_length=256)
reference_model.load_state_dict(base_state)
pairs = generate_preference_pairs(reference_model, dataset, num_pairs=100, completions_per_prompt=4)
print(f"Generated {len(pairs)} shared preference pairs\n")


def generate_sample(model, prompt="ROMEO:", length=200, temperature=0.8, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    ids = dataset.encode(prompt).unsqueeze(0)
    with torch.no_grad():
        for _ in range(length):
            logits, _ = model(ids)
            next_logits = logits[0, -1, :] / temperature
            probs = torch.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            ids = torch.cat([ids, next_id.unsqueeze(0)], dim=1)
    return dataset.decode(ids[0].tolist())


def averaged_repetition(model, n_samples=20, seed_start=1000):
    rates = []
    for i in range(n_samples):
        text = generate_sample(model, seed=seed_start + i)
        rates.append(compute_repetition_rate(dataset.encode(text).tolist()))
    return float(np.mean(rates)), float(np.std(rates)), rates


N_SAMPLES = 20

print(f"=== Baseline: averaged repetition rate over {N_SAMPLES} samples, BASE model ===")
base_model = GPT(vocab_size=dataset.vocab_size, d_model=64, num_heads=2, num_layers=2, d_ff=256, max_seq_length=256)
base_model.load_state_dict(base_state)
base_mean, base_std, _ = averaged_repetition(base_model, n_samples=N_SAMPLES)
print(f"Base model: mean={base_mean:.4f}, std={base_std:.4f}\n")

results = {}

for beta in [0.05, 0.1, 0.5, 2.0]:
    print(f"=== Training with beta={beta} ===")
    torch.manual_seed(42)
    model = GPT(vocab_size=dataset.vocab_size, d_model=64, num_heads=2, num_layers=2, d_ff=256, max_seq_length=256)
    model.load_state_dict(base_state)

    trainer = DPOTrainer(model, beta=beta, learning_rate=0.00005)
    history = trainer.fit(pairs, epochs=20, verbose=False)

    mean_rep, std_rep, all_rates = averaged_repetition(model, n_samples=N_SAMPLES)

    results[beta] = {
        "final_margin": history["reward_margin"][-1],
        "final_kl": history["approx_kl"][-1],
        "final_win_rate": history["win_rate"][-1],
        "repetition_mean": mean_rep,
        "repetition_std": std_rep,
    }

    print(f"  final_margin={history['reward_margin'][-1]:.4f} | "
          f"final_kl={history['approx_kl'][-1]:.4f} | "
          f"repetition: mean={mean_rep:.4f}, std={std_rep:.4f} (n={N_SAMPLES})")

print("\n\n=== BETA ABLATION SUMMARY (averaged over 20 samples each) ===")
print(f"{'Beta':<8}{'Reward Margin':>16}{'Approx KL':>14}{'Win Rate':>12}{'Rep Mean':>12}{'Rep Std':>10}")
print(f"{'base':<8}{'--':>16}{'--':>14}{'--':>12}{base_mean:>12.4f}{base_std:>10.4f}")
for beta, r in results.items():
    print(f"{beta:<8}{r['final_margin']:>16.4f}{r['final_kl']:>14.4f}{r['final_win_rate']:>12.4f}{r['repetition_mean']:>12.4f}{r['repetition_std']:>10.4f}")