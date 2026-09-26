import torch
from src.phase3_rnn.sequence_data import CharDataset
from src.phase4_transformer.gpt_model import GPT
from src.phase5_alignment.preference_data import generate_preference_pairs
from src.phase5_alignment.dpo_trainer import DPOTrainer

torch.manual_seed(42)
dataset = CharDataset()

print("=== Loading trained Phase 4 GPT ===")
policy_model = GPT(vocab_size=dataset.vocab_size, d_model=64, num_heads=2, num_layers=2, d_ff=256, max_seq_length=256)
policy_model.load_state_dict(torch.load("checkpoints/gpt_extended.pt"))

print("\n=== Generating preference pairs (anti-repetition) ===")
pairs = generate_preference_pairs(policy_model, dataset, num_pairs=100, completions_per_prompt=4)
print(f"Generated {len(pairs)} preference pairs")

avg_winner_rep = sum(p.winner_repetition_rate for p in pairs) / len(pairs)
avg_loser_rep = sum(p.loser_repetition_rate for p in pairs) / len(pairs)
print(f"Average winner repetition rate: {avg_winner_rep:.4f}")
print(f"Average loser repetition rate:  {avg_loser_rep:.4f}")

print("\n=== Training with DPO ===")
trainer = DPOTrainer(policy_model, beta=0.1, learning_rate=0.00005)
history = trainer.fit(pairs, epochs=20, verbose=True)

print("\n=== Training Summary ===")
print(f"Initial reward margin: {history['reward_margin'][0]:.4f}")
print(f"Final reward margin:   {history['reward_margin'][-1]:.4f}")
print(f"Initial win rate: {history['win_rate'][0]:.4f}")
print(f"Final win rate:   {history['win_rate'][-1]:.4f}")
print(f"Initial approx KL: {history['approx_kl'][0]:.4f}")
print(f"Final approx KL:   {history['approx_kl'][-1]:.4f}")

torch.save(policy_model.state_dict(), "checkpoints/gpt_dpo_aligned.pt")
print("\nAligned model saved to checkpoints/gpt_dpo_aligned.pt")