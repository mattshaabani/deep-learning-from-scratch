import torch
from src.phase3_rnn.sequence_data import CharDataset
from src.phase4_transformer.gpt_model import GPT
from src.phase5_alignment.preference_data import generate_preference_pairs, compute_repetition_rate

torch.manual_seed(42)
dataset = CharDataset()

print("Loading trained Phase 4 GPT checkpoint...")
model = GPT(vocab_size=dataset.vocab_size, d_model=64, num_heads=2, num_layers=2, d_ff=256, max_seq_length=256)
model.load_state_dict(torch.load("checkpoints/gpt_extended.pt"))

print("\nGenerating 5 preference pairs (small test batch)...\n")
pairs = generate_preference_pairs(model, dataset, num_pairs=5, completions_per_prompt=4)

for i, pair in enumerate(pairs):
    prompt_text = dataset.decode(pair.prompt_ids.tolist())
    winner_text = dataset.decode(pair.winner_ids.tolist())
    loser_text  = dataset.decode(pair.loser_ids.tolist())

    print(f"--- Pair {i+1} ---")
    print(f"Prompt: {repr(prompt_text)}")
    print(f"WINNER (repetition={pair.winner_repetition_rate:.3f}): {repr(winner_text)}")
    print(f"LOSER  (repetition={pair.loser_repetition_rate:.3f}): {repr(loser_text)}")
    print()