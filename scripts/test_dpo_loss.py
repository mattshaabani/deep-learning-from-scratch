import torch
import copy
from src.phase3_rnn.sequence_data import CharDataset
from src.phase4_transformer.gpt_model import GPT
from src.phase5_alignment.dpo_loss import compute_sequence_logprob, dpo_loss

torch.manual_seed(42)
dataset = CharDataset()

print("=== Loading model and creating a frozen reference copy ===")
policy_model = GPT(vocab_size=dataset.vocab_size, d_model=64, num_heads=2, num_layers=2, d_ff=256, max_seq_length=256)
policy_model.load_state_dict(torch.load("checkpoints/gpt_extended.pt"))

ref_model = copy.deepcopy(policy_model)
for p in ref_model.parameters():
    p.requires_grad = False

print("\n=== Sanity check: log-probability computation ===")
x, y = dataset.get_batch(seq_length=50, batch_size=1, split="train")
prompt_ids = x[0, :20]
completion_ids = x[0, 20:]

logprob = compute_sequence_logprob(policy_model, prompt_ids, completion_ids)
print(f"Log-probability of a REAL Shakespeare continuation: {logprob.item():.4f}")
print(f"(should be a reasonably large negative number, e.g. -50 to -150 for 30 tokens,")
print(f" corresponding to roughly {(-logprob.item()/30):.2f} nats/token average)")

# Compare to a RANDOM (nonsense) completion -- should have MUCH lower log-prob
random_completion = torch.randint(0, dataset.vocab_size, (30,))
random_logprob = compute_sequence_logprob(policy_model, prompt_ids, random_completion)
print(f"\nLog-probability of a RANDOM nonsense completion: {random_logprob.item():.4f}")
print(f"(should be substantially MORE negative than the real continuation)")

print("\n\n=== DPO loss sanity check: identical policy and reference ===")
prompt_ids2 = x[0, :20]
winner_ids = x[0, 20:35]
loser_ids  = torch.randint(0, dataset.vocab_size, (15,))   # nonsense as the "loser"

loss, metrics = dpo_loss(policy_model, ref_model, prompt_ids2, winner_ids, loser_ids, beta=0.1)
print(f"Loss (policy == reference, real winner vs nonsense loser): {loss.item():.4f}")
print(f"Implicit reward margin: {metrics['implicit_reward_margin']:.4f}")
print(f"(margin should be POSITIVE, since the real Shakespeare text should already")
print(f" have higher log-prob than random nonsense, even before any DPO training)")

print("\n=== Gradient check: does the loss produce sensible gradients? ===")
loss.backward()
total_grad_norm = sum(p.grad.norm(2).item() for p in policy_model.parameters() if p.grad is not None)
print(f"Total gradient norm across policy model: {total_grad_norm:.4f}")
print(f"(should be non-zero and finite -- confirms backprop through the DPO loss works)")