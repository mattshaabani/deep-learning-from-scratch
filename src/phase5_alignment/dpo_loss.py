"""
src/phase5_alignment/dpo_loss.py

Direct Preference Optimization (DPO) loss, derived from the KL-regularized
RLHF objective and the Bradley-Terry preference model, implemented from
scratch.

L_DPO(theta) = -log(sigmoid(
    beta * [ log(pi_theta(y_w|x)/pi_ref(y_w|x)) - log(pi_theta(y_l|x)/pi_ref(y_l|x)) ]
))

Where:
    pi_theta = the policy being trained (our GPT, being fine-tuned)
    pi_ref   = a FROZEN reference copy of the policy (before alignment)
    y_w      = the "winning" (preferred) completion
    y_l      = the "losing" (dispreferred) completion
    beta     = KL penalty strength (higher beta = stay closer to reference)

This eliminates the need for a separate reward model or RL loop --
see docs/05_dpo_derivation.md for the full step-by-step derivation.

Usage:
    from src.phase5_alignment.dpo_loss import compute_sequence_logprob, dpo_loss
    logprob = compute_sequence_logprob(model, prompt_ids, completion_ids)
    loss, metrics = dpo_loss(policy_model, ref_model, prompt_ids, winner_ids, loser_ids, beta=0.1)
"""

import torch
import torch.nn.functional as F


def compute_sequence_logprob(
    model,
    prompt_ids: torch.Tensor,
    completion_ids: torch.Tensor,
) -> torch.Tensor:
    """
    Compute log pi(completion | prompt) -- the sum of per-token log
    probabilities the model assigns to each token in the completion,
    conditioned on the prompt and all prior completion tokens.

    Args:
        model:          a GPT model (policy or reference)
        prompt_ids:     (seq_len_prompt,) token ids
        completion_ids: (seq_len_completion,) token ids

    Returns:
        scalar tensor: sum of log-probabilities across the completion
    """
    full_sequence = torch.cat([prompt_ids, completion_ids]).unsqueeze(0)   # (1, total_len)

    logits, _ = model(full_sequence)   # (1, total_len, vocab_size)

    # We want the model's predicted distribution AT each position that
    # predicts the NEXT token -- so logits[:, t] predicts token at t+1.
    # The completion starts right after the prompt, so we need logits
    # from position (prompt_len - 1) through (total_len - 2) to predict
    # completion tokens at positions prompt_len through total_len - 1.
    prompt_len = prompt_ids.shape[0]

    relevant_logits = logits[0, prompt_len - 1 : -1, :]   # (completion_len, vocab_size)
    log_probs_all = F.log_softmax(relevant_logits, dim=-1)

    # Gather the log-probability the model assigned to the ACTUAL
    # completion tokens (not just the max/argmax)
    token_log_probs = log_probs_all.gather(
        dim=-1, index=completion_ids.unsqueeze(-1)
    ).squeeze(-1)   # (completion_len,)

    return token_log_probs.sum()


def dpo_loss(
    policy_model,
    ref_model,
    prompt_ids: torch.Tensor,
    winner_ids: torch.Tensor,
    loser_ids: torch.Tensor,
    beta: float = 0.1,
) -> tuple[torch.Tensor, dict]:
    """
    Compute the DPO loss for a single preference pair.

    Args:
        policy_model: the model being trained (gradients flow through this)
        ref_model:    a FROZEN copy of the model before training (no gradients)
        prompt_ids, winner_ids, loser_ids: token id tensors
        beta:         KL penalty strength

    Returns:
        loss: scalar tensor, ready for .backward()
        metrics: dict with implicit reward margin and individual log-probs,
                 for monitoring training progress (does the model actually
                 learn to prefer winners over losers?)
    """
    # Policy model log-probs -- WITH gradients, since we're training this
    policy_winner_logprob = compute_sequence_logprob(policy_model, prompt_ids, winner_ids)
    policy_loser_logprob  = compute_sequence_logprob(policy_model, prompt_ids, loser_ids)

    # Reference model log-probs -- NO gradients, this model is frozen
    with torch.no_grad():
        ref_winner_logprob = compute_sequence_logprob(ref_model, prompt_ids, winner_ids)
        ref_loser_logprob  = compute_sequence_logprob(ref_model, prompt_ids, loser_ids)

    # The log-ratio terms: log(pi_theta(y|x) / pi_ref(y|x)) = log(pi_theta) - log(pi_ref)
    policy_winner_ratio = policy_winner_logprob - ref_winner_logprob
    policy_loser_ratio  = policy_loser_logprob - ref_loser_logprob

    # The DPO logits: beta * [winner_ratio - loser_ratio]
    dpo_logits = beta * (policy_winner_ratio - policy_loser_ratio)

    # L_DPO = -log(sigmoid(dpo_logits))
    # Using F.logsigmoid for numerical stability (avoids computing
    # sigmoid then log separately, which can underflow)
    loss = -F.logsigmoid(dpo_logits)

    # Implicit reward margin: how much more the policy "rewards" the
    # winner over the loser, in the DPO-implied reward sense.
    # This should INCREASE over training if DPO is working -- our
    # primary training-progress signal, since there's no explicit
    # reward model to inspect directly.
    implicit_reward_margin = dpo_logits.detach()

    metrics = {
        "loss": loss.item(),
        "implicit_reward_margin": implicit_reward_margin.item(),
        "policy_winner_logprob": policy_winner_logprob.item(),
        "policy_loser_logprob": policy_loser_logprob.item(),
        "ref_winner_logprob": ref_winner_logprob.item(),
        "ref_loser_logprob": ref_loser_logprob.item(),
    }

    return loss, metrics