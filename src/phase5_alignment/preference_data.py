"""
src/phase5_alignment/preference_data.py

Generates synthetic preference pairs for DPO training using an
anti-repetition heuristic: generate multiple completions from the
base model for the same prompt, score each by n-gram repetition
rate, and label the least-repetitive completion as the "winner"
and the most-repetitive as the "loser".

This gives DPO a genuine, automatically-labeled preference signal
without requiring human annotation, while targeting a real,
common failure mode of small language models (repetitive/looping
generation).

Usage:
    from src.phase5_alignment.preference_data import generate_preference_pairs
    pairs = generate_preference_pairs(model, dataset, num_pairs=500)
"""

import torch
from dataclasses import dataclass
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PreferencePair:
    """One preference comparison: prompt, winning completion, losing completion."""
    prompt_ids:  torch.Tensor
    winner_ids:  torch.Tensor
    loser_ids:   torch.Tensor
    winner_repetition_rate: float
    loser_repetition_rate: float


def compute_repetition_rate(token_ids: list[int], n: int = None) -> float:
    """
    Fraction of n-grams in the sequence that are REPEATS of an
    earlier n-gram in the same sequence.

    High repetition rate = degenerate, looping generation (a real,
    common failure mode of small/undertrained language models).
    Low repetition rate = more varied, natural-looking text.
    """
    n = n or settings.phase5_eval.repetition_window

    if len(token_ids) < n + 1:
        return 0.0

    ngrams = [tuple(token_ids[i:i+n]) for i in range(len(token_ids) - n + 1)]
    seen = set()
    repeats = 0

    for ng in ngrams:
        if ng in seen:
            repeats += 1
        seen.add(ng)

    return repeats / len(ngrams)


def generate_completions(model, prompt_ids: torch.Tensor, num_samples: int, length: int, temperature: float = 1.0):
    """
    Generate multiple completions for the same prompt, sampling with
    temperature (not greedy) so completions genuinely differ in
    quality/repetitiveness -- this variance is exactly what we need
    to construct meaningful winner/loser pairs.
    """
    model.eval()
    completions = []

    with torch.no_grad():
        for _ in range(num_samples):
            ids = prompt_ids.clone()
            for _ in range(length):
                logits, _ = model(ids.unsqueeze(0))
                next_logits = logits[0, -1, :] / temperature
                probs = torch.softmax(next_logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
                ids = torch.cat([ids, next_id])
            completions.append(ids)

    return completions


def generate_preference_pairs(
    model,
    dataset,
    num_pairs: int = None,
    prompt_length: int = 20,
    completion_length: int = 40,
    completions_per_prompt: int = 4,
) -> list[PreferencePair]:
    """
    Build a synthetic preference dataset by generating multiple
    completions per prompt and ranking by repetition rate.

    For each prompt: generate `completions_per_prompt` completions,
    take the LEAST repetitive as winner, the MOST repetitive as loser.
    """
    num_pairs = num_pairs or settings.phase5_preference.num_pairs

    pairs = []

    logger.info(f"Generating preference pairs", extra={
        "num_pairs": num_pairs,
        "completions_per_prompt": completions_per_prompt,
    })

    while len(pairs) < num_pairs:
        # Sample a random prompt from the corpus
        x, _ = dataset.get_batch(seq_length=prompt_length, batch_size=1, split="train")
        prompt_ids = x[0]

        completions = generate_completions(
            model, prompt_ids,
            num_samples=completions_per_prompt,
            length=completion_length,
            temperature=1.2,   # higher temp -> more variance in repetitiveness
        )

        scored = [
            (c, compute_repetition_rate(c[prompt_length:].tolist()))
            for c in completions
        ]
        scored.sort(key=lambda x: x[1])   # ascending repetition rate

        winner, winner_rate = scored[0]     # least repetitive
        loser, loser_rate   = scored[-1]     # most repetitive

        # Skip pairs with no meaningful difference -- not informative for DPO
        if loser_rate - winner_rate < 0.05:
            continue

        pairs.append(PreferencePair(
            prompt_ids=prompt_ids,
            winner_ids=winner[prompt_length:],
            loser_ids=loser[prompt_length:],
            winner_repetition_rate=winner_rate,
            loser_repetition_rate=loser_rate,
        ))

        if len(pairs) % 50 == 0:
            logger.info(f"Generated {len(pairs)}/{num_pairs} preference pairs")

    return pairs