"""
src/phase5_alignment/dpo_trainer.py

DPO training loop: fine-tunes a policy model on preference pairs
against a frozen reference copy, tracking the implicit reward margin,
approximate KL divergence from the reference, and win-rate accuracy
over training -- our primary evidence that alignment is genuinely
occurring.

Usage:
    from src.phase5_alignment.dpo_trainer import DPOTrainer
    trainer = DPOTrainer(policy_model, ref_model, beta=0.1)
    history = trainer.fit(preference_pairs, epochs=30)
"""

import copy
import torch
import numpy as np
from src.phase5_alignment.dpo_loss import dpo_loss, compute_sequence_logprob
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DPOTrainer:
    """
    Trains a policy model using Direct Preference Optimization.

    The reference model is a FROZEN deep copy of the policy model,
    taken at the start of training -- it never updates, providing
    the fixed anchor point the KL-style penalty (implicit in the
    DPO loss) measures distance from.
    """

    def __init__(
        self,
        policy_model,
        beta: float = None,
        learning_rate: float = None,
    ):
        self.policy_model = policy_model
        self.beta = beta or settings.phase5_dpo.beta
        self.lr = learning_rate or settings.phase5_dpo.learning_rate

        # Freeze a reference copy at the CURRENT state of the policy --
        # this is the anchor point DPO trains the policy not to drift
        # too far from.
        self.ref_model = copy.deepcopy(policy_model)
        for p in self.ref_model.parameters():
            p.requires_grad = False
        self.ref_model.eval()

        self.optimizer = torch.optim.Adam(self.policy_model.parameters(), lr=self.lr)

        logger.info(f"Initialized DPOTrainer", extra={
            "beta": self.beta,
            "learning_rate": self.lr,
        })

    def _approximate_kl_divergence(self, pairs, sample_size: int = 20) -> float:
        """
        Approximate KL(policy || reference) by averaging
        |log policy(y|x) - log ref(y|x)| over a sample of completions.

        This is a simplified proxy, not an exact KL computation (which
        would require summing over the full output distribution), but
        it directly measures what we actually care about: how far has
        the policy's assigned probability to these SPECIFIC sequences
        drifted from the reference's.
        """
        self.policy_model.eval()
        sample = pairs[:sample_size]

        diffs = []
        with torch.no_grad():
            for pair in sample:
                policy_lp = compute_sequence_logprob(self.policy_model, pair.prompt_ids, pair.winner_ids)
                ref_lp = compute_sequence_logprob(self.ref_model, pair.prompt_ids, pair.winner_ids)
                diffs.append(abs(policy_lp.item() - ref_lp.item()))

        self.policy_model.train()
        return float(np.mean(diffs))

    def fit(self, preference_pairs: list, epochs: int = None, verbose: bool = True) -> dict:
        """
        Train on the full preference dataset for the given number of epochs.

        Returns:
            history dict with per-epoch: mean loss, mean implicit reward
            margin, win rate (fraction of pairs where policy correctly
            ranks winner > loser), and approximate KL divergence
        """
        epochs = epochs or settings.phase5_dpo.epochs

        history = {
            "loss": [], "reward_margin": [], "win_rate": [], "approx_kl": [],
        }

        for epoch in range(epochs):
            epoch_losses = []
            epoch_margins = []
            correct = 0

            for pair in preference_pairs:
                self.optimizer.zero_grad()

                loss, metrics = dpo_loss(
                    self.policy_model, self.ref_model,
                    pair.prompt_ids, pair.winner_ids, pair.loser_ids,
                    beta=self.beta,
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 1.0)
                self.optimizer.step()

                epoch_losses.append(metrics["loss"])
                epoch_margins.append(metrics["implicit_reward_margin"])
                if metrics["implicit_reward_margin"] > 0:
                    correct += 1

            mean_loss = float(np.mean(epoch_losses))
            mean_margin = float(np.mean(epoch_margins))
            win_rate = correct / len(preference_pairs)
            approx_kl = self._approximate_kl_divergence(preference_pairs)

            history["loss"].append(mean_loss)
            history["reward_margin"].append(mean_margin)
            history["win_rate"].append(win_rate)
            history["approx_kl"].append(approx_kl)

            if verbose:
                logger.info(
                    f"Epoch {epoch+1}/{epochs} | loss={mean_loss:.4f} | "
                    f"reward_margin={mean_margin:.4f} | win_rate={win_rate:.4f} | "
                    f"approx_kl={approx_kl:.4f}"
                )

        return history