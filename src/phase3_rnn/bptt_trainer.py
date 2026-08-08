"""
src/phase3_rnn/bptt_trainer.py

Training loop for sequence models (VanillaRNN, LSTM) with per-timestep
gradient norm tracking -- the temporal analog of Phase 2's per-layer
gradient norm tracking.

Usage:
    from src.phase3_rnn.bptt_trainer import BPTTTrainer
    trainer = BPTTTrainer(model, dataset)
    history = trainer.fit(epochs=10, seq_length=100)
"""

import time
import torch
import torch.nn as nn
import numpy as np
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BPTTTrainer:
    """
    Trains a character-level sequence model (VanillaRNN or LSTM) with
    next-character prediction, tracking per-timestep gradient norms
    to directly measure vanishing gradients through time.

    Key technique -- retain_grad() on intermediate hidden states:
        By default PyTorch only keeps .grad for LEAF tensors (like
        model parameters). Hidden states produced at each timestep
        are intermediate (non-leaf) tensors, so we must explicitly
        call .retain_grad() on each one BEFORE backward(), or their
        gradients get discarded immediately after use.
    """

    def __init__(self, model: nn.Module, dataset, device: str = None, learning_rate: float = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model  = model.to(self.device)
        self.dataset = dataset
        self.lr      = learning_rate or settings.phase3_training.learning_rate

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()
        self.clip_norm = settings.phase3_training.gradient_clip_norm

        logger.info(f"Initialized BPTTTrainer", extra={
            "device": self.device,
            "model":  model.__class__.__name__,
        })

    def _train_step_with_grad_tracking(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> tuple[float, list[float]]:
        """
        One training step, with gradient norms captured at EVERY timestep.

        Returns:
            loss value, list of gradient norms per timestep
                        (index 0 = earliest timestep, index -1 = latest)
        """
        self.optimizer.zero_grad()

        batch_size, seq_length = x.shape
        embedded = self.model.embedding(x)

        # Manually unroll the model's cell so we can retain_grad()
        # on every intermediate hidden state -- this requires
        # duplicating the unroll logic from VanillaRNN/LSTM.forward(),
        # since we need access to each h_t BEFORE it gets stacked away.
        is_lstm = hasattr(self.model.cell, "candidate_gate")

        if is_lstm:
            h_t, c_t = self.model.cell.init_hidden(batch_size, device=x.device)
        else:
            h_t = self.model.cell.init_hidden(batch_size, device=x.device)

        all_hidden_states = []
        for t in range(seq_length):
            if is_lstm:
                h_t, c_t = self.model.cell(embedded[:, t, :], h_t, c_t)
            else:
                h_t = self.model.cell(embedded[:, t, :], h_t)

            h_t.retain_grad()   # THE key line -- keep gradient for this intermediate tensor
            all_hidden_states.append(h_t)

        hidden_stack = torch.stack(all_hidden_states, dim=1)
        logits = self.model.output_layer(hidden_stack)

        loss = self.criterion(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
        )

        loss.backward()

        # Now every h_t in all_hidden_states has a populated .grad
        grad_norms = [h.grad.norm(2).item() for h in all_hidden_states]

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)
        self.optimizer.step()

        return loss.item(), grad_norms

    def fit(
        self,
        epochs: int = None,
        seq_length: int = None,
        steps_per_epoch: int = 50,
        verbose: bool = True,
    ) -> dict:
        """
        Train for a fixed number of epochs, each consisting of
        steps_per_epoch random batches sampled from the corpus.

        Returns:
            history dict with train_loss, val_loss per epoch, and
            grad_norms_last_step (per-timestep gradient norms from
            the very last training step, for gradient-through-time analysis)
        """
        epochs = epochs or settings.phase3_training.epochs
        seq_length = seq_length or settings.phase3_sequence.default_seq_length
        batch_size = settings.phase3_sequence.batch_size

        history = {"train_loss": [], "val_loss": [], "grad_norms_last_step": None}

        for epoch in range(epochs):
            start = time.time()
            self.model.train()

            epoch_losses = []
            last_grad_norms = None

            for step in range(steps_per_epoch):
                x, y = self.dataset.get_batch(seq_length=seq_length, batch_size=batch_size, split="train")
                x, y = x.to(self.device), y.to(self.device)

                loss, grad_norms = self._train_step_with_grad_tracking(x, y)
                epoch_losses.append(loss)
                last_grad_norms = grad_norms   # keep overwriting -- final step's norms survive

            train_loss = float(np.mean(epoch_losses))

            # Validation
            self.model.eval()
            with torch.no_grad():
                x_val, y_val = self.dataset.get_batch(seq_length=seq_length, batch_size=batch_size, split="val")
                x_val, y_val = x_val.to(self.device), y_val.to(self.device)
                logits, _, _ = self.model(x_val)
                val_loss = self.criterion(
                    logits.reshape(-1, logits.size(-1)),
                    y_val.reshape(-1),
                ).item()

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            elapsed = time.time() - start

            if verbose:
                logger.info(
                    f"Epoch {epoch+1}/{epochs} | train_loss={train_loss:.4f} | "
                    f"val_loss={val_loss:.4f} | {elapsed:.1f}s"
                )

        history["grad_norms_last_step"] = last_grad_norms
        return history