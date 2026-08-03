"""
src/phase2_cnn/trainer.py

Training loop for Phase 2 CNN architectures, with per-layer gradient
norm tracking to directly measure and visualize the vanishing gradient
phenomenon (and how residual connections mitigate it).

Usage:
    from src.phase2_cnn.trainer import CNNTrainer
    trainer = CNNTrainer(model, device="cpu")
    history = trainer.fit(train_loader, val_loader, epochs=20)
"""

import time
import torch
import torch.nn as nn
import numpy as np
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CNNTrainer:
    """
    Trains a PyTorch CNN with standard cross-entropy loss, tracks
    per-layer gradient norms for vanishing-gradient analysis, and
    reports train/val loss and accuracy per epoch.
    """

    def __init__(self, model: nn.Module, device: str = None, learning_rate: float = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model  = model.to(self.device)
        self.lr     = learning_rate or settings.phase2_training.learning_rate

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=settings.phase2_training.weight_decay,
        )
        self.criterion = nn.CrossEntropyLoss()

        logger.info(f"Initialized CNNTrainer", extra={
            "device": self.device,
            "model":  model.__class__.__name__,
        })

    def _get_conv_layer_grad_norms(self) -> dict[str, float]:
        """
        Compute the L2 gradient norm for every Conv2d layer's weights,
        in order, right after a backward() call.

        Returns:
            dict mapping "layer_0", "layer_1", ... to their gradient norm,
            in the order they appear in the model (input-side layers first).
        """
        grad_norms = {}
        conv_idx = 0
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d) and module.weight.grad is not None:
                norm = module.weight.grad.norm(2).item()
                grad_norms[f"layer_{conv_idx}_{name}"] = norm
                conv_idx += 1
        return grad_norms

    def _run_one_epoch(self, loader, training: bool) -> tuple[float, float, dict]:
        """
        Run one full pass over the data. Returns (avg_loss, accuracy, grad_norms).
        grad_norms is only meaningfully populated on the LAST batch of a
        training epoch (representative snapshot, not averaged across
        the whole epoch, to keep this cheap).
        """
        self.model.train() if training else self.model.eval()

        total_loss, total_correct, total_samples = 0.0, 0, 0
        last_batch_grad_norms = {}

        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)

                if training:
                    self.optimizer.zero_grad()

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                if training:
                    loss.backward()
                    last_batch_grad_norms = self._get_conv_layer_grad_norms()
                    self.optimizer.step()

                total_loss += loss.item() * images.size(0)
                total_correct += (outputs.argmax(dim=1) == labels).sum().item()
                total_samples += images.size(0)

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples

        return avg_loss, accuracy, last_batch_grad_norms

    def fit(
        self,
        train_loader,
        val_loader,
        epochs: int = None,
        verbose: bool = True,
    ) -> dict:
        """
        Full training loop.

        Returns:
            history dict with train_loss, val_loss, train_acc, val_acc
            per epoch, plus grad_norms_by_epoch (per-layer gradient
            norms captured on the last training batch of each epoch).
        """
        epochs = epochs or settings.phase2_training.epochs

        history = {
            "train_loss": [], "val_loss": [],
            "train_acc":  [], "val_acc":  [],
            "grad_norms_by_epoch": [],
        }

        for epoch in range(epochs):
            start = time.time()

            train_loss, train_acc, grad_norms = self._run_one_epoch(train_loader, training=True)
            val_loss, val_acc, _ = self._run_one_epoch(val_loader, training=False)

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["grad_norms_by_epoch"].append(grad_norms)

            elapsed = time.time() - start

            if verbose:
                logger.info(
                    f"Epoch {epoch+1}/{epochs} | "
                    f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                    f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
                    f"{elapsed:.1f}s"
                )

        return history