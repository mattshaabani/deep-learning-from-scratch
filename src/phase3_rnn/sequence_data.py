"""
src/phase3_rnn/sequence_data.py

Character-level text data loading for the Tiny Shakespeare corpus.
Same corpus popularized by Karpathy's char-rnn -- small, real,
and complex enough to show genuine sequence-modeling behavior.

Usage:
    from src.phase3_rnn.sequence_data import CharDataset
    dataset = CharDataset()
    x_batch, y_batch = dataset.get_batch(seq_length=100, batch_size=32)
"""

import urllib.request
import torch
import numpy as np
from pathlib import Path
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CharDataset:
    """
    Character-level tokenizer and batch sampler for Shakespeare text.

    Why character-level: keeps vocabulary tiny (~65 unique chars vs
    tens of thousands of words), so our from-scratch RNN/LSTM cells
    stay small and fast to train while still learning genuine
    sequential structure (spelling, punctuation, dialogue patterns).
    """

    def __init__(self):
        self.data_path = settings.root_dir / "data" / "raw" / "shakespeare.txt"
        self.data_path.parent.mkdir(parents=True, exist_ok=True)

        self._download_if_needed()
        self.text = self.data_path.read_text(encoding="utf-8")

        self.chars = sorted(list(set(self.text)))
        self.vocab_size = len(self.chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}

        self.encoded = np.array([self.char_to_idx[c] for c in self.text], dtype=np.int64)

        split_idx = int(len(self.encoded) * settings.phase3_dataset.train_split)
        self.train_data = self.encoded[:split_idx]
        self.val_data   = self.encoded[split_idx:]

        logger.info(f"Loaded Shakespeare corpus", extra={
            "total_chars": len(self.text),
            "vocab_size":  self.vocab_size,
            "train_chars": len(self.train_data),
            "val_chars":   len(self.val_data),
        })

    def _download_if_needed(self):
        if self.data_path.exists():
            return
        logger.info(f"Downloading Tiny Shakespeare corpus")
        try:
            urllib.request.urlretrieve(settings.phase3_dataset.source_url, self.data_path)
        except Exception as e:
            logger.warning(f"Download failed, creating fallback text", extra={"error": str(e)})
            fallback = "To be, or not to be, that is the question:\n" * 200
            self.data_path.write_text(fallback)

    def encode(self, text: str) -> torch.Tensor:
        return torch.tensor([self.char_to_idx.get(c, 0) for c in text], dtype=torch.long)

    def decode(self, indices) -> str:
        return "".join(self.idx_to_char[int(i)] for i in indices)

    def get_batch(
        self,
        seq_length: int = None,
        batch_size: int = None,
        split: str = "train",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample a random batch of (input, target) sequences.

        Target is the input shifted by one character -- standard
        next-character-prediction setup: given chars[0:t], predict chars[t+1].

        Returns:
            x: (batch_size, seq_length) input character indices
            y: (batch_size, seq_length) target character indices (shifted by 1)
        """
        seq_length = seq_length or settings.phase3_sequence.default_seq_length
        batch_size = batch_size or settings.phase3_sequence.batch_size

        data = self.train_data if split == "train" else self.val_data

        max_start = len(data) - seq_length - 1
        starts = np.random.randint(0, max_start, size=batch_size)

        x = np.stack([data[s : s + seq_length] for s in starts])
        y = np.stack([data[s + 1 : s + seq_length + 1] for s in starts])

        return torch.from_numpy(x).long(), torch.from_numpy(y).long()