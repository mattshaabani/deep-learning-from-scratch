"""
src/phase3_rnn/copy_task.py

The "copy task" -- a synthetic long-range memory benchmark, originally
used by Hochreiter & Schmidhuber (1997) to demonstrate LSTM's advantage
over vanilla RNNs. Unlike natural language (which has strong local
structure a model can exploit without long-range memory), this task
is specifically designed to REQUIRE carrying information across a
long temporal gap, with nothing else to rely on.

Task structure:
    Input:  [s1, s2, ..., sk, blank, blank, ..., blank, GO]
    Target: [ignore, ...,     ignore,         ..., s1, s2, ..., sk]

The model must output the k symbols in order, immediately after
seeing the GO token -- which requires remembering them across the
entire blank gap.

Usage:
    from src.phase3_rnn.copy_task import CopyTaskDataset
    dataset = CopyTaskDataset(num_symbols=8, gap_length=100)
    x, y = dataset.get_batch(batch_size=32)
"""

import torch
import numpy as np


class CopyTaskDataset:
    """
    Generates copy-task sequences with a configurable gap length --
    our direct "dial" for testing how far back each architecture can
    successfully carry information.

    Vocabulary:
        0 to num_symbols-1  : the symbols to copy
        num_symbols          : BLANK token (filler during the gap)
        num_symbols + 1       : GO token (signals "now reproduce the symbols")
    """

    def __init__(self, num_symbols: int = 8, key_length: int = 4, gap_length: int = 50):
        self.num_symbols = num_symbols
        self.key_length   = key_length
        self.gap_length    = gap_length

        self.blank_token = num_symbols
        self.go_token     = num_symbols + 1
        self.vocab_size    = num_symbols + 2

        self.seq_length = key_length + gap_length + 1 + key_length

    def get_batch(self, batch_size: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            x: (batch, seq_length) input sequence including symbols, blanks, GO
            y: (batch, seq_length) target sequence -- symbols to predict ONLY
               at the positions after GO; all other positions use -100
               (PyTorch's ignore_index convention for CrossEntropyLoss)
        """
        keys = np.random.randint(0, self.num_symbols, size=(batch_size, self.key_length))

        x = np.full((batch_size, self.seq_length), self.blank_token, dtype=np.int64)
        y = np.full((batch_size, self.seq_length), -100, dtype=np.int64)   # -100 = ignore in loss

        x[:, :self.key_length] = keys
        go_position = self.key_length + self.gap_length
        x[:, go_position] = self.go_token

        # Target: reproduce the keys immediately after the GO token
        y[:, go_position + 1 : go_position + 1 + self.key_length] = keys

        return torch.from_numpy(x), torch.from_numpy(y)

    def compute_copy_accuracy(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Accuracy computed ONLY on the positions that matter -- the
        key-reproduction positions after GO (where target != -100).
        """
        mask = targets != -100
        correct = (predictions == targets) & mask
        return (correct.sum().float() / mask.sum().float()).item()