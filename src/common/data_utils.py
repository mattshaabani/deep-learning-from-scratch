"""
src/common/data_utils.py

Shared data loading and cross-validation utilities used across
all phases of the deep learning from scratch project.

Usage:
    from src.common.data_utils import load_moons, load_breast_cancer_data, KFoldSplitter
"""

import numpy as np
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_moons() -> tuple[np.ndarray, np.ndarray]:
    """
    Load the make_moons synthetic dataset — two interleaving half-circles.

    Why this dataset for visualization:
        It's the classic example of a NON-LINEARLY separable problem.
        A single-layer (logistic regression style) network CANNOT solve
        this — no straight line separates the two moons. This makes it
        the perfect visual proof that depth/non-linearity matters.

    Returns:
        (X, y) where X is (n_samples, 2), y is (n_samples,) binary labels
    """
    from sklearn.datasets import make_moons

    cfg = settings.dataset.moons
    X, y = make_moons(
        n_samples=cfg["n_samples"],
        noise=cfg["noise"],
        random_state=cfg["random_state"],
    )

    logger.info(f"Loaded make_moons dataset", extra={
        "n_samples": len(X), "n_features": X.shape[1]
    })

    return X, y.reshape(-1, 1)


def load_breast_cancer_data() -> tuple[np.ndarray, np.ndarray]:
    """
    Load the Breast Cancer Wisconsin dataset — 30 real features,
    569 samples, binary classification (malignant vs benign).

    Why this dataset for rigorous experiments:
        Real medical data with genuine noise and feature correlations.
        Small enough for fast pure-NumPy training across many k-fold
        runs, large enough to have meaningful train/val/test splits.

    Returns:
        (X, y) — X standardized to zero mean unit variance, y binary labels
    """
    from sklearn.datasets import load_breast_cancer as sk_load_breast_cancer
    from sklearn.preprocessing import StandardScaler

    data = sk_load_breast_cancer()
    X, y = data.data, data.target

    # Standardize features — critical for neural net training stability
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    logger.info(f"Loaded Breast Cancer Wisconsin dataset", extra={
        "n_samples": len(X), "n_features": X.shape[1],
        "class_balance": f"{y.mean():.2%} positive",
    })

    return X, y.reshape(-1, 1)


def train_test_split_manual(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    """
    Manual train/test split implementation — no sklearn.

    Why implement this ourselves?
        Phase 1's entire point is understanding fundamentals.
        A random permutation + slicing is simple but worth doing
        explicitly rather than importing it, since we're about to
        build k-fold CV manually too — same underlying mechanism.
    """
    rng = np.random.RandomState(random_state)
    n_samples = len(X)
    indices   = rng.permutation(n_samples)

    n_test = int(n_samples * test_size)
    test_idx  = indices[:n_test]
    train_idx = indices[n_test:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


class KFoldSplitter:
    """
    Manual k-fold cross-validation splitter — implemented from scratch.

    Why implement k-fold manually instead of sklearn.model_selection.KFold?
        K-fold CV is conceptually simple (split into k chunks, rotate
        which chunk is held out) but deeply important to understand
        exactly, since we're about to use it to make real claims about
        generalization. Implementing it removes any "black box" feeling.

    Algorithm:
        1. Shuffle indices (if shuffle=True)
        2. Split into k roughly-equal contiguous chunks
        3. For each fold i: chunk i is validation, rest is training
    """

    def __init__(
        self,
        k_folds: int = None,
        shuffle: bool = None,
        random_state: int = None,
    ):
        self.k_folds      = k_folds or settings.cross_validation.k_folds
        self.shuffle       = shuffle if shuffle is not None else settings.cross_validation.shuffle
        self.random_state  = random_state or settings.cross_validation.random_state

    def split(self, X: np.ndarray):
        """
        Generate train/val indices for each fold.

        Yields:
            (train_indices, val_indices) for each of the k_folds folds
        """
        n_samples = len(X)
        indices   = np.arange(n_samples)

        if self.shuffle:
            rng = np.random.RandomState(self.random_state)
            rng.shuffle(indices)

        fold_sizes = np.full(self.k_folds, n_samples // self.k_folds, dtype=int)
        fold_sizes[: n_samples % self.k_folds] += 1   # distribute remainder

        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            val_indices   = indices[start:stop]
            train_indices = np.concatenate([indices[:start], indices[stop:]])
            yield train_indices, val_indices
            current = stop