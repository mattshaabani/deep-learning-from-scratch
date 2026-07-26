"""
src/common/cross_validator.py

Wraps KFoldSplitter with a model training/evaluation loop to run
full k-fold cross-validation studies in one call.

Usage:
    from src.common.cross_validator import CrossValidator
    from src.phase1_mlp.network import NeuralNetwork

    def model_factory():
        return NeuralNetwork(
            layer_sizes=[30, 16, 8, 1],
            activations=["relu", "relu", "sigmoid"],
            optimizer="adam",
            learning_rate=0.01,
        )

    cv = CrossValidator(k_folds=5)
    results = cv.run(X, y, model_factory, epochs=100)
    print(results["mean_val_acc"], results["std_val_acc"])
"""

import numpy as np
from src.common.data_utils import KFoldSplitter
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CrossValidator:
    """
    Runs k-fold cross-validation for any model exposing .fit() and
    .predict_proba() -- specifically our from-scratch NeuralNetwork,
    but written generically enough to also validate against sklearn
    models for comparison.

    Why a NEW model per fold matters:
        If we reused the same NeuralNetwork object across folds, its
        weights from fold 1 would carry over into fold 2's training --
        contaminating the "independent estimate" that k-fold CV is
        supposed to give us. Each fold must start from a fresh,
        freshly-initialized model.
    """

    def __init__(self, k_folds: int = None, shuffle: bool = None, random_state: int = None):
        self.splitter = KFoldSplitter(k_folds=k_folds, shuffle=shuffle, random_state=random_state)

    def run(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_factory,
        epochs: int = 100,
        batch_size: int = 32,
        verbose: bool = False,
    ) -> dict:
        """
        Run full k-fold cross-validation.

        Args:
            X, y:          Full dataset (will be split into folds internally)
            model_factory: A ZERO-ARGUMENT function that returns a freshly
                          initialized model each time it's called.
            epochs, batch_size: Passed through to each fold's .fit() call

        Returns:
            dict with per-fold metrics AND aggregated mean/std across folds --
            the mean/std pair is the actual statistically honest way to
            report cross-validated performance, since a single number
            hides how much it varies fold-to-fold.
        """
        fold_train_acc = []
        fold_val_acc    = []
        fold_val_loss    = []
        fold_histories    = []

        for fold_idx, (train_idx, val_idx) in enumerate(self.splitter.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val   = y[train_idx], y[val_idx]

            model = model_factory()
            history = model.fit(
                X_train, y_train, X_val, y_val,
                epochs=epochs, batch_size=batch_size, verbose=False,
            )

            final_train_acc = history["train_acc"][-1]
            final_val_acc     = history["val_acc"][-1]
            final_val_loss     = history["val_loss"][-1]

            fold_train_acc.append(final_train_acc)
            fold_val_acc.append(final_val_acc)
            fold_val_loss.append(final_val_loss)
            fold_histories.append(history)

            if verbose:
                logger.info(f"Fold {fold_idx+1} complete", extra={
                    "train_acc": round(final_train_acc, 4),
                    "val_acc":   round(final_val_acc, 4),
                    "val_loss":  round(final_val_loss, 4),
                })

        results = {
            "fold_train_acc": fold_train_acc,
            "fold_val_acc":    fold_val_acc,
            "fold_val_loss":    fold_val_loss,
            "mean_train_acc":  float(np.mean(fold_train_acc)),
            "std_train_acc":    float(np.std(fold_train_acc)),
            "mean_val_acc":      float(np.mean(fold_val_acc)),
            "std_val_acc":        float(np.std(fold_val_acc)),
            "mean_val_loss":       float(np.mean(fold_val_loss)),
            "std_val_loss":         float(np.std(fold_val_loss)),
            "histories":             fold_histories,
        }

        logger.info(f"K-fold CV complete", extra={
            "k_folds":         len(fold_val_acc),
            "mean_val_acc":    round(results["mean_val_acc"], 4),
            "std_val_acc":      round(results["std_val_acc"], 4),
        })

        return results