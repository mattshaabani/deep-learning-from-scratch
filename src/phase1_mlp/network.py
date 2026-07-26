"""
src/phase1_mlp/network.py

A full multi-layer perceptron built from DenseLayer objects,
with a complete training loop: forward pass, backward pass,
optimizer updates, and support for L1/L2 regularization and dropout.

Usage:
    from src.phase1_mlp.network import NeuralNetwork
    net = NeuralNetwork(layer_sizes=[30, 16, 8, 1], activations=["relu","relu","sigmoid"])
    history = net.fit(X_train, y_train, X_val, y_val, epochs=200)
"""

import numpy as np
from src.phase1_mlp.layers import DenseLayer
from src.phase1_mlp.losses import BinaryCrossEntropy
from src.phase1_mlp.optimizers import get_optimizer
from src.utils.logger import get_logger

logger = get_logger(__name__)


class NeuralNetwork:
    """
    A feedforward neural network assembled from DenseLayer objects.

    Supports:
        - Arbitrary depth/width via layer_sizes
        - Any optimizer from optimizers.py
        - L1/L2 weight regularization
        - Dropout (inverted-dropout scaling, active only during training)
        - Early stopping based on validation loss
    """

    def __init__(
        self,
        layer_sizes:  list[int],
        activations:  list[str],
        weight_init:  str = "he",
        optimizer:    str = "adam",
        learning_rate: float = 0.01,
        l1_lambda:    float = 0.0,
        l2_lambda:    float = 0.0,
        dropout_rate:  float = 0.0,
    ):
        assert len(layer_sizes) - 1 == len(activations), (
            "Need exactly one activation per layer transition"
        )

        self.layers = []
        for i in range(len(layer_sizes) - 1):
            self.layers.append(DenseLayer(
                input_dim=layer_sizes[i],
                output_dim=layer_sizes[i + 1],
                activation=activations[i],
                weight_init=weight_init,
            ))

        self.loss_fn      = BinaryCrossEntropy()
        self.optimizer    = get_optimizer(optimizer, learning_rate=learning_rate)
        self.l1_lambda     = l1_lambda
        self.l2_lambda      = l2_lambda
        self.dropout_rate   = dropout_rate

        # Cache of dropout masks per layer, only used during training
        self._dropout_masks = [None] * (len(self.layers) - 1)   # not applied to output layer

        logger.info(f"Initialized NeuralNetwork", extra={
            "architecture": layer_sizes,
            "activations":  activations,
            "optimizer":    optimizer,
            "l1_lambda":    l1_lambda,
            "l2_lambda":    l2_lambda,
            "dropout_rate": dropout_rate,
        })

    def _forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Forward pass through all layers, applying dropout after
        every HIDDEN layer (not the output layer) when training=True.

        Inverted dropout: during training we scale surviving activations
        by 1/(1-dropout_rate) so that at TEST TIME (training=False) we
        can use the raw activations unmodified -- no test-time rescaling needed.
        """
        a = X
        for i, layer in enumerate(self.layers):
            a = layer.forward(a)

            is_hidden_layer = i < len(self.layers) - 1
            if training and is_hidden_layer and self.dropout_rate > 0:
                mask = (np.random.rand(*a.shape) > self.dropout_rate).astype(np.float64)
                mask /= (1.0 - self.dropout_rate)   # inverted dropout scaling
                self._dropout_masks[i] = mask
                a = a * mask
            elif is_hidden_layer:
                self._dropout_masks[i] = None

        return a

    def _backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """
        Backward pass through all layers in reverse order.

        The OUTPUT layer uses the fused sigmoid+BCE gradient (dL/dz = y_hat-y)
        directly, skipping the separate activation-derivative multiplication,
        since that simplification is both correct and numerically stable.

        All HIDDEN layers use the standard chain rule, and additionally
        multiply by the cached dropout mask (dropout's backward pass simply
        zeroes/scales the same units that were zeroed/scaled going forward).
        """
        grad = self.loss_fn.fused_sigmoid_gradient(y_true, y_pred)

        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            is_output_layer = (i == len(self.layers) - 1)

            if is_output_layer:
                grad = layer.backward(grad, skip_activation_grad=True)
            else:
                if self._dropout_masks[i] is not None:
                    grad = grad * self._dropout_masks[i]
                grad = layer.backward(grad, skip_activation_grad=False)

    def _apply_regularization_and_update(self) -> None:
        """
        Apply L1/L2 gradient penalties, then update every layer's
        weights and biases via the optimizer.

        L2 penalty adds lambda*W to the gradient (equivalent to weight decay).
        L1 penalty adds lambda*sign(W) to the gradient (encourages sparsity).
        Biases are typically NOT regularized (standard convention).
        """
        for i, layer in enumerate(self.layers):
            dW = layer.dW.copy()

            if self.l2_lambda > 0:
                dW += self.l2_lambda * layer.W
            if self.l1_lambda > 0:
                dW += self.l1_lambda * np.sign(layer.W)

            layer.W = self.optimizer.update(layer.W, dW, param_id=f"layer{i}_W")
            layer.b = self.optimizer.update(layer.b, layer.db, param_id=f"layer{i}_b")

    def _compute_regularization_loss(self) -> float:
        """Add the regularization penalty terms to the reported loss."""
        reg_loss = 0.0
        for layer in self.layers:
            if self.l2_lambda > 0:
                reg_loss += self.l2_lambda * np.sum(layer.W ** 2) / 2
            if self.l1_lambda > 0:
                reg_loss += self.l1_lambda * np.sum(np.abs(layer.W))
        return reg_loss

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities (no dropout applied -- training=False)."""
        return self._forward(X, training=False)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict binary class labels."""
        return (self.predict_proba(X) >= threshold).astype(int)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val:    np.ndarray = None,
        y_val:     np.ndarray = None,
        epochs:     int = 200,
        batch_size:  int = 32,
        early_stopping_patience: int = None,
        verbose:      bool = True,
    ) -> dict:
        """
        Full training loop with mini-batch gradient descent.

        Returns:
            history dict with train_loss, val_loss, train_acc, val_acc per epoch
        """
        n_samples = X_train.shape[0]
        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

        best_val_loss = np.inf
        patience_counter = 0
        best_weights = None

        for epoch in range(epochs):
            # Shuffle each epoch -- standard SGD practice
            perm = np.random.permutation(n_samples)
            X_shuffled, y_shuffled = X_train[perm], y_train[perm]

            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                y_pred = self._forward(X_batch, training=True)
                self._backward(y_batch, y_pred)
                self._apply_regularization_and_update()

            # End-of-epoch metrics (no dropout, full dataset)
            train_pred = self.predict_proba(X_train)
            train_loss = self.loss_fn.forward(y_train, train_pred) + self._compute_regularization_loss()
            train_acc  = np.mean((train_pred >= 0.5) == y_train)

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)

            if X_val is not None:
                val_pred = self.predict_proba(X_val)
                val_loss  = self.loss_fn.forward(y_val, val_pred)
                val_acc    = np.mean((val_pred >= 0.5) == y_val)

                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)

                if early_stopping_patience is not None:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        best_weights = [(l.W.copy(), l.b.copy()) for l in self.layers]
                    else:
                        patience_counter += 1
                        if patience_counter >= early_stopping_patience:
                            logger.info(f"Early stopping at epoch {epoch+1}", extra={
                                "best_val_loss": round(best_val_loss, 4)
                            })
                            if best_weights:
                                for layer, (W, b) in zip(self.layers, best_weights):
                                    layer.W, layer.b = W, b
                            break

            if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
                msg = f"Epoch {epoch+1}/{epochs} | train_loss={train_loss:.4f} | train_acc={train_acc:.4f}"
                if X_val is not None:
                    msg += f" | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}"
                logger.info(msg)

        return history