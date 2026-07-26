import numpy as np
from src.common.data_utils import load_breast_cancer_data
from src.common.cross_validator import CrossValidator
from src.phase1_mlp.network import NeuralNetwork

np.random.seed(42)

X, y = load_breast_cancer_data()

def model_factory():
    return NeuralNetwork(
        layer_sizes=[30, 16, 8, 1],
        activations=["relu", "relu", "sigmoid"],
        optimizer="adam",
        learning_rate=0.01,
        l2_lambda=0.001,   # small L2 regularization this time
    )

cv = CrossValidator(k_folds=5)
results = cv.run(X, y, model_factory, epochs=100, verbose=True)

print("\n=== 5-FOLD CROSS-VALIDATION RESULTS (Adam + L2=0.001) ===")
print(f"Per-fold val accuracy: {[round(a, 4) for a in results['fold_val_acc']]}")
print(f"Mean val accuracy: {results['mean_val_acc']:.4f} +/- {results['std_val_acc']:.4f}")
print(f"Mean val loss:     {results['mean_val_loss']:.4f} +/- {results['std_val_loss']:.4f}")

print("\n=== Compare: Adam WITHOUT L2 regularization ===")
def model_factory_no_reg():
    return NeuralNetwork(
        layer_sizes=[30, 16, 8, 1],
        activations=["relu", "relu", "sigmoid"],
        optimizer="adam",
        learning_rate=0.01,
        l2_lambda=0.0,
    )

cv2 = CrossValidator(k_folds=5)
results_no_reg = cv2.run(X, y, model_factory_no_reg, epochs=100, verbose=False)

print(f"Mean val accuracy: {results_no_reg['mean_val_acc']:.4f} +/- {results_no_reg['std_val_acc']:.4f}")
print(f"Mean val loss:     {results_no_reg['mean_val_loss']:.4f} +/- {results_no_reg['std_val_loss']:.4f}")

print(f"\n=== L2 regularization effect ===")
print(f"Val loss WITHOUT L2: {results_no_reg['mean_val_loss']:.4f}")
print(f"Val loss WITH L2:    {results['mean_val_loss']:.4f}")