import numpy as np
from src.common.data_utils import load_breast_cancer_data, train_test_split_manual
from src.phase1_mlp.network import NeuralNetwork

np.random.seed(42)

X, y = load_breast_cancer_data()
X_train, X_test, y_train, y_test = train_test_split_manual(X, y, test_size=0.2)

print(f"Train: {X_train.shape}, Test: {X_test.shape}\n")

for opt_name in ["sgd", "momentum", "rmsprop", "adam"]:
    np.random.seed(42)   # reset seed so all optimizers start from identical weights
    net = NeuralNetwork(
        layer_sizes=[30, 16, 8, 1],
        activations=["relu", "relu", "sigmoid"],
        optimizer=opt_name,
        learning_rate=0.01,
    )
    history = net.fit(X_train, y_train, X_test, y_test, epochs=100, batch_size=32, verbose=False)

    final_train_acc = history["train_acc"][-1]
    final_val_acc   = history["val_acc"][-1]
    final_val_loss  = history["val_loss"][-1]

    print(f"{opt_name:10s} | final_train_acc={final_train_acc:.4f} | final_val_acc={final_val_acc:.4f} | final_val_loss={final_val_loss:.4f}")

print("\nTraining complete across all four optimizers.")