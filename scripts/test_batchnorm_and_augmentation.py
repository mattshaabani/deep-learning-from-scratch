import numpy as np
import matplotlib.pyplot as plt
from src.phase2_cnn.cnn_architectures import PlainCNN, BatchNormCNN
from src.phase2_cnn.trainer import CNNTrainer
from src.phase2_cnn.data_augmentation import get_cifar10_loaders

np.random.seed(42)


print("=== Experiment 1: BatchNorm stability at higher LR ===")

train_loader, val_loader, test_loader = get_cifar10_loaders(use_augmentation=True)

results_stability = {}
for name, model_cls in [("PlainCNN", PlainCNN), ("BatchNormCNN", BatchNormCNN)]:
    for lr in [0.001, 0.01, 0.05]:
        np.random.seed(42)
        model = model_cls(num_classes=4, depth=6)
        trainer = CNNTrainer(model, device="cpu", learning_rate=lr)
        history = trainer.fit(train_loader, val_loader, epochs=8, verbose=False)

        key = f"{name}_lr{lr}"
        results_stability[key] = history
        final_loss = history["train_loss"][-1]
        diverged = np.isnan(final_loss) or final_loss > 5.0
        print(f"{key:20s}: final_train_loss={final_loss:.4f} diverged={diverged} final_val_acc={history['val_acc'][-1]:.4f}")



print("\n=== Experiment 2: Data Augmentation as Regularization ===")

train_loader_aug, val_loader_aug, _ = get_cifar10_loaders(use_augmentation=True)
train_loader_noaug, val_loader_noaug, _ = get_cifar10_loaders(use_augmentation=False)

results_aug = {}
for name, (t_loader, v_loader) in [
    ("With augmentation", (train_loader_aug, val_loader_aug)),
    ("No augmentation",    (train_loader_noaug, val_loader_noaug)),
]:
    np.random.seed(42)
    model = BatchNormCNN(num_classes=4, depth=6)
    trainer = CNNTrainer(model, device="cpu", learning_rate=0.001)
    history = trainer.fit(t_loader, v_loader, epochs=15, verbose=False)
    results_aug[name] = history

    train_val_gap = history["train_acc"][-1] - history["val_acc"][-1]
    print(f"{name:20s}: train_acc={history['train_acc'][-1]:.4f} val_acc={history['val_acc'][-1]:.4f} "
          f"val_loss={history['val_loss'][-1]:.4f} train_val_gap={train_val_gap:.4f}")


fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for key, history in results_stability.items():
    if "lr0.05" in key:  # the LR most likely to reveal instability
        axes[0].plot(history["train_loss"], label=key)
axes[0].set_title("Training Loss at LR=0.05 (Plain vs BatchNorm)")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Train Loss")
axes[0].legend()
axes[0].grid(alpha=0.3)

for name, history in results_aug.items():
    axes[1].plot(history["val_loss"], label=name)
axes[1].set_title("Validation Loss: Augmentation as Regularization")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Val Loss")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("data/batchnorm_and_augmentation.png", dpi=150)
plt.show()

print("\nSaved plot to data/batchnorm_and_augmentation.png")