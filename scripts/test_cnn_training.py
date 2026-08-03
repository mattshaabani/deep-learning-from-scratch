import numpy as np
from src.phase2_cnn.data_augmentation import get_cifar10_loaders
from src.phase2_cnn.cnn_architectures import PlainCNN, BatchNormCNN, ResNetCNN
from src.phase2_cnn.trainer import CNNTrainer

train_loader, val_loader, test_loader = get_cifar10_loaders(use_augmentation=True)

results = {}

for name, model_cls in [("PlainCNN", PlainCNN), ("BatchNormCNN", BatchNormCNN), ("ResNetCNN", ResNetCNN)]:
    print(f"\n=== Training {name} ===")
    model = model_cls(num_classes=4, depth=6)   # deeper, to make vanishing gradients visible
    trainer = CNNTrainer(model, device="cpu")
    history = trainer.fit(train_loader, val_loader, epochs=10, verbose=True)
    results[name] = history

print("\n\n=== FINAL COMPARISON ===")
print(f"{'Model':<15}{'Train Acc':>12}{'Val Acc':>12}{'Val Loss':>12}")
for name, history in results.items():
    print(f"{name:<15}{history['train_acc'][-1]:>12.4f}{history['val_acc'][-1]:>12.4f}{history['val_loss'][-1]:>12.4f}")

print("\n\n=== GRADIENT NORMS (last epoch, layer 0 = closest to input) ===")
for name, history in results.items():
    last_grad_norms = history["grad_norms_by_epoch"][-1]
    print(f"\n{name}:")
    for layer_name, norm in last_grad_norms.items():
        print(f"  {layer_name}: {norm:.6f}")