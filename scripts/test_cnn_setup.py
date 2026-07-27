import torch
from src.phase2_cnn.data_augmentation import get_cifar10_loaders, CIFAR10_CLASSES
from src.phase2_cnn.cnn_architectures import PlainCNN, BatchNormCNN, ResNetCNN, count_parameters

print("=== Loading CIFAR-10 (subset) ===")
train_loader, val_loader, test_loader = get_cifar10_loaders(use_augmentation=True)

images, labels = next(iter(train_loader))
print(f"Batch shape: {images.shape}")
print(f"Labels: {labels[:10].tolist()}")
print(f"Classes in this batch: {[CIFAR10_CLASSES[l] for l in labels[:5].tolist()]}")

print("\n=== Testing architectures with a forward pass ===")
num_classes = 4  # matches our subset config

for name, model_cls in [("PlainCNN", PlainCNN), ("BatchNormCNN", BatchNormCNN), ("ResNetCNN", ResNetCNN)]:
    model = model_cls(num_classes=num_classes, depth=4)
    output = model(images)
    n_params = count_parameters(model)
    print(f"{name:15s} | output shape: {output.shape} | trainable params: {n_params:,}")