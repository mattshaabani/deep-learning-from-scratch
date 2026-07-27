"""
src/phase2_cnn/data_augmentation.py

CIFAR-10 data loading with a configurable subset for fast iteration,
plus data augmentation transforms used as a regularization technique
(the image analog of Phase 1's dropout/L2).

Usage:
    from src.phase2_cnn.data_augmentation import get_cifar10_loaders
    train_loader, val_loader, test_loader = get_cifar10_loaders(use_augmentation=True)
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def get_transforms(use_augmentation: bool = True):
    """
    Build the train and eval transform pipelines.

    Augmentation transforms (train only):
        RandomHorizontalFlip — images are naturally flip-invariant
        RandomCrop with padding — simulates slight translation/occlusion
        Normalization with CIFAR-10's known per-channel mean/std
    """
    normalize = transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616),
    )

    if use_augmentation:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=settings.phase2_augmentation.random_crop_padding),
            transforms.RandomHorizontalFlip() if settings.phase2_augmentation.horizontal_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    return train_transform, eval_transform


def _subset_by_classes(dataset, n_classes: int, per_class_limit: int):
    """
    Select only the first n_classes and cap samples per class --
    keeps iteration fast during development without needing the full
    50,000-image CIFAR-10 training set every run.
    """
    targets = np.array(dataset.targets)
    selected_indices = []

    for class_idx in range(n_classes):
        class_indices = np.where(targets == class_idx)[0]
        class_indices = class_indices[:per_class_limit]
        selected_indices.extend(class_indices.tolist())

    return Subset(dataset, selected_indices)


def get_cifar10_loaders(
    use_augmentation: bool = True,
    batch_size: int = None,
    use_full_dataset: bool = None,
):
    """
    Download (if needed) and load CIFAR-10, returning train/val/test DataLoaders.

    Uses a class-and-sample-limited subset by default (configured in
    phase2_config.yaml) for fast experimentation; set use_full_dataset=True
    for the final reported results.
    """
    batch_size = batch_size or settings.phase2_training.batch_size
    use_full   = use_full_dataset if use_full_dataset is not None else settings.phase2_dataset.full_run_all_classes

    train_transform, eval_transform = get_transforms(use_augmentation)

    data_dir = str(settings.root_dir / "data" / "raw" / "cifar10")

    logger.info(f"Loading CIFAR-10 dataset", extra={
        "use_augmentation": use_augmentation,
        "use_full_dataset": use_full,
    })

    full_train = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    full_test = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=eval_transform
    )

    if not use_full:
        n_classes  = settings.phase2_dataset.subset_classes
        per_class  = settings.phase2_dataset.subset_size_per_class
        full_train = _subset_by_classes(full_train, n_classes, per_class)
        full_test  = _subset_by_classes(full_test, n_classes, per_class // 5)

    # Split train into train/val
    n_train = len(full_train)
    n_val   = int(n_train * settings.phase2_dataset.test_size)
    n_train_final = n_train - n_val

    generator = torch.Generator().manual_seed(settings.phase2_dataset.random_state)
    train_subset, val_subset = torch.utils.data.random_split(
        full_train, [n_train_final, n_val], generator=generator
    )

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(full_test, batch_size=batch_size, shuffle=False, num_workers=0)

    logger.info(f"CIFAR-10 loaders ready", extra={
        "train_size": len(train_subset),
        "val_size":   len(val_subset),
        "test_size":  len(full_test),
    })

    return train_loader, val_loader, test_loader