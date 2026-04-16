"""
CIFAR-10 / CIFAR-100 data loaders.

Images are returned in [-1, 1] with random horizontal flip for training.
"""

import torch
import torch.utils.data
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_loaders(
    dataset: str = "cifar10",
    batch_size: int = 128,
    num_workers: int = 4,
    data_root: str = "./data",
):
    """
    Returns (train_loader, test_loader).

    dataset : "cifar10" | "cifar100"
    """
    assert dataset in ("cifar10", "cifar100"), f"Unknown dataset: {dataset!r}"

    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # → [-1, 1]
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    Cls = datasets.CIFAR10 if dataset == "cifar10" else datasets.CIFAR100

    train_ds = Cls(data_root, train=True,  download=True, transform=train_tf)
    test_ds  = Cls(data_root, train=False, download=True, transform=test_tf)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=True,
        persistent_workers=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=False,
    )
    return train_loader, test_loader
