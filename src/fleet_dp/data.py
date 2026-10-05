"""Data loading, partitioning, and distribution utilities.

Provides MNIST/CIFAR-10 loading and non-IID Dirichlet partitioning
to simulate heterogeneous data across fleet vehicle clients.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def get_mnist_transforms() -> transforms.Compose:
    """Standard MNIST preprocessing transforms."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])


def get_cifar10_transforms(train: bool = True) -> transforms.Compose:
    """CIFAR-10 preprocessing transforms."""
    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])


def load_dataset(
    name: str = "mnist",
    data_dir: str = "./data",
) -> tuple[datasets.VisionDataset, datasets.VisionDataset]:
    """Load train and test datasets.

    Args:
        name: Dataset name ('mnist' or 'cifar10').
        data_dir: Directory to store/load data.

    Returns:
        Tuple of (train_dataset, test_dataset).
    """
    if name == "mnist":
        transform = get_mnist_transforms()
        train_ds = datasets.MNIST(
            data_dir, train=True, download=True, transform=transform
        )
        test_ds = datasets.MNIST(
            data_dir, train=False, download=True, transform=transform
        )
    elif name == "cifar10":
        train_ds = datasets.CIFAR10(
            data_dir, train=True, download=True,
            transform=get_cifar10_transforms(train=True),
        )
        test_ds = datasets.CIFAR10(
            data_dir, train=False, download=True,
            transform=get_cifar10_transforms(train=False),
        )
    else:
        raise ValueError(f"Unsupported dataset: {name}. Use 'mnist' or 'cifar10'.")

    return train_ds, test_ds


def partition_dirichlet(
    targets: np.ndarray,
    num_clients: int,
    alpha: float,
    seed: int = 42,
) -> list[np.ndarray]:
    """Partition dataset indices using Dirichlet distribution for non-IID splits.

    This creates heterogeneous data distributions across clients, simulating
    the uneven local data distributions that may occur across vehicles in a fleet.

    Args:
        targets: Array of class labels for each sample.
        num_clients: Number of clients (vehicles) to partition data across.
        alpha: Dirichlet concentration parameter. Lower values create
               more heterogeneous distributions.
        seed: Random seed for reproducibility.

    Returns:
        List of arrays, where each array contains the indices assigned
        to that client.
    """
    rng = np.random.default_rng(seed)
    num_classes = len(np.unique(targets))
    client_indices: list[list[int]] = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        class_indices = np.where(targets == c)[0]
        rng.shuffle(class_indices)

        # Draw Dirichlet proportions for this class
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))

        # Ensure no client gets zero samples (minimum 1 if available)
        proportions = proportions / proportions.sum()

        # Split class indices according to proportions
        splits = (proportions * len(class_indices)).astype(int)

        # Distribute remainder to avoid losing samples
        remainder = len(class_indices) - splits.sum()
        for i in range(remainder):
            splits[i % num_clients] += 1

        current = 0
        for client_id in range(num_clients):
            end = current + splits[client_id]
            client_indices[client_id].extend(class_indices[current:end].tolist())
            current = end

    # Convert to numpy arrays and shuffle each client's data
    result = []
    for indices in client_indices:
        arr = np.array(indices, dtype=np.int64)
        rng.shuffle(arr)
        result.append(arr)

    return result


def partition_iid(
    num_samples: int,
    num_clients: int,
    seed: int = 42,
) -> list[np.ndarray]:
    """Partition dataset indices equally (IID) across clients.

    Args:
        num_samples: Total number of samples.
        num_clients: Number of clients.
        seed: Random seed for reproducibility.

    Returns:
        List of arrays with indices for each client.
    """
    rng = np.random.default_rng(seed)
    indices = np.arange(num_samples)
    rng.shuffle(indices)

    splits = np.array_split(indices, num_clients)
    return [s.astype(np.int64) for s in splits]


def get_client_loaders(
    train_dataset: datasets.VisionDataset,
    num_clients: int,
    batch_size: int,
    iid: bool = False,
    dirichlet_alpha: float = 0.3,
    seed: int = 42,
) -> list[DataLoader]:
    """Create data loaders for each client with specified partitioning.

    Args:
        train_dataset: Full training dataset.
        num_clients: Number of clients/vehicles.
        batch_size: Batch size for training.
        iid: If True, use IID partitioning; otherwise Dirichlet.
        dirichlet_alpha: Alpha parameter for Dirichlet distribution.
        seed: Random seed.

    Returns:
        List of DataLoader objects, one per client.
    """
    if iid:
        client_indices = partition_iid(len(train_dataset), num_clients, seed)
    else:
        targets = np.array(train_dataset.targets)
        client_indices = partition_dirichlet(
            targets, num_clients, dirichlet_alpha, seed
        )

    loaders = []
    for indices in client_indices:
        if len(indices) == 0:
            raise ValueError(
                f"Client received 0 samples. Try increasing alpha or "
                f"reducing num_clients. alpha={dirichlet_alpha}, "
                f"num_clients={num_clients}"
            )
        subset = Subset(train_dataset, indices.tolist())
        loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
        )
        loaders.append(loader)

    return loaders


def get_client_label_distribution(
    train_dataset: datasets.VisionDataset,
    client_indices: list[np.ndarray],
) -> np.ndarray:
    """Compute label distribution for each client.

    Args:
        train_dataset: Full training dataset.
        client_indices: List of index arrays per client.

    Returns:
        Array of shape (num_clients, num_classes) with sample counts.
    """
    targets = np.array(train_dataset.targets)
    num_classes = len(np.unique(targets))
    num_clients = len(client_indices)

    distribution = np.zeros((num_clients, num_classes), dtype=int)
    for client_id, indices in enumerate(client_indices):
        for label in targets[indices]:
            distribution[client_id, label] += 1

    return distribution


def get_test_loader(
    test_dataset: datasets.VisionDataset,
    batch_size: int = 256,
) -> DataLoader:
    """Create a DataLoader for the test set.

    Args:
        test_dataset: Test dataset.
        batch_size: Batch size for evaluation.

    Returns:
        DataLoader for the test set.
    """
    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
