"""Unit tests for dataset loading and partitioning."""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from fleet_dp.data import (
    get_client_label_distribution,
    get_client_loaders,
    partition_dirichlet,
    partition_iid,
)


@pytest.fixture
def synthetic_dataset():
    """Create a small synthetic dataset resembling MNIST."""
    num_samples = 1000
    features = torch.randn(num_samples, 1, 28, 28)
    targets = torch.tensor(np.repeat(np.arange(10), num_samples // 10), dtype=torch.long)
    dataset = TensorDataset(features, targets)
    dataset.targets = targets.numpy()
    return dataset


def test_partition_iid():
    num_samples = 800
    num_clients = 8
    splits = partition_iid(num_samples=num_samples, num_clients=num_clients, seed=42)

    assert len(splits) == num_clients
    total_assigned = sum(len(s) for s in splits)
    assert total_assigned == num_samples

    # Check mutually disjoint
    all_indices = np.concatenate(splits)
    assert len(np.unique(all_indices)) == num_samples


def test_partition_dirichlet(synthetic_dataset):
    targets = synthetic_dataset.targets
    num_clients = 5
    alpha = 0.3
    splits = partition_dirichlet(targets, num_clients=num_clients, alpha=alpha, seed=42)

    assert len(splits) == num_clients
    total_assigned = sum(len(s) for s in splits)
    assert total_assigned == len(targets)

    # Check all clients have non-empty data
    for i, s in enumerate(splits):
        assert len(s) > 0, f"Client {i} has 0 samples"

    # Check non-IID skew: variance of class distributions across clients should be positive
    dist = get_client_label_distribution(synthetic_dataset, splits)
    assert dist.shape == (num_clients, 10)
    # The distributions should not be uniform
    col_stds = dist.std(axis=0)
    assert np.mean(col_stds) > 1.0


def test_get_client_loaders(synthetic_dataset):
    loaders = get_client_loaders(
        synthetic_dataset,
        num_clients=4,
        batch_size=32,
        iid=False,
        dirichlet_alpha=0.5,
        seed=123,
    )
    assert len(loaders) == 4
    for loader in loaders:
        assert isinstance(loader, DataLoader)
        batch = next(iter(loader))
        assert batch[0].shape[1:] == (1, 28, 28)
