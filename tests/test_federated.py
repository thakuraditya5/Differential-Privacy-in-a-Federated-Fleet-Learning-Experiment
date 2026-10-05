"""Unit tests for federated client and server interaction."""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from fleet_dp.client import FleetClient
from fleet_dp.model import MnistCNN, create_model


@pytest.fixture
def dummy_loader():
    x = torch.randn(64, 1, 28, 28)
    y = torch.randint(0, 10, (64,))
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=16, shuffle=True)


def test_fleet_client_baseline_fit(dummy_loader):
    client = FleetClient(
        client_id=0,
        train_loader=dummy_loader,
        model_name="mnist_cnn",
        device="cpu",
        local_epochs=1,
        learning_rate=0.01,
        dp_enabled=False,
    )

    init_model = create_model("mnist_cnn")
    init_params = [v.cpu().numpy() for _, v in init_model.state_dict().items()]

    updated_params, num_samples, metrics = client.fit(init_params, config={})

    assert num_samples == 64
    assert len(updated_params) == len(init_params)
    assert "train_loss" in metrics
    assert "train_accuracy" in metrics
    # Verify parameters actually changed after training
    assert not np.array_equal(updated_params[0], init_params[0])


def test_fleet_client_dp_fit(dummy_loader):
    client = FleetClient(
        client_id=1,
        train_loader=dummy_loader,
        model_name="mnist_cnn",
        device="cpu",
        local_epochs=1,
        learning_rate=0.01,
        dp_enabled=True,
        target_epsilon=5.0,
        target_delta=1e-5,
        max_grad_norm=1.0,
    )

    init_model = create_model("mnist_cnn")
    init_params = [v.cpu().numpy() for _, v in init_model.state_dict().items()]

    updated_params, num_samples, metrics = client.fit(init_params, config={})

    assert num_samples == 64
    assert len(updated_params) == len(init_params)
    assert "epsilon_spent" in metrics
    assert metrics["epsilon_spent"] > 0.0
    assert not np.array_equal(updated_params[0], init_params[0])
