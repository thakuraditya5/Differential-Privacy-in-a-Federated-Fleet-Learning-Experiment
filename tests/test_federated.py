"""Unit tests for federated client and server interaction."""

import csv

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
    assert "noise_multiplier" in metrics
    assert metrics["noise_multiplier"] > 0.0
    assert not np.array_equal(updated_params[0], init_params[0])


def test_fleet_client_dp_fit_loose_epsilon(dummy_loader):
    """High epsilon (ε=1000) still reports calibrated noise and spend."""
    client = FleetClient(
        client_id=1,
        train_loader=dummy_loader,
        model_name="mnist_cnn",
        device="cpu",
        local_epochs=1,
        learning_rate=0.01,
        dp_enabled=True,
        target_epsilon=1000.0,
        target_delta=1e-5,
        max_grad_norm=1.0,
    )

    init_model = create_model("mnist_cnn")
    init_params = [v.cpu().numpy() for _, v in init_model.state_dict().items()]

    _, _, metrics = client.fit(init_params, config={})

    assert "epsilon_spent" in metrics
    assert metrics["epsilon_spent"] > 0.0
    assert "noise_multiplier" in metrics
    assert metrics["noise_multiplier"] > 0.0


def test_client_accounting_is_cumulative_across_rounds(dummy_loader):
    client = FleetClient(
        client_id=0,
        train_loader=dummy_loader,
        dp_enabled=True,
        target_epsilon=5.0,
        noise_multiplier=0.8,
    )
    model = create_model("mnist_cnn")
    parameters = [value.detach().numpy() for value in model.state_dict().values()]

    _, _, first_metrics = client.fit(parameters, config={"server_round": 1})
    _, _, second_metrics = client.fit(parameters, config={"server_round": 2})

    assert second_metrics["epsilon_spent"] > first_metrics["epsilon_spent"]
    assert second_metrics["noise_multiplier"] == pytest.approx(0.8)
    assert second_metrics["noise_norm"] > 0.0
    assert second_metrics["clipped_grad_norm"] > 0.0


def test_clipping_only_control_adds_no_noise(dummy_loader):
    client = FleetClient(
        client_id=0,
        train_loader=dummy_loader,
        clipping_only=True,
        max_grad_norm=0.5,
    )
    model = create_model("mnist_cnn")
    parameters = [value.detach().numpy() for value in model.state_dict().values()]

    _, _, metrics = client.fit(parameters, config={"server_round": 1})

    assert metrics["clipping_only"] is True
    assert metrics["noise_norm"] == pytest.approx(0.0)
    assert metrics["clipped_sample_grad_norm"] <= 0.5


def test_epsilon_sweep_resets_calibration_for_each_condition(monkeypatch):
    from fleet_dp.config import ExperimentConfig
    from fleet_dp.experiment import run_epsilon_sweep

    config = ExperimentConfig()
    config.dp.noise_multiplier = 9.0
    config.dp.current_epsilon = 8.0
    observed = []

    def capture_run(config, seed):
        observed.append((
            config.dp.enabled,
            config.dp.clipping_only,
            config.dp.target_epsilon,
            config.dp.noise_multiplier,
            config.dp.current_epsilon,
        ))
        return []

    monkeypatch.setattr("fleet_dp.experiment.run_single_experiment", capture_run)
    run_epsilon_sweep(config, [None, 0.1, 100.0], [42], include_clipping_only=True)

    assert observed == [
        (False, True, None, None, None),
        (False, False, None, None, None),
        (True, False, 0.1, None, None),
        (True, False, 100.0, None, None),
    ]


def test_fit_metrics_aggregation_records_calibrated_dp():
    from fleet_dp.config import ExperimentConfig
    from fleet_dp.server import get_fit_metrics_aggregation_fn

    config = ExperimentConfig()
    config.dp.enabled = True
    config.dp.target_epsilon = 1000.0
    assert config.dp.noise_multiplier is None
    assert config.dp.current_epsilon is None

    aggregate = get_fit_metrics_aggregation_fn(config)
    result = aggregate(
        [
            (32, {
                "train_loss": 1.0,
                "train_accuracy": 0.5,
                "epsilon_spent": 12.5,
                "noise_multiplier": 0.4,
            }),
            (32, {
                "train_loss": 2.0,
                "train_accuracy": 0.25,
                "epsilon_spent": 10.0,
                "noise_multiplier": 0.6,
            }),
        ]
    )

    assert result["noise_multiplier"] == pytest.approx(0.5)
    assert result["max_epsilon_spent"] == pytest.approx(12.5)
    assert config.dp.noise_multiplier == pytest.approx(0.5)
    assert config.dp.current_epsilon == pytest.approx(12.5)
    serialized = config.to_dict()["dp"]
    assert serialized["noise_multiplier"] == pytest.approx(0.5)
    assert serialized["current_epsilon"] == pytest.approx(12.5)


def test_calibrated_privacy_is_written_to_processed_csv(dummy_loader, tmp_path):
    from fleet_dp.config import ExperimentConfig
    from fleet_dp.metrics import MetricsCollector
    from fleet_dp.server import (
        get_evaluate_fn,
        get_fit_metrics_aggregation_fn,
    )

    config = ExperimentConfig()
    config.dp.enabled = True
    config.dp.target_epsilon = 1000.0
    config.dp.noise_multiplier = 0.5
    config.dp.current_epsilon = 0.0
    collector = MetricsCollector(tmp_path)
    evaluate = get_evaluate_fn(
        model_name="mnist_cnn",
        test_loader=dummy_loader,
        device=torch.device("cpu"),
        config=config,
        collector=collector,
        experiment_name="dp_eps1000_seed42",
        seed=42,
    )
    model = create_model("mnist_cnn")
    parameters = [value.detach().numpy() for value in model.state_dict().values()]

    evaluate(0, parameters, {})
    get_fit_metrics_aggregation_fn(config)(
        [(64, {"noise_multiplier": 0.5, "epsilon_spent": 12.5})]
    )
    evaluate(1, parameters, {})
    path = collector.save_processed(collector.rounds)

    with path.open(newline="") as results_file:
        rows = list(csv.DictReader(results_file))

    assert rows[0]["noise_multiplier"] == "0.500000"
    assert rows[0]["current_epsilon"] == "0.0000"
    assert rows[1]["noise_multiplier"] == "0.500000"
    assert rows[1]["current_epsilon"] == "12.5000"
