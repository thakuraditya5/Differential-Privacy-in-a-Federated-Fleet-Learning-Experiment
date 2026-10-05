"""Federated learning server and simulation runner.

Implements FedAvg-based federated learning using Flower's simulation
API. Handles global model evaluation, metric aggregation, and
round-level logging.
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from typing import Any, Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from torch.utils.data import DataLoader

import flwr
from flwr.client import Client, NumPyClient
from flwr.server import ServerConfig
from flwr.server.strategy import FedAvg
from flwr.simulation import start_simulation

from fleet_dp.client import FleetClient
from fleet_dp.config import ExperimentConfig
from fleet_dp.data import (
    get_client_loaders,
    get_test_loader,
    load_dataset,
)
from fleet_dp.metrics import MetricsCollector, RoundMetrics
from fleet_dp.model import create_model
from fleet_dp.utils import get_device, set_seed

logger = logging.getLogger(__name__)


def get_evaluate_fn(
    model_name: str,
    test_loader: DataLoader,
    device: torch.device,
    config: ExperimentConfig,
    collector: MetricsCollector,
    experiment_name: str,
    seed: int,
) -> Callable:
    """Create a server-side evaluation function.

    This function evaluates the global model on the full test set
    after each federated round.

    Args:
        model_name: Name of the model architecture.
        test_loader: Test set DataLoader.
        device: Torch device.
        config: Experiment configuration.
        collector: Metrics collector for recording results.
        experiment_name: Name of this experiment run.
        seed: Random seed used.

    Returns:
        Evaluation function for the Flower strategy.
    """
    start_time = time.time()

    def evaluate(
        server_round: int,
        parameters: list[NDArray],
        eval_config: dict[str, Any],
    ) -> Optional[tuple[float, dict[str, Any]]]:
        """Evaluate global model on test set."""
        model = create_model(model_name).to(device)
        keys = list(model.state_dict().keys())
        state_dict = OrderedDict(
            {k: torch.tensor(v) for k, v in zip(keys, parameters)}
        )
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        loss_fn = nn.CrossEntropyLoss()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                total_loss += loss_fn(output, target).item() * data.size(0)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += data.size(0)

        avg_loss = total_loss / total if total > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0
        elapsed = time.time() - start_time

        # Record metrics
        metrics = RoundMetrics(
            experiment=experiment_name,
            epsilon=config.dp.target_epsilon if config.dp.enabled else None,
            delta=config.dp.target_delta if config.dp.enabled else None,
            seed=seed,
            round_num=server_round,
            train_loss=avg_loss,
            test_accuracy=accuracy,
            num_clients=config.dataset.num_clients,
            local_epochs=config.federated.local_epochs,
            batch_size=config.federated.batch_size,
            learning_rate=config.federated.learning_rate,
            max_grad_norm=(
                config.dp.max_grad_norm if config.dp.enabled else None
            ),
            noise_multiplier=config.dp.noise_multiplier,
            num_samples=total,
            elapsed_time=elapsed,
            dp_enabled=config.dp.enabled,
        )
        collector.log_round(metrics)

        print(
            f"\n{'='*50}\n"
            f"Round {server_round}/{config.federated.num_rounds}\n"
            f"{'='*50}\n"
            f"DP: {'enabled' if config.dp.enabled else 'disabled'}"
            + (f" | Target ε={config.dp.target_epsilon}" if config.dp.enabled else "")
            + f"\nGlobal test accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)"
            f"\nGlobal test loss: {avg_loss:.4f}"
            f"\nElapsed time: {elapsed:.1f}s\n"
        )

        return avg_loss, {"accuracy": accuracy}

    return evaluate


def get_fit_metrics_aggregation_fn() -> Callable:
    """Create a function to aggregate training metrics from clients."""

    def aggregate_fit_metrics(
        metrics: list[tuple[int, dict[str, Any]]]
    ) -> dict[str, Any]:
        """Aggregate fit metrics from all clients."""
        total_samples = sum(n for n, _ in metrics)
        if total_samples == 0:
            return {}

        avg_loss = sum(
            n * m.get("train_loss", 0.0) for n, m in metrics
        ) / total_samples
        avg_acc = sum(
            n * m.get("train_accuracy", 0.0) for n, m in metrics
        ) / total_samples

        # Collect epsilon from DP clients
        epsilons = [
            m["epsilon_spent"]
            for _, m in metrics
            if "epsilon_spent" in m
        ]
        noise_mults = [
            m["noise_multiplier"]
            for _, m in metrics
            if "noise_multiplier" in m
        ]

        result: dict[str, Any] = {
            "train_loss": float(avg_loss),
            "train_accuracy": float(avg_acc),
        }
        if epsilons:
            result["max_epsilon_spent"] = float(max(epsilons))
            result["avg_epsilon_spent"] = float(np.mean(epsilons))
        if noise_mults:
            result["noise_multiplier"] = float(noise_mults[0])

        return result

    return aggregate_fit_metrics


def run_federated_experiment(
    config: ExperimentConfig,
    seed: Optional[int] = None,
    output_dir: Optional[str] = None,
) -> MetricsCollector:
    """Run a complete federated learning experiment.

    Args:
        config: Experiment configuration.
        seed: Override seed (uses config.seed if None).
        output_dir: Override output directory.

    Returns:
        MetricsCollector with recorded experiment metrics.
    """
    seed = seed if seed is not None else config.seed
    out_dir = output_dir or config.output_dir
    set_seed(seed)

    device = get_device(config.device)
    logger.info(f"Using device: {device}")

    # Determine experiment name
    if config.dp.enabled and config.dp.target_epsilon is not None:
        exp_name = f"dp_eps{config.dp.target_epsilon}_seed{seed}"
    else:
        exp_name = f"baseline_seed{seed}"

    logger.info(f"Starting experiment: {exp_name}")
    collector = MetricsCollector(out_dir)

    # Load data
    logger.info(f"Loading {config.dataset.name} dataset...")
    train_ds, test_ds = load_dataset(config.dataset.name)
    test_loader = get_test_loader(test_ds)

    # Create client data loaders
    logger.info(
        f"Partitioning data for {config.dataset.num_clients} clients "
        f"(IID={config.dataset.iid}, alpha={config.dataset.dirichlet_alpha})"
    )
    client_loaders = get_client_loaders(
        train_ds,
        num_clients=config.dataset.num_clients,
        batch_size=config.federated.batch_size,
        iid=config.dataset.iid,
        dirichlet_alpha=config.dataset.dirichlet_alpha,
        seed=seed,
    )

    # Create initial model and get parameters
    init_model = create_model(config.model.name).to(device)
    init_params = [
        val.cpu().numpy() for _, val in init_model.state_dict().items()
    ]

    # Client factory function for Flower simulation
    def client_fn(context) -> Client:
        """Create a FleetClient for a given context."""
        # Extract partition ID from context
        partition_id = int(context.node_config.get("partition-id", 0))
        client = FleetClient(
            client_id=partition_id,
            train_loader=client_loaders[partition_id],
            model_name=config.model.name,
            device=str(device),
            local_epochs=config.federated.local_epochs,
            learning_rate=config.federated.learning_rate,
            dp_enabled=config.dp.enabled,
            target_epsilon=config.dp.target_epsilon,
            target_delta=config.dp.target_delta,
            max_grad_norm=config.dp.max_grad_norm,
            noise_multiplier=config.dp.noise_multiplier,
        )
        return client.to_client()

    # Strategy: FedAvg
    strategy = FedAvg(
        fraction_fit=config.federated.fraction_fit,
        fraction_evaluate=0.0,  # Don't evaluate on clients
        min_fit_clients=config.dataset.num_clients,
        min_evaluate_clients=0,
        min_available_clients=config.dataset.num_clients,
        evaluate_fn=get_evaluate_fn(
            model_name=config.model.name,
            test_loader=test_loader,
            device=device,
            config=config,
            collector=collector,
            experiment_name=exp_name,
            seed=seed,
        ),
        initial_parameters=flwr.common.ndarrays_to_parameters(init_params),
        fit_metrics_aggregation_fn=get_fit_metrics_aggregation_fn(),
    )

    # Run simulation
    logger.info(
        f"Starting Flower simulation with {config.dataset.num_clients} "
        f"clients for {config.federated.num_rounds} rounds..."
    )

    start_simulation(
        client_fn=client_fn,
        num_clients=config.dataset.num_clients,
        config=ServerConfig(num_rounds=config.federated.num_rounds),
        strategy=strategy,
        ray_init_args={"num_cpus": 2, "include_dashboard": False},
    )

    # Save results
    collector.save_raw(exp_name)
    collector.save_config(config.to_dict(), exp_name)

    logger.info(f"Experiment {exp_name} completed.")
    return collector
