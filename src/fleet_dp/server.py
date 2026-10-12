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
from fleet_dp.dp import calibrate_noise_multiplier
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
    round_fit_metrics: Optional[dict[int, dict[str, float]]] = None,
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
        fit_metrics = (round_fit_metrics or {}).get(server_round, {})
        metrics = RoundMetrics(
            experiment=experiment_name,
            epsilon=config.dp.target_epsilon if config.dp.enabled else None,
            delta=config.dp.target_delta if config.dp.enabled else None,
            seed=seed,
            round_num=server_round,
            train_loss=fit_metrics.get("train_loss", 0.0),
            train_accuracy=fit_metrics.get("train_accuracy", 0.0),
            test_loss=avg_loss,
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
            current_epsilon=config.dp.current_epsilon,
            clipped_grad_norm=fit_metrics.get("clipped_grad_norm"),
            noise_norm=fit_metrics.get("noise_norm"),
            preclip_sample_grad_norm=fit_metrics.get(
                "preclip_sample_grad_norm"
            ),
            clipped_sample_grad_norm=fit_metrics.get(
                "clipped_sample_grad_norm"
            ),
            elapsed_time=elapsed,
            dp_enabled=config.dp.enabled,
            clipping_only=config.dp.clipping_only,
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


def get_fit_metrics_aggregation_fn(
    config: Optional[ExperimentConfig] = None,
    round_fit_metrics: Optional[dict[int, dict[str, float]]] = None,
) -> Callable:
    """Create a function to aggregate training metrics from clients.

    When ``config`` is provided, calibrated DP values reported by clients
    are written back onto the experiment config so evaluation logging and
    ``experiment_results.csv`` record the actual noise multiplier and
    privacy spent (not the pre-calibration ``None`` defaults).
    """

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
        for name in (
            "clipped_grad_norm",
            "noise_norm",
            "preclip_sample_grad_norm",
            "clipped_sample_grad_norm",
        ):
            values = [
                (n, m[name]) for n, m in metrics if name in m
            ]
            if values:
                weight = sum(n for n, _ in values)
                result[name] = float(
                    sum(n * value for n, value in values) / weight
                )
        if epsilons:
            result["max_epsilon_spent"] = float(max(epsilons))
            result["avg_epsilon_spent"] = float(np.mean(epsilons))
        if noise_mults:
            result["noise_multiplier"] = float(np.mean(noise_mults))

        rounds = [m["server_round"] for _, m in metrics if "server_round" in m]
        if rounds and round_fit_metrics is not None:
            round_fit_metrics[int(rounds[0])] = {
                key: float(value)
                for key, value in result.items()
                if isinstance(value, (int, float))
            }

        if config is not None:
            config.record_calibrated_privacy(
                noise_multiplier=result.get("noise_multiplier"),
                current_epsilon=result.get("max_epsilon_spent"),
            )

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
    if config.dp.clipping_only:
        exp_name = f"clip_only_seed{seed}"
    elif config.dp.enabled and config.dp.target_epsilon is not None:
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

    # Calibrate noise on the server so CSV rows are not empty even if
    # client metrics are not echoed back through Flower/Ray.
    if (
        config.dp.enabled
        and config.dp.target_epsilon is not None
        and config.dp.noise_multiplier is None
    ):
        sizes = [
            len(loader.dataset)
            for loader in client_loaders
            if getattr(loader, "dataset", None) is not None
        ]
        min_n = float(min(sizes)) if sizes else 1.0
        sample_rate = min(1.0, config.federated.batch_size / max(min_n, 1.0))
        config.dp.noise_multiplier = calibrate_noise_multiplier(
            target_epsilon=config.dp.target_epsilon,
            target_delta=config.dp.target_delta,
            sample_rate=sample_rate,
            epochs=(
                config.federated.local_epochs
                * config.federated.num_rounds
            ),
            max_grad_norm=config.dp.max_grad_norm,
        )
        logger.info(
            f"Server-calibrated noise_multiplier="
            f"{config.dp.noise_multiplier:.4f} "
            f"(minimum client n={min_n:.0f}, sample_rate={sample_rate:.4f}, "
            f"total local epochs={config.federated.local_epochs * config.federated.num_rounds})"
        )

    if config.dp.enabled and config.dp.current_epsilon is None:
        config.dp.current_epsilon = 0.0

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
            clipping_only=config.dp.clipping_only,
        )
        return client.to_client()

    round_fit_metrics: dict[int, dict[str, float]] = {}

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
            round_fit_metrics=round_fit_metrics,
        ),
        initial_parameters=flwr.common.ndarrays_to_parameters(init_params),
        fit_metrics_aggregation_fn=get_fit_metrics_aggregation_fn(
            config, round_fit_metrics
        ),
        on_fit_config_fn=lambda server_round: {"server_round": server_round},
    )

    # Run simulation
    logger.info(
        f"Starting Flower simulation with {config.dataset.num_clients} "
        f"clients for {config.federated.num_rounds} rounds..."
    )

    history = start_simulation(
        client_fn=client_fn,
        num_clients=config.dataset.num_clients,
        config=ServerConfig(num_rounds=config.federated.num_rounds),
        strategy=strategy,
        ray_init_args={"num_cpus": 2, "include_dashboard": False},
    )

    nm_by_round: dict[int, float] = {}
    eps_by_round: dict[int, float] = {}
    if history is not None:
        nm_by_round = {
            int(rnd): float(val)
            for rnd, val in history.metrics_distributed_fit.get(
                "noise_multiplier", []
            )
        }
        eps_by_round = {
            int(rnd): float(val)
            for rnd, val in history.metrics_distributed_fit.get(
                "max_epsilon_spent", []
            )
        }
        if nm_by_round and config.dp.noise_multiplier is None:
            config.dp.noise_multiplier = next(iter(nm_by_round.values()))
        if eps_by_round:
            config.dp.current_epsilon = eps_by_round[max(eps_by_round)]

    # Backfill calibrated DP values if evaluate ran before fit metrics
    # were available (Flower evaluates round 0 before any client fit).
    if config.dp.enabled:
        for metrics in collector.rounds:
            if metrics.round_num in nm_by_round:
                metrics.noise_multiplier = nm_by_round[metrics.round_num]
            elif metrics.noise_multiplier is None:
                metrics.noise_multiplier = config.dp.noise_multiplier
            if metrics.round_num in eps_by_round:
                metrics.current_epsilon = eps_by_round[metrics.round_num]
            elif metrics.round_num > 0 and metrics.current_epsilon is None:
                metrics.current_epsilon = config.dp.current_epsilon
    elif config.dp.clipping_only:
        for metrics in collector.rounds:
            metrics.noise_multiplier = 0.0

    # Save results
    collector.save_raw(exp_name)
    collector.save_config(config.to_dict(), exp_name)

    logger.info(f"Experiment {exp_name} completed.")
    return collector
