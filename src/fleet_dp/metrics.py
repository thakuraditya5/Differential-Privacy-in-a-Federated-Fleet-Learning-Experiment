"""Metrics collection, logging, and CSV export.

Records per-round, per-client, and experiment-level metrics for
analysis and reproducibility.
"""

from __future__ import annotations

import csv
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class RoundMetrics:
    """Metrics for a single federated round."""
    experiment: str = ""
    epsilon: Optional[float] = None
    delta: Optional[float] = None
    seed: int = 42
    round_num: int = 0
    train_loss: float = 0.0
    train_accuracy: float = 0.0
    test_loss: float = 0.0
    test_accuracy: float = 0.0
    num_clients: int = 0
    local_epochs: int = 0
    batch_size: int = 0
    learning_rate: float = 0.0
    max_grad_norm: Optional[float] = None
    noise_multiplier: Optional[float] = None
    num_samples: int = 0
    current_epsilon: Optional[float] = None
    clipped_grad_norm: Optional[float] = None
    noise_norm: Optional[float] = None
    preclip_sample_grad_norm: Optional[float] = None
    clipped_sample_grad_norm: Optional[float] = None
    elapsed_time: float = 0.0
    dp_enabled: bool = False
    clipping_only: bool = False


class MetricsCollector:
    """Collects and exports experiment metrics."""

    CSV_COLUMNS = [
        "experiment", "epsilon", "delta", "seed", "round",
        "train_loss", "train_accuracy", "test_loss", "test_accuracy",
        "num_clients", "local_epochs", "batch_size", "learning_rate",
        "max_grad_norm", "noise_multiplier", "num_samples",
        "current_epsilon", "clipped_grad_norm", "noise_norm",
        "preclip_sample_grad_norm", "clipped_sample_grad_norm",
        "elapsed_time", "dp_enabled", "clipping_only",
    ]

    def __init__(self, output_dir: str | Path = "results"):
        self.output_dir = Path(output_dir)
        self.raw_dir = self.output_dir / "raw"
        self.processed_dir = self.output_dir / "processed"
        self.plots_dir = self.output_dir / "plots"

        for d in [self.raw_dir, self.processed_dir, self.plots_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self.rounds: list[RoundMetrics] = []
        self._start_time = time.time()

    def log_round(self, metrics: RoundMetrics) -> None:
        """Record metrics for one federated round."""
        self.rounds.append(metrics)
        logger.info(
            f"Round {metrics.round_num}: "
            f"loss={metrics.train_loss:.4f}, "
            f"test_acc={metrics.test_accuracy:.4f}"
            + (f", ε_spent={metrics.current_epsilon:.2f}" if metrics.current_epsilon else "")
        )

    def save_raw(self, experiment_name: str) -> Path:
        """Save raw metrics to CSV.

        Args:
            experiment_name: Name identifier for this experiment run.

        Returns:
            Path to the saved CSV file.
        """
        path = self.raw_dir / f"{experiment_name}.csv"
        self._write_csv(path, self.rounds)
        logger.info(f"Raw metrics saved to {path}")
        return path

    def save_processed(self, all_rounds: list[RoundMetrics]) -> Path:
        """Save processed/aggregated metrics to the main results CSV.

        Args:
            all_rounds: All round metrics across all experiments.

        Returns:
            Path to the saved CSV file.
        """
        path = self.processed_dir / "experiment_results.csv"
        self._write_csv(path, all_rounds)
        logger.info(f"Processed results saved to {path}")
        return path

    def _write_csv(self, path: Path, rounds: list[RoundMetrics]) -> None:
        """Write metrics to CSV file."""
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.CSV_COLUMNS)
            writer.writeheader()
            for r in rounds:
                row = {
                    "experiment": r.experiment,
                    "epsilon": r.epsilon if r.epsilon is not None else "",
                    "delta": r.delta if r.delta is not None else "",
                    "seed": r.seed,
                    "round": r.round_num,
                    "train_loss": f"{r.train_loss:.6f}",
                    "train_accuracy": f"{r.train_accuracy:.6f}",
                    "test_loss": f"{r.test_loss:.6f}",
                    "test_accuracy": f"{r.test_accuracy:.6f}",
                    "num_clients": r.num_clients,
                    "local_epochs": r.local_epochs,
                    "batch_size": r.batch_size,
                    "learning_rate": r.learning_rate,
                    "max_grad_norm": (
                        r.max_grad_norm if r.max_grad_norm is not None else ""
                    ),
                    "noise_multiplier": (
                        f"{r.noise_multiplier:.6f}"
                        if r.noise_multiplier is not None
                        else ""
                    ),
                    "num_samples": r.num_samples,
                    "current_epsilon": (
                        f"{r.current_epsilon:.4f}"
                        if r.current_epsilon is not None
                        else ""
                    ),
                    "clipped_grad_norm": (
                        f"{r.clipped_grad_norm:.6f}"
                        if r.clipped_grad_norm is not None else ""
                    ),
                    "noise_norm": (
                        f"{r.noise_norm:.6f}"
                        if r.noise_norm is not None else ""
                    ),
                    "preclip_sample_grad_norm": (
                        f"{r.preclip_sample_grad_norm:.6f}"
                        if r.preclip_sample_grad_norm is not None else ""
                    ),
                    "clipped_sample_grad_norm": (
                        f"{r.clipped_sample_grad_norm:.6f}"
                        if r.clipped_sample_grad_norm is not None else ""
                    ),
                    "elapsed_time": f"{r.elapsed_time:.2f}",
                    "dp_enabled": r.dp_enabled,
                    "clipping_only": r.clipping_only,
                }
                writer.writerow(row)

    def save_config(self, config_dict: dict, experiment_name: str) -> Path:
        """Save experiment configuration alongside results.

        Args:
            config_dict: Configuration dictionary.
            experiment_name: Experiment identifier.

        Returns:
            Path to saved config JSON.
        """
        path = self.raw_dir / f"{experiment_name}_config.json"
        with open(path, "w") as f:
            json.dump(config_dict, f, indent=2, default=str)
        return path

    def elapsed(self) -> float:
        """Get elapsed time since collector creation."""
        return time.time() - self._start_time


def setup_logging(level: int = logging.INFO) -> None:
    """Configure structured logging for the experiment."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
