"""Experiment orchestration for multi-seed, multi-epsilon runs.

Runs baseline and DP experiments across multiple seeds, collects
results, and produces aggregated CSVs for analysis.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import pandas as pd

from fleet_dp.config import ExperimentConfig
from fleet_dp.metrics import MetricsCollector, RoundMetrics, setup_logging
from fleet_dp.server import run_federated_experiment

logger = logging.getLogger(__name__)


def run_single_experiment(
    config: ExperimentConfig,
    seed: int,
) -> list[RoundMetrics]:
    """Run a single experiment with the given seed.

    Args:
        config: Experiment configuration.
        seed: Random seed for this run.

    Returns:
        List of RoundMetrics from this experiment.
    """
    cfg = config.copy()
    cfg.seed = seed
    collector = run_federated_experiment(cfg, seed=seed)
    return collector.rounds


def run_epsilon_sweep(
    base_config: ExperimentConfig,
    epsilons: list[Optional[float]],
    seeds: list[int],
) -> list[RoundMetrics]:
    """Run experiments across multiple epsilon values and seeds.

    Args:
        base_config: Base experiment configuration.
        epsilons: List of epsilon values (None = baseline/no DP).
        seeds: List of random seeds.

    Returns:
        Aggregated list of all RoundMetrics.
    """
    all_rounds: list[RoundMetrics] = []

    for eps in epsilons:
        cfg = base_config.copy()
        if eps is None:
            cfg.dp.enabled = False
            cfg.dp.target_epsilon = None
            label = "baseline"
        else:
            cfg.dp.enabled = True
            cfg.dp.target_epsilon = eps
            label = f"dp_eps{eps}"

        logger.info(f"\n{'#'*60}")
        logger.info(f"# Experiment: {label}")
        logger.info(f"{'#'*60}")

        for seed in seeds:
            logger.info(f"\n--- Seed {seed} ---")
            try:
                rounds = run_single_experiment(cfg, seed)
                all_rounds.extend(rounds)
            except Exception as e:
                logger.error(f"Experiment {label} seed={seed} failed: {e}")
                raise

    return all_rounds


def aggregate_results(
    all_rounds: list[RoundMetrics],
    output_dir: str = "results",
) -> pd.DataFrame:
    """Aggregate results and save to CSV.

    Args:
        all_rounds: All round metrics from experiments.
        output_dir: Output directory.

    Returns:
        DataFrame with aggregated results.
    """
    collector = MetricsCollector(output_dir)
    collector.save_processed(all_rounds)

    # Build DataFrame for analysis
    records = []
    for r in all_rounds:
        records.append({
            "experiment": r.experiment,
            "epsilon": r.epsilon,
            "delta": r.delta,
            "seed": r.seed,
            "round": r.round_num,
            "train_loss": r.train_loss,
            "train_accuracy": r.train_accuracy,
            "test_accuracy": r.test_accuracy,
            "num_clients": r.num_clients,
            "local_epochs": r.local_epochs,
            "batch_size": r.batch_size,
            "learning_rate": r.learning_rate,
            "max_grad_norm": r.max_grad_norm,
            "noise_multiplier": r.noise_multiplier,
            "num_samples": r.num_samples,
            "current_epsilon": r.current_epsilon,
            "elapsed_time": r.elapsed_time,
            "dp_enabled": r.dp_enabled,
        })

    df = pd.DataFrame(records)
    return df


def run_full_experiment(
    config_path: Optional[str] = None,
    epsilons: Optional[list[Optional[float]]] = None,
    seeds: Optional[list[int]] = None,
    output_dir: str = "results",
) -> pd.DataFrame:
    """Run the full experiment pipeline.

    Args:
        config_path: Path to base configuration YAML.
        epsilons: Epsilon values to test (None = baseline).
        seeds: Random seeds.
        output_dir: Output directory.

    Returns:
        DataFrame with all results.
    """
    setup_logging()

    if config_path:
        config = ExperimentConfig.from_yaml(config_path)
    else:
        config = ExperimentConfig()

    config.output_dir = output_dir

    if epsilons is None:
        epsilons = [None, 1.0, 5.0, 20.0]

    if seeds is None:
        seeds = config.seeds

    start_time = time.time()
    logger.info("Starting full experiment pipeline")
    logger.info(f"Epsilons: {epsilons}")
    logger.info(f"Seeds: {seeds}")

    all_rounds = run_epsilon_sweep(config, epsilons, seeds)
    df = aggregate_results(all_rounds, output_dir)

    elapsed = time.time() - start_time
    logger.info(f"\nFull experiment completed in {elapsed:.1f}s")

    return df
