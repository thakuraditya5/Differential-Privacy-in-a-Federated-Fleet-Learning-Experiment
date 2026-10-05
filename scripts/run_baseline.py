#!/usr/bin/env python3
"""Run the federated learning baseline (no differential privacy).

Usage:
    python scripts/run_baseline.py
    python scripts/run_baseline.py --rounds 10 --clients 8 --seed 42
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fleet_dp.config import ExperimentConfig
from fleet_dp.metrics import setup_logging
from fleet_dp.server import run_federated_experiment


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run federated learning baseline (no DP)"
    )
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file")
    parser.add_argument("--rounds", type=int, default=20,
                        help="Number of federated rounds")
    parser.add_argument("--clients", type=int, default=8,
                        help="Number of clients/vehicles")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Local training epochs per round")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Training batch size")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Learning rate")
    parser.add_argument("--alpha", type=float, default=0.3,
                        help="Dirichlet alpha for non-IID partitioning")
    parser.add_argument("--iid", action="store_true",
                        help="Use IID data partitioning")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Output directory for results")
    args = parser.parse_args()

    setup_logging()

    if args.config:
        config = ExperimentConfig.from_yaml(args.config)
    else:
        config = ExperimentConfig()

    # Override with CLI args
    config.federated.num_rounds = args.rounds
    config.dataset.num_clients = args.clients
    config.seed = args.seed
    config.federated.local_epochs = args.epochs
    config.federated.batch_size = args.batch_size
    config.federated.learning_rate = args.lr
    config.dataset.dirichlet_alpha = args.alpha
    config.dataset.iid = args.iid
    config.dp.enabled = False
    config.output_dir = args.output_dir

    print("\n" + "="*60)
    print("FEDERATED LEARNING BASELINE (No DP)")
    print("="*60)
    print(f"Clients: {config.dataset.num_clients}")
    print(f"Rounds: {config.federated.num_rounds}")
    print(f"Local epochs: {config.federated.local_epochs}")
    print(f"Batch size: {config.federated.batch_size}")
    print(f"Learning rate: {config.federated.learning_rate}")
    print(f"Non-IID alpha: {config.dataset.dirichlet_alpha}")
    print(f"Seed: {config.seed}")
    print("="*60 + "\n")

    collector = run_federated_experiment(config, seed=config.seed)

    # Sanity check
    if collector.rounds:
        final_acc = collector.rounds[-1].test_accuracy
        print(f"\n{'='*60}")
        print(f"BASELINE RESULT: Final test accuracy = {final_acc:.4f} "
              f"({final_acc*100:.2f}%)")
        if final_acc < 0.5:
            print("WARNING: Baseline accuracy is below 50%. "
                  "The model may not be learning properly.")
        elif final_acc < 0.8:
            print("NOTE: Baseline accuracy is moderate. Consider more "
                  "rounds or tuning hyperparameters.")
        else:
            print("OK: Baseline accuracy looks reasonable.")
        print("="*60)


if __name__ == "__main__":
    main()
