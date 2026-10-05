#!/usr/bin/env python3
"""Run a federated learning experiment with differential privacy.

Usage:
    python scripts/run_dp_experiment.py --epsilon 1
    python scripts/run_dp_experiment.py --epsilon 5
    python scripts/run_dp_experiment.py --epsilon 20
    python scripts/run_dp_experiment.py --epsilon 5 --clients 8 --rounds 20 --seed 42
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fleet_dp.config import ExperimentConfig
from fleet_dp.metrics import setup_logging
from fleet_dp.server import run_federated_experiment


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run federated learning with differential privacy"
    )
    parser.add_argument("--epsilon", type=float, required=True,
                        help="Target privacy budget epsilon")
    parser.add_argument("--delta", type=float, default=1e-5,
                        help="Privacy delta parameter")
    parser.add_argument("--max-grad-norm", type=float, default=1.0,
                        help="Maximum gradient norm for clipping")
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

    if args.epsilon <= 0:
        print(f"ERROR: epsilon must be positive, got {args.epsilon}")
        sys.exit(1)

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
    config.dp.enabled = True
    config.dp.target_epsilon = args.epsilon
    config.dp.target_delta = args.delta
    config.dp.max_grad_norm = args.max_grad_norm
    config.output_dir = args.output_dir

    print("\n" + "="*60)
    print(f"FEDERATED LEARNING WITH DP (ε={args.epsilon})")
    print("="*60)
    print(f"Clients: {config.dataset.num_clients}")
    print(f"Rounds: {config.federated.num_rounds}")
    print(f"Target epsilon: {config.dp.target_epsilon}")
    print(f"Delta: {config.dp.target_delta}")
    print(f"Max grad norm: {config.dp.max_grad_norm}")
    print(f"Local epochs: {config.federated.local_epochs}")
    print(f"Batch size: {config.federated.batch_size}")
    print(f"Learning rate: {config.federated.learning_rate}")
    print(f"Non-IID alpha: {config.dataset.dirichlet_alpha}")
    print(f"Seed: {config.seed}")
    print("="*60 + "\n")

    collector = run_federated_experiment(config, seed=config.seed)

    if collector.rounds:
        final_acc = collector.rounds[-1].test_accuracy
        print(f"\n{'='*60}")
        print(f"DP RESULT (ε={args.epsilon}): Final test accuracy = "
              f"{final_acc:.4f} ({final_acc*100:.2f}%)")
        print("="*60)


if __name__ == "__main__":
    main()
