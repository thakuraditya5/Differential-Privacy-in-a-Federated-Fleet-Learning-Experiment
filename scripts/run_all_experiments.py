#!/usr/bin/env python3
"""Run all experiments: baseline + DP at multiple epsilon values.

Runs each experiment across multiple seeds for statistical robustness,
then generates the combined results CSV.

Usage:
    python scripts/run_all_experiments.py
    python scripts/run_all_experiments.py --seeds 42 43 44
    python scripts/run_all_experiments.py --rounds 10
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fleet_dp.config import ExperimentConfig
from fleet_dp.experiment import run_full_experiment
from fleet_dp.metrics import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all federated DP experiments"
    )
    parser.add_argument("--config", type=str, default=None,
                        help="Path to base YAML config file")
    parser.add_argument("--epsilons", type=float, nargs="+",
                        default=[1.0, 5.0, 20.0],
                        help="Epsilon values to test")
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[42, 43, 44],
                        help="Random seeds for multiple runs")
    parser.add_argument("--rounds", type=int, default=None,
                        help="Override number of federated rounds")
    parser.add_argument("--clients", type=int, default=None,
                        help="Override number of clients")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Output directory")
    parser.add_argument("--skip-baseline", action="store_true",
                        help="Skip the baseline experiment")
    args = parser.parse_args()

    setup_logging()

    # Build epsilon list (None = baseline)
    epsilons = []
    if not args.skip_baseline:
        epsilons.append(None)
    epsilons.extend(args.epsilons)

    print("\n" + "#"*60)
    print("# FULL EXPERIMENT PIPELINE")
    print("#"*60)
    print(f"# Experiments: baseline + ε={args.epsilons}")
    print(f"# Seeds: {args.seeds}")
    print(f"# Total runs: {len(epsilons) * len(args.seeds)}")
    print("#"*60 + "\n")

    # If we have a config file, load overrides
    config_path = args.config

    # For overrides, we'll modify after loading
    if config_path:
        config = ExperimentConfig.from_yaml(config_path)
    else:
        config = ExperimentConfig()

    if args.rounds:
        config.federated.num_rounds = args.rounds
    if args.clients:
        config.dataset.num_clients = args.clients

    # Save updated config temporarily
    import tempfile
    import yaml
    tmpdir = Path(args.output_dir) / "raw"
    tmpdir.mkdir(parents=True, exist_ok=True)
    tmp_config = tmpdir / "_run_config.yaml"
    with open(tmp_config, "w") as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False)

    df = run_full_experiment(
        config_path=str(tmp_config),
        epsilons=epsilons,
        seeds=args.seeds,
        output_dir=args.output_dir,
    )

    # Print summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)

    # Group by experiment condition across seeds
    df["condition"] = df.apply(
        lambda r: "No DP (Baseline)" if not r["dp_enabled"] or pd.isna(r["epsilon"]) or r["epsilon"] == "" else f"DP (ε = {float(r['epsilon']):g})",
        axis=1
    )

    for cond, group in df.groupby("condition"):
        final_rounds = group[group["round"] == group["round"].max()]
        mean_acc = final_rounds["test_accuracy"].mean()
        std_acc = final_rounds["test_accuracy"].std() if len(final_rounds) > 1 else 0.0
        print(f"  {cond:20s}: {mean_acc:.4f} ± {std_acc:.4f} ({mean_acc*100:.2f}% ± {std_acc*100:.2f}%) [n={len(final_rounds)}]")

    print("="*60)
    print(f"\nResults saved to {args.output_dir}/processed/experiment_results.csv")

    # Automatically generate plots and report
    from scripts.plot_results import main as plot_main
    from scripts.generate_report import generate_report

    print("\nGenerating publication plots...")
    plot_dir = Path(args.output_dir) / "plots"
    res_file = Path(args.output_dir) / "processed" / "experiment_results.csv"
    from scripts.plot_results import (
        plot_accuracy_vs_round,
        plot_final_accuracy_vs_epsilon,
        plot_accuracy_degradation,
        plot_client_data_distribution,
        set_plot_style,
    )
    set_plot_style()
    plot_client_data_distribution(num_clients=config.dataset.num_clients, alpha=config.dataset.dirichlet_alpha, seed=42, output_dir=plot_dir)
    plot_accuracy_vs_round(df, plot_dir)
    plot_final_accuracy_vs_epsilon(df, plot_dir)
    plot_accuracy_degradation(df, plot_dir)

    print("\nGenerating markdown experiment report...")
    report_file = Path("reports/experiment_report.md")
    generate_report(res_file, report_file)
    print(f"Done! Report saved to {report_file}")


if __name__ == "__main__":
    main()
