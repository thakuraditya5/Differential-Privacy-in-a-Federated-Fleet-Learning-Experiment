#!/usr/bin/env python3
"""Generate publication-quality plots from federated DP experiment results.

Generates:
1. Plot 1 — Accuracy vs Federated Round (convergence curves)
2. Plot 2 — Final Accuracy vs Epsilon (privacy-utility tradeoff)
3. Plot 3 — Accuracy Degradation (accuracy drop vs baseline)
4. Plot 4 — Client Data Distribution (Dirichlet non-IID heterogeneity heatmap)

Usage:
    python scripts/plot_results.py
    python scripts/plot_results.py --results-file results/processed/experiment_results.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fleet_dp.data import (
    get_client_label_distribution,
    load_dataset,
    partition_dirichlet,
)


def set_plot_style() -> None:
    """Set clean, professional plotting style."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def plot_accuracy_vs_round(df: pd.DataFrame, output_dir: Path) -> Path:
    """Plot 1: Global Test Accuracy vs Federated Round across DP settings."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # Group by (experiment_type, round) to aggregate across seeds
    # Determine experiment display label
    def get_exp_label(row):
        if row.get("clipping_only", False):
            return "Clipping only (no noise)"
        if not row["dp_enabled"] or pd.isna(row["epsilon"]) or row["epsilon"] == "":
            return "No DP (Baseline)"
        return f"DP (ε = {float(row['epsilon']):g})"

    df["display_label"] = df.apply(get_exp_label, axis=1)

    # Order of curves
    unique_labels = sorted(
        df["display_label"].unique(),
        key=lambda x: (
            0 if "No DP" in x else 1 if "Clipping only" in x else 2,
            float(x.split("=")[-1].replace(")", "").strip())
            if "ε" in x else 999,
        ),
    )

    colors = sns.color_palette("tab10", len(unique_labels))

    for idx, label in enumerate(unique_labels):
        sub_df = df[df["display_label"] == label]
        round_stats = sub_df.groupby("round")["test_accuracy"].agg(["mean", "std"]).reset_index()

        rounds = round_stats["round"]
        mean_acc = round_stats["mean"] * 100.0  # in %
        std_acc = round_stats["std"].fillna(0) * 100.0

        line, = ax.plot(rounds, mean_acc, label=label, color=colors[idx], marker="o", markersize=4, linewidth=1.8)
        ax.fill_between(rounds, mean_acc - std_acc, mean_acc + std_acc, color=colors[idx], alpha=0.18)

    ax.set_xlabel("Federated Round")
    ax.set_ylabel("Global Test Accuracy (%)")
    ax.set_title("Plot 1: Model Convergence across Privacy Budgets")
    ax.legend(title="Privacy Budget", loc="lower right", frameon=True)
    ax.set_ylim([0, 100])
    ax.set_xlim([1, df["round"].max()])
    ax.grid(True, linestyle="--", alpha=0.6)

    out_path = output_dir / "plot1_accuracy_vs_round.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Generated: {out_path}")
    return out_path


def plot_final_accuracy_vs_epsilon(df: pd.DataFrame, output_dir: Path) -> Path:
    """Plot 2: Final Test Accuracy vs Privacy Budget Epsilon."""
    fig, ax = plt.subplots(figsize=(7, 5))

    # Take the final round for each experiment and seed
    max_round = df["round"].max()
    final_df = df[df["round"] == max_round].copy()

    # Split into baseline vs DP
    clipping_only = final_df[final_df.get("clipping_only", False).astype(bool)] if "clipping_only" in final_df else final_df.iloc[0:0]
    baseline = final_df[
        ~final_df["dp_enabled"]
        & ~final_df.get("clipping_only", pd.Series(False, index=final_df.index)).astype(bool)
    ]
    dp_df = final_df[
        final_df["dp_enabled"]
        & final_df["epsilon"].notna()
        & (final_df["epsilon"] != "")
    ].copy()
    dp_df["epsilon"] = dp_df["epsilon"].astype(float)

    # Plot baseline as horizontal dashed reference band
    if not baseline.empty:
        base_mean = baseline["test_accuracy"].mean() * 100.0
        base_std = baseline["test_accuracy"].std() * 100.0 if len(baseline) > 1 else 0.0
        ax.axhline(base_mean, color="red", linestyle="--", linewidth=1.8, label=f"Baseline (No DP): {base_mean:.1f}%")
        ax.axhspan(base_mean - base_std, base_mean + base_std, color="red", alpha=0.12)

    if not clipping_only.empty:
        clip_mean = clipping_only["test_accuracy"].mean() * 100.0
        clip_std = clipping_only["test_accuracy"].std() * 100.0 if len(clipping_only) > 1 else 0.0
        ax.axhline(clip_mean, color="green", linestyle=":", linewidth=1.8, label=f"Clipping only: {clip_mean:.1f}%")
        ax.axhspan(clip_mean - clip_std, clip_mean + clip_std, color="green", alpha=0.10)

    if not dp_df.empty:
        # Group by epsilon
        stats = dp_df.groupby("epsilon")["test_accuracy"].agg(["mean", "std"]).reset_index()
        stats["mean"] *= 100.0
        stats["std"] = stats["std"].fillna(0) * 100.0
        stats = stats.sort_values("epsilon")

        ax.errorbar(
            stats["epsilon"],
            stats["mean"],
            yerr=stats["std"],
            fmt="o-",
            color="#1f77b4",
            linewidth=2,
            capsize=5,
            markersize=7,
            label="DP-SGD Model",
        )

        for _, row in stats.iterrows():
            ax.annotate(
                f"{row['mean']:.1f}%",
                (row["epsilon"], row["mean"]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontweight="bold",
                fontsize=9,
            )

    ax.set_xlabel("Privacy Budget (ε) [Log Scale]")
    ax.set_ylabel("Final Global Test Accuracy (%)")
    ax.set_title("Plot 2: Final Accuracy vs Privacy Budget (ε)")
    ax.set_xscale("log")
    ax.legend(loc="lower right", frameon=True)
    ax.grid(True, which="both", linestyle="--", alpha=0.6)

    out_path = output_dir / "plot2_accuracy_vs_epsilon.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Generated: {out_path}")
    return out_path


def plot_accuracy_degradation(df: pd.DataFrame, output_dir: Path) -> Path:
    """Plot 3: Accuracy Degradation (drop from baseline) vs Epsilon."""
    fig, ax = plt.subplots(figsize=(7, 5))

    max_round = df["round"].max()
    final_df = df[df["round"] == max_round].copy()

    baseline = final_df[
        ~final_df["dp_enabled"]
        & ~final_df.get("clipping_only", pd.Series(False, index=final_df.index)).astype(bool)
    ]
    if baseline.empty:
        print("Warning: No baseline found for accuracy degradation plot.")
        return output_dir / "plot3_accuracy_degradation.png"

    base_mean = baseline["test_accuracy"].mean() * 100.0

    dp_df = final_df[final_df["dp_enabled"] & final_df["epsilon"].notna() & (final_df["epsilon"] != "")].copy()
    dp_df["epsilon"] = dp_df["epsilon"].astype(float)

    if not dp_df.empty:
        stats = dp_df.groupby("epsilon")["test_accuracy"].agg(["mean", "std"]).reset_index()
        stats["mean"] *= 100.0
        stats["std"] = stats["std"].fillna(0) * 100.0
        stats["drop"] = base_mean - stats["mean"]
        stats = stats.sort_values("epsilon")

        ax.bar(
            [str(eps) for eps in stats["epsilon"]],
            stats["drop"],
            yerr=stats["std"],
            capsize=5,
            color="#d62728",
            alpha=0.8,
            edgecolor="black",
            width=0.45,
        )

        for idx, row in stats.iterrows():
            ax.annotate(
                f"-{row['drop']:.2f}%",
                (idx, row["drop"] / 2 if row["drop"] > 0 else 0.5),
                ha="center",
                va="center",
                color="white" if row["drop"] > 2 else "black",
                fontweight="bold",
            )

    ax.set_xlabel("Privacy Budget (ε)")
    ax.set_ylabel("Accuracy Drop vs Baseline (% points)")
    ax.set_title("Plot 3: Privacy Cost (Accuracy Degradation)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)

    out_path = output_dir / "plot3_accuracy_degradation.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Generated: {out_path}")
    return out_path


def plot_client_data_distribution(
    num_clients: int = 8,
    alpha: float = 0.3,
    seed: int = 42,
    output_dir: Path = Path("results/plots"),
) -> Path:
    """Plot 4: Non-IID Dirichlet distribution heatmap across fleet vehicles."""
    train_ds, _ = load_dataset("mnist")
    targets = np.array(train_ds.targets)

    client_indices = partition_dirichlet(targets, num_clients=num_clients, alpha=alpha, seed=seed)
    dist = get_client_label_distribution(train_ds, client_indices)

    fig, ax = plt.subplots(figsize=(9, 6))

    client_labels = [f"Vehicle {i+1}" for i in range(num_clients)]
    digit_labels = [str(d) for d in range(10)]

    sns.heatmap(
        dist,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=digit_labels,
        yticklabels=client_labels,
        ax=ax,
        cbar_kws={"label": "Sample Count"},
    )

    ax.set_xlabel("MNIST Digit Class")
    ax.set_ylabel("Simulated Fleet Vehicle")
    ax.set_title(f"Plot 4: Fleet Vehicle Data Heterogeneity (Dirichlet α = {alpha})")

    out_path = output_dir / "plot4_client_distribution.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Generated: {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate plots from experiment results")
    parser.add_argument(
        "--results-file",
        type=str,
        default="results/processed/experiment_results.csv",
        help="Path to processed experiment results CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/plots",
        help="Directory to save plots",
    )
    parser.add_argument(
        "--num-clients",
        type=int,
        default=8,
        help="Number of simulated fleet vehicles",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.3,
        help="Dirichlet concentration parameter",
    )
    args = parser.parse_args()

    set_plot_style()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Plot Client Data Distribution (can always be plotted)
    plot_client_data_distribution(
        num_clients=args.num_clients,
        alpha=args.alpha,
        seed=42,
        output_dir=output_dir,
    )

    # 2. Check if results CSV exists for results plots
    res_path = Path(args.results_file)
    if not res_path.exists():
        print(f"Results CSV not found at {res_path}. Run experiments first.")
        return

    df = pd.read_csv(res_path)
    if df.empty:
        print(f"Results CSV at {res_path} is empty.")
        return

    plot_accuracy_vs_round(df, output_dir)
    plot_final_accuracy_vs_epsilon(df, output_dir)
    plot_accuracy_degradation(df, output_dir)
    print("\nAll plots successfully generated in:", output_dir)


if __name__ == "__main__":
    main()
