"""Compare old vs new experiment results with statistical tests.

Usage:
    python experiments/compare_results.py --old results/metrics_v1.csv --new results/metrics.csv
"""
import argparse
import logging

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def load_metrics(path: str) -> pd.DataFrame:
    """Load metrics CSV."""
    return pd.read_csv(path)


def compute_deltas(
    old: pd.DataFrame,
    new: pd.DataFrame,
    agent_type: str = "embeddings",
) -> pd.DataFrame:
    """Compute per-seed deltas between old and new metrics."""
    old_agent = old[old["agent_type"] == agent_type].sort_values("seed").reset_index(drop=True)
    new_agent = new[new["agent_type"] == agent_type].sort_values("seed").reset_index(drop=True)

    merged = old_agent.merge(new_agent, on="seed", suffixes=("_old", "_new"))
    merged["sharpe_delta"] = merged["sharpe_ratio_new"] - merged["sharpe_ratio_old"]
    merged["return_delta"] = merged["total_return_new"] - merged["total_return_old"]
    return merged


def run_ttest(
    old_values: np.ndarray,
    new_values: np.ndarray,
) -> tuple[float, float]:
    """Run paired t-test (if same size) or independent t-test."""
    if len(old_values) == len(new_values):
        t_stat, p_value = stats.ttest_rel(new_values, old_values)
    else:
        t_stat, p_value = stats.ttest_ind(new_values, old_values)
    return float(t_stat), float(p_value)


def plot_comparison(old: pd.DataFrame, new: pd.DataFrame, save_path: str):
    """Bar plot: old vs new Sharpe ratios side by side."""
    fig, ax = plt.subplots(figsize=(10, 6))

    agent_types = sorted(set(old["agent_type"]) | set(new["agent_type"]))
    x = np.arange(len(agent_types))
    width = 0.35

    old_means = []
    old_stds = []
    new_means = []
    new_stds = []

    for at in agent_types:
        old_s = old[old["agent_type"] == at]["sharpe_ratio"]
        new_s = new[new["agent_type"] == at]["sharpe_ratio"]
        old_means.append(old_s.mean() if len(old_s) > 0 else 0)
        old_stds.append(old_s.std() if len(old_s) > 1 else 0)
        new_means.append(new_s.mean() if len(new_s) > 0 else 0)
        new_stds.append(new_s.std() if len(new_s) > 1 else 0)

    ax.bar(x - width / 2, old_means, width, yerr=old_stds, label="Old", alpha=0.8)
    ax.bar(x + width / 2, new_means, width, yerr=new_stds, label="New", alpha=0.8)

    ax.set_ylabel("Sharpe Ratio")
    ax.set_title("Old vs New: Sharpe Ratio by Agent Type")
    ax.set_xticks(x)
    ax.set_xticklabels(agent_types)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved comparison plot to {save_path}")


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Compare old vs new results")
    parser.add_argument("--old", required=True, help="Path to old metrics CSV")
    parser.add_argument("--new", required=True, help="Path to new metrics CSV")
    parser.add_argument("--plot", default="results/figures/comparison.png")
    args = parser.parse_args()

    old = load_metrics(args.old)
    new = load_metrics(args.new)

    print("\n" + "=" * 70)
    print("COMPARISON: Old vs New Results")
    print("=" * 70)

    for agent_type in sorted(set(old["agent_type"]) | set(new["agent_type"])):
        old_s = old[old["agent_type"] == agent_type]["sharpe_ratio"].values
        new_s = new[new["agent_type"] == agent_type]["sharpe_ratio"].values

        print(f"\n--- {agent_type} ---")
        if len(old_s) > 0:
            print(f"  Old Sharpe: {old_s.mean():.3f} +/- {old_s.std():.3f} (n={len(old_s)})")
        if len(new_s) > 0:
            print(f"  New Sharpe: {new_s.mean():.3f} +/- {new_s.std():.3f} (n={len(new_s)})")

        if len(old_s) >= 2 and len(new_s) >= 2:
            t_stat, p_val = run_ttest(old_s, new_s)
            sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else ""
            print(f"  t-test: t={t_stat:.3f}, p={p_val:.4f} {sig}")

    # Embeddings vs baseline (new)
    emb_new = new[new["agent_type"] == "embeddings"]["sharpe_ratio"].values
    base_new = new[new["agent_type"] == "baseline"]["sharpe_ratio"].values
    if len(emb_new) >= 2 and len(base_new) >= 2:
        t_stat, p_val = run_ttest(base_new, emb_new)
        print(f"\n--- embeddings vs baseline (new) ---")
        print(f"  t={t_stat:.3f}, p={p_val:.4f}")

    # Deltas
    deltas = compute_deltas(old, new, "embeddings")
    if len(deltas) > 0:
        print(f"\n--- Per-seed deltas (embeddings) ---")
        for _, row in deltas.iterrows():
            print(f"  seed={int(row['seed'])}: Sharpe {row.get('sharpe_ratio_old', 0):.3f} -> {row.get('sharpe_ratio_new', 0):.3f} (delta={row['sharpe_delta']:+.3f})")

    plot_comparison(old, new, args.plot)
    print(f"\nPlot saved to {args.plot}")


if __name__ == "__main__":
    main()
