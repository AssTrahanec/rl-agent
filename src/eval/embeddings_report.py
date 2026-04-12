"""Aggregate OOS embeddings results: mean±std table, Mann-Whitney U, figures."""
from __future__ import annotations

import logging
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

logger = logging.getLogger(__name__)


def generate_embeddings_report(
    oos_csv: Path,
    report_md: Path,
    figures_dir: Path,
    bh_sharpe: float = 0.0,
    bh_total_return: float = 0.0,
) -> Path:
    oos_csv = Path(oos_csv)
    report_md = Path(report_md)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(oos_csv)
    summary = df.groupby("algorithm").agg(
        sharpe_mean=("sharpe", "mean"),
        sharpe_std=("sharpe", "std"),
        total_return_mean=("total_return", "mean"),
        total_return_std=("total_return", "std"),
        max_dd_mean=("max_drawdown", "mean"),
        calmar_mean=("calmar", "mean"),
    ).round(4)

    pairs = list(combinations(sorted(df["algorithm"].unique()), 2))
    mw_rows = []
    for a, b in pairs:
        sa = df[df["algorithm"] == a]["sharpe"].values
        sb = df[df["algorithm"] == b]["sharpe"].values
        stat, p = mannwhitneyu(sa, sb, alternative="two-sided")
        mw_rows.append({"pair": f"{a} vs {b}", "U": float(stat), "p_value": float(p)})
    mw_df = pd.DataFrame(mw_rows)

    fig, ax = plt.subplots(figsize=(7, 4))
    algos = summary.index.tolist()
    means = summary["sharpe_mean"].values
    stds = summary["sharpe_std"].values
    ax.bar(algos, means, yerr=stds, capsize=6, color=["#4C72B0", "#55A868", "#C44E52"])
    ax.axhline(bh_sharpe, color="gray", linestyle="--", label=f"Buy&Hold ({bh_sharpe:.2f})")
    ax.set_ylabel("Sharpe (OOS 2024)")
    ax.set_title("Embeddings agents — OOS 2024 Sharpe (mean ± std over 5 seeds)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "sharpe_bar.png", dpi=150)
    plt.close(fig)

    lines = []
    lines.append("# Embeddings 3-Agents — OOS 2024 Report\n")
    lines.append(f"Source: `{oos_csv.name}`  (n={len(df)} runs)\n")
    lines.append("\n## Summary (mean ± std over seeds)\n")
    lines.append(summary.to_markdown())
    lines.append("\n\n## Buy & Hold baseline\n")
    lines.append(f"- Sharpe: {bh_sharpe:.4f}")
    lines.append(f"- Total Return: {bh_total_return:.4f}\n")
    lines.append("\n## Mann-Whitney U (Sharpe, two-sided)\n")
    lines.append(mw_df.to_markdown(index=False))
    lines.append("\n\n## Figures\n")
    lines.append("- `figures/sharpe_bar.png` — Sharpe bar chart with ±std error bars and B&H line\n")
    report_md.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Wrote report {report_md}")
    return report_md


if __name__ == "__main__":
    from src.eval.baselines import buy_and_hold
    from src.data.load_features import load_features_for_agent
    import numpy as np

    logging.basicConfig(level=logging.INFO)
    result = load_features_for_agent(
        agent_type="embeddings", asset="BTC/USDT",
        train_start="2024-01-01", train_end="2024-12-31",
        data_dir="data/processed", timeframe="4h",
        return_sentiment=False,
    )
    _, prices = (result[:2] if len(result) == 3 else result)
    bh = buy_and_hold(prices)
    generate_embeddings_report(
        oos_csv=Path("results/oos_2024_embeddings.csv"),
        report_md=Path("results/oos_2024_embeddings_report.md"),
        figures_dir=Path("results/figures/embeddings_v4"),
        bh_sharpe=bh["sharpe"],
        bh_total_return=bh["total_return"],
    )
