"""Two-period OOS comparison report: 2024 vs 2025."""
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

PERIODS = {
    "2024 (OOS baseline)": "results/oos_2024_embeddings.csv",
    "2025 (NLP)": "results/oos_2025_embeddings.csv",
}

# Pre-computed B&H for 2024 from existing report
BH_2024 = {"sharpe": 0.7212, "total_return": 1.1854}


def _compute_bh(feature_path: str, test_start: str, test_end: str) -> dict:
    """Compute Buy & Hold metrics for a period using raw_close prices."""
    from src.eval.baselines import buy_and_hold
    df = pd.read_parquet(feature_path)
    start_ts = pd.Timestamp(test_start)
    end_ts = pd.Timestamp(test_end)
    if df.index.tz is not None:
        if start_ts.tzinfo is None:
            start_ts = start_ts.tz_localize("UTC")
        if end_ts.tzinfo is None:
            end_ts = end_ts.tz_localize("UTC")
    df = df.loc[start_ts:end_ts]
    prices = df["raw_close"].values if "raw_close" in df.columns else df["close"].values
    bh = buy_and_hold(prices)
    return {
        "sharpe": bh["metrics"]["sharpe_ratio"],
        "total_return": bh["metrics"]["total_return"],
    }


def generate_two_period_report(
    output_md: Path = Path("results/oos_two_period_report.md"),
    figures_dir: Path = Path("results/figures/two_period"),
) -> Path:
    output_md = Path(output_md)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    # Load both periods
    dfs = {}
    for period, csv_path in PERIODS.items():
        p = Path(csv_path)
        if p.exists():
            dfs[period] = pd.read_csv(p)
        else:
            logger.warning(f"Missing CSV: {csv_path}")

    if len(dfs) == 0:
        raise FileNotFoundError("No OOS CSV files found. Run backtests first.")

    # Compute B&H for 2025
    bh_stats = {
        "2024 (OOS baseline)": BH_2024,
        "2025 (NLP)": None,
    }
    fp_2025 = "data/processed/btc_4h_2025_embedding_features.parquet"
    if Path(fp_2025).exists():
        try:
            bh_stats["2025 (NLP)"] = _compute_bh(fp_2025, "2025-01-01", "2025-04-10")
        except Exception as e:
            logger.warning(f"Could not compute B&H for 2025: {e}")

    lines = ["# Embeddings 3-Agents — OOS 2024 vs 2025 Report\n"]
    lines.append("Periods: **2024** (in-distribution OOS) · **2025** (full NLP, unseen)\n")
    lines.append("Models: 15 (PPO×5, A2C×5, SAC×5), trained on BTC/USDT 4h 2020–2023\n")

    # Per-period summary tables
    for period, df in dfs.items():
        summary = df.groupby("algorithm").agg(
            sharpe_mean=("sharpe", "mean"),
            sharpe_std=("sharpe", "std"),
            total_return_mean=("total_return", "mean"),
            total_return_std=("total_return", "std"),
            max_dd_mean=("max_drawdown", "mean"),
            calmar_mean=("calmar", "mean"),
        ).round(4)
        bh = bh_stats.get(period)
        lines.append(f"\n## {period}\n")
        if bh:
            lines.append(
                f"**Buy & Hold**: Sharpe={bh['sharpe']:.4f}, "
                f"Total Return={bh['total_return']:.4f}\n"
            )
        lines.append(summary.to_markdown())
        lines.append("\n")

    # Sharpe bar chart: side-by-side for both periods
    algos = ["PPO", "A2C", "SAC"]
    colors = {"PPO": "#4C72B0", "A2C": "#55A868", "SAC": "#C44E52"}
    n_periods = len(dfs)
    fig, axes = plt.subplots(1, n_periods, figsize=(5 * n_periods, 4), sharey=True)
    if n_periods == 1:
        axes = [axes]
    for ax, (period, df) in zip(axes, dfs.items()):
        summary = df.groupby("algorithm")["sharpe"].agg(["mean", "std"])
        means = [summary.loc[a, "mean"] if a in summary.index else 0.0 for a in algos]
        stds = [summary.loc[a, "std"] if a in summary.index else 0.0 for a in algos]
        ax.bar(algos, means, yerr=stds, capsize=6, color=[colors[a] for a in algos])
        bh = bh_stats.get(period)
        if bh:
            ax.axhline(bh["sharpe"], color="gray", linestyle="--",
                       label=f"B&H {bh['sharpe']:.2f}")
            ax.legend(fontsize=8)
        ax.set_title(period, fontsize=10)
        ax.set_ylabel("Sharpe")
        ax.axhline(0, color="black", linewidth=0.5)
    fig.suptitle("OOS Sharpe by Period (mean ± std over 5 seeds)", fontsize=12)
    fig.tight_layout()
    chart_path = figures_dir / "sharpe_by_period.png"
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)

    lines.append("\n## Figures\n")
    lines.append(f"- `{chart_path}` — Sharpe bar chart per period\n")

    # Mann-Whitney U for SAC Sharpe across periods
    lines.append("\n## Mann-Whitney U (SAC Sharpe: 2024 vs 2025, two-sided)\n")
    sac_sharpes = {}
    for period, df in dfs.items():
        sac = df[df["algorithm"] == "SAC"]["sharpe"].values
        if len(sac) > 0:
            sac_sharpes[period] = sac
    mw_rows = []
    period_list = list(sac_sharpes.keys())
    for a, b in combinations(period_list, 2):
        if len(sac_sharpes[a]) >= 2 and len(sac_sharpes[b]) >= 2:
            stat, p = mannwhitneyu(sac_sharpes[a], sac_sharpes[b], alternative="two-sided")
            mw_rows.append({"pair": f"{a} vs {b}", "U": float(stat), "p_value": round(float(p), 5)})
    if mw_rows:
        lines.append(pd.DataFrame(mw_rows).to_markdown(index=False))
        lines.append("\n")
    else:
        lines.append("Not enough data for Mann-Whitney U test.\n")

    output_md.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Wrote two-period report to {output_md}")
    return output_md


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    generate_two_period_report()
