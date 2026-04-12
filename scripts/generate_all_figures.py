"""Generate all experiment figures:
1. Sharpe bar chart per period (3 panels)
2. Equity curves per period (3 panels, all 15 models)
3. Sharpe heatmap (algo x seed x period)
4. Scatter Sharpe vs Max Drawdown
5. Total Return comparison bar
"""
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PERIODS = {
    "2024 OOS": {
        "csv": "results/oos_2024_embeddings.csv",
        "equity_json": "results/oos_2024_embeddings_equity.json",
        "feature_path": "data/processed/btc_4h_embedding_features.parquet",
        "start": "2024-01-01", "end": "2024-12-31",
    },
    "2025 OOS (NLP)": {
        "csv": "results/oos_2025_embeddings.csv",
        "equity_json": "results/oos_2025_embeddings_equity.json",
        "feature_path": "data/processed/btc_4h_2025_embedding_features.parquet",
        "start": "2025-01-01", "end": "2025-04-10",
    },
    "2026 YTD (no NLP)": {
        "csv": "results/oos_2026_embeddings.csv",
        "equity_json": "results/oos_2026_embeddings_equity.json",
        "feature_path": "data/processed/btc_4h_2026_embedding_features.parquet",
        "start": "2026-01-01", "end": "2026-04-12",
    },
}

ALGO_COLORS = {"PPO": "#4C72B0", "A2C": "#55A868", "SAC": "#C44E52"}
FIGURES_DIR = Path("results/figures/all_periods")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_bh(feature_path, start, end):
    from src.eval.baselines import buy_and_hold
    df = pd.read_parquet(feature_path)
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    if df.index.tz is not None:
        s = s.tz_localize("UTC") if s.tzinfo is None else s
        e = e.tz_localize("UTC") if e.tzinfo is None else e
    df = df.loc[s:e]
    prices = df["raw_close"].values if "raw_close" in df.columns else df["close"].values
    bh = buy_and_hold(prices)
    return bh["metrics"]


def fig_sharpe_bar(dfs, bh_stats):
    periods = [p for p in PERIODS if p in dfs]
    fig, axes = plt.subplots(1, len(periods), figsize=(5 * len(periods), 4), sharey=False)
    if len(periods) == 1:
        axes = [axes]
    algos = ["PPO", "A2C", "SAC"]
    for ax, period in zip(axes, periods):
        df = dfs[period]
        summary = df.groupby("algorithm")["sharpe"].agg(["mean", "std"])
        means = [summary.loc[a, "mean"] if a in summary.index else 0 for a in algos]
        stds = [summary.loc[a, "std"] if a in summary.index else 0 for a in algos]
        ax.bar(algos, means, yerr=stds, capsize=6, color=[ALGO_COLORS[a] for a in algos], alpha=0.85)
        bh = bh_stats.get(period)
        if bh:
            ax.axhline(bh["sharpe_ratio"], color="gray", linestyle="--", linewidth=1.5,
                       label=f"B&H {bh['sharpe_ratio']:.2f}")
            ax.legend(fontsize=8)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_title(period, fontsize=10)
        ax.set_ylabel("Sharpe Ratio")
    fig.suptitle("OOS Sharpe (mean +/- std, 5 seeds)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = FIGURES_DIR / "sharpe_by_period.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved {path}")


def fig_equity_curves():
    available = {p: v for p, v in PERIODS.items() if Path(v["equity_json"]).exists()}
    if not available:
        logger.warning("No equity JSON files found, skipping equity curves")
        return
    fig, axes = plt.subplots(1, len(available), figsize=(6 * len(available), 4))
    if len(available) == 1:
        axes = [axes]
    for ax, (period, info) in zip(axes, available.items()):
        with open(info["equity_json"]) as f:
            eq_data = json.load(f)
        for key, curve in eq_data.items():
            algo = key.split("_")[0]
            color = ALGO_COLORS.get(algo, "gray")
            ax.plot(curve, color=color, alpha=0.4, linewidth=0.8)
        for algo in ["PPO", "A2C", "SAC"]:
            curves = [v for k, v in eq_data.items() if k.startswith(algo)]
            if curves:
                min_len = min(len(c) for c in curves)
                mean_curve = np.mean([c[:min_len] for c in curves], axis=0)
                ax.plot(mean_curve, color=ALGO_COLORS[algo], linewidth=2, label=algo)
        ax.axhline(1.0, color="black", linewidth=0.5, linestyle="--")
        ax.set_title(period, fontsize=10)
        ax.set_ylabel("Portfolio Value")
        ax.set_xlabel("Steps")
        ax.legend(fontsize=8)
    fig.suptitle("Equity Curves (thin=individual seeds, thick=mean)", fontsize=11, fontweight="bold")
    fig.tight_layout()
    path = FIGURES_DIR / "equity_curves.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved {path}")


def fig_sharpe_heatmap(dfs):
    periods = [p for p in PERIODS if p in dfs]
    algos = ["PPO", "A2C", "SAC"]
    seeds = [42, 123, 7, 2024, 99]

    fig, axes = plt.subplots(1, len(periods), figsize=(4 * len(periods), 4))
    if len(periods) == 1:
        axes = [axes]
    for ax, period in zip(axes, periods):
        df = dfs[period]
        matrix = np.zeros((len(algos), len(seeds)))
        for i, algo in enumerate(algos):
            for j, seed in enumerate(seeds):
                row = df[(df["algorithm"] == algo) & (df["seed"] == seed)]
                if len(row):
                    matrix[i, j] = row["sharpe"].values[0]
        im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=-0.5, vmax=1.5)
        ax.set_xticks(range(len(seeds)))
        ax.set_xticklabels(seeds, fontsize=8)
        ax.set_yticks(range(len(algos)))
        ax.set_yticklabels(algos)
        ax.set_title(period, fontsize=9)
        for i in range(len(algos)):
            for j in range(len(seeds)):
                ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center", fontsize=7)
        plt.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle("Sharpe Heatmap (algo x seed)", fontsize=11, fontweight="bold")
    fig.tight_layout()
    path = FIGURES_DIR / "sharpe_heatmap.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved {path}")


def fig_scatter_sharpe_dd(dfs):
    periods = [p for p in PERIODS if p in dfs]
    fig, axes = plt.subplots(1, len(periods), figsize=(4.5 * len(periods), 4))
    if len(periods) == 1:
        axes = [axes]
    for ax, period in zip(axes, periods):
        df = dfs[period]
        for algo in ["PPO", "A2C", "SAC"]:
            sub = df[df["algorithm"] == algo]
            ax.scatter(sub["max_drawdown"], sub["sharpe"],
                       color=ALGO_COLORS[algo], label=algo, s=60, alpha=0.8)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xlabel("Max Drawdown")
        ax.set_ylabel("Sharpe Ratio")
        ax.set_title(period, fontsize=9)
        ax.legend(fontsize=8)
    fig.suptitle("Sharpe vs Max Drawdown (per seed)", fontsize=11, fontweight="bold")
    fig.tight_layout()
    path = FIGURES_DIR / "sharpe_vs_drawdown.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved {path}")


def fig_total_return(dfs, bh_stats):
    periods = [p for p in PERIODS if p in dfs]
    algos = ["PPO", "A2C", "SAC"]
    fig, axes = plt.subplots(1, len(periods), figsize=(5 * len(periods), 4), sharey=False)
    if len(periods) == 1:
        axes = [axes]
    for ax, period in zip(axes, periods):
        df = dfs[period]
        summary = df.groupby("algorithm")["total_return"].agg(["mean", "std"])
        means = [summary.loc[a, "mean"] if a in summary.index else 0 for a in algos]
        stds = [summary.loc[a, "std"] if a in summary.index else 0 for a in algos]
        ax.bar(algos, means, yerr=stds, capsize=6, color=[ALGO_COLORS[a] for a in algos], alpha=0.85)
        bh = bh_stats.get(period)
        if bh:
            ax.axhline(bh["total_return"], color="gray", linestyle="--", linewidth=1.5,
                       label=f"B&H {bh['total_return']:.2f}")
            ax.legend(fontsize=8)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_title(period, fontsize=10)
        ax.set_ylabel("Total Return")
    fig.suptitle("OOS Total Return (mean +/- std, 5 seeds)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = FIGURES_DIR / "total_return_by_period.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved {path}")


def main():
    dfs = {}
    for period, info in PERIODS.items():
        if Path(info["csv"]).exists():
            dfs[period] = pd.read_csv(info["csv"])
            logger.info(f"Loaded {period}: {len(dfs[period])} rows")
        else:
            logger.warning(f"Missing: {info['csv']}")

    bh_stats = {}
    for period, info in PERIODS.items():
        if Path(info["feature_path"]).exists():
            try:
                bh_stats[period] = load_bh(info["feature_path"], info["start"], info["end"])
                logger.info(f"B&H {period}: sharpe={bh_stats[period]['sharpe_ratio']:.3f}")
            except Exception as e:
                logger.warning(f"B&H failed for {period}: {e}")

    if not dfs:
        logger.error("No OOS CSVs found. Run backtests first.")
        return

    fig_sharpe_bar(dfs, bh_stats)
    fig_equity_curves()
    fig_sharpe_heatmap(dfs)
    fig_scatter_sharpe_dd(dfs)
    fig_total_return(dfs, bh_stats)

    logger.info(f"All figures saved to {FIGURES_DIR}/")
    print("\n=== FIGURES GENERATED ===")
    for f in sorted(FIGURES_DIR.glob("*.png")):
        print(f"  {f}")


if __name__ == "__main__":
    main()
