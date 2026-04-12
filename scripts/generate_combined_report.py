"""Generate combined multi-period markdown report."""
import logging
import sys
from pathlib import Path
from itertools import combinations

import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PERIODS = {
    "2024 OOS": {
        "csv": "results/oos_2024_embeddings.csv",
        "feature_path": "data/processed/btc_4h_embedding_features.parquet",
        "start": "2024-01-01", "end": "2024-12-31",
        "note": "Full NLP (real news + embeddings)",
    },
    "2025 OOS": {
        "csv": "results/oos_2025_embeddings.csv",
        "feature_path": "data/processed/btc_4h_2025_embedding_features.parquet",
        "start": "2025-01-01", "end": "2025-04-10",
        "note": "Full NLP (real news + embeddings)",
    },
    "2026 YTD": {
        "csv": "results/oos_2026_embeddings.csv",
        "feature_path": "data/processed/btc_4h_2026_embedding_features.parquet",
        "start": "2026-01-01", "end": "2026-04-12",
        "note": "Price only (NLP = zeros, no news data available)",
    },
}


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


def main():
    lines = []
    lines.append("# Embeddings 3-Agents -- Multi-Period OOS Report\n")
    lines.append("**Generated:** 2026-04-12  \n")
    lines.append("**Models:** 15 (PPO x5, A2C x5, SAC x5), trained BTC/USDT 4h 2020-2023  \n")
    lines.append("**Features:** 95 (18 tech + 64d PCA embeddings + 13 NLP)  \n\n")
    lines.append("---\n")

    all_dfs = {}
    for period, info in PERIODS.items():
        if not Path(info["csv"]).exists():
            logger.warning(f"Missing {info['csv']}, skipping {period}")
            continue
        df = pd.read_csv(info["csv"])
        all_dfs[period] = df

        bh = None
        if Path(info["feature_path"]).exists():
            try:
                bh = load_bh(info["feature_path"], info["start"], info["end"])
            except Exception as e:
                logger.warning(f"B&H error {period}: {e}")

        summary = df.groupby("algorithm").agg(
            sharpe_mean=("sharpe", "mean"),
            sharpe_std=("sharpe", "std"),
            total_return_mean=("total_return", "mean"),
            total_return_std=("total_return", "std"),
            max_dd_mean=("max_drawdown", "mean"),
            calmar_mean=("calmar", "mean"),
            sortino_mean=("sortino", "mean"),
        ).round(4)

        lines.append(f"\n## {period}\n")
        lines.append(f"*{info['note']}* -- {info['start']} to {info['end']}  \n\n")
        if bh:
            lines.append(f"**Buy & Hold:** Sharpe={bh['sharpe_ratio']:.4f}, "
                         f"Total Return={bh['total_return']:.4f}, "
                         f"Max DD={bh['max_drawdown']:.4f}  \n\n")
        lines.append("### Summary (mean +/- std over 5 seeds)\n\n")
        lines.append(summary.to_markdown())
        lines.append("\n\n### Per-seed results\n\n")
        per_seed = df[["algorithm", "seed", "sharpe", "total_return", "max_drawdown", "calmar"]].copy()
        per_seed = per_seed.sort_values(["algorithm", "seed"]).round(4)
        lines.append(per_seed.to_markdown(index=False))
        lines.append("\n\n### Mann-Whitney U (Sharpe, two-sided)\n\n")
        algos = sorted(df["algorithm"].unique())
        mw_rows = []
        for a, b in combinations(algos, 2):
            sa = df[df["algorithm"] == a]["sharpe"].values
            sb = df[df["algorithm"] == b]["sharpe"].values
            stat, p = mannwhitneyu(sa, sb, alternative="two-sided")
            sig = "**yes**" if p < 0.05 else "no"
            mw_rows.append({"pair": f"{a} vs {b}", "U": int(stat),
                            "p_value": round(p, 5), "significant": sig})
        lines.append(pd.DataFrame(mw_rows).to_markdown(index=False))
        lines.append("\n\n---\n")

    # Cross-period SAC comparison
    if len(all_dfs) > 1:
        lines.append("\n## Cross-Period SAC Comparison\n\n")
        rows = []
        for period, df in all_dfs.items():
            sac = df[df["algorithm"] == "SAC"]
            rows.append({
                "Period": period,
                "Sharpe mean": round(sac["sharpe"].mean(), 4),
                "Sharpe std": round(sac["sharpe"].std(), 4),
                "Total Return mean": round(sac["total_return"].mean(), 4),
                "Max DD mean": round(sac["max_drawdown"].mean(), 4),
            })
        lines.append(pd.DataFrame(rows).to_markdown(index=False))
        lines.append("\n\n")

    lines.append("## Figures\n\n")
    lines.append("All figures in `results/figures/all_periods/`:\n\n")
    lines.append("- `sharpe_by_period.png` -- Sharpe bar chart (3 panels, +/-std, B&H line)\n")
    lines.append("- `equity_curves.png` -- Equity curves (thin=seeds, thick=mean per algo)\n")
    lines.append("- `sharpe_heatmap.png` -- Sharpe heatmap (algo x seed, per period)\n")
    lines.append("- `sharpe_vs_drawdown.png` -- Scatter Sharpe vs Max Drawdown\n")
    lines.append("- `total_return_by_period.png` -- Total Return bar chart\n")

    out_path = Path("results/oos_multi_period_report.md")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Wrote {out_path}")
    print(f"\nReport saved to {out_path}")


if __name__ == "__main__":
    main()
