"""Run ensemble backtests on multiple market periods.

Usage:
    python -m scripts.run_ensemble
    python -m scripts.run_ensemble --period 2024
"""
import argparse
import csv
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.load_features import load_features_for_agent
from src.eval.ensemble_backtest import run_ensemble_backtest
from src.eval.backtest import run_backtest
from src.eval.baselines import buy_and_hold

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_CSV = Path("results/all_experiments.csv")
PERIODS = {
    "2024": ("2024-01-01", "2024-12-31"),
    "2022": ("2022-01-01", "2022-12-31"),
    "2025q1": ("2025-01-01", "2025-03-19"),
}


def load_experiment_results() -> pd.DataFrame:
    """Load all experiment results and return sorted by Sharpe."""
    df = pd.read_csv(RESULTS_CSV)
    return df.sort_values("sharpe_ratio", ascending=False)


def _model_path_from_label(label) -> Path | None:
    """Extract model path from CSV label like 'btc/baseline/20260313_124941'."""
    if not isinstance(label, str):
        return None
    parts = label.split("/")
    if len(parts) >= 3:
        agent_type = parts[1]
        timestamp = parts[2]
        mp = Path("experiments") / agent_type / timestamp / "model.zip"
        if mp.exists():
            return mp
    return None


def select_top_models(df: pd.DataFrame, n: int = 3,
                      agent_type: str | None = None) -> list[str]:
    """Select top N model paths by Sharpe ratio."""
    sub = df[df["agent_type"] == agent_type] if agent_type else df
    paths = []
    for _, row in sub.iterrows():
        mp = _model_path_from_label(row["label"])
        if mp is not None:
            paths.append(str(mp))
        if len(paths) >= n:
            break
    return paths


def _print_metrics(label: str, m: dict):
    print(f"  {label:<35} Sharpe={m['sharpe_ratio']:.3f}  "
          f"Return={m['total_return']*100:.1f}%  MaxDD={m['max_drawdown']*100:.1f}%")


def run_period(period_name: str, start: str, end: str, df: pd.DataFrame):
    """Run all ensemble strategies on one period."""
    print(f"\n{'#'*80}")
    print(f"#  ENSEMBLE BACKTEST: {period_name} ({start} -> {end})")
    print(f"{'#'*80}")

    # Load baseline features (18d) — works for baseline models
    try:
        feat_bl, prices_bl = load_features_for_agent("baseline", "BTC/USDT", start, end)
    except Exception as e:
        logger.error(f"Cannot load baseline features for {period_name}: {e}")
        return []

    # Buy & Hold
    bh_result = buy_and_hold(prices_bl)
    bh = bh_result["metrics"]
    _print_metrics("Buy & Hold", bh)

    results = []

    # Strategy 1: Top-3 baseline models (same feature space)
    top3_bl = select_top_models(df, n=3, agent_type="baseline")
    if len(top3_bl) >= 2:
        r = run_ensemble_backtest(top3_bl, feat_bl, prices_bl)
        m = r["metrics"]
        label = f"Top-{len(top3_bl)} Baseline Ensemble"
        _print_metrics(label, m)
        results.append({"period": period_name, "strategy": label, **m})

    # Strategy 2: Top-3 embeddings models (50d features)
    top3_emb = select_top_models(df, n=3, agent_type="embeddings")
    if len(top3_emb) >= 2:
        try:
            feat_emb, prices_emb = load_features_for_agent(
                "embeddings", "BTC/USDT", start, end)
            r = run_ensemble_backtest(top3_emb, feat_emb, prices_emb)
            m = r["metrics"]
            label = f"Top-{len(top3_emb)} Embeddings Ensemble"
            _print_metrics(label, m)
            results.append({"period": period_name, "strategy": label, **m})
        except Exception as e:
            logger.warning(f"Cannot run embeddings ensemble: {e}")

    # Strategy 3: Top-5 baseline models
    top5_bl = select_top_models(df, n=5, agent_type="baseline")
    if len(top5_bl) >= 3:
        r = run_ensemble_backtest(top5_bl, feat_bl, prices_bl)
        m = r["metrics"]
        label = f"Top-{len(top5_bl)} Baseline Ensemble"
        _print_metrics(label, m)
        results.append({"period": period_name, "strategy": label, **m})

    # Single best model for comparison
    best_path = select_top_models(df, n=1, agent_type="baseline")
    if best_path:
        try:
            r = run_backtest(feat_bl, prices_bl, best_path[0])
            m = r["metrics"]
            _print_metrics("Single Best (baseline)", m)
            results.append({"period": period_name, "strategy": "Single Best", **m})
        except Exception as e:
            logger.warning(f"Cannot run single best: {e}")

    # Buy & Hold row
    results.append({"period": period_name, "strategy": "Buy & Hold", **bh})

    return results


def main():
    parser = argparse.ArgumentParser(description="Run ensemble backtests")
    parser.add_argument("--period", type=str, default=None,
                        help="Period to test (2024, 2022, 2025q1). Default: all")
    args = parser.parse_args()

    if not RESULTS_CSV.exists():
        logger.error(f"{RESULTS_CSV} not found. Run scripts/analyze_all.py first.")
        return

    df = load_experiment_results()
    logger.info(f"Loaded {len(df)} experiment results")

    periods = {args.period: PERIODS[args.period]} if args.period else PERIODS
    all_results = []

    for name, (start, end) in periods.items():
        results = run_period(name, start, end, df)
        all_results.extend(results)

    # Save CSV
    if all_results:
        csv_path = Path("results/ensemble_results.csv")
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        fields = list(all_results[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nResults saved to {csv_path}")


if __name__ == "__main__":
    main()
