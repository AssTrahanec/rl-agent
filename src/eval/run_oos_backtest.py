"""OOS backtest runner for embeddings experiments.

For each trained model in the runs manifest, runs backtest on the test slice,
computes metrics + Bootstrap 95% CI, and appends to output CSV.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from src.data.load_features import load_features_for_agent
from src.eval.backtest import run_backtest
from src.eval.bootstrap import bootstrap_ci
from src.eval.metrics import compute_metrics

logger = logging.getLogger(__name__)


def _load_test_features(
    asset: str,
    timeframe: str,
    test_start: str,
    test_end: str,
) -> Tuple[np.ndarray, np.ndarray]:
    result = load_features_for_agent(
        agent_type="embeddings",
        asset=asset,
        train_start=test_start,
        train_end=test_end,
        data_dir="data/processed",
        timeframe=timeframe,
        return_sentiment=False,
    )
    if isinstance(result, tuple) and len(result) == 3:
        features, prices, _ = result
    else:
        features, prices = result
    return features, prices


def run_oos_backtest(
    runs_csv: Path,
    output_csv: Path,
    asset: str = "BTC/USDT",
    timeframe: str = "4h",
    test_start: str = "2024-01-01",
    test_end: str = "2024-12-31",
    window: int = 30,
    tx_cost: float = 0.001,
) -> Path:
    runs_csv = Path(runs_csv)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    runs = pd.read_csv(runs_csv)
    features, prices = _load_test_features(asset, timeframe, test_start, test_end)
    logger.info(f"Loaded test features shape={features.shape}, prices={len(prices)}")

    rows = []
    rng = np.random.RandomState(0)
    for _, run in runs.iterrows():
        algo = run["algorithm"]
        seed = int(run["seed"])
        model_path = run["model_path"]
        vecnorm_path = run["vecnorm_path"] if isinstance(run["vecnorm_path"], str) and run["vecnorm_path"] else None

        logger.info(f"Backtesting {algo} seed={seed}")
        bt = run_backtest(
            model_path=model_path,
            features=features,
            prices=prices,
            window=window,
            tx_cost=tx_cost,
            vecnorm_path=vecnorm_path,
        )
        returns = bt["daily_returns"]
        metrics = compute_metrics(returns)
        sharpe_low, sharpe_high = bootstrap_ci(
            returns,
            statistic=lambda r: compute_metrics(r)["sharpe"],
            n_iter=1000,
            ci=0.95,
            seed=seed,
        )
        tr_low, tr_high = bootstrap_ci(
            returns,
            statistic=lambda r: compute_metrics(r)["total_return"],
            n_iter=1000,
            ci=0.95,
            seed=seed,
        )
        rows.append({
            "algorithm": algo,
            "seed": seed,
            "sharpe": metrics["sharpe"],
            "sharpe_ci_low": sharpe_low,
            "sharpe_ci_high": sharpe_high,
            "sortino": metrics["sortino"],
            "max_drawdown": metrics["max_drawdown"],
            "calmar": metrics["calmar"],
            "total_return": metrics["total_return"],
            "total_return_ci_low": tr_low,
            "total_return_ci_high": tr_high,
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    logger.info(f"Wrote {len(df)} OOS rows to {output_csv}")
    return output_csv


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_oos_backtest(
        runs_csv=Path("results/embeddings_runs_v4.csv"),
        output_csv=Path("results/oos_2024_embeddings.csv"),
    )
