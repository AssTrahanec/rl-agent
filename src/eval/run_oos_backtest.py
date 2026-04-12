"""OOS backtest runner for embeddings experiments."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from src.data.load_features import load_features_for_agent
from src.eval.backtest import run_backtest
from src.eval.bootstrap import bootstrap_ci
from src.eval.metrics import compute_metrics

logger = logging.getLogger(__name__)

_PRICE_COLUMNS_EXT = {"open", "high", "low", "close", "volume", "raw_close"}


def _load_test_features(
    asset: str,
    timeframe: str,
    test_start: str,
    test_end: str,
    feature_path: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if feature_path is not None:
        df = pd.read_parquet(feature_path)
        start_ts = pd.Timestamp(test_start)
        end_ts = pd.Timestamp(test_end)
        if df.index.tz is not None:
            start_ts = start_ts.tz_localize("UTC") if start_ts.tzinfo is None else start_ts
            end_ts = end_ts.tz_localize("UTC") if end_ts.tzinfo is None else end_ts
        df = df.loc[start_ts:end_ts]
        if "raw_close" in df.columns:
            prices = df["raw_close"].to_numpy(dtype=np.float64)
        else:
            prices = df["close"].to_numpy(dtype=np.float64)
        feature_cols = [c for c in df.columns if c.lower() not in _PRICE_COLUMNS_EXT]
        features = df[feature_cols].to_numpy(dtype=np.float32)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        logger.info(f"Loaded from {feature_path}: shape={features.shape}, prices={len(prices)}")
        return features, prices

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
    feature_path: Optional[str] = None,
) -> Path:
    runs_csv = Path(runs_csv)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    runs = pd.read_csv(runs_csv)
    features, prices = _load_test_features(asset, timeframe, test_start, test_end, feature_path)
    logger.info(f"Loaded test features shape={features.shape}, prices={len(prices)}")

    rows = []
    algo_returns: dict = {}
    for _, run in runs.iterrows():
        algo = run["algorithm"]
        seed = int(run["seed"])
        model_path = run["model_path"]
        vecnorm_path = (
            run["vecnorm_path"]
            if isinstance(run["vecnorm_path"], str) and run["vecnorm_path"]
            else None
        )

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
        equity = bt["equity_curve"]
        metrics = compute_metrics(returns)
        algo_returns.setdefault(algo, []).append(returns)
        rows.append({
            "algorithm": algo,
            "seed": seed,
            "sharpe": metrics["sharpe_ratio"],
            "sortino": metrics["sortino_ratio"],
            "max_drawdown": metrics["max_drawdown"],
            "calmar": metrics["calmar_ratio"],
            "total_return": metrics["total_return"],
            "equity_curve": equity.tolist(),
        })

    algo_ci: dict = {}
    for algo, returns_list in algo_returns.items():
        ci = bootstrap_ci(returns_list, n_bootstrap=1000, confidence=0.95, seed=0)
        algo_ci[algo] = ci

    for row in rows:
        ci = algo_ci[row["algorithm"]]
        row["sharpe_ci_low"] = ci["sharpe_ratio"]["ci_lower"]
        row["sharpe_ci_high"] = ci["sharpe_ratio"]["ci_upper"]
        row["total_return_ci_low"] = ci["total_return"]["ci_lower"]
        row["total_return_ci_high"] = ci["total_return"]["ci_upper"]

    # Save equity curves separately, drop from main CSV
    equity_data = {f"{r['algorithm']}_seed{r['seed']}": r.pop("equity_curve") for r in rows}
    import json
    equity_path = Path(str(output_csv).replace(".csv", "_equity.json"))
    with open(equity_path, "w") as f:
        json.dump(equity_data, f)

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    logger.info(f"Wrote {len(df)} OOS rows to {output_csv}")
    return output_csv


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-csv", default="results/embeddings_runs_v4.csv")
    parser.add_argument("--output-csv", default="results/oos_2024_embeddings.csv")
    parser.add_argument("--test-start", default="2024-01-01")
    parser.add_argument("--test-end", default="2024-12-31")
    parser.add_argument("--feature-path", default=None)
    args = parser.parse_args()
    run_oos_backtest(
        runs_csv=Path(args.runs_csv),
        output_csv=Path(args.output_csv),
        test_start=args.test_start,
        test_end=args.test_end,
        feature_path=args.feature_path,
    )
