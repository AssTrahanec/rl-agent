"""Out-of-sample backtest: run trained models on new time periods.

Takes already-trained models from experiments/ and backtests them on
a user-specified date range — no retraining needed.

Usage:
    python -m scripts.oos_backtest --start 2025-01-01 --end 2025-03-19
    python -m scripts.oos_backtest --start 2022-01-01 --end 2022-12-31
    python -m scripts.oos_backtest --start 2025-01-01 --end 2025-03-19 --baseline-only
"""
import argparse
import csv
import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.price_collector import fetch_ohlcv
from src.features.technical import add_technical_indicators
from src.features.normalizer import rolling_zscore_normalize
from src.eval.backtest import run_backtest
from src.eval.metrics import compute_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

EXPERIMENTS_DIR = Path("experiments")
RESULTS_DIR = Path("results")
WARMUP_DAYS = 90  # extra days before test start for indicator warm-up

FEATURE_COUNTS = {
    "baseline": 18,
    "sentiment": 19,
    "embeddings": 50,
}


def _build_baseline_features(symbol: str, start: str, end: str) -> tuple[pd.DataFrame, int]:
    """Fetch prices, compute indicators, normalize. Return full DataFrame."""
    # Fetch extra days for indicator warm-up
    start_dt = datetime.strptime(start, "%Y-%m-%d") - timedelta(days=WARMUP_DAYS)
    warmup_start = start_dt.strftime("%Y-%m-%d")

    logger.info(f"Fetching {symbol} from {warmup_start} to {end} (incl. {WARMUP_DAYS}d warm-up)")
    df = fetch_ohlcv(symbol, warmup_start, end)
    logger.info(f"Fetched {len(df)} rows")

    # Keep raw close for prices array
    df_with_indicators = add_technical_indicators(df)

    # Separate raw close before normalization
    raw_close = df_with_indicators["close"].copy()

    # Normalize all columns
    df_norm = rolling_zscore_normalize(df_with_indicators, window=30)

    # Add raw close back
    df_norm["raw_close"] = raw_close

    # Trim to test period (remove warm-up)
    start_ts = pd.Timestamp(start, tz="UTC") if df_norm.index.tz else pd.Timestamp(start)
    df_norm = df_norm.loc[start_ts:]

    return df_norm, len(df)


def _extract_features_prices(df: pd.DataFrame, n_features: int):
    """Extract feature matrix and price array from DataFrame."""
    prices = df["raw_close"].to_numpy(dtype=np.float64)

    # Drop OHLCV and raw_close, keep only normalized indicator columns
    drop_cols = {"open", "high", "low", "close", "volume", "raw_close"}
    feature_cols = [c for c in df.columns if c.lower() not in drop_cols]
    features = df[feature_cols].to_numpy(dtype=np.float32)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    actual = features.shape[1]
    if actual != n_features:
        if n_features > actual:
            # Pad with zeros (e.g., sentiment/embedding columns missing)
            pad = np.zeros((features.shape[0], n_features - actual), dtype=np.float32)
            features = np.concatenate([features, pad], axis=1)
            logger.warning(
                f"Padded {n_features - actual} zero columns "
                f"(model expects {n_features}, got {actual})"
            )
        else:
            # Truncate (shouldn't happen normally)
            features = features[:, :n_features]
            logger.warning(f"Truncated features from {actual} to {n_features}")

    return features, prices


def _buy_and_hold_metrics(prices: np.ndarray) -> dict:
    """Compute Buy & Hold metrics from price series."""
    daily_returns = np.diff(prices) / prices[:-1]
    return compute_metrics(daily_returns)


def _find_models(agent_type: str) -> list[Path]:
    """Find all model.zip files for an agent type."""
    agent_dir = EXPERIMENTS_DIR / agent_type
    if not agent_dir.exists():
        return []
    return sorted([
        d / "model.zip"
        for d in agent_dir.iterdir()
        if d.is_dir() and (d / "model.zip").exists()
    ])


def main():
    parser = argparse.ArgumentParser(description="Out-of-sample backtest")
    parser.add_argument("--start", default="2025-01-01", help="Test start date")
    parser.add_argument("--end", default="2025-03-19", help="Test end date")
    parser.add_argument("--asset", default="BTC/USDT", help="Trading pair")
    parser.add_argument(
        "--baseline-only", action="store_true",
        help="Only test baseline agent (skip sentiment/embeddings)"
    )
    args = parser.parse_args()

    agent_types = ["baseline"] if args.baseline_only else list(FEATURE_COUNTS.keys())

    # Step 1: Build features
    logger.info(f"=== OOS Backtest: {args.asset} {args.start} -> {args.end} ===")
    df, total_rows = _build_baseline_features(args.asset, args.start, args.end)
    logger.info(f"Test period: {len(df)} trading days")

    if len(df) < 31:
        logger.error(f"Only {len(df)} days in test period — need at least 31 (30 window + 1)")
        return

    # Step 2: Buy & Hold
    _, bh_prices = _extract_features_prices(df, FEATURE_COUNTS["baseline"])
    bh_metrics = _buy_and_hold_metrics(bh_prices)

    print("\n" + "=" * 90)
    print(f"OOS BACKTEST: {args.asset}  {args.start} -> {args.end}  ({len(df)} days)")
    print("=" * 90)
    print(
        f"\n{'Strategy':<30} {'Return':>10} {'Sharpe':>10} {'Sortino':>10} "
        f"{'MaxDD':>10} {'Calmar':>10}"
    )
    print("-" * 90)
    print(
        f"{'Buy & Hold':<30} "
        f"{bh_metrics['total_return']*100:>9.1f}% "
        f"{bh_metrics['sharpe_ratio']:>10.3f} "
        f"{bh_metrics['sortino_ratio']:>10.3f} "
        f"{bh_metrics['max_drawdown']*100:>9.1f}% "
        f"{bh_metrics['calmar_ratio']:>10.3f}"
    )

    # Step 3: Run each agent type
    all_results = []

    for agent_type in agent_types:
        n_feat = FEATURE_COUNTS[agent_type]
        features, prices = _extract_features_prices(df, n_feat)
        models = _find_models(agent_type)

        if not models:
            logger.warning(f"No models found for {agent_type}")
            continue

        for model_path in models:
            timestamp = model_path.parent.name
            label = f"{agent_type}/{timestamp}"

            try:
                result = run_backtest(
                    features=features,
                    prices=prices,
                    model_path=str(model_path),
                    window=30,
                    tx_cost=0.001,
                )
                m = result["metrics"]
                print(
                    f"{label:<30} "
                    f"{m['total_return']*100:>9.1f}% "
                    f"{m['sharpe_ratio']:>10.3f} "
                    f"{m['sortino_ratio']:>10.3f} "
                    f"{m['max_drawdown']*100:>9.1f}% "
                    f"{m['calmar_ratio']:>10.3f}"
                )
                all_results.append({
                    "agent_type": agent_type,
                    "model_timestamp": timestamp,
                    "asset": args.asset,
                    "test_start": args.start,
                    "test_end": args.end,
                    **m,
                })
            except Exception as e:
                logger.error(f"  FAILED {label}: {e}")

    # Step 4: Random agent
    features_bl, prices_bl = _extract_features_prices(df, FEATURE_COUNTS["baseline"])
    rand_result = run_backtest(
        features=features_bl, prices=prices_bl,
        model_path=None, window=30, tx_cost=0.001,
    )
    rm = rand_result["metrics"]
    print(
        f"{'Random Agent':<30} "
        f"{rm['total_return']*100:>9.1f}% "
        f"{rm['sharpe_ratio']:>10.3f} "
        f"{rm['sortino_ratio']:>10.3f} "
        f"{rm['max_drawdown']*100:>9.1f}% "
        f"{rm['calmar_ratio']:>10.3f}"
    )

    print("=" * 90)

    # Step 5: Summary by agent type
    if all_results:
        print(f"\n{'SUMMARY (mean across models)':<30}")
        print("-" * 70)
        from collections import defaultdict
        grouped = defaultdict(list)
        for r in all_results:
            grouped[r["agent_type"]].append(r)

        for at in agent_types:
            rows = grouped.get(at, [])
            if not rows:
                continue
            sharpes = [r["sharpe_ratio"] for r in rows]
            returns = [r["total_return"] for r in rows]
            maxdds = [r["max_drawdown"] for r in rows]
            print(
                f"{at:<30} "
                f"{np.mean(returns)*100:>6.1f}±{np.std(returns)*100:.1f}%  "
                f"Sharpe {np.mean(sharpes):>6.3f}±{np.std(sharpes):.3f}  "
                f"MaxDD {np.mean(maxdds)*100:>5.1f}±{np.std(maxdds)*100:.1f}%"
            )

        print(
            f"{'Buy & Hold':<30} "
            f"{bh_metrics['total_return']*100:>6.1f}%        "
            f"Sharpe {bh_metrics['sharpe_ratio']:>6.3f}        "
            f"MaxDD {bh_metrics['max_drawdown']*100:>5.1f}%"
        )

    # Step 6: Save CSV
    if all_results:
        csv_name = f"oos_{args.start}_{args.end}.csv"
        csv_path = RESULTS_DIR / csv_name
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        fields = list(all_results[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nResults saved to {csv_path}")


if __name__ == "__main__":
    main()
