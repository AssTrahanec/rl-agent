"""Rebuild metrics.csv by running backtest on all saved models.

Usage:
    python scripts/rebuild_metrics.py
"""
import csv
import logging
from pathlib import Path

import numpy as np

from src.data.load_features import load_features_for_agent
from src.eval.backtest import run_backtest

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

EXPERIMENTS_DIR = Path("experiments")
RESULTS_CSV = Path("results/metrics.csv")
CSV_FIELDS = [
    "agent_type", "asset", "seed", "model_timestamp",
    "total_return", "sharpe_ratio", "sortino_ratio", "max_drawdown", "calmar_ratio",
]

# We only have BTC data processed, check for ETH too
ASSETS = ["BTC/USDT"]
TEST_START = "2024-01-01"
TEST_END = "2024-12-31"

AGENT_TYPES = ["baseline", "sentiment", "embeddings"]
# Seeds used: 42, 43, 44 — but we don't know which model corresponds to which seed
# We'll just enumerate models by timestamp


def main():
    all_results = []

    for agent_type in AGENT_TYPES:
        agent_dir = EXPERIMENTS_DIR / agent_type
        if not agent_dir.exists():
            logger.warning(f"No directory for {agent_type}")
            continue

        # Find all model.zip files
        model_dirs = sorted([d for d in agent_dir.iterdir() if d.is_dir() and (d / "model.zip").exists()])
        logger.info(f"{agent_type}: found {len(model_dirs)} models")

        for asset in ASSETS:
            try:
                test_features, test_prices = load_features_for_agent(
                    agent_type=agent_type,
                    asset=asset,
                    train_start=TEST_START,
                    train_end=TEST_END,
                )
            except (FileNotFoundError, ValueError) as e:
                logger.error(f"Cannot load test data for {agent_type}/{asset}: {e}")
                continue

            for i, model_dir in enumerate(model_dirs):
                model_path = str(model_dir / "model.zip")
                timestamp = model_dir.name
                seed = 42 + i  # approximate seed assignment by order

                try:
                    result = run_backtest(
                        features=test_features,
                        prices=test_prices,
                        model_path=model_path,
                        window=30,
                        tx_cost=0.001,
                    )
                    metrics = result["metrics"]
                    row = {
                        "agent_type": agent_type,
                        "asset": asset,
                        "seed": seed,
                        "model_timestamp": timestamp,
                        "total_return": metrics["total_return"],
                        "sharpe_ratio": metrics["sharpe_ratio"],
                        "sortino_ratio": metrics["sortino_ratio"],
                        "max_drawdown": metrics["max_drawdown"],
                        "calmar_ratio": metrics["calmar_ratio"],
                    }
                    all_results.append(row)
                    logger.info(
                        f"  {agent_type}/{timestamp} on {asset}: "
                        f"Sharpe={metrics['sharpe_ratio']:.3f} "
                        f"Return={metrics['total_return']*100:.1f}% "
                        f"MaxDD={metrics['max_drawdown']*100:.1f}%"
                    )
                except Exception as e:
                    logger.error(f"  FAILED {agent_type}/{timestamp}: {e}")

    # Save
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(all_results)

    logger.info(f"\nSaved {len(all_results)} rows to {RESULTS_CSV}")

    # Print summary
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in all_results:
        grouped[r["agent_type"]].append(r)

    print("\n" + "=" * 80)
    print(f"{'Agent':<15} {'N':>3} {'Sharpe':>12} {'Return':>12} {'MaxDD':>12} {'Sortino':>12}")
    print("=" * 80)
    for agent_type in AGENT_TYPES:
        rows = grouped.get(agent_type, [])
        if not rows:
            continue
        sharpes = [r["sharpe_ratio"] for r in rows]
        returns = [r["total_return"] for r in rows]
        maxdds = [r["max_drawdown"] for r in rows]
        sortinos = [r["sortino_ratio"] for r in rows]
        print(
            f"{agent_type:<15} {len(rows):>3} "
            f"{np.mean(sharpes):>6.3f}±{np.std(sharpes):.3f}  "
            f"{np.mean(returns)*100:>5.1f}±{np.std(returns)*100:.1f}%  "
            f"{np.mean(maxdds)*100:>5.1f}±{np.std(maxdds)*100:.1f}%  "
            f"{np.mean(sortinos):>6.3f}±{np.std(sortinos):.3f}"
        )
    print("=" * 80)


if __name__ == "__main__":
    main()
