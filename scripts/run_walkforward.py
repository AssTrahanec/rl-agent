"""Walk-forward validation: rolling train/test splits.

Usage:
    python -m scripts.run_walkforward --dummy
    python -m scripts.run_walkforward --agent-type baseline --total-timesteps 500000
"""
import argparse
import csv
import logging
from pathlib import Path

import numpy as np

from src.agents.config import AgentConfig
from src.agents.train import train_agent, FEATURE_COUNTS
from src.data.load_features import load_features_for_agent
from src.eval.backtest import run_backtest
from src.eval.metrics import compute_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def generate_splits(
    start_year: int = 2020,
    end_year: int = 2024,
    train_years: int = 2,
    test_years: int = 1,
) -> list[dict]:
    """Generate rolling train/test date splits.

    Args:
        start_year: First year of data.
        end_year: Last year (inclusive for test).
        train_years: Number of years in each training window.
        test_years: Number of years in each test window.

    Returns:
        List of dicts with train_start, train_end, test_start, test_end.
    """
    splits = []
    year = start_year
    while year + train_years + test_years - 1 <= end_year:
        train_start = f"{year}-01-01"
        train_end = f"{year + train_years - 1}-12-31"
        test_start = f"{year + train_years}-01-01"
        test_end = f"{year + train_years + test_years - 1}-12-31"
        splits.append({
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
        })
        year += 1
    return splits


def run_walkforward(
    agent_type: str = "baseline",
    total_timesteps: int = 500_000,
    seed: int = 42,
    asset: str = "BTC/USDT",
    dummy: bool = False,
) -> list[dict]:
    """Run walk-forward validation."""
    splits = generate_splits()
    results = []

    for i, split in enumerate(splits):
        print(f"\n{'='*60}")
        print(f"  Walk-Forward Split {i+1}/{len(splits)}")
        print(f"  Train: {split['train_start']} -> {split['train_end']}")
        print(f"  Test:  {split['test_start']} -> {split['test_end']}")
        print(f"{'='*60}")

        config = AgentConfig(
            agent_type=agent_type,
            algorithm="PPO",
            total_timesteps=total_timesteps,
            seed=seed,
            save_dir=f"experiments/walkforward/{agent_type}",
        )

        model_path = train_agent(
            config, dummy=dummy,
            train_start=split["train_start"],
            train_end=split["train_end"],
            asset=asset,
        )

        if dummy:
            rng = np.random.default_rng(seed + 1000)
            n_feat = FEATURE_COUNTS.get(agent_type, 18)
            test_feat = rng.standard_normal((150, n_feat)).astype(np.float32)
            test_prices = (100 + np.cumsum(rng.standard_normal(150) * 0.5)).astype(np.float64)
            test_prices = np.maximum(test_prices, 1.0)
        else:
            test_feat, test_prices = load_features_for_agent(
                agent_type, asset,
                split["test_start"], split["test_end"],
            )

        backtest = run_backtest(
            features=test_feat, prices=test_prices,
            model_path=str(model_path),
            window=config.window, tx_cost=config.tx_cost,
        )
        m = backtest["metrics"]

        bh_returns = np.diff(test_prices) / test_prices[:-1]
        bh = compute_metrics(bh_returns)

        result = {
            "split": i + 1,
            "train_start": split["train_start"],
            "train_end": split["train_end"],
            "test_start": split["test_start"],
            "test_end": split["test_end"],
            "agent_type": agent_type,
            "agent_sharpe": m["sharpe_ratio"],
            "agent_return": m["total_return"],
            "agent_maxdd": m["max_drawdown"],
            "bh_sharpe": bh["sharpe_ratio"],
            "bh_return": bh["total_return"],
            "bh_maxdd": bh["max_drawdown"],
        }
        results.append(result)

        print(f"  Agent:    Sharpe={m['sharpe_ratio']:.3f}  "
              f"Return={m['total_return']*100:.1f}%  MaxDD={m['max_drawdown']*100:.1f}%")
        print(f"  Buy&Hold: Sharpe={bh['sharpe_ratio']:.3f}  "
              f"Return={bh['total_return']*100:.1f}%  MaxDD={bh['max_drawdown']*100:.1f}%")

    return results


def main():
    parser = argparse.ArgumentParser(description="Walk-forward validation")
    parser.add_argument("--agent-type", type=str, default="baseline")
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--asset", type=str, default="BTC/USDT")
    parser.add_argument("--dummy", action="store_true")
    args = parser.parse_args()

    results = run_walkforward(
        agent_type=args.agent_type,
        total_timesteps=args.total_timesteps,
        seed=args.seed,
        asset=args.asset,
        dummy=args.dummy,
    )

    print(f"\n{'='*70}")
    print("WALK-FORWARD SUMMARY")
    print(f"{'='*70}")
    print(f"{'Split':<8} {'Test Period':<25} {'Agent Sharpe':>13} {'B&H Sharpe':>12} {'Agent MaxDD':>13}")
    print(f"{'-'*70}")
    for r in results:
        print(f"{r['split']:<8} {r['test_start']}->{r['test_end']:<14} "
              f"{r['agent_sharpe']:>13.3f} {r['bh_sharpe']:>12.3f} "
              f"{r['agent_maxdd']*100:>12.1f}%")

    csv_path = Path(f"results/walkforward_{args.agent_type}.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(results[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved to {csv_path}")


if __name__ == "__main__":
    main()
