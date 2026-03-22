"""Run improved experiments: risk-adjusted reward + short positions.

Usage:
    python experiments/run_improved.py --dummy
    python experiments/run_improved.py --total-timesteps 500000 --seeds 42,43,44
"""
import argparse
import csv
import logging
import sys
from pathlib import Path

# Ensure the project root is on sys.path when run as a script
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from src.agents.config import AgentConfig
from src.agents.train import train_agent, FEATURE_COUNTS
from src.eval.backtest import run_backtest
from src.data.load_features import load_features_for_agent

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_CSV = Path("results/improved_metrics.csv")
CSV_FIELDS = [
    "experiment", "agent_type", "reward_type", "allow_short",
    "asset", "seed",
    "total_return", "sharpe_ratio", "sortino_ratio", "max_drawdown", "calmar_ratio",
]

EXPERIMENTS = [
    {"label": "risk_adjusted", "reward_type": "risk_adjusted", "allow_short": False},
    {"label": "risk_adjusted_short", "reward_type": "risk_adjusted", "allow_short": True},
]


def _make_dummy_data(agent_type, seed, n=150):
    rng = np.random.default_rng(seed + 2000)
    n_features = FEATURE_COUNTS.get(agent_type, 18)
    features = rng.standard_normal((n, n_features)).astype(np.float32)
    prices = (100 + np.cumsum(rng.standard_normal(n) * 0.5)).astype(np.float64)
    prices = np.maximum(prices, 1.0)
    return features, prices


def main():
    parser = argparse.ArgumentParser(description="Run improved experiments")
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--agent-type", type=str, default="baseline")
    parser.add_argument("--asset", type=str, default="BTC/USDT")
    parser.add_argument("--dummy", action="store_true")
    parser.add_argument("--train-start", type=str, default="2020-01-01")
    parser.add_argument("--train-end", type=str, default="2023-12-31")
    parser.add_argument("--test-start", type=str, default="2024-01-01")
    parser.add_argument("--test-end", type=str, default="2024-12-31")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    all_results = []

    for exp in EXPERIMENTS:
        for seed in seeds:
            print(f"\n{'='*60}", flush=True)
            print(f"  {exp['label']}  seed={seed}  agent={args.agent_type}", flush=True)
            print(f"{'='*60}", flush=True)

            config = AgentConfig(
                agent_type=args.agent_type,
                algorithm="PPO",
                total_timesteps=args.total_timesteps,
                seed=seed,
                save_dir=f"experiments/improved/{exp['label']}",
                reward_type=exp["reward_type"],
                allow_short=exp["allow_short"],
            )

            model_path = train_agent(
                config, dummy=args.dummy,
                train_start=args.train_start,
                train_end=args.train_end,
                asset=args.asset,
            )

            if args.dummy:
                test_feat, test_prices = _make_dummy_data(args.agent_type, seed)
            else:
                test_feat, test_prices = load_features_for_agent(
                    args.agent_type, args.asset,
                    args.test_start, args.test_end,
                )

            backtest = run_backtest(
                features=test_feat, prices=test_prices,
                model_path=str(model_path),
                window=config.window, tx_cost=config.tx_cost,
                allow_short=config.allow_short,
            )
            m = backtest["metrics"]

            row = {
                "experiment": exp["label"],
                "agent_type": args.agent_type,
                "reward_type": exp["reward_type"],
                "allow_short": exp["allow_short"],
                "asset": args.asset,
                "seed": seed,
                **{k: m[k] for k in ["total_return", "sharpe_ratio", "sortino_ratio",
                                      "max_drawdown", "calmar_ratio"]},
            }
            all_results.append(row)

            print(f"  DONE  Sharpe={m['sharpe_ratio']:.3f}  "
                  f"Return={m['total_return']*100:.1f}%  "
                  f"MaxDD={m['max_drawdown']*100:.1f}%", flush=True)

    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for exp in EXPERIMENTS:
        rows = [r for r in all_results if r["experiment"] == exp["label"]]
        sharpes = [r["sharpe_ratio"] for r in rows]
        returns = [r["total_return"] for r in rows]
        maxdds = [r["max_drawdown"] for r in rows]
        print(f"{exp['label']:<25} Sharpe={np.mean(sharpes):.3f}+-{np.std(sharpes):.2f}  "
              f"Return={np.mean(returns)*100:.1f}%  MaxDD={np.mean(maxdds)*100:.1f}%")
    print(f"\nSaved to {RESULTS_CSV}")
    print("\nDONE")


if __name__ == "__main__":
    main()
