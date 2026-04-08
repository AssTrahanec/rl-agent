"""Run all ablation experiments: trains baseline/sentiment/embeddings × seeds, saves metrics.

Usage (dummy data — no setup needed):
    python experiments/run_all.py --dummy

Usage (real data — run build scripts first):
    python -m src.data.build_price_features
    python -m src.features.build_sentiment_features
    python -m src.features.build_embedding_features
    python experiments/run_all.py --total-timesteps 500000 --asset BTC/USDT
"""
import argparse
import csv
import logging
from pathlib import Path

import numpy as np

from src.agents.config import AgentConfig
from src.agents.train import train_agent, FEATURE_COUNTS
from src.eval.backtest import run_backtest
from src.data.load_features import load_features_for_agent

logger = logging.getLogger(__name__)

AGENT_TYPES = ["baseline", "sentiment", "embeddings"]
RESULTS_CSV = Path("results/metrics.csv")
CSV_FIELDS = [
    "agent_type", "asset", "seed",
    "total_return", "sharpe_ratio", "sortino_ratio", "max_drawdown", "calmar_ratio",
]

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


def _make_dummy_data(agent_type: str, seed: int, n: int = 150):
    """Create dummy features and prices (for testing without real data)."""
    rng = np.random.default_rng(seed + 1000)
    n_features = FEATURE_COUNTS.get(agent_type, 20)
    features = rng.standard_normal((n, n_features)).astype(np.float32)
    prices = (100 + np.cumsum(rng.standard_normal(n) * 0.5)).astype(np.float64)
    prices = np.maximum(prices, 1.0)
    return features, prices


def run_all(
    total_timesteps: int,
    seeds: list[int],
    save_dir: str,
    dummy: bool = False,
    asset: str = "BTC/USDT",
    train_start: str = "2020-01-01",
    train_end: str = "2023-12-31",
    test_start: str = "2024-01-01",
    test_end: str = "2024-12-31",
    data_dir: str = "data/processed",
    agent_types: list[str] | None = None,
    timeframe: str = "1d",
    algorithm: str = "PPO",
    reward_type: str = "basic",
) -> list[dict]:
    """Train all agent types × seeds and collect backtest metrics."""
    agent_types = agent_types or AGENT_TYPES
    all_results = []
    total_runs = len(agent_types) * len(seeds)
    run_idx = 0

    if _WANDB_AVAILABLE:
        wandb.init(project="rl-trading-ablation", config={
            "total_timesteps": total_timesteps,
            "seeds": seeds,
            "agent_types": AGENT_TYPES,
            "asset": asset,
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
            "dummy": dummy,
        })
        logger.info("wandb initialized")
    else:
        logger.info("wandb not available — skipping wandb logging")

    for agent_type in agent_types:
        for seed in seeds:
            run_idx += 1
            mode = "[dummy]" if dummy else f"[{asset} {train_start}–{train_end}]"
            print(f"\n{'='*60}", flush=True)
            print(f"  Run {run_idx}/{total_runs}: agent={agent_type}  seed={seed}  {mode}", flush=True)
            print(f"{'='*60}", flush=True)
            logger.info(f"[{run_idx}/{total_runs}] Training {agent_type}, seed={seed} {mode}")

            config = AgentConfig(
                agent_type=agent_type,
                algorithm=algorithm,
                total_timesteps=total_timesteps,
                seed=seed,
                save_dir=save_dir,
                lr_schedule="linear",
                reward_type=reward_type,
            )

            # Embeddings-specific hyperparameter overrides
            if agent_type in ("embeddings", "fusion"):
                config.gamma = 0.95
                config.ent_coef = 0.01
                config.clip_range = 0.15
                config.reward_type = "dsr"

            model_path = train_agent(
                config,
                dummy=dummy,
                train_start=train_start,
                train_end=train_end,
                asset=asset,
                data_dir=data_dir,
                timeframe=timeframe,
            )
            logger.info(f"  Model saved: {model_path}")

            # Backtest on test period
            if dummy:
                test_features, test_prices = _make_dummy_data(agent_type, seed)
            else:
                test_features, test_prices = load_features_for_agent(
                    agent_type=agent_type,
                    asset=asset,
                    train_start=test_start,
                    train_end=test_end,
                    data_dir=data_dir,
                    timeframe=timeframe,
                )

            vecnorm_path = str(Path(model_path).parent / "vecnormalize.pkl")

            backtest = run_backtest(
                features=test_features,
                prices=test_prices,
                model_path=str(model_path),
                window=config.window,
                tx_cost=config.tx_cost,
                vecnorm_path=vecnorm_path,
            )
            metrics = backtest["metrics"]

            row = {
                "agent_type": agent_type,
                "asset": asset,
                "seed": seed,
                **{k: metrics[k] for k in ["total_return", "sharpe_ratio", "sortino_ratio", "max_drawdown", "calmar_ratio"]},
            }
            all_results.append(row)

            if _WANDB_AVAILABLE:
                wandb.log({
                    "agent_type": agent_type,
                    "seed": seed,
                    **{k: metrics[k] for k in ["total_return", "sharpe_ratio", "sortino_ratio", "max_drawdown", "calmar_ratio"]},
                })

            logger.info(
                f"  Sharpe={metrics['sharpe_ratio']:.3f}  "
                f"Return={metrics['total_return']:.3f}  "
                f"MaxDD={metrics['max_drawdown']:.3f}"
            )
            print(
                f"  DONE  Sharpe={metrics['sharpe_ratio']:.3f}  "
                f"Return={metrics['total_return']*100:.1f}%  "
                f"MaxDD={metrics['max_drawdown']*100:.1f}%",
                flush=True,
            )

    if _WANDB_AVAILABLE:
        wandb.finish()

    return all_results


def save_csv(results: list[dict]) -> None:
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(results)
    logger.info(f"Saved metrics to {RESULTS_CSV}")


def print_summary(results: list[dict]) -> None:
    from collections import defaultdict

    grouped: dict[str, list] = defaultdict(list)
    for r in results:
        grouped[r["agent_type"]].append(r)

    print("\n" + "=" * 70)
    print(f"{'Agent':<15} {'Sharpe':>8} {'Return':>8} {'MaxDD':>8} {'Sortino':>8}")
    print("=" * 70)
    present_types = [a for a in AGENT_TYPES if a in grouped]
    for agent_type in present_types:
        rows = grouped.get(agent_type, [])
        if not rows:
            continue
        sharpes = [r["sharpe_ratio"] for r in rows]
        returns = [r["total_return"] for r in rows]
        maxdds = [r["max_drawdown"] for r in rows]
        sortinos = [r["sortino_ratio"] for r in rows]
        print(
            f"{agent_type:<15} "
            f"{np.mean(sharpes):>6.3f}±{np.std(sharpes):.2f}  "
            f"{np.mean(returns):>6.3f}±{np.std(returns):.2f}  "
            f"{np.mean(maxdds):>6.3f}±{np.std(maxdds):.2f}  "
            f"{np.mean(sortinos):>6.3f}±{np.std(sortinos):.2f}"
        )
    print("=" * 70)
    print(f"\nResults saved to: {RESULTS_CSV}")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Run all ablation experiments")
    parser.add_argument(
        "--total-timesteps", type=int, default=500_000,
        help="Training steps per run (default: 500000)"
    )
    parser.add_argument(
        "--seeds", type=str, default="42,43,44,45,46",
        help="Comma-separated seeds (default: 42,43,44,45,46)"
    )
    parser.add_argument(
        "--save-dir", type=str, default="experiments",
        help="Directory for saved models (default: experiments)"
    )
    parser.add_argument(
        "--agent-types", type=str, default=None,
        help="Comma-separated agent types to run, e.g. 'baseline,sentiment' (default: all)"
    )
    parser.add_argument(
        "--dummy", action="store_true",
        help="Use random dummy data (no real data needed)"
    )
    parser.add_argument(
        "--asset", type=str, default="BTC/USDT",
        help="Asset symbol (default: BTC/USDT)"
    )
    parser.add_argument(
        "--train-start", type=str, default="2020-01-01",
        help="Train period start (default: 2020-01-01)"
    )
    parser.add_argument(
        "--train-end", type=str, default="2023-12-31",
        help="Train period end (default: 2023-12-31)"
    )
    parser.add_argument(
        "--test-start", type=str, default="2024-01-01",
        help="Test period start (default: 2024-01-01)"
    )
    parser.add_argument(
        "--test-end", type=str, default="2024-12-31",
        help="Test period end (default: 2024-12-31)"
    )
    parser.add_argument(
        "--data-dir", type=str, default="data/processed",
        help="Directory with preprocessed parquet files (default: data/processed)"
    )
    parser.add_argument(
        "--timeframe", type=str, default="1d",
        help="Candle timeframe: '1d' or '4h' (default: 1d)"
    )
    parser.add_argument(
        "--algorithm", type=str, default="PPO",
        help="RL algorithm: PPO, A2C, or SAC (default: PPO)"
    )
    parser.add_argument(
        "--reward-type", type=str, default="basic",
        help="Reward function: 'basic' or 'risk_adjusted' (default: basic)"
    )
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    agent_types = [s.strip() for s in args.agent_types.split(",")] if args.agent_types else None
    mode = "dummy" if args.dummy else f"real data ({args.asset})"
    logger.info(
        f"Starting ablation: agents={agent_types or AGENT_TYPES}, seeds={seeds}, "
        f"timesteps={args.total_timesteps}, mode={mode}"
    )

    results = run_all(
        total_timesteps=args.total_timesteps,
        seeds=seeds,
        save_dir=args.save_dir,
        dummy=args.dummy,
        asset=args.asset,
        train_start=args.train_start,
        train_end=args.train_end,
        test_start=args.test_start,
        test_end=args.test_end,
        data_dir=args.data_dir,
        agent_types=agent_types,
        timeframe=args.timeframe,
        algorithm=args.algorithm,
        reward_type=args.reward_type,
    )
    save_csv(results)
    print_summary(results)


if __name__ == "__main__":
    main()
