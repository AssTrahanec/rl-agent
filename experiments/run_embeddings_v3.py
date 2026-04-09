"""Run embeddings v3 experiment: MI feature selection + tuned hyperparams.

Usage:
    PYTHONPATH=. python experiments/run_embeddings_v3.py
    PYTHONPATH=. python experiments/run_embeddings_v3.py --total-timesteps 1000000
"""
import argparse
import csv
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.agents.config import AgentConfig
from src.agents.train import train_agent, FEATURE_COUNTS
from src.eval.backtest import run_backtest
from src.data.load_features import load_features_for_agent
from src.features.feature_selector import MIFeatureSelector
from src.env.trading_env import TradingEnv

logger = logging.getLogger(__name__)

RESULTS_CSV = Path("results/metrics_v3.csv")
CSV_FIELDS = [
    "agent_type", "asset", "seed",
    "total_return", "sharpe_ratio", "sortino_ratio", "max_drawdown", "calmar_ratio",
    "n_features",
]

# Price/tech features that are always kept (never dropped by MI)
PROTECTED_FEATURES = [
    "sma_7", "sma_25", "ema_12", "ema_26", "macd", "macd_signal", "macd_hist",
    "rsi_14", "bb_upper", "bb_lower", "bb_mid", "bb_width", "atr_14", "obv",
    "stoch_k", "stoch_d", "return_1d", "return_5d",
]


def run_v3(
    total_timesteps: int = 1_000_000,
    seeds: list[int] = None,
    asset: str = "BTC/USDT",
    train_start: str = "2020-01-01",
    train_end: str = "2023-12-31",
    test_start: str = "2024-01-01",
    test_end: str = "2024-12-31",
    data_dir: str = "data/processed",
    save_dir: str = "experiments",
    mi_threshold: float = 0.005,
) -> list[dict]:
    """Train embeddings agent with MI feature selection."""
    seeds = seeds or [42, 43, 44, 45, 46]
    all_results = []

    # --- Load full features ---
    features_train, prices_train, sentiment_train = load_features_for_agent(
        agent_type="embeddings", asset=asset,
        train_start=train_start, train_end=train_end,
        data_dir=data_dir, return_sentiment=True,
    )
    features_test, prices_test, sentiment_test = load_features_for_agent(
        agent_type="embeddings", asset=asset,
        train_start=test_start, train_end=test_end,
        data_dir=data_dir, return_sentiment=True,
    )

    # --- Load column names from parquet ---
    parquet_path = Path(data_dir) / f"{asset.split('/')[0].lower()}_embedding_features.parquet"
    df_full = pd.read_parquet(parquet_path)
    price_cols = {"open", "high", "low", "close", "volume", "raw_close"}
    feature_cols = [c for c in df_full.columns if c.lower() not in price_cols]

    # --- MI feature selection on train data ---
    df_train = df_full.loc[train_start:train_end]
    raw_close = df_train["raw_close"].values
    next_returns = np.log(raw_close[1:] / raw_close[:-1])

    X_train = df_train[feature_cols].iloc[:-1]
    selector = MIFeatureSelector(
        mi_threshold=mi_threshold,
        protected_columns=PROTECTED_FEATURES,
        random_state=42,
    )
    selector.fit(X_train, next_returns)

    # Apply selection to train and test
    selected = selector.selected_features_
    n_selected = len(selected)
    print(f"\nFeature selection: {n_selected}/{len(feature_cols)} features kept")
    print(f"MI scores: {selector.mi_scores_}")

    # Get column indices for selection
    col_indices = [feature_cols.index(c) for c in selected]
    features_train_sel = features_train[:, col_indices]
    features_test_sel = features_test[:, col_indices]

    print(f"Train shape: {features_train_sel.shape}, Test shape: {features_test_sel.shape}")

    for i, seed in enumerate(seeds):
        print(f"\n{'='*60}")
        print(f"  Run {i+1}/{len(seeds)}: embeddings_v3  seed={seed}")
        print(f"{'='*60}")

        config = AgentConfig(
            agent_type="embeddings",
            algorithm="PPO",
            total_timesteps=total_timesteps,
            seed=seed,
            save_dir=save_dir,
            lr_schedule="warmup_linear",
            reward_type="dsr",
            gamma=0.95,
            ent_coef=0.01,
            clip_range=0.15,
            max_grad_norm=0.3,
            batch_size=256,
            n_steps=4096,
        )

        # Build env directly with selected features
        env = TradingEnv(
            features=features_train_sel,
            prices=prices_train,
            window=config.window,
            tx_cost=config.tx_cost,
            reward_type=config.reward_type,
            allow_short=config.allow_short,
            sentiment_signal=sentiment_train,
            sentiment_lambda=config.sentiment_lambda,
        )

        # Wrap in VecNormalize
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        vec_env = DummyVecEnv([lambda: env])
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=5.0)

        from src.agents.train import _warmup_linear_schedule, EMBEDDINGS_NET_ARCH
        lr = _warmup_linear_schedule(config.learning_rate, warmup_frac=0.1)

        policy_kwargs = config.policy_kwargs()
        policy_kwargs["net_arch"] = EMBEDDINGS_NET_ARCH

        from stable_baselines3 import PPO
        from src.agents.train import ProgressCallback

        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=lr,
            seed=config.seed,
            verbose=0,
            device="cpu",
            policy_kwargs=policy_kwargs,
            n_steps=config.n_steps,
            batch_size=config.batch_size,
            n_epochs=config.n_epochs,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            clip_range=config.clip_range,
            ent_coef=config.ent_coef,
            max_grad_norm=config.max_grad_norm,
        )

        logger.info(f"Training PPO (embeddings_v3) for {total_timesteps} steps, seed={seed}")
        callback = ProgressCallback(total_timesteps=total_timesteps)
        model.learn(total_timesteps=total_timesteps, callback=callback)

        # Save model + vecnorm
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_save_dir = Path(save_dir) / "embeddings_v3" / timestamp
        model_save_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_save_dir / "model.zip"
        model.save(str(model_path.with_suffix("")))
        vecnorm_path = model_save_dir / "vecnormalize.pkl"
        vec_env.save(str(vecnorm_path))

        # Backtest on test data with selected features
        backtest = run_backtest(
            features=features_test_sel,
            prices=prices_test,
            model_path=str(model_path),
            window=config.window,
            tx_cost=config.tx_cost,
            vecnorm_path=str(vecnorm_path),
        )
        metrics = backtest["metrics"]

        row = {
            "agent_type": "embeddings_v3",
            "asset": asset,
            "seed": seed,
            "n_features": n_selected,
            **{k: metrics[k] for k in ["total_return", "sharpe_ratio", "sortino_ratio", "max_drawdown", "calmar_ratio"]},
        }
        all_results.append(row)

        print(
            f"  DONE  Sharpe={metrics['sharpe_ratio']:.3f}  "
            f"Return={metrics['total_return']*100:.1f}%  "
            f"MaxDD={metrics['max_drawdown']*100:.1f}%"
        )

    return all_results


def save_csv(results: list[dict]) -> None:
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved to {RESULTS_CSV}")


def print_summary(results: list[dict]) -> None:
    sharpes = [r["sharpe_ratio"] for r in results]
    returns = [r["total_return"] for r in results]
    maxdds = [r["max_drawdown"] for r in results]
    print(f"\n{'='*60}")
    print(f"embeddings_v3 ({len(results)} seeds, {results[0]['n_features']} features)")
    print(f"  Sharpe:  {np.mean(sharpes):.3f} +/- {np.std(sharpes):.3f}")
    print(f"  Return:  {np.mean(returns)*100:.1f}% +/- {np.std(returns)*100:.1f}%")
    print(f"  MaxDD:   {np.mean(maxdds)*100:.1f}% +/- {np.std(maxdds)*100:.1f}%")
    print(f"{'='*60}")

    # Compare with v2 and baseline from metrics.csv
    try:
        old = pd.read_csv("results/metrics.csv")
        for agent in ["embeddings", "baseline"]:
            s = old[old["agent_type"] == agent]["sharpe_ratio"]
            if len(s) > 0:
                print(f"  vs {agent} (v2): {s.mean():.3f} +/- {s.std():.3f}")
    except FileNotFoundError:
        pass


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description="Run embeddings v3 experiment")
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--seeds", type=str, default="42,43,44,45,46")
    parser.add_argument("--asset", type=str, default="BTC/USDT")
    parser.add_argument("--mi-threshold", type=float, default=0.005)
    parser.add_argument("--save-dir", type=str, default="experiments")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    results = run_v3(
        total_timesteps=args.total_timesteps,
        seeds=seeds,
        asset=args.asset,
        mi_threshold=args.mi_threshold,
        save_dir=args.save_dir,
    )
    save_csv(results)
    print_summary(results)


if __name__ == "__main__":
    main()
