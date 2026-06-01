"""Train embeddings agents with DSR+sentiment reward, run OOS on 2024 and 2025.

Usage:
    # Train SAC only (faster, GPU):
    PYTHONPATH=. venv/Scripts/python.exe -m scripts.run_dsr_experiment --algo SAC

    # Train PPO only (CPU):
    PYTHONPATH=. venv/Scripts/python.exe -m scripts.run_dsr_experiment --algo PPO

    # Train both:
    PYTHONPATH=. venv/Scripts/python.exe -m scripts.run_dsr_experiment
"""
import argparse
import csv
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.agents.config import sac_embeddings_dsr_config, ppo_embeddings_dsr_config
from src.agents.train import train_agent
from src.data.load_features import load_features_for_agent
from src.eval.backtest import run_backtest
from src.eval.metrics import compute_metrics
from src.eval.bootstrap import bootstrap_ci

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

SEEDS = [42, 123, 7, 2024, 99]
RESULTS_DIR = Path("results")

PERIODS = {
    "2024": ("2024-01-01", "2024-12-31"),
    "2025": ("2025-01-01", "2025-04-10"),
}

FEATURE_FILE_MAP = {
    "2024": "data/processed/btc_4h_embedding_features.parquet",
    "2025": "data/processed/btc_4h_2025_embedding_features.parquet",
}


def _load_oos_features(period: str):
    """Load OOS feature matrix and prices for a given period."""
    path = Path(FEATURE_FILE_MAP[period])
    start, end = PERIODS[period]

    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index, utc=True)
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")
    df = df.loc[start_ts:end_ts]

    if df.empty:
        raise ValueError(f"No data for period {period} in {path}")

    prices = df["raw_close"].to_numpy(dtype=np.float64)
    drop = {"open", "high", "low", "close", "volume", "raw_close"}
    feat_cols = [c for c in df.columns if c not in drop]
    features = df[feat_cols].to_numpy(dtype=np.float32)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    logger.info(f"OOS {period}: {len(df)} rows, {features.shape[1]} features")
    return features, prices


def _buy_and_hold(prices: np.ndarray) -> dict:
    returns = np.diff(prices) / prices[:-1]
    return compute_metrics(returns)


def train_all(algo: str) -> list[Path]:
    """Train algo x 5 seeds, return list of model paths."""
    config_fn = sac_embeddings_dsr_config if algo == "SAC" else ppo_embeddings_dsr_config
    model_paths = []
    for seed in SEEDS:
        logger.info(f"Training {algo} DSR seed={seed}")
        cfg = config_fn(seed)
        path = train_agent(
            config=cfg,
            dummy=False,
            train_start="2020-01-01",
            train_end="2023-12-31",
            asset="BTC/USDT",
            data_dir="data/processed",
            timeframe="4h",
        )
        model_paths.append(path)
        logger.info(f"Saved: {path}")
    return model_paths


def run_oos_for_period(model_paths: list[Path], algo: str, period: str) -> list[dict]:
    """Run OOS backtest for all models on a period, return rows."""
    features, prices = _load_oos_features(period)
    rows = []
    for path, seed in zip(model_paths, SEEDS):
        try:
            result = run_backtest(
                features=features,
                prices=prices,
                model_path=str(path),
                window=30,
                tx_cost=0.001,
            )
            m = result["metrics"]
            rows.append({
                "algorithm": algo,
                "seed": seed,
                "reward_type": "dsr",
                "period": period,
                **m,
            })
            logger.info(
                f"  {algo} seed={seed} {period}: "
                f"Sharpe={m['sharpe_ratio']:.3f} Return={m['total_return']*100:.1f}%"
            )
        except Exception as e:
            logger.error(f"  FAILED {algo} seed={seed}: {e}")
    return rows


def save_results(rows: list[dict], period: str) -> Path:
    """Save OOS results to CSV."""
    if not rows:
        return None
    csv_path = RESULTS_DIR / f"oos_{period}_dsr.csv"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Saved {len(rows)} rows to {csv_path}")
    return csv_path


def print_comparison(period: str):
    """Print side-by-side DSR vs basic results."""
    basic_path = RESULTS_DIR / f"oos_{period}_embeddings.csv"
    dsr_path = RESULTS_DIR / f"oos_{period}_dsr.csv"

    if not basic_path.exists() or not dsr_path.exists():
        logger.warning(f"Cannot compare: missing files for {period}")
        return

    df_basic = pd.read_csv(basic_path)
    df_dsr = pd.read_csv(dsr_path)

    print(f"\n{'='*70}")
    print(f"COMPARISON: basic vs DSR reward -- {period}")
    print(f"{'='*70}")

    for algo in df_basic["algorithm"].unique():
        b = df_basic[df_basic["algorithm"] == algo]
        d = df_dsr[df_dsr["algorithm"] == algo] if "algorithm" in df_dsr.columns else df_dsr

        b_sharpe = b["sharpe"].mean() if "sharpe" in b.columns else b["sharpe_ratio"].mean()
        d_sharpe = d["sharpe"].mean() if "sharpe" in d.columns else d["sharpe_ratio"].mean()
        b_ret = b["total_return"].mean() if "total_return" in b.columns else 0
        d_ret = d["total_return"].mean()
        b_dd = b["max_drawdown"].mean() if "max_drawdown" in b.columns else 0
        d_dd = d["max_drawdown"].mean()

        print(f"\n  {algo}:")
        print(f"    {'Metric':<15} {'basic':>10} {'DSR':>10} {'delta':>10}")
        print(f"    {'Sharpe':<15} {b_sharpe:>10.3f} {d_sharpe:>10.3f} {d_sharpe-b_sharpe:>+10.3f}")
        print(f"    {'Return':<15} {b_ret*100:>9.1f}% {d_ret*100:>9.1f}% {(d_ret-b_ret)*100:>+9.1f}%")
        print(f"    {'MaxDD':<15} {b_dd*100:>9.1f}% {d_dd*100:>9.1f}% {(d_dd-b_dd)*100:>+9.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["SAC", "PPO", "both"], default="both",
                        help="Which algorithm to train (default: both)")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training, only run OOS on existing DSR models")
    args = parser.parse_args()

    algos = ["SAC", "PPO"] if args.algo == "both" else [args.algo]
    all_rows = {period: [] for period in PERIODS}

    for algo in algos:
        if args.skip_train:
            # Find existing DSR models
            agent_dir = Path("experiments/embeddings")
            model_paths = sorted([
                d / "model.zip"
                for d in agent_dir.iterdir()
                if d.is_dir() and (d / "model.zip").exists()
            ])[-len(SEEDS):]
            logger.info(f"Using {len(model_paths)} existing models for {algo}")
        else:
            model_paths = train_all(algo)

        for period in PERIODS:
            rows = run_oos_for_period(model_paths, algo, period)
            all_rows[period].extend(rows)

    for period, rows in all_rows.items():
        if rows:
            save_results(rows, period)
            print_comparison(period)


if __name__ == "__main__":
    main()
