"""Train SAC/PPO with N seeds, run OOS backtest on configured periods.

Examples:
    python run.py
    python run.py --algo SAC
    python run.py --skip-train
    python run.py --skip-train --oos oos_2025
    python run.py --skip-oos
"""
import argparse
import csv
import logging
from pathlib import Path

import numpy as np

from lib.config_loader import load_config, Config
from lib.data_loader import load_train, load_oos
from lib.train import train_agent
from lib.backtest import run_backtest

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = "models"
RESULTS_DIR = "results"


def train_phase(cfg: Config, algos: list) -> dict:
    """Train algos x seeds; return {algo: [model_path, ...]}."""
    features, prices, sentiment = load_train(cfg)
    out: dict = {algo: [] for algo in algos}
    for algo in algos:
        for seed in cfg.experiment.seeds:
            logger.info(f"--- Training {algo} seed={seed} ---")
            path = train_agent(
                cfg=cfg, algo=algo, seed=seed,
                features=features, prices=prices, sentiment=sentiment,
                models_dir=MODELS_DIR,
            )
            out[algo].append(path)
    return out


def find_latest_models(algo: str, n: int) -> list:
    """Return the N latest model.zip paths for the given algo."""
    algo_dir = Path(MODELS_DIR) / algo
    if not algo_dir.exists():
        raise FileNotFoundError(f"No models directory for {algo}: {algo_dir}")
    runs = sorted(
        [d for d in algo_dir.iterdir() if d.is_dir() and (d / "model.zip").exists()],
        key=lambda d: d.stat().st_mtime,
    )
    if len(runs) < n:
        logger.warning(f"{algo}: requested {n} models, found {len(runs)}")
    return [r / "model.zip" for r in runs[-n:]]


def oos_phase(cfg: Config, algos: list, model_paths: dict, oos_keys: list):
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    for period_key in oos_keys:
        features, prices, _ = load_oos(cfg, period_key)
        rows = []
        for algo in algos:
            for path, seed in zip(model_paths[algo], cfg.experiment.seeds):
                vecnorm = path.parent / "vecnormalize.pkl"
                try:
                    res = run_backtest(
                        features=features, prices=prices,
                        model_path=str(path),
                        window=cfg.env.window,
                        tx_cost=cfg.env.tx_cost,
                        allow_short=cfg.env.allow_short,
                        vecnorm_path=str(vecnorm) if vecnorm.exists() else None,
                    )
                    m = res["metrics"]
                    rows.append({
                        "algorithm": algo, "seed": seed, "period": period_key,
                        "reward_type": cfg.env.reward_type, **m,
                    })
                    logger.info(
                        f"  {algo} seed={seed} {period_key}: "
                        f"Sharpe={m['sharpe_ratio']:.3f} Ret={m['total_return']*100:.1f}%"
                    )
                except Exception as e:
                    logger.error(f"  FAILED {algo} seed={seed}: {e}")

        if rows:
            csv_path = Path(RESULTS_DIR) / f"oos_{period_key}.csv"
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            logger.info(f"Saved {len(rows)} rows -> {csv_path}")
            _print_summary(rows, period_key)


def _print_summary(rows: list, period_key: str):
    print(f"\n=== {period_key} (mean ± std across seeds) ===")
    by_algo: dict = {}
    for r in rows:
        by_algo.setdefault(r["algorithm"], []).append(r)
    for algo, items in by_algo.items():
        sharpe = np.array([r["sharpe_ratio"] for r in items])
        ret = np.array([r["total_return"] for r in items])
        dd = np.array([r["max_drawdown"] for r in items])
        print(f"  {algo}: Sharpe {sharpe.mean():+.3f}±{sharpe.std():.3f} | "
              f"Return {ret.mean()*100:+.1f}%±{ret.std()*100:.1f}% | "
              f"MaxDD {dd.mean()*100:.1f}%±{dd.std()*100:.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--algo", choices=["SAC", "PPO"], help="Train only this algo")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-oos", action="store_true")
    ap.add_argument("--oos", action="append",
                    help="OOS period key (repeatable); default = experiment.oos_periods")
    args = ap.parse_args()

    cfg = load_config(args.config)
    algos = [args.algo] if args.algo else cfg.experiment.algos

    if args.skip_train:
        model_paths = {algo: find_latest_models(algo, len(cfg.experiment.seeds)) for algo in algos}
    else:
        model_paths = train_phase(cfg, algos)

    if args.skip_oos:
        return

    oos_keys = args.oos if args.oos else cfg.experiment.oos_periods
    for k in oos_keys:
        if k not in cfg.periods or k == "train":
            raise SystemExit(f"Invalid --oos: {k}")

    oos_phase(cfg, algos, model_paths, oos_keys)


if __name__ == "__main__":
    main()
