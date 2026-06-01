"""Train SAC/PPO with N seeds, run OOS backtest on configured periods.

Examples:
    python run.py
    python run.py --algo SAC
    python run.py --skip-train
    python run.py --skip-train --oos oos_2025
    python run.py --skip-oos
    python run.py --no-news --models-dir models_no_news --results-dir results_no_news
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

DEFAULT_MODELS_DIR = "models"
DEFAULT_RESULTS_DIR = "results"


def train_phase(
    cfg: Config,
    algos: list,
    models_dir: str = DEFAULT_MODELS_DIR,
    exclude_news: bool = False,
) -> dict:
    """Train algos x seeds; return {algo: [model_path, ...]}.

    Args:
        cfg: Config.
        algos: List of algorithm names (e.g. ["DQN", "SAC"]).
        models_dir: Output directory for trained models.
        exclude_news: If True, drop news features from state and zero the sentiment
            channel (ablation: no-news baseline).
    """
    features, prices, sentiment = load_train(cfg, exclude_news=exclude_news)
    out: dict = {algo: [] for algo in algos}
    for algo in algos:
        for seed in cfg.experiment.seeds:
            logger.info(f"--- Training {algo} seed={seed} ---")
            path = train_agent(
                cfg=cfg, algo=algo, seed=seed,
                features=features, prices=prices, sentiment=sentiment,
                models_dir=models_dir,
            )
            out[algo].append(path)
    return out


def find_latest_models(algo: str, n: int, models_dir: str = DEFAULT_MODELS_DIR) -> list:
    """Return the N latest model.zip paths for the given algo."""
    algo_dir = Path(models_dir) / algo
    if not algo_dir.exists():
        raise FileNotFoundError(f"No models directory for {algo}: {algo_dir}")
    runs = sorted(
        [d for d in algo_dir.iterdir() if d.is_dir() and (d / "model.zip").exists()],
        key=lambda d: d.stat().st_mtime,
    )
    if len(runs) < n:
        logger.warning(f"{algo}: requested {n} models, found {len(runs)}")
    return [r / "model.zip" for r in runs[-n:]]


def oos_phase(
    cfg: Config,
    algos: list,
    model_paths: dict,
    oos_keys: list,
    results_dir: str = DEFAULT_RESULTS_DIR,
    exclude_news: bool = False,
):
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    for period_key in oos_keys:
        features, prices, _ = load_oos(cfg, period_key, exclude_news=exclude_news)
        rows = []
        for algo in algos:
            for path, seed in zip(model_paths[algo], cfg.experiment.seeds):
                try:
                    res = run_backtest(
                        features=features, prices=prices,
                        model_path=str(path),
                        window=cfg.env.window,
                        tx_cost=cfg.env.tx_cost,
                        action_space_type=getattr(cfg.env, "action_space_type", "discrete"),
                    )
                    m = res["metrics"]
                    rows.append({
                        "algorithm": algo, "seed": seed, "period": period_key, **m,
                    })
                    np.savez(
                        Path(results_dir) / f"oos_{period_key}_{algo}_seed{seed}.npz",
                        daily_returns=res["daily_returns"],
                        allocations=res["allocations"],
                        equity_curve=res["equity_curve"],
                    )
                    logger.info(
                        f"  {algo} seed={seed} {period_key}: "
                        f"Sharpe={m['sharpe_ratio']:.3f} Ret={m['total_return']*100:.1f}%"
                    )
                except Exception as e:
                    logger.error(f"  FAILED {algo} seed={seed}: {e}")

        if rows:
            csv_path = Path(results_dir) / f"oos_{period_key}.csv"
            # Merge with any existing rows so that running --algo on one algorithm
            # at a time accumulates results instead of truncating the CSV. A row
            # is identified by (algorithm, seed, period); new rows replace older
            # ones for the same key.
            existing: list[dict] = []
            if csv_path.exists():
                with open(csv_path, "r", newline="", encoding="utf-8") as f:
                    existing = list(csv.DictReader(f))
            new_keys = {(r["algorithm"], str(r["seed"]), r["period"]) for r in rows}
            kept = [r for r in existing
                    if (r.get("algorithm"), str(r.get("seed")), r.get("period")) not in new_keys]
            merged = kept + rows
            fieldnames = list(rows[0].keys())
            # Reconcile fieldnames in case a stored row has extra columns (forward-compat).
            for r in kept:
                for k in r:
                    if k not in fieldnames:
                        fieldnames.append(k)
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(merged)
            logger.info(
                f"Saved {len(rows)} new + {len(kept)} kept = {len(merged)} rows -> {csv_path}"
            )
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
    ap.add_argument("--algo", choices=["SAC", "DQN"], help="Train only this algo")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-oos", action="store_true")
    ap.add_argument("--oos", action="append",
                    help="OOS period key (repeatable); default = experiment.oos_periods")
    ap.add_argument("--no-news", action="store_true",
                    help="Ablation: drop news features (sentiment_*/news_count*/emb_*) from "
                         "state, zero out sentiment channel, force sentiment_lambda=0 in reward.")
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="Override experiment.seeds with this explicit list (e.g. 42 123 7).")
    ap.add_argument("--models-dir", default=DEFAULT_MODELS_DIR,
                    help=f"Output dir for trained models (default: {DEFAULT_MODELS_DIR}). "
                         f"For ablation use e.g. 'models_no_news'.")
    ap.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR,
                    help=f"Output dir for backtest results (default: {DEFAULT_RESULTS_DIR}). "
                         f"For ablation use e.g. 'results_no_news'.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    algos = [args.algo] if args.algo else cfg.experiment.algos

    if args.seeds:
        original_seeds = cfg.experiment.seeds
        cfg.experiment.seeds = list(args.seeds)
        logger.info(f"--seeds override: {original_seeds} -> {cfg.experiment.seeds}")

    if args.no_news:
        # Disable sentiment bonus in reward — otherwise the env would expect a meaningful
        # sentiment signal but receive zeros, which is inconsistent. Clean ablation
        # requires both: no news in state AND no sentiment-driven reward shaping.
        original_lambda = getattr(cfg.env, "sentiment_lambda", None)
        cfg.env.sentiment_lambda = 0.0
        logger.info(
            f"--no-news enabled: sentiment_lambda {original_lambda} -> 0.0, "
            f"models_dir={args.models_dir}, results_dir={args.results_dir}"
        )

    if args.skip_train:
        model_paths = {
            algo: find_latest_models(algo, len(cfg.experiment.seeds), models_dir=args.models_dir)
            for algo in algos
        }
    else:
        model_paths = train_phase(
            cfg, algos, models_dir=args.models_dir, exclude_news=args.no_news
        )

    if args.skip_oos:
        return

    oos_keys = args.oos if args.oos else cfg.experiment.oos_periods
    for k in oos_keys:
        if k not in cfg.periods or k == "train":
            raise SystemExit(f"Invalid --oos: {k}")

    oos_phase(
        cfg, algos, model_paths, oos_keys,
        results_dir=args.results_dir, exclude_news=args.no_news,
    )


if __name__ == "__main__":
    main()
