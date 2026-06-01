"""Permutation importance on trained models: which feature group does the agent use?"""
import argparse
import csv
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.config_loader import load_config
from lib.data_loader import load_oos
from lib.interpretability import identify_feature_groups, permutation_importance

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

_PRICE_EXT = {"open", "high", "low", "close", "volume", "raw_close"}


def _feature_names_from_parquet(path):
    df = pd.read_parquet(path, columns=None)
    return [c for c in df.columns if c.lower() not in _PRICE_EXT]


def _find_model_runs(models_dir, algo):
    algo_dir = models_dir / algo
    if not algo_dir.exists():
        return []
    return sorted(d for d in algo_dir.iterdir()
                  if d.is_dir() and (d / "model.zip").exists())


def _seed_from_run_name(run_dir):
    for part in run_dir.name.split("_"):
        if part.startswith("seed"):
            try:
                return int(part[4:])
            except ValueError:
                return None
    return None


def _action_space_type_for_algo(cfg, algo):
    if algo == "DQN":
        return "discrete"
    if algo == "SAC":
        return "continuous"
    return getattr(cfg.env, "action_space_type", "discrete")


def run_permutation_phase(cfg, periods, algos, models_dir, results_dir,
                          n_repeats=1, mode="zero"):
    per_seed_rows = []

    for period in periods:
        features, prices, _sentiment = load_oos(cfg, period, exclude_news=False)
        feat_names = _feature_names_from_parquet(cfg.oos_features_path(period))
        if len(feat_names) != features.shape[1]:
            raise ValueError(f"{period}: feature_names ({len(feat_names)}) != "
                             f"features.shape[1] ({features.shape[1]})")
        groups = identify_feature_groups(feat_names)
        logger.info(f"=== {period}: {features.shape[0]} rows x {features.shape[1]} features ===")

        for algo in algos:
            runs = _find_model_runs(models_dir, algo)
            if not runs:
                logger.warning(f"  no models for {algo} in {models_dir} — skip")
                continue
            action_type = _action_space_type_for_algo(cfg, algo)
            for run_dir in runs:
                seed = _seed_from_run_name(run_dir)
                model_path = str(run_dir / "model.zip")
                logger.info(f"--- {algo} seed={seed} on {period} ---")

                try:
                    result = permutation_importance(
                        model_path=model_path,
                        features=features,
                        prices=prices,
                        feature_groups=groups,
                        backtest_kwargs=dict(
                            window=cfg.env.window,
                            tx_cost=cfg.env.tx_cost,
                            action_space_type=action_type,
                        ),
                        n_repeats=n_repeats,
                        mode=mode,
                    )
                except Exception as e:
                    logger.error(f"  FAILED permutation: {e}")
                    continue

                for group_name, metrics in result["groups"].items():
                    for metric_name, stats in metrics.items():
                        per_seed_rows.append({
                            "period": period,
                            "algorithm": algo,
                            "seed": seed,
                            "group": group_name,
                            "metric": metric_name,
                            "baseline": result["baseline"][metric_name],
                            "perturbed_mean": stats["perturbed_mean"],
                            "drop_mean": stats["drop_mean"],
                            "n_cols_in_group": len(groups[group_name]),
                            "mode": result["mode"],
                        })

    return per_seed_rows


def summarize_permutation(rows):
    if not rows:
        return []
    df = pd.DataFrame(rows)
    grouped = (
        df.groupby(["period", "algorithm", "group", "metric"])
        .agg(
            baseline_mean=("baseline", "mean"),
            perturbed_mean=("perturbed_mean", "mean"),
            drop_mean=("drop_mean", "mean"),
            drop_std=("drop_mean", "std"),
            n_seeds=("seed", "count"),
        )
        .reset_index()
    )
    grouped["drop_std"] = grouped["drop_std"].fillna(0.0)
    return grouped.to_dict("records")


def save_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Saved {len(rows)} rows -> {path}")


def plot_permutation(summary_rows, path):
    if not summary_rows:
        return
    import matplotlib.pyplot as plt

    df = pd.DataFrame(summary_rows)
    df = df[df["metric"] == "sharpe_ratio"]
    if df.empty:
        return
    periods = sorted(df["period"].unique())
    algos = sorted(df["algorithm"].unique())
    groups = [g for g in ("price", "sentiment", "embeddings", "news_count", "news")
              if g in df["group"].unique()]

    fig, axes = plt.subplots(len(periods), 1, figsize=(10, 4 * len(periods)),
                             squeeze=False, sharex=True)
    for ax_row, period in enumerate(periods):
        ax = axes[ax_row, 0]
        sub = df[df["period"] == period]
        x = np.arange(len(groups))
        width = 0.8 / max(len(algos), 1)
        for i, algo in enumerate(algos):
            algo_sub = sub[sub["algorithm"] == algo].set_index("group")
            means = [algo_sub.loc[g, "drop_mean"] if g in algo_sub.index else 0.0 for g in groups]
            stds = [algo_sub.loc[g, "drop_std"] if g in algo_sub.index else 0.0 for g in groups]
            offsets = x + (i - (len(algos) - 1) / 2) * width
            ax.bar(offsets, means, width, yerr=stds, capsize=4, label=algo)
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(groups)
        ax.set_ylabel("Sharpe drop (baseline - perturbed)")
        ax.set_title(f"Permutation importance ({period}) — positive = group is useful")
        ax.legend(loc="best")
        ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved figure -> {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--periods", nargs="+", default=None)
    ap.add_argument("--algos", nargs="+", default=None)
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--mode", choices=["zero", "shuffle"], default="zero")
    ap.add_argument("--n-repeats", type=int, default=1)
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    cfg = load_config(args.config)
    periods = args.periods or cfg.experiment.oos_periods
    algos = args.algos or cfg.experiment.algos

    models_dir = Path(args.models_dir)
    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Periods: {periods}  |  Algos: {algos}  |  Mode: {args.mode}")

    per_seed_rows = run_permutation_phase(
        cfg, periods, algos, models_dir, results_dir,
        n_repeats=args.n_repeats, mode=args.mode,
    )
    save_csv(per_seed_rows, out_dir / "permutation_importance.csv")
    summary_rows = summarize_permutation(per_seed_rows)
    save_csv(summary_rows, out_dir / "permutation_importance_summary.csv")
    plot_permutation(summary_rows, figures_dir / "permutation_importance.png")

    if summary_rows:
        df = pd.DataFrame(summary_rows)
        news_summary = df[(df["group"] == "news") & (df["metric"] == "sharpe_ratio")]
        if not news_summary.empty:
            print("\n=== News-channel Sharpe drop (positive = news helps) ===")
            for _, r in news_summary.iterrows():
                print(f"  {r['period']:<10} {r['algorithm']:<5} "
                      f"drop = {r['drop_mean']:+.3f} +/- {r['drop_std']:.3f}  "
                      f"(n_seeds={int(r['n_seeds'])})")


if __name__ == "__main__":
    main()
