"""Bootstrap CI for the difference of metrics between with-news and no-news models."""
import argparse
import csv
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

METRICS = (
    "total_return",
    "annualized_return",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "calmar_ratio",
    "win_rate",
    "profit_factor",
    "time_in_market",
)

METRIC_HIGHER_IS_BETTER = {
    "total_return": True,
    "annualized_return": True,
    "sharpe_ratio": True,
    "sortino_ratio": True,
    "max_drawdown": False,
    "calmar_ratio": True,
    "win_rate": True,
    "profit_factor": True,
    "time_in_market": True,
}


def bootstrap_diff_ci(with_news, no_news, n_bootstrap=5000, confidence=0.95, seed=42):
    rng = np.random.RandomState(seed)
    with_news = np.asarray(with_news, dtype=np.float64)
    no_news = np.asarray(no_news, dtype=np.float64)
    paired = len(with_news) == len(no_news)

    if paired:
        diffs = with_news - no_news
        boot = np.array([rng.choice(diffs, size=len(diffs), replace=True).mean()
                         for _ in range(n_bootstrap)])
        diff_std = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0
    else:
        boot = np.empty(n_bootstrap)
        for i in range(n_bootstrap):
            a = rng.choice(with_news, size=len(with_news), replace=True)
            b = rng.choice(no_news, size=len(no_news), replace=True)
            boot[i] = a.mean() - b.mean()
        diff_std = float("nan")

    alpha = (1 - confidence) / 2
    return {
        "with_news_mean": float(with_news.mean()),
        "with_news_std": float(np.std(with_news, ddof=1)) if len(with_news) > 1 else 0.0,
        "no_news_mean": float(no_news.mean()),
        "no_news_std": float(np.std(no_news, ddof=1)) if len(no_news) > 1 else 0.0,
        "diff_mean": float(with_news.mean() - no_news.mean()),
        "diff_std": diff_std,
        "ci_lower": float(np.percentile(boot, alpha * 100)),
        "ci_upper": float(np.percentile(boot, (1 - alpha) * 100)),
        "n_with": int(len(with_news)),
        "n_no": int(len(no_news)),
        "paired": bool(paired),
    }


def is_significant(ci_lower, ci_upper):
    return not (ci_lower <= 0.0 <= ci_upper)


def _load_results_csv(path):
    if not path.exists():
        raise FileNotFoundError(f"Results CSV not found: {path}")
    df = pd.read_csv(path)
    missing = {"algorithm", "seed", *METRICS} - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    return df


def compare_period(with_news_dir, no_news_dir, period, algos=None, n_bootstrap=5000):
    with_df = _load_results_csv(with_news_dir / f"oos_{period}.csv")
    no_df = _load_results_csv(no_news_dir / f"oos_{period}.csv")

    if algos is None:
        algos = sorted(set(with_df["algorithm"]) & set(no_df["algorithm"]))
    if not algos:
        raise ValueError(f"No common algorithms between {with_news_dir} and {no_news_dir}")

    rows = []
    for algo in algos:
        a_with = with_df[with_df["algorithm"] == algo].sort_values("seed")
        a_no = no_df[no_df["algorithm"] == algo].sort_values("seed")
        if a_with.empty or a_no.empty:
            logger.warning(f"  {algo}/{period}: empty (with={len(a_with)} no={len(a_no)}) — skip")
            continue
        for metric in METRICS:
            ci = bootstrap_diff_ci(
                a_with[metric].to_numpy(),
                a_no[metric].to_numpy(),
                n_bootstrap=n_bootstrap,
            )
            sig = is_significant(ci["ci_lower"], ci["ci_upper"])
            news_helps = (
                ci["diff_mean"] > 0
                if METRIC_HIGHER_IS_BETTER[metric]
                else ci["diff_mean"] < 0
            ) and sig
            rows.append({
                "period": period,
                "algorithm": algo,
                "metric": metric,
                **ci,
                "significant": sig,
                "news_helps": bool(news_helps),
            })
    return rows


def save_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        logger.warning(f"No rows to save to {path}")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Saved {len(rows)} comparison rows -> {path}")


METRIC_LABELS = {
    "total_return": ("Общая доходность", True),
    "annualized_return": ("Годовая доходность", True),
    "sharpe_ratio": ("Коэффициент Шарпа", False),
    "sortino_ratio": ("Коэффициент Сортино", False),
    "max_drawdown": ("Максимальная просадка", True),
    "calmar_ratio": ("Коэффициент Кальмара", False),
    "win_rate": ("Доля прибыльных шагов", True),
    "profit_factor": ("Profit Factor", False),
    "time_in_market": ("Время в позиции", True),
}


def plot_comparison(rows, path):
    # Один сабплот = одна метрика для одного алгоритма. Два столбика: «с новостями»
    # (синий) и «без новостей» (оранжевый), подписаны реальные значения. Зелёная
    # рамка вокруг сабплота — разница статистически значима (CI не пересекает 0).
    if not rows:
        return
    import matplotlib.pyplot as plt

    df = pd.DataFrame(rows)
    periods = sorted(df["period"].unique())
    algos = sorted(df["algorithm"].unique())

    for period in periods:
        for algo in algos:
            sub = df[(df["period"] == period) & (df["algorithm"] == algo)]
            if sub.empty:
                continue
            fig, axes = plt.subplots(3, 3, figsize=(14, 11))
            for ax, metric in zip(axes.flat, METRICS):
                label, is_pct = METRIC_LABELS[metric]
                r = sub[sub["metric"] == metric]
                if r.empty:
                    ax.set_visible(False)
                    continue
                r = r.iloc[0]
                with_val = r["with_news_mean"]
                no_val = r["no_news_mean"]
                sig = is_significant(r["ci_lower"], r["ci_upper"])

                bars = ax.bar(["С новостями", "Без новостей"], [with_val, no_val],
                              color=["tab:blue", "tab:orange"], width=0.55)
                fmt = (lambda v: f"{v * 100:.1f}%") if is_pct else (lambda v: f"{v:.2f}")
                for bar, val in zip(bars, [with_val, no_val]):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                            fmt(val), ha="center", va="bottom", fontsize=11)

                title = label + ("   ✓ разница значима" if sig else "")
                ax.set_title(title, fontsize=11)
                ax.grid(True, axis="y", alpha=0.3)
                ymin, ymax = ax.get_ylim()
                ax.set_ylim(ymin, ymax + (ymax - ymin) * 0.15)
                if sig:
                    for spine in ax.spines.values():
                        spine.set_edgecolor("tab:green")
                        spine.set_linewidth(2)

            year = period.replace("oos_", "")
            fig.suptitle(
                f"{algo} на {year} год: с новостями против без новостей\n"
                f"зелёная рамка — разница подтверждена статистически (бутстрап, 5000 итераций, 95% CI)",
                fontsize=13,
            )
            fig.tight_layout()
            out = path.parent / f"{path.stem}_{algo}_{period}.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out, dpi=150)
            plt.close(fig)
            logger.info(f"Saved figure -> {out}")


def _print_summary(rows):
    if not rows:
        return
    print("\n=== Ablation comparison (with-news - no-news) ===")
    print(f"{'period':<10} {'algo':<6} {'metric':<18} {'with':>8} {'no':>8} "
          f"{'diff':>8} {'CI':>22} {'sig':>4}")
    for r in rows:
        ci = f"[{r['ci_lower']:+.3f}; {r['ci_upper']:+.3f}]"
        print(f"{r['period']:<10} {r['algorithm']:<6} {r['metric']:<18} "
              f"{r['with_news_mean']:>+8.3f} {r['no_news_mean']:>+8.3f} "
              f"{r['diff_mean']:>+8.3f} {ci:>22} "
              f"{'YES' if r['significant'] else '-':>4}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--with-news-dir", default="results")
    ap.add_argument("--no-news-dir", default="results_no_news")
    ap.add_argument("--out-csv", default="results/ablation_comparison.csv")
    ap.add_argument("--out-fig", default="results/figures/ablation_comparison.png")
    ap.add_argument("--periods", nargs="+", default=None)
    ap.add_argument("--algos", nargs="+", default=None)
    ap.add_argument("--n-bootstrap", type=int, default=5000)
    args = ap.parse_args()

    with_dir = Path(args.with_news_dir)
    no_dir = Path(args.no_news_dir)

    if args.periods:
        periods = args.periods
    else:
        periods = sorted({
            p.stem.replace("oos_", "", 1)
            for p in with_dir.glob("oos_oos_*.csv")
            if (no_dir / p.name).exists()
        })
        logger.info(f"Discovered periods: {periods}")

    if not periods:
        raise SystemExit(f"No common oos_oos_*.csv in {with_dir} and {no_dir}.")

    all_rows = []
    for period in periods:
        logger.info(f"--- Comparing {period} ---")
        try:
            all_rows.extend(compare_period(with_dir, no_dir, period,
                                           args.algos, args.n_bootstrap))
        except FileNotFoundError as e:
            logger.error(str(e))

    if not all_rows:
        raise SystemExit("No comparison rows produced.")

    save_csv(all_rows, Path(args.out_csv))
    plot_comparison(all_rows, Path(args.out_fig))
    _print_summary(all_rows)


if __name__ == "__main__":
    main()
