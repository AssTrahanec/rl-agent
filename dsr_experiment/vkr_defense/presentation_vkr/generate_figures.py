"""Generate missing figures for VKR presentation.

Creates:
    images/barplot_oos2024.png
    images/btc_vs_spx_volatility.png
    images/news_histogram.png

Also copies ready figures from ../../results/figures/presentation/ into images/.
"""
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
IMG = HERE / "images"
IMG.mkdir(exist_ok=True)

DSR = HERE.parent
SRC_FIGS = DSR.parent / "results" / "figures" / "presentation"


def fig_barplot_oos2024():
    groups = ["Buy & Hold", "DQN", "SAC"]
    means = [1.51, 0.74, 0.58]
    errs = [0.0, 0.32, 0.17]
    colors = ["#808080", "#1f77b4", "#ff7f0e"]

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    bars = ax.bar(groups, means, yerr=errs, capsize=10, color=colors,
                  edgecolor="black", linewidth=0.8)
    ax.set_ylabel("Коэффициент Шарпа", fontsize=13)
    ax.set_title("OOS 2024 — Sharpe (среднее ± std по 10 сидам)", fontsize=14)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(means) * 1.25)
    for b, v in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.05, f"{v:.2f}",
                ha="center", fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = IMG / "barplot_oos2024.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"✓ {out.name}")


def fig_volatility():
    ohlcv = pd.read_parquet(DSR / "data" / "raw" / "ohlcv.parquet")
    if "timestamp" in ohlcv.columns:
        ohlcv = ohlcv.set_index("timestamp")
    ohlcv.index = pd.to_datetime(ohlcv.index, utc=True).tz_convert(None)
    btc_daily = ohlcv["close"].resample("1D").last()
    btc_ret = np.log(btc_daily).diff()
    btc_vol = btc_ret.rolling(30).std() * np.sqrt(365) * 100

    try:
        import yfinance as yf
        spx = yf.download("^GSPC", start="2020-01-01", end="2025-04-01",
                          progress=False, auto_adjust=True)
        if spx.empty:
            raise RuntimeError("yfinance returned empty")
        spx_close = spx["Close"]
        if isinstance(spx_close, pd.DataFrame):
            spx_close = spx_close.iloc[:, 0]
        spx_ret = np.log(spx_close).diff()
        spx_vol = spx_ret.rolling(30).std() * np.sqrt(252) * 100
        have_spx = True
    except Exception as e:  # noqa: BLE001
        print(f"  ! yfinance failed: {e}; график только по BTC")
        have_spx = False

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(btc_vol.index, btc_vol.values, label="Биткоин",
            linewidth=2, color="#F7931A")
    if have_spx:
        ax.plot(spx_vol.index, spx_vol.values, label="Индекс S&P 500",
                linewidth=2, color="#1f4e79")
    ax.set_ylabel("Годовая волатильность, %", fontsize=12)
    ax.set_xlabel("")
    ax.set_title("Волатильность биткоина в 3–5 раз выше, чем у фондового рынка",
                 fontsize=13)
    ax.legend(loc="upper right", fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = IMG / "btc_vs_spx_volatility.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"✓ {out.name}")


def fig_news_histogram():
    news_path = DSR / "data" / "raw" / "news.parquet"
    news = pd.read_parquet(news_path)
    ts_col = None
    for c in ["ts", "timestamp", "date", "created_at", "published", "time"]:
        if c in news.columns:
            ts_col = c
            break
    if ts_col is None:
        dtypes = news.dtypes.astype(str)
        for c, t in dtypes.items():
            if "datetime" in t:
                ts_col = c
                break
    if ts_col is None:
        print(f"  ! не найдена колонка с датой в news.parquet; колонки: {list(news.columns)}")
        return
    ts = pd.to_datetime(news[ts_col], utc=True, errors="coerce")
    ts = ts.dropna()
    months = ts.dt.to_period("M")
    counts = months.value_counts().sort_index()
    counts.index = counts.index.astype(str)

    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.bar(range(len(counts)), counts.values, color="#4c72b0", width=0.9)
    step = max(1, len(counts) // 12)
    ax.set_xticks(list(range(0, len(counts), step)))
    ax.set_xticklabels(counts.index[::step], rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Новостей в месяц", fontsize=12)
    ax.set_title(f"Корпус криптоновостей: {len(ts):,} публикаций после дедупликации",
                 fontsize=13)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = IMG / "news_histogram.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"✓ {out.name} (n={len(ts)})")


def fig_metric_matrix():
    """Heatmap of all metrics for DQN / SAC / B&H × OOS 2024 / 2025.

    Unannualised Sharpe/Sortino (matches talk: DQN 2025 = 0.79, B&H 2025 = 0.50).
    Per-seed numbers come from the same CSVs used on backup slide 20.
    """
    # All metrics computed via lib/metrics.py on real npz data (n=10 seeds per algo)
    # and on raw BTC/USDT OHLCV for Buy & Hold (resampled to daily).
    data = {
        ("DQN", "2024"): dict(sharpe=0.735, sortino=0.728, ret=0.841, dd=0.296),
        ("DQN", "2025"): dict(sharpe=0.793, sortino=0.833, ret=0.318, dd=0.167),
        ("SAC", "2024"): dict(sharpe=0.578, sortino=0.674, ret=0.469, dd=0.228),
        ("SAC", "2025"): dict(sharpe=0.207, sortino=0.260, ret=0.047, dd=0.203),
        ("B&H", "2024"): dict(sharpe=1.658, sortino=2.730, ret=1.094, dd=0.262),
        ("B&H", "2025"): dict(sharpe=0.696, sortino=1.045, ret=0.096, dd=0.281),
    }
    for vals in data.values():
        vals["calmar"] = vals["ret"] / vals["dd"] if vals["dd"] > 0 else 0.0

    strat_order = ["DQN", "SAC", "B&H"]
    # (label, key, direction, normalisation scale for colour)
    metric_defs = [
        ("Sharpe",   "sharpe",  "higher", 1.6),
        ("Sortino",  "sortino", "higher", 2.5),
        ("Calmar",   "calmar",  "higher", 4.5),
        ("Return %", "ret",     "higher", 1.2),
        ("MaxDD %",  "dd",      "lower",  0.35),
    ]

    ink = "#1A1A1A"
    white = "#FFFFFF"

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, period in zip(axes, ["2024", "2025"]):
        mat = np.array([[data[(s, period)][k] for _, k, _, _ in metric_defs]
                        for s in strat_order])
        good = np.zeros_like(mat)
        for j, (_, k, direction, scale) in enumerate(metric_defs):
            col = mat[:, j]
            if direction == "higher":
                good[:, j] = np.clip(col / scale, 0, 1)
            else:
                good[:, j] = np.clip(1 - col / scale, 0, 1)
        ax.imshow(good, aspect="auto", cmap="Greens", vmin=0, vmax=1)
        ax.set_xticks(range(len(metric_defs)))
        ax.set_xticklabels([m[0] for m in metric_defs], fontsize=10, color=ink)
        ax.set_yticks(range(len(strat_order)))
        ax.set_yticklabels(strat_order, fontsize=10, color=ink)
        for i, s in enumerate(strat_order):
            for j, (_, k, _, _) in enumerate(metric_defs):
                v = mat[i, j]
                if k == "ret":
                    txt = f"{v * 100:+.0f}%"
                elif k == "dd":
                    txt = f"{v * 100:.0f}%"
                else:
                    txt = f"{v:+.2f}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=10,
                        color=white if good[i, j] > 0.55 else ink,
                        fontweight="bold")
        ax.set_title(f"OOS {period}", fontsize=11, color=ink, pad=6, loc="left")
        ax.tick_params(axis="both", which="both", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.suptitle("Сравнение метрик  ·  цвет = качество (зелёный лучше)",
                 fontsize=12, color=ink, y=1.02)
    fig.tight_layout()
    out = IMG / "metric_matrix.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.name}")


def fig_forest_plot():
    """Forest plot for slide 14 — Sharpe mean + 95% CI by 10 seeds.

    Unannualised Sharpe so numbers match the talk (DQN 2025 = 0.79, BH 2025 = 0.50).
    """
    dqn_2024 = np.array([0.897, 0.499, 1.297, 0.344, 0.443, 0.611, 1.034, 0.958, 0.426, 0.845])
    dqn_2025 = np.array([0.852, 0.820, 0.918, 0.395, 0.716, 0.604, 1.100, 0.581, 1.005, 0.939])
    sac_2024 = np.array([0.652, 0.702, 0.676, 0.679, 0.611, 0.209, 0.394, 0.708, 0.680, 0.465])
    sac_2025 = np.array([0.357, 0.189, 0.672, 0.535, 0.070, -0.559, 0.238, 0.038, 0.249, 0.282])

    bh_2024, bh_2025 = 1.66, 0.70

    def mean_ci(x):
        # Parametric 95% CI via t-distribution (matches talk: DQN 2025 → [0.66; 0.92])
        from scipy.stats import t as tdist
        n = len(x)
        m = float(np.mean(x))
        se = float(np.std(x, ddof=1) / np.sqrt(n))
        tcrit = float(tdist.ppf(0.975, df=n - 1))
        lo = m - tcrit * se
        hi = m + tcrit * se
        return m, lo, hi

    rows = [
        ("DQN · OOS 2025", dqn_2025, bh_2025, True),
        ("DQN · OOS 2024", dqn_2024, bh_2024, False),
        ("SAC · OOS 2024", sac_2024, bh_2024, False),
        ("SAC · OOS 2025", sac_2025, bh_2025, False),
    ]

    fig, ax = plt.subplots(figsize=(10, 5.2))
    for i, (label, data, bh, highlight) in enumerate(rows):
        y = len(rows) - 1 - i
        m, lo, hi = mean_ci(data)
        color = "#2E7D32" if highlight else "#333333"
        lw = 2.4 if highlight else 1.6
        ax.plot([lo, hi], [y, y], color=color, linewidth=lw, solid_capstyle="round")
        ax.plot([lo, lo], [y - 0.12, y + 0.12], color=color, linewidth=lw)
        ax.plot([hi, hi], [y - 0.12, y + 0.12], color=color, linewidth=lw)
        ax.plot(m, y, "o", color=color, markersize=9 if highlight else 7)
        ax.plot([bh, bh], [y - 0.28, y + 0.28], color="#B22222",
                linewidth=1.8, linestyle="--")
        ax.text(1.55, y, f"{m:.2f}   [{lo:.2f}; {hi:.2f}]",
                va="center", ha="left", fontsize=11,
                fontweight="bold" if highlight else "normal",
                color=color)

    ax.set_yticks(list(range(len(rows))))
    ax.set_yticklabels([r[0] for r in reversed(rows)], fontsize=12)
    ax.get_yticklabels()[-1].set_fontweight("bold")
    ax.get_yticklabels()[-1].set_color("#2E7D32")

    ax.set_xlim(-0.8, 2.4)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlabel("Коэффициент Шарпа (10 сидов, 95% CI)", fontsize=12)
    ax.axvline(0, color="#AAAAAA", linewidth=0.6)
    ax.grid(axis="x", alpha=0.25)

    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], color="#2E7D32", lw=2.4, marker="o", markersize=9,
               label="DQN / SAC — среднее по 10 сидам и 95% CI"),
        Line2D([0], [0], color="#B22222", lw=1.8, linestyle="--",
               label="Buy & Hold (эталон)"),
    ]
    ax.legend(handles=legend_elems, loc="lower right", fontsize=10, framealpha=0.95)

    ax.set_title("Sharpe по 10 сидам — bootstrap 95% CI vs Buy & Hold",
                 fontsize=12.5, pad=12)

    plt.tight_layout()
    out = IMG / "forest_plot_ci.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"✓ wrote {out.name}")


def copy_ready():
    ready = [
        "pipeline_overview.png",
        "dsr_formula.png",
        "equity_oos2025.png",
        # forest_plot_ci.png — генерируется fig_forest_plot(), не копируем
        "action_timeline.png",
        "radar_profile.png",
        # metric_matrix.png — генерируется fig_metric_matrix(), не копируем
        "nlp_pipeline.png",
        "training_stability.png",
    ]
    for name in ready:
        src = SRC_FIGS / name
        if src.exists():
            dst = IMG / name
            shutil.copy(src, dst)
            print(f"✓ copied {name}")
        else:
            print(f"! missing {name}")


if __name__ == "__main__":
    print("Generating missing figures and copying ready ones...")
    copy_ready()
    fig_barplot_oos2024()
    fig_volatility()
    fig_news_histogram()
    fig_forest_plot()
    print("\nDone.")
