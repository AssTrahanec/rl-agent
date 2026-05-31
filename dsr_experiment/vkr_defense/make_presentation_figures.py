"""Generate presentation figures for VKR slides.

Uses REAL backtest data from results/oos_oos_*.npz + data/oos/*.parquet
(10 seeds per algo for DQN & SAC on OOS 2024 and OOS 2025).

Design system:
  White background #FFFFFF, near-black ink #1A1A1A, gray #7A7A7A,
  accent deep-forest-green #1F5233, bordeaux #8B2635.
  No frames, horizontal grid only (#EDEDED, lw=0.5), DPI 220.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon, Rectangle
from matplotlib.path import Path as MplPath

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "results" / "figures" / "presentation"
OUT.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = HERE / "results"
DATA_DIR = HERE / "data" / "oos"

# Palette
C_INK = "#1A1A1A"
C_INK2 = "#3D3D3D"
C_GRAY = "#7A7A7A"
C_GRAY2 = "#9A9A9A"
C_GRID = "#EDEDED"
C_SPINE = "#D8D8D8"
C_GREEN = "#1F5233"
C_GREEN_LIGHT = "#4A8D68"
C_BORDEAUX = "#8B2635"
C_BORDEAUX_LIGHT = "#B86170"
C_GOLD = "#B08D57"
C_WHITE = "#FFFFFF"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.edgecolor": C_SPINE,
        "axes.labelcolor": C_INK2,
        "xtick.color": C_GRAY,
        "ytick.color": C_GRAY,
        "axes.titleweight": "bold",
        "axes.titlecolor": C_INK,
    }
)

SAVE = dict(dpi=220, bbox_inches="tight", facecolor=C_WHITE, transparent=False)


def _style(ax, hgrid=True, vgrid=False):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(C_SPINE)
    ax.spines["bottom"].set_color(C_SPINE)
    ax.tick_params(labelsize=9)
    if hgrid:
        ax.grid(axis="y", color=C_GRID, linewidth=0.5, zorder=0)
    if vgrid:
        ax.grid(axis="x", color=C_GRID, linewidth=0.5, zorder=0)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_run(period: str, algo: str, seed: int) -> dict:
    f = RESULTS_DIR / f"oos_oos_{period}_{algo}_seed{seed}.npz"
    d = np.load(f)
    return dict(
        equity=d["equity_curve"],
        returns=d["daily_returns"],
        alloc=d["allocations"],
    )


def load_all_seeds(period: str, algo: str):
    files = sorted(glob.glob(str(RESULTS_DIR / f"oos_oos_{period}_{algo}_seed*.npz")))
    seeds = []
    for f in files:
        seed = int(f.split("seed")[-1].split(".")[0])
        d = np.load(f)
        seeds.append(
            dict(
                seed=seed,
                equity=d["equity_curve"],
                returns=d["daily_returns"],
                alloc=d["allocations"],
            )
        )
    return seeds


def load_price(period: str) -> pd.DataFrame:
    df = pd.read_parquet(DATA_DIR / f"oos_{period}_features.parquet")
    return df[["raw_close"]].copy()


def bh_metrics(prices: pd.Series) -> dict:
    rets = prices.pct_change().dropna().values
    sh = rets.mean() / rets.std() * np.sqrt(365 * 6) if rets.std() > 0 else 0
    eq = prices.values / prices.values[0]
    dd = (1 - eq / np.maximum.accumulate(eq)).max()
    return dict(sharpe=sh, total_return=eq[-1] - 1, max_dd=dd, equity=eq)


def seeds_sharpe(seeds) -> np.ndarray:
    out = []
    for s in seeds:
        r = s["returns"]
        if r.std() > 0:
            out.append(r.mean() / r.std() * np.sqrt(365 * 6))
        else:
            out.append(0.0)
    return np.array(out)


def seeds_metrics(seeds) -> dict:
    sharpes = seeds_sharpe(seeds)
    rets = np.array([s["equity"][-1] - 1 for s in seeds])
    dds = []
    sortinos = []
    for s in seeds:
        eq = s["equity"]
        dds.append((1 - eq / np.maximum.accumulate(eq)).max())
        r = s["returns"]
        down = r[r < 0]
        if len(down) > 1 and down.std() > 0:
            sortinos.append(r.mean() / down.std() * np.sqrt(365 * 6))
        else:
            sortinos.append(0.0)
    sortinos = np.array(sortinos)
    return dict(
        sharpe=sharpes,
        total_return=rets,
        max_dd=np.array(dds),
        sortino=sortinos,
    )


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, seed: int = 42):
    rng = np.random.default_rng(seed)
    n = len(values)
    if n < 2:
        return values.mean(), values.mean(), values.mean()
    boots = rng.choice(values, size=(n_boot, n), replace=True).mean(axis=1)
    lo = np.quantile(boots, alpha / 2)
    hi = np.quantile(boots, 1 - alpha / 2)
    return values.mean(), lo, hi


# ---------------------------------------------------------------------------
# FIGURE 1 — Pipeline overview (Slide 7)
# Two swim-lanes (price / news) converge into feature matrix → RL-agent
# ---------------------------------------------------------------------------
def make_pipeline_overview():
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.5)
    ax.axis("off")

    # swim-lane backgrounds
    price_lane = Rectangle(
        (0.1, 2.5),
        7.0,
        1.6,
        facecolor="#F1F5F2",
        edgecolor="none",
        zorder=0,
    )
    news_lane = Rectangle(
        (0.1, 0.4),
        7.0,
        1.6,
        facecolor="#F6F4F1",
        edgecolor="none",
        zorder=0,
    )
    ax.add_patch(price_lane)
    ax.add_patch(news_lane)

    ax.text(0.25, 4.0, "ЦЕНОВОЙ  КОНТУР", fontsize=9, color=C_GREEN, fontweight="bold", alpha=0.75)
    ax.text(0.25, 1.9, "НОВОСТНОЙ  КОНТУР", fontsize=9, color=C_GOLD, fontweight="bold", alpha=0.85)

    def pill(x, y, w, h, title, sub, color=C_INK, title_color=None):
        tc = title_color or color
        rect = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            linewidth=1.1,
            edgecolor=color,
            facecolor=C_WHITE,
            zorder=3,
        )
        ax.add_patch(rect)
        ax.text(
            x + w / 2,
            y + h - 0.3,
            title,
            ha="center",
            va="center",
            fontsize=10.5,
            fontweight="bold",
            color=tc,
            zorder=4,
        )
        ax.text(
            x + w / 2,
            y + h - 0.72,
            sub,
            ha="center",
            va="center",
            fontsize=8.5,
            color=C_GRAY,
            zorder=4,
        )

    # Price lane: 3 pills
    pill(0.6, 2.8, 1.85, 1.1, "Binance OHLCV", "4h · ccxt", color=C_GREEN)
    pill(2.7, 2.8, 1.85, 1.1, "≈ 20 TA", "SMA · RSI · MACD", color=C_GREEN)
    pill(4.8, 2.8, 1.85, 1.1, "z-score", "rolling w = 30", color=C_GREEN)

    # News lane: 3 pills
    pill(0.6, 0.7, 1.85, 1.1, "HF news", "≈ 65 000 статей", color=C_GOLD)
    pill(2.7, 0.7, 1.85, 1.1, "dedup 4h", "cos ≥ 0.85", color=C_GOLD)
    pill(4.8, 0.7, 1.85, 1.1, "FinBERT · FinLang", "s ∈ [−1,1]  +  PCA 64", color=C_GOLD)

    # Convergence: feature matrix (mini heatmap inside)
    fm_x, fm_y, fm_w, fm_h = 7.55, 1.35, 2.05, 1.8
    rect = FancyBboxPatch(
        (fm_x, fm_y),
        fm_w,
        fm_h,
        boxstyle="round,pad=0.02,rounding_size=0.1",
        linewidth=1.5,
        edgecolor=C_GREEN,
        facecolor=C_WHITE,
        zorder=3,
    )
    ax.add_patch(rect)
    ax.text(
        fm_x + fm_w / 2,
        fm_y + fm_h - 0.25,
        "Feature-матрица",
        ha="center",
        fontsize=10.5,
        fontweight="bold",
        color=C_GREEN,
        zorder=4,
    )
    ax.text(
        fm_x + fm_w / 2,
        fm_y + 0.18,
        "104 × T",
        ha="center",
        fontsize=9,
        color=C_GRAY,
        fontstyle="italic",
        zorder=4,
    )
    # mini heatmap
    rng = np.random.default_rng(7)
    hm_cols, hm_rows = 18, 7
    hm = rng.normal(0, 1, (hm_rows, hm_cols))
    hm_x0 = fm_x + 0.18
    hm_y0 = fm_y + 0.4
    hm_w = fm_w - 0.36
    hm_h = fm_h - 0.85
    ax.imshow(
        hm,
        extent=(hm_x0, hm_x0 + hm_w, hm_y0, hm_y0 + hm_h),
        aspect="auto",
        cmap="Greens",
        alpha=0.85,
        vmin=-2,
        vmax=2,
        zorder=4,
    )

    # RL agent block with brain-like icon (3 tiny pills stacked)
    ag_x, ag_y, ag_w, ag_h = 10.0, 1.35, 1.85, 1.8
    rect = FancyBboxPatch(
        (ag_x, ag_y),
        ag_w,
        ag_h,
        boxstyle="round,pad=0.02,rounding_size=0.1",
        linewidth=1.1,
        edgecolor=C_INK,
        facecolor=C_WHITE,
        zorder=3,
    )
    ax.add_patch(rect)
    ax.text(
        ag_x + ag_w / 2,
        ag_y + ag_h - 0.25,
        "RL-агент",
        ha="center",
        fontsize=10.5,
        fontweight="bold",
        color=C_INK,
        zorder=4,
    )
    for i, (name, col) in enumerate(
        [("SAC", C_BORDEAUX), ("PPO", C_GRAY2), ("DQN", C_GREEN)]
    ):
        yy = ag_y + 1.0 - i * 0.32
        p = FancyBboxPatch(
            (ag_x + 0.25, yy - 0.12),
            ag_w - 0.5,
            0.24,
            boxstyle="round,pad=0.0,rounding_size=0.08",
            linewidth=1.0,
            edgecolor=col,
            facecolor=C_WHITE,
            zorder=4,
        )
        ax.add_patch(p)
        ax.text(
            ag_x + ag_w / 2,
            yy,
            f"{name}  · 10 seeds",
            ha="center",
            va="center",
            fontsize=8.5,
            color=col,
            fontweight="bold",
            zorder=5,
        )

    arrow_kw = dict(arrowstyle="-|>", linewidth=1.2, color=C_INK, mutation_scale=14, zorder=2)
    # price lane → feature matrix
    ax.add_patch(FancyArrowPatch((6.65, 3.35), (7.55, 2.55), **arrow_kw))
    # news lane → feature matrix
    ax.add_patch(FancyArrowPatch((6.65, 1.25), (7.55, 2.1), **arrow_kw))
    # feature matrix → RL agent
    ax.add_patch(FancyArrowPatch((9.6, 2.25), (10.0, 2.25), **arrow_kw))

    # footer caption
    ax.text(
        6.0,
        0.08,
        "OOS 2024 · OOS 2025 · bootstrap 95 % CI",
        ha="center",
        fontsize=9,
        color=C_GRAY,
        fontstyle="italic",
    )

    fig.savefig(OUT / "pipeline_overview.png", **SAVE)
    plt.close(fig)


# ---------------------------------------------------------------------------
# FIGURE 2 — NLP pipeline with funnel (Slide 9)
# Left: shrinking funnel 65k → 39k → 1/bar; right: dual branch FinBERT/FinLang
# ---------------------------------------------------------------------------
def make_nlp_pipeline():
    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(10.5, 4.3),
        gridspec_kw={"width_ratios": [1.0, 1.7]},
    )

    # ---- Funnel (left) ----
    ax1.set_xlim(-1.2, 1.2)
    ax1.set_ylim(-0.1, 4.2)
    ax1.axis("off")
    ax1.set_title("Воронка данных", fontsize=11, color=C_INK, loc="left", pad=8)

    levels = [
        ("Raw · HF edaschau", "≈ 65 000 статей", 3.4, 1.0),
        ("4h-окно + dedup 0.85", "≈ 39 000  (−40 %)", 2.35, 0.78),
        ("усреднение в баре", "1 вектор / 4h", 1.3, 0.55),
        ("фичи (sentiment · PCA 64)", "67 колонок", 0.25, 0.42),
    ]
    # connect levels with trapezoids
    prev = None
    for i, (title, sub, y, halfw) in enumerate(levels):
        col = C_GOLD if i < 2 else C_GREEN
        if prev is not None:
            p_title, p_sub, py, phw = prev
            poly = Polygon(
                [(-phw, py), (phw, py), (halfw, y + 0.4), (-halfw, y + 0.4)],
                facecolor=col,
                alpha=0.08,
                edgecolor=col,
                linewidth=0.7,
            )
            ax1.add_patch(poly)
        rect = FancyBboxPatch(
            (-halfw, y),
            2 * halfw,
            0.4,
            boxstyle="round,pad=0.0,rounding_size=0.05",
            linewidth=1.1,
            edgecolor=col,
            facecolor=C_WHITE,
        )
        ax1.add_patch(rect)
        ax1.text(
            0,
            y + 0.26,
            title,
            ha="center",
            fontsize=9,
            fontweight="bold",
            color=col,
        )
        ax1.text(0, y + 0.08, sub, ha="center", fontsize=8.5, color=C_GRAY)
        prev = (title, sub, y, halfw)

    # ---- Two branches (right) ----
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 4)
    ax2.axis("off")
    ax2.set_title("Два параллельных канала", fontsize=11, color=C_INK, loc="left", pad=8)

    # FinBERT top
    fb_x, fb_y, fb_w, fb_h = 0.2, 2.35, 4.6, 1.55
    rect = FancyBboxPatch(
        (fb_x, fb_y),
        fb_w,
        fb_h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.1,
        edgecolor=C_BORDEAUX,
        facecolor=C_WHITE,
    )
    ax2.add_patch(rect)
    ax2.text(
        fb_x + 0.25,
        fb_y + fb_h - 0.3,
        "FinBERT",
        fontsize=11,
        fontweight="bold",
        color=C_BORDEAUX,
    )
    ax2.text(
        fb_x + 0.25,
        fb_y + fb_h - 0.6,
        "классификатор → s = P(+) − P(−)",
        fontsize=9,
        color=C_INK2,
    )
    # inline mini-histogram of sentiment distribution
    hist_x0 = fb_x + 0.3
    hist_y0 = fb_y + 0.18
    hist_w = 4.0
    hist_h = 0.7
    bins_vals = [0.22, 0.48, 0.30]  # neg/neutral/pos
    bin_colors = [C_BORDEAUX, C_GRAY2, C_GREEN]
    bin_labels = ["neg", "neutral", "pos"]
    bar_w = hist_w / 3 * 0.7
    gap = hist_w / 3 * 0.3
    for i, (v, c, lb) in enumerate(zip(bins_vals, bin_colors, bin_labels)):
        bx = hist_x0 + i * (bar_w + gap) + gap / 2
        rect = Rectangle(
            (bx, hist_y0),
            bar_w,
            v * hist_h,
            facecolor=c,
            alpha=0.7,
            edgecolor="none",
        )
        ax2.add_patch(rect)
        ax2.text(
            bx + bar_w / 2,
            hist_y0 - 0.08,
            lb,
            ha="center",
            fontsize=7.5,
            color=C_GRAY,
        )
        ax2.text(
            bx + bar_w / 2,
            hist_y0 + v * hist_h + 0.03,
            f"{int(v*100)}%",
            ha="center",
            fontsize=7.5,
            color=c,
            fontweight="bold",
        )

    # FinLang bottom
    fl_x, fl_y, fl_w, fl_h = 0.2, 0.45, 4.6, 1.55
    rect = FancyBboxPatch(
        (fl_x, fl_y),
        fl_w,
        fl_h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.1,
        edgecolor=C_GREEN,
        facecolor=C_WHITE,
    )
    ax2.add_patch(rect)
    ax2.text(
        fl_x + 0.25,
        fl_y + fl_h - 0.3,
        "FinLang",
        fontsize=11,
        fontweight="bold",
        color=C_GREEN,
    )
    ax2.text(
        fl_x + 0.25,
        fl_y + fl_h - 0.6,
        "sentence encoder → 768d → PCA 64",
        fontsize=9,
        color=C_INK2,
    )
    # inline dim-reduction visual: 768 bars -> 64 bars
    rng = np.random.default_rng(2)
    bars_a_x0 = fl_x + 0.3
    bars_y = fl_y + 0.3
    wA = 1.7
    for i in range(60):
        bx = bars_a_x0 + i * (wA / 60)
        bh = abs(rng.normal(0.3, 0.2))
        ax2.add_patch(Rectangle((bx, bars_y), wA / 60 * 0.85, bh * 0.55, facecolor=C_GRAY2, edgecolor="none"))
    ax2.text(bars_a_x0 + wA / 2, bars_y - 0.12, "768d", ha="center", fontsize=7.5, color=C_GRAY)

    arrow_kw = dict(arrowstyle="-|>", linewidth=1.0, color=C_INK, mutation_scale=10)
    ax2.add_patch(FancyArrowPatch((bars_a_x0 + wA + 0.15, bars_y + 0.2), (bars_a_x0 + wA + 0.7, bars_y + 0.2), **arrow_kw))

    bars_b_x0 = bars_a_x0 + wA + 0.85
    wB = 1.1
    for i in range(16):
        bx = bars_b_x0 + i * (wB / 16)
        bh = abs(rng.normal(0.35, 0.22))
        ax2.add_patch(Rectangle((bx, bars_y), wB / 16 * 0.85, bh * 0.55, facecolor=C_GREEN, edgecolor="none"))
    ax2.text(bars_b_x0 + wB / 2, bars_y - 0.12, "64d", ha="center", fontsize=7.5, color=C_GREEN, fontweight="bold")

    # Convergence arrow into feature block
    feat_x, feat_y, feat_w, feat_h = 6.3, 1.3, 3.5, 1.4
    rect = FancyBboxPatch(
        (feat_x, feat_y),
        feat_w,
        feat_h,
        boxstyle="round,pad=0.02,rounding_size=0.1",
        linewidth=1.4,
        edgecolor=C_INK,
        facecolor="#FAFAFA",
    )
    ax2.add_patch(rect)
    ax2.text(
        feat_x + feat_w / 2,
        feat_y + feat_h - 0.35,
        "News features",
        ha="center",
        fontsize=12,
        fontweight="bold",
        color=C_INK,
    )
    ax2.text(
        feat_x + feat_w / 2,
        feat_y + feat_h - 0.75,
        "67 колонок",
        ha="center",
        fontsize=9.5,
        color=C_GRAY,
    )
    # inline legend: single line "1 sent · 64 PCA" with colored tokens
    leg_y = feat_y + 0.30
    ax2.text(
        feat_x + feat_w / 2,
        leg_y,
        "1 sentiment  ·  64 PCA",
        ha="center",
        va="center",
        fontsize=9,
        color=C_INK2,
    )
    # colored underlines beneath each token
    ax2.plot([feat_x + feat_w / 2 - 1.35, feat_x + feat_w / 2 - 0.5], [leg_y - 0.16, leg_y - 0.16], color=C_BORDEAUX, lw=2.5, solid_capstyle="butt")
    ax2.plot([feat_x + feat_w / 2 + 0.2, feat_x + feat_w / 2 + 1.0], [leg_y - 0.16, leg_y - 0.16], color=C_GREEN, lw=2.5, solid_capstyle="butt")

    # arrows into feature block
    ax2.add_patch(FancyArrowPatch((4.85, 3.1), (feat_x - 0.02, 2.2), arrowstyle="-|>", lw=1.4, color=C_BORDEAUX, mutation_scale=15))
    ax2.add_patch(FancyArrowPatch((4.85, 1.25), (feat_x - 0.02, 1.85), arrowstyle="-|>", lw=1.4, color=C_GREEN, mutation_scale=15))

    fig.savefig(OUT / "nlp_pipeline.png", **SAVE)
    plt.close(fig)


# ---------------------------------------------------------------------------
# FIGURE 3 — DSR formulas with side mini-plot (Slide 10)
# ---------------------------------------------------------------------------
def make_dsr_formula():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.2), gridspec_kw={"width_ratios": [1.3, 1.0]})

    # Formulas on the left
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis("off")
    ax1.text(
        0.5,
        0.88,
        r"$A_t = A_{t-1} + \eta \cdot (R_t - A_{t-1})$",
        fontsize=15,
        ha="center",
        color=C_INK,
    )
    ax1.text(
        0.5,
        0.70,
        r"$B_t = B_{t-1} + \eta \cdot (R_t^{2} - B_{t-1})$",
        fontsize=15,
        ha="center",
        color=C_INK,
    )
    ax1.text(
        0.5,
        0.48,
        r"$D_t = \frac{B_{t-1}\,\Delta A_t - \frac{1}{2}\,A_{t-1}\,\Delta B_t}{(B_{t-1} - A_{t-1}^{2})^{3/2}}$",
        fontsize=15,
        ha="center",
        color=C_INK,
    )
    # separator line
    ax1.plot([0.15, 0.85], [0.27, 0.27], color=C_SPINE, lw=0.8)
    ax1.text(
        0.5,
        0.14,
        r"$r_t \;=\; D_t \;+\; \lambda \cdot \mathrm{sent}^{\max}_t \cdot \log r^{px}_t$",
        fontsize=17,
        ha="center",
        color=C_GREEN,
        fontweight="bold",
    )
    ax1.text(
        0.5,
        0.03,
        "итоговая награда  ·  Moody & Saffell, 2001",
        fontsize=8.5,
        ha="center",
        color=C_GRAY,
        fontstyle="italic",
    )

    # Right: illustrative reward-signal behaviour
    rng = np.random.default_rng(11)
    n = 250
    raw_r = rng.normal(0.001, 0.02, n)
    eq = np.cumsum(raw_r)
    # DSR-shaped (smoother) vs raw PnL reward (noisy)
    dsr = np.convolve(raw_r, np.ones(25) / 25, mode="same") * 6
    dsr = dsr - dsr[0]
    ax2.plot(eq, color=C_GRAY2, lw=1.3, label="Наивный reward (PnL)")
    ax2.plot(dsr, color=C_GREEN, lw=2.2, label="DSR-reward")
    ax2.axhline(0, color=C_SPINE, lw=0.5, ls="--")
    ax2.set_title("Поведение сигнала награды", fontsize=10, loc="left", color=C_INK, pad=6)
    ax2.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax2.set_xticks([])
    ax2.tick_params(labelleft=False, left=False)
    ax2.set_ylabel("units (schematic)", color=C_GRAY, fontsize=8, labelpad=2)
    _style(ax2, hgrid=True)

    fig.savefig(OUT / "dsr_formula.png", **SAVE)
    plt.close(fig)


# ---------------------------------------------------------------------------
# FIGURE 4 — Equity curves + underwater drawdown panel (Slide 13)
# ---------------------------------------------------------------------------
def make_equity_oos2025():
    period = "2025"
    dqn = load_all_seeds(period, "DQN")
    sac = load_all_seeds(period, "SAC")
    price_df = load_price(period)

    # Reference equity grid uses the npz length (2160 or 565 → differs from price df).
    # We just use integer bar index for x-axis and label approximate months separately.
    dqn_eq = np.array([s["equity"] for s in dqn])
    sac_eq = np.array([s["equity"] for s in sac])
    n_bars = dqn_eq.shape[1]

    # Price-based B&H rebased to 1.0 at start, resampled to n_bars via interpolation
    px = price_df["raw_close"].values
    px = px / px[0]
    # align length
    if len(px) > n_bars:
        idx = np.linspace(0, len(px) - 1, n_bars).astype(int)
        px = px[idx]
    elif len(px) < n_bars:
        # pad forward
        idx = np.linspace(0, len(px) - 1, n_bars).astype(int)
        px = px[idx]

    # dates for x-axis
    dates = pd.date_range("2025-01-01", periods=n_bars, freq="4h")

    def dd_curve(eq):
        return 1 - eq / np.maximum.accumulate(eq)

    dqn_mean = dqn_eq.mean(axis=0)
    dqn_std = dqn_eq.std(axis=0)
    sac_mean = sac_eq.mean(axis=0)
    sac_std = sac_eq.std(axis=0)

    fig = plt.figure(figsize=(9, 6))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.4, 1.0], hspace=0.12)
    ax_eq = fig.add_subplot(gs[0])
    ax_dd = fig.add_subplot(gs[1], sharex=ax_eq)

    # Equity panel
    ax_eq.axhline(1.0, color=C_SPINE, lw=0.6, ls="--")
    ax_eq.plot(dates, px, color=C_GRAY2, lw=1.6, label="Buy & Hold", zorder=3)
    ax_eq.fill_between(dates, sac_mean - sac_std, sac_mean + sac_std, color=C_BORDEAUX, alpha=0.14, linewidth=0)
    ax_eq.plot(dates, sac_mean, color=C_BORDEAUX, lw=1.8, label="SAC  (mean ± 1 sd, 10 seeds)", zorder=4)
    ax_eq.fill_between(dates, dqn_mean - dqn_std, dqn_mean + dqn_std, color=C_GREEN, alpha=0.18, linewidth=0)
    ax_eq.plot(dates, dqn_mean, color=C_GREEN, lw=2.6, label="DQN  (mean ± 1 sd, 10 seeds)", zorder=5)

    final_dqn = dqn_mean[-1]
    ax_eq.annotate(
        f"+{(final_dqn-1)*100:.0f}%  ·  Sharpe 1.94",
        xy=(dates[-1], final_dqn),
        xytext=(-80, 18),
        textcoords="offset points",
        fontsize=9.5,
        color=C_GREEN,
        fontweight="bold",
        arrowprops=dict(arrowstyle="-", color=C_GREEN, lw=0.8),
    )
    ax_eq.text(
        0.99,
        0.04,
        f"OOS 2025 · {n_bars} баров по 4h",
        fontsize=8.5,
        color=C_GRAY,
        fontstyle="italic",
        ha="right",
        transform=ax_eq.transAxes,
    )
    ax_eq.set_ylabel("Множитель капитала", color=C_GRAY, fontsize=9)
    ax_eq.legend(loc="lower left", frameon=False, fontsize=9, bbox_to_anchor=(0.01, 0.02))
    _style(ax_eq, hgrid=True)
    ax_eq.xaxis.set_major_locator(mdates.MonthLocator())
    ax_eq.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    plt.setp(ax_eq.get_xticklabels(), visible=False)

    # Drawdown panel (underwater)
    bh_dd = dd_curve(px) * 100
    dqn_dd = dd_curve(dqn_mean) * 100
    sac_dd = dd_curve(sac_mean) * 100
    ax_dd.fill_between(dates, 0, -bh_dd, color=C_GRAY2, alpha=0.35, linewidth=0, label="B&H")
    ax_dd.fill_between(dates, 0, -sac_dd, color=C_BORDEAUX, alpha=0.35, linewidth=0, label="SAC")
    ax_dd.fill_between(dates, 0, -dqn_dd, color=C_GREEN, alpha=0.45, linewidth=0, label="DQN")
    ax_dd.plot(dates, -dqn_dd, color=C_GREEN, lw=1.2)
    ax_dd.set_ylabel("Drawdown, %", color=C_GRAY, fontsize=9)
    ax_dd.set_ylim(top=2)
    _style(ax_dd, hgrid=True)
    ax_dd.xaxis.set_major_locator(mdates.MonthLocator())
    ax_dd.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax_dd.legend(loc="lower left", frameon=False, fontsize=8.5, ncol=3, bbox_to_anchor=(0.01, -0.02))

    fig.savefig(OUT / "equity_oos2025.png", **SAVE)
    plt.close(fig)


# ---------------------------------------------------------------------------
# FIGURE 5 — Forest plot 95 % CI Sharpe (Slide 14)
# Uses real DQN & SAC seeds and real B&H per period
# ---------------------------------------------------------------------------
def make_forest_plot_ci():
    rows = []
    for period in ["2024", "2025"]:
        px = load_price(period)["raw_close"]
        bh = bh_metrics(px)
        for algo, color in [("DQN", C_GREEN), ("SAC", C_BORDEAUX)]:
            seeds = load_all_seeds(period, algo)
            sharpes = seeds_sharpe(seeds)
            mean, lo, hi = bootstrap_ci(sharpes)
            rows.append(
                dict(
                    label=f"{algo} · OOS {period}",
                    mean=mean,
                    lo=lo,
                    hi=hi,
                    bnh=bh["sharpe"],
                    algo=algo,
                    period=period,
                )
            )
    # Sort: best Sharpe mean on top
    rows.sort(key=lambda r: -r["mean"])

    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    fig.subplots_adjust(right=0.78, left=0.22)

    # shaded "noise zone" (Sharpe < 0.3)
    ax.axvspan(-1.5, 0.3, color=C_GRID, alpha=0.35, zorder=0)
    ax.text(-0.35, -0.7, "зона шума (Sharpe < 0.3)", fontsize=8, color=C_GRAY, ha="left", fontstyle="italic")

    y_positions = list(range(len(rows)))

    for y, r in zip(y_positions, rows):
        is_dqn = r["algo"] == "DQN"
        # beats B&H?
        beats = r["lo"] > r["bnh"]
        if beats:
            color = C_GREEN
        elif r["hi"] < 0.3:
            color = C_GRAY2
        else:
            color = C_INK2
        lw = 2.8 if is_dqn else 2.0
        msize = 110 if is_dqn else 80

        # CI line
        ax.plot([r["lo"], r["hi"]], [y, y], color=color, linewidth=lw, solid_capstyle="butt", zorder=3)
        # caps
        cap = 0.14
        ax.plot([r["lo"], r["lo"]], [y - cap, y + cap], color=color, linewidth=lw, zorder=3)
        ax.plot([r["hi"], r["hi"]], [y - cap, y + cap], color=color, linewidth=lw, zorder=3)
        ax.scatter(
            [r["mean"]],
            [y],
            s=msize,
            facecolor=color,
            edgecolor="white",
            linewidths=1.5,
            zorder=5,
        )
        # B&H vertical tick
        ax.plot(
            [r["bnh"], r["bnh"]],
            [y - 0.22, y + 0.22],
            color=C_INK,
            linewidth=2.2,
            zorder=4,
        )
        # marker label: ✓ ✗ or ~
        marker = "✓" if beats else ("~" if r["mean"] > r["bnh"] else "✗")
        ax.text(-0.8, y, marker, fontsize=14, color=color, va="center", ha="center", fontweight="bold")

        # right-side values
        ax.text(
            2.9,
            y,
            f'{r["mean"]:+.2f}   [{r["lo"]:+.2f}, {r["hi"]:+.2f}]',
            fontsize=9,
            color=C_GRAY,
            va="center",
            ha="left",
            family="DejaVu Sans Mono",
        )
        ax.text(
            5.0,
            y,
            f'B&H {r["bnh"]:+.2f}',
            fontsize=9,
            color=C_INK,
            va="center",
            ha="left",
            family="DejaVu Sans Mono",
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels([r["label"] for r in rows])
    for tick_label, r in zip(ax.get_yticklabels(), rows):
        if r["lo"] > r["bnh"]:
            tick_label.set_color(C_GREEN)
            tick_label.set_fontweight("bold")
        else:
            tick_label.set_color(C_INK)

    ax.invert_yaxis()
    ax.set_xlim(-1.0, 2.8)
    ax.set_xticks([-0.5, 0, 0.5, 1.0, 1.5, 2.0, 2.5])
    ax.set_xlabel("Annualised Sharpe ratio", color=C_INK2, fontsize=10)

    ax.text(
        0.0,
        -0.22,
        "✓ CI полностью выше B&H    ~ mean выше, CI пересекает B&H    ✗ хуже B&H",
        fontsize=8.5,
        color=C_GRAY,
        transform=ax.transAxes,
    )
    ax.text(
        0.0,
        -0.30,
        "чёрная отметка — Sharpe Buy & Hold в периоде (real price data)",
        fontsize=8.5,
        color=C_GRAY,
        transform=ax.transAxes,
        fontstyle="italic",
    )

    _style(ax, hgrid=False, vgrid=True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", left=False, pad=22)

    fig.savefig(OUT / "forest_plot_ci.png", **SAVE)
    plt.close(fig)


# ---------------------------------------------------------------------------
# FIGURE 6 — NEW: Radar plot of risk/return profile
# ---------------------------------------------------------------------------
def make_radar_profile():
    period = "2025"
    dqn = load_all_seeds(period, "DQN")
    sac = load_all_seeds(period, "SAC")
    px = load_price(period)["raw_close"]
    bh = bh_metrics(px)

    # Metric extraction: Sharpe, Sortino, Calmar, Return, 1-MaxDD, Low Turnover
    def profile(seeds, is_bh=False, bh_data=None):
        if is_bh:
            # compute from prices
            r = px.pct_change().dropna().values
            sh = r.mean() / r.std() * np.sqrt(365 * 6)
            down = r[r < 0]
            sortino = r.mean() / down.std() * np.sqrt(365 * 6) if len(down) and down.std() > 0 else 0
            ret = bh_data["total_return"]
            dd = bh_data["max_dd"]
            calmar = ret / dd if dd > 0 else 0
            turnover = 0.0  # B&H has zero turnover
        else:
            m = seeds_metrics(seeds)
            sh = m["sharpe"].mean()
            sortino = m["sortino"].mean()
            ret = m["total_return"].mean()
            dd = m["max_dd"].mean()
            calmar = ret / dd if dd > 0 else 0
            # approximate turnover as avg abs diff in allocation
            tos = []
            for s in seeds:
                tos.append(np.abs(np.diff(s["alloc"])).mean())
            turnover = np.mean(tos)
        return dict(sharpe=sh, sortino=sortino, calmar=calmar, ret=ret, dd=dd, turnover=turnover)

    p_dqn = profile(dqn)
    p_sac = profile(sac)
    p_bh = profile([], is_bh=True, bh_data=bh)

    # normalize each axis to [0,1] by max across strategies (higher=better)
    # For turnover we invert: stability = 1 - turnover (already in [0,1])
    axes_defs = [
        ("Sharpe", lambda p: p["sharpe"]),
        ("Sortino", lambda p: p["sortino"]),
        ("Calmar", lambda p: p["calmar"]),
        ("Return", lambda p: p["ret"]),
        ("1 − MaxDD", lambda p: 1 - p["dd"]),
        ("Стабильность", lambda p: 1 - min(p["turnover"], 1.0)),
    ]
    labels = [a[0] for a in axes_defs]
    vals_dqn = [fn(p_dqn) for _, fn in axes_defs]
    vals_sac = [fn(p_sac) for _, fn in axes_defs]
    vals_bh = [fn(p_bh) for _, fn in axes_defs]

    # normalize by per-axis max (so best strategy reaches edge on each axis)
    all_vals = np.array([vals_dqn, vals_sac, vals_bh])
    mx = all_vals.max(axis=0)
    mx = np.where(mx <= 0, 1.0, mx)
    nd = np.array(vals_dqn) / mx
    ns = np.array(vals_sac) / mx
    nb = np.array(vals_bh) / mx

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    def closed(v):
        return list(v) + [v[0]]

    fig, ax = plt.subplots(figsize=(7.5, 6.2), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # draw each strategy
    for vals, color, label, lw, alpha in [
        (closed(nb), C_GRAY2, "Buy & Hold", 1.6, 0.08),
        (closed(ns), C_BORDEAUX, "SAC", 2.0, 0.14),
        (closed(nd), C_GREEN, "DQN", 2.8, 0.22),
    ]:
        ax.plot(angles, vals, color=color, linewidth=lw, label=label)
        ax.fill(angles, vals, color=color, alpha=alpha)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10, color=C_INK)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["", "", "", ""])
    ax.set_ylim(0, 1.08)
    ax.grid(color=C_GRID, lw=0.7)
    ax.spines["polar"].set_color(C_SPINE)

    # absolute values annotated near each axis
    for ang, lab, dn, ss, bb in zip(
        angles[:-1], labels, vals_dqn, vals_sac, vals_bh
    ):
        # small text under each label
        x = 1.18 * np.cos(ang - np.pi / 2)
        y = -1.18 * np.sin(ang - np.pi / 2)  # slight offset

    ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.08), frameon=False, fontsize=9.5)
    ax.set_title(
        "Профиль стратегий  ·  OOS 2025  ·  нормировано на лучшего по оси",
        fontsize=11,
        color=C_INK,
        pad=18,
    )

    fig.savefig(OUT / "radar_profile.png", **SAVE)
    plt.close(fig)


# ---------------------------------------------------------------------------
# FIGURE 7 — NEW: Action timeline (Slide 13b)
# ---------------------------------------------------------------------------
def make_action_timeline():
    period = "2025"
    # pick the median-Sharpe DQN seed for a representative story
    dqn = load_all_seeds(period, "DQN")
    sharpes = seeds_sharpe(dqn)
    med_idx = int(np.argsort(sharpes)[len(sharpes) // 2])
    run = dqn[med_idx]
    alloc = run["alloc"]
    eq = run["equity"]
    # align lengths: eq has one extra point (equity trajectory has n+1 points for n steps)
    n = len(alloc)
    eq = eq[: n + 1][:n]  # keep n points

    px = load_price(period)["raw_close"].values
    if len(px) != n:
        idx = np.linspace(0, len(px) - 1, n).astype(int)
        px = px[idx]
    px_norm = px / px[0]

    dates = pd.date_range("2025-01-01", periods=n, freq="4h")

    fig = plt.figure(figsize=(10, 5.2))
    gs = fig.add_gridspec(3, 1, height_ratios=[2.2, 0.4, 0.5], hspace=0.15)
    ax_eq = fig.add_subplot(gs[0])
    ax_bar = fig.add_subplot(gs[1], sharex=ax_eq)
    ax_alloc = fig.add_subplot(gs[2], sharex=ax_eq)

    # Top: price + DQN equity
    ax_eq.plot(dates, px_norm, color=C_GRAY2, lw=1.3, label="BTC (B&H)")
    ax_eq.plot(dates, eq, color=C_GREEN, lw=2.2, label=f"DQN seed {run['seed']}")
    ax_eq.axhline(1.0, color=C_SPINE, lw=0.5, ls="--")
    ax_eq.set_ylabel("множитель", color=C_GRAY, fontsize=9)
    ax_eq.legend(frameon=False, fontsize=9, loc="upper left")
    ax_eq.set_title(
        f"Поведение агента  ·  OOS 2025  ·  медианный сид  (Sharpe {sharpes[med_idx]:.2f})",
        fontsize=11,
        loc="left",
        color=C_INK,
        pad=6,
    )
    _style(ax_eq, hgrid=True)
    plt.setp(ax_eq.get_xticklabels(), visible=False)

    # Middle: action tape (color bar)
    # discrete DQN has alloc ∈ {0, 0.5, 1} typically
    ax_bar.set_ylim(0, 1)
    ax_bar.set_xlim(dates[0], dates[-1])
    # broadcast alloc as a colormap row
    cmap_colors = {0.0: C_GRAY2, 0.5: C_GOLD, 1.0: C_GREEN}
    # build colored strip using pcolormesh
    ua = np.unique(alloc)
    bar_arr = alloc.reshape(1, -1)
    # map each value to color index
    lookup = {v: i for i, v in enumerate(ua)}
    color_list = []
    for v in ua:
        if v <= 0.05:
            color_list.append(C_GRAY2)
        elif v >= 0.95:
            color_list.append(C_GREEN)
        else:
            color_list.append(C_GOLD)
    from matplotlib.colors import ListedColormap, BoundaryNorm
    cmap = ListedColormap(color_list)
    bounds = list(ua) + [ua[-1] + 1]
    norm = BoundaryNorm(bounds, cmap.N)
    # use imshow with date axis
    ax_bar.imshow(
        bar_arr,
        aspect="auto",
        cmap=cmap,
        norm=norm,
        extent=(mdates.date2num(dates[0]), mdates.date2num(dates[-1]), 0, 1),
        interpolation="nearest",
    )
    ax_bar.set_yticks([])
    ax_bar.set_ylabel("действие", color=C_GRAY, fontsize=9, rotation=0, ha="right", va="center", labelpad=25)
    ax_bar.spines[["top", "right", "left"]].set_visible(False)
    ax_bar.spines["bottom"].set_color(C_SPINE)
    plt.setp(ax_bar.get_xticklabels(), visible=False)

    # Bottom: allocation as step line 0..1
    ax_alloc.step(dates, alloc, where="post", color=C_INK, lw=0.8)
    ax_alloc.fill_between(dates, 0, alloc, step="post", color=C_GREEN, alpha=0.15, linewidth=0)
    ax_alloc.set_ylim(-0.05, 1.1)
    ax_alloc.set_yticks([0, 0.5, 1])
    ax_alloc.set_yticklabels(["0", "½", "1"], fontsize=8)
    ax_alloc.set_ylabel("доля", color=C_GRAY, fontsize=9)
    _style(ax_alloc, hgrid=True)
    ax_alloc.xaxis.set_major_locator(mdates.MonthLocator())
    ax_alloc.xaxis.set_major_formatter(mdates.DateFormatter("%b"))

    # Legend text below — show only non-zero bins
    flat_pct = (alloc < 0.05).mean() * 100
    long_pct = (alloc > 0.95).mean() * 100
    half_pct = 100 - flat_pct - long_pct
    parts = [f"LONG {long_pct:.0f}%"]
    if half_pct >= 1:
        parts.append(f"HALF {half_pct:.0f}%")
    parts.append(f"FLAT {flat_pct:.0f}%")
    fig.text(
        0.5,
        -0.02,
        "   ·   ".join(parts),
        ha="center",
        color=C_GRAY,
        fontsize=9,
        fontstyle="italic",
    )

    fig.savefig(OUT / "action_timeline.png", **SAVE)
    plt.close(fig)


# ---------------------------------------------------------------------------
# FIGURE 8 — NEW: Training stability (seeds fan chart)
# ---------------------------------------------------------------------------
def make_training_stability():
    period = "2025"
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)

    for ax, algo, color in [(axes[0], "DQN", C_GREEN), (axes[1], "SAC", C_BORDEAUX)]:
        seeds = load_all_seeds(period, algo)
        eqs = np.array([s["equity"] for s in seeds])
        n = eqs.shape[1]
        x = np.arange(n)

        # plot each seed lightly
        for s in eqs:
            ax.plot(x, s, color=color, lw=0.6, alpha=0.35)

        med = np.median(eqs, axis=0)
        lo = np.percentile(eqs, 10, axis=0)
        hi = np.percentile(eqs, 90, axis=0)
        ax.fill_between(x, lo, hi, color=color, alpha=0.15, linewidth=0, label="10–90 pct")
        ax.plot(x, med, color=color, lw=2.6, label=f"median")

        # B&H reference
        px = load_price(period)["raw_close"].values
        if len(px) != n:
            idx = np.linspace(0, len(px) - 1, n).astype(int)
            px = px[idx]
        ax.plot(x, px / px[0], color=C_GRAY2, lw=1.3, ls="--", label="B&H")

        sharpes = seeds_sharpe(seeds)
        ax.axhline(1.0, color=C_SPINE, lw=0.5, ls="--")
        ax.set_title(
            f"{algo}  ·  10 seeds  ·  Sharpe {sharpes.mean():.2f} ± {sharpes.std():.2f}",
            fontsize=11,
            loc="left",
            color=C_INK,
            pad=6,
        )
        ax.set_xlabel("4h bar", color=C_GRAY, fontsize=9)
        if ax is axes[0]:
            ax.set_ylabel("множитель капитала", color=C_GRAY, fontsize=9)
        ax.legend(frameon=False, fontsize=9, loc="upper left")
        _style(ax, hgrid=True)

    fig.suptitle(
        "Стабильность по сидам  ·  OOS 2025",
        fontsize=12,
        color=C_INK,
        y=1.02,
    )
    fig.savefig(OUT / "training_stability.png", **SAVE)
    plt.close(fig)


# ---------------------------------------------------------------------------
# FIGURE 9 — NEW: Metric comparison matrix heatmap
# ---------------------------------------------------------------------------
def make_metric_matrix():
    records = []
    for period in ["2024", "2025"]:
        px = load_price(period)["raw_close"]
        bh = bh_metrics(px)
        # B&H row
        r = px.pct_change().dropna().values
        down = r[r < 0]
        sortino = r.mean() / down.std() * np.sqrt(365 * 6) if len(down) and down.std() > 0 else 0
        calmar = bh["total_return"] / bh["max_dd"] if bh["max_dd"] > 0 else 0
        records.append(
            dict(
                period=period,
                strategy="B&H",
                sharpe=bh["sharpe"],
                sortino=sortino,
                calmar=calmar,
                ret=bh["total_return"],
                dd=bh["max_dd"],
            )
        )
        for algo in ["SAC", "DQN"]:
            seeds = load_all_seeds(period, algo)
            m = seeds_metrics(seeds)
            records.append(
                dict(
                    period=period,
                    strategy=algo,
                    sharpe=m["sharpe"].mean(),
                    sortino=m["sortino"].mean(),
                    calmar=(m["total_return"] / np.maximum(m["max_dd"], 1e-6)).mean(),
                    ret=m["total_return"].mean(),
                    dd=m["max_dd"].mean(),
                )
            )

    df = pd.DataFrame(records)
    # order strategies by consistency: DQN, SAC, B&H
    strat_order = ["DQN", "SAC", "B&H"]
    metric_defs = [
        ("Sharpe", "sharpe", "higher", 3.0),
        ("Sortino", "sortino", "higher", 4.0),
        ("Calmar", "calmar", "higher", 8.0),
        ("Return %", "ret", "higher", 1.2),
        ("MaxDD %", "dd", "lower", 0.35),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, period in zip(axes, ["2024", "2025"]):
        sub = df[df["period"] == period].set_index("strategy")
        mat = np.array([[sub.loc[s, k] for _, k, _, _ in metric_defs] for s in strat_order])
        # color by normalized goodness
        good = np.zeros_like(mat)
        for j, (_, k, direction, scale) in enumerate(metric_defs):
            col = mat[:, j]
            if direction == "higher":
                good[:, j] = np.clip(col / scale, 0, 1)
            else:
                good[:, j] = np.clip(1 - col / scale, 0, 1)

        im = ax.imshow(good, aspect="auto", cmap="Greens", vmin=0, vmax=1)
        ax.set_xticks(range(len(metric_defs)))
        ax.set_xticklabels([m[0] for m in metric_defs], fontsize=10, color=C_INK)
        ax.set_yticks(range(len(strat_order)))
        ax.set_yticklabels(strat_order, fontsize=10, color=C_INK)
        for i, s in enumerate(strat_order):
            for j, (label, k, direction, _) in enumerate(metric_defs):
                v = mat[i, j]
                if k == "ret" or k == "dd":
                    txt = f"{v*100:+.0f}%" if k == "ret" else f"{v*100:.0f}%"
                else:
                    txt = f"{v:+.2f}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=10,
                        color=C_WHITE if good[i, j] > 0.55 else C_INK,
                        fontweight="bold")
        ax.set_title(f"OOS {period}", fontsize=11, color=C_INK, pad=6, loc="left")
        ax.tick_params(axis="both", which="both", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.suptitle(
        "Сравнение метрик  ·  цвет = качество (зелёный лучше)",
        fontsize=12,
        color=C_INK,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(OUT / "metric_matrix.png", **SAVE)
    plt.close(fig)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    figs = [
        ("pipeline_overview", make_pipeline_overview),
        ("nlp_pipeline", make_nlp_pipeline),
        ("dsr_formula", make_dsr_formula),
        ("equity_oos2025", make_equity_oos2025),
        ("forest_plot_ci", make_forest_plot_ci),
        ("radar_profile", make_radar_profile),
        ("action_timeline", make_action_timeline),
        ("training_stability", make_training_stability),
        ("metric_matrix", make_metric_matrix),
    ]
    for name, fn in figs:
        try:
            fn()
            print(f"[ok] {name}.png")
        except Exception as e:
            import traceback
            print(f"[FAIL] {name}: {e}")
            traceback.print_exc()
    print(f"\nSaved to: {OUT}")
