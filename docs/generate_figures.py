"""Generate figures for the practice report.

Usage (from repo root):
    python docs/generate_figures.py

Output:
    docs/figures/*.png
"""

import sys
import os

# Add repo root to path so we can import src.*
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

from src.features.technical import add_technical_indicators
from src.features.normalizer import rolling_zscore_normalize

OUT_DIR = Path(__file__).parent / "figures"
OUT_DIR.mkdir(exist_ok=True)

# Shared style
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})


def make_synthetic_ohlcv(n=500, seed=42):
    """Generate realistic-looking synthetic OHLCV data."""
    np.random.seed(seed)
    # Random walk with drift and volatility clusters
    returns = np.random.normal(0.0005, 0.025, n)
    # Add a trend: bull -> bear -> recovery
    trend = np.concatenate([
        np.linspace(0, 0.002, n // 3),
        np.linspace(0.002, -0.003, n // 3),
        np.linspace(-0.003, 0.001, n - 2 * (n // 3)),
    ])
    returns += trend
    close = 30000 * np.exp(np.cumsum(returns))
    high = close * (1 + np.abs(np.random.normal(0, 0.01, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.01, n)))
    open_ = close * (1 + np.random.normal(0, 0.005, n))
    volume = np.random.lognormal(10, 1, n)

    dates = pd.date_range("2022-01-01", periods=n, freq="D")
    df = pd.DataFrame({
        "open": open_, "high": high, "low": low,
        "close": close, "volume": volume,
    }, index=dates)
    return df


# ── Figure 1: OHLCV + Technical Indicators ──────────────────────────

def fig1_price_with_indicators():
    """Price chart with SMA, Bollinger Bands, RSI, MACD — 3 subplots."""
    df = make_synthetic_ohlcv(400)
    df_ind = add_technical_indicators(df)

    fig, axes = plt.subplots(3, 1, figsize=(12, 9),
                             gridspec_kw={"height_ratios": [3, 1, 1]},
                             sharex=True)

    ax1, ax2, ax3 = axes

    # ── Price + SMA + Bollinger ──
    ax1.plot(df_ind.index, df_ind["close"], color="#2c3e50", lw=1.2,
             label="Close", zorder=3)
    ax1.plot(df_ind.index, df_ind["sma_7"], color="#e74c3c", lw=0.9,
             label="SMA(7)", alpha=0.8)
    ax1.plot(df_ind.index, df_ind["sma_25"], color="#3498db", lw=0.9,
             label="SMA(25)", alpha=0.8)
    ax1.fill_between(df_ind.index, df_ind["bb_upper"], df_ind["bb_lower"],
                     color="#3498db", alpha=0.1, label="Bollinger Bands(20,2)")
    ax1.set_ylabel("Price (synthetic BTC)")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.set_title("Рисунок 1 — Ценовой график с техническими индикаторами")
    ax1.grid(True, alpha=0.3)

    # ── RSI ──
    ax2.plot(df_ind.index, df_ind["rsi_14"], color="#8e44ad", lw=1)
    ax2.axhline(70, color="#e74c3c", ls="--", lw=0.8, alpha=0.7)
    ax2.axhline(30, color="#27ae60", ls="--", lw=0.8, alpha=0.7)
    ax2.fill_between(df_ind.index, 70, 100, color="#e74c3c", alpha=0.05)
    ax2.fill_between(df_ind.index, 0, 30, color="#27ae60", alpha=0.05)
    ax2.set_ylabel("RSI(14)")
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3)

    # ── MACD ──
    ax3.plot(df_ind.index, df_ind["macd"], color="#2980b9", lw=1,
             label="MACD")
    ax3.plot(df_ind.index, df_ind["macd_signal"], color="#e67e22", lw=1,
             label="Signal")
    colors = ["#27ae60" if v >= 0 else "#e74c3c"
              for v in df_ind["macd_hist"].fillna(0)]
    ax3.bar(df_ind.index, df_ind["macd_hist"], color=colors, alpha=0.4,
            width=1.0)
    ax3.set_ylabel("MACD")
    ax3.legend(loc="upper left", fontsize=9)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig1_price_indicators.png")
    plt.close(fig)
    print(f"  [OK] fig1_price_indicators.png")


# ── Figure 2: Normalization Before/After ─────────────────────────────

def fig2_normalization_effect():
    """Side-by-side: raw features vs rolling z-score normalized."""
    df = make_synthetic_ohlcv(300)
    df_ind = add_technical_indicators(df)
    df_norm = rolling_zscore_normalize(df_ind, window=30)

    cols_to_show = ["close", "rsi_14", "macd", "obv"]
    titles_ru = ["Цена закрытия", "RSI(14)", "MACD", "OBV"]

    fig, axes = plt.subplots(4, 2, figsize=(13, 10), sharex="col")
    fig.suptitle("Рисунок 2 — Эффект rolling z-score нормализации (window=30)",
                 fontsize=13, y=1.01)

    for i, (col, title) in enumerate(zip(cols_to_show, titles_ru)):
        # Raw
        axes[i, 0].plot(df_ind.index, df_ind[col], color="#2c3e50", lw=1)
        axes[i, 0].set_ylabel(title, fontsize=9)
        axes[i, 0].grid(True, alpha=0.3)
        if i == 0:
            axes[i, 0].set_title("До нормализации", fontsize=11)

        # Normalized
        axes[i, 1].plot(df_norm.index, df_norm[col], color="#2980b9", lw=1)
        axes[i, 1].axhline(0, color="gray", ls="--", lw=0.7, alpha=0.5)
        axes[i, 1].grid(True, alpha=0.3)
        if i == 0:
            axes[i, 1].set_title("После нормализации (z-score)", fontsize=11)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig2_normalization_effect.png")
    plt.close(fig)
    print(f"  [OK] fig2_normalization_effect.png")


# ── Figure 3: Agent Observation Spaces ───────────────────────────────

def fig3_agent_observation_spaces():
    """Visual comparison of 3 agent observation spaces."""
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.axis("off")
    ax.set_title("Рисунок 3 — Пространства наблюдений трёх агентов",
                 fontsize=13, pad=15)

    agents = [
        {
            "name": "Agent-1\n(Baseline)",
            "y": 45,
            "blocks": [("OHLCV + Tech.\nIndikators\n(24 feat.)", "#3498db", 35)],
            "dim": "30 × 24 = 720",
        },
        {
            "name": "Agent-2\n(+Sentiment)",
            "y": 27,
            "blocks": [
                ("OHLCV + Tech.\nIndikators\n(24 feat.)", "#3498db", 35),
                ("Sentiment\n(1 feat.)", "#e74c3c", 8),
            ],
            "dim": "30 × 25 = 750",
        },
        {
            "name": "Agent-3\n(+Embeddings)",
            "y": 9,
            "blocks": [
                ("OHLCV + Tech.\nIndikators\n(24 feat.)", "#3498db", 35),
                ("Embeddings\n(32 feat.)", "#27ae60", 25),
            ],
            "dim": "30 × 56 = 1680",
        },
    ]

    for agent in agents:
        y = agent["y"]
        ax.text(5, y + 3.5, agent["name"], fontsize=9, fontweight="bold",
                ha="center", va="center")

        x_start = 16
        for label, color, width in agent["blocks"]:
            rect = FancyBboxPatch(
                (x_start, y), width, 7,
                boxstyle="round,pad=0.3",
                facecolor=color, alpha=0.25, edgecolor=color, lw=1.5
            )
            ax.add_patch(rect)
            ax.text(x_start + width / 2, y + 3.5, label,
                    ha="center", va="center", fontsize=7.5)
            x_start += width + 1

        # Dimension label
        ax.text(x_start + 5, y + 3.5, f"→ {agent['dim']}",
                fontsize=9, va="center", color="#555",
                fontweight="bold")

    # Window label
    ax.annotate("30 days window", xy=(33.5, 54), fontsize=9, color="#888",
                ha="center")

    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig3_agent_observations.png")
    plt.close(fig)
    print(f"  [OK] fig3_agent_observations.png")


# ── Figure 4: Pipeline Architecture ─────────────────────────────────

def fig4_pipeline_architecture():
    """Block diagram of the data pipeline + training architecture."""
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.set_xlim(0, 130)
    ax.set_ylim(0, 60)
    ax.axis("off")
    ax.set_title("Рисунок 4 — Архитектура конвейера данных и обучения агента",
                 fontsize=13, pad=15)

    def draw_box(x, y, w, h, text, color, fontsize=8):
        rect = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.4",
            facecolor=color, alpha=0.2,
            edgecolor=color, lw=1.5
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text,
                ha="center", va="center", fontsize=fontsize,
                fontweight="bold")

    def draw_arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#555", lw=1.5))

    # Row 1: Data Sources
    draw_box(2, 45, 22, 10, "Binance API\n(OHLCV)", "#3498db")
    draw_box(2, 28, 22, 10, "HuggingFace\n(News)", "#e67e22")

    # Row 2: Processing
    draw_box(32, 45, 20, 10, "Technical\nIndicators\n(19 feat.)", "#2ecc71")
    draw_box(32, 28, 20, 10, "NLP\nPipeline", "#e67e22")

    # NLP branches
    draw_box(60, 33, 18, 8, "FinBERT\n→ sentiment", "#e74c3c")
    draw_box(60, 22, 18, 8, "MiniLM\n→ 384d → 32d", "#27ae60")

    # Normalization
    draw_box(60, 45, 18, 10, "Rolling\nZ-Score\nNormalize", "#9b59b6")

    # Merge
    draw_box(85, 32, 16, 26, "Feature\nMatrix\n(N × F)", "#34495e", fontsize=9)

    # Environment
    draw_box(107, 38, 18, 14, "Trading\nEnvironment\n(Gymnasium)", "#1abc9c",
             fontsize=9)

    # Agent
    draw_box(107, 18, 18, 14, "PPO Agent\n(SB3)\nMLP [256, 256]", "#3498db",
             fontsize=9)

    # Arrows
    draw_arrow(24, 50, 32, 50)     # Binance -> Indicators
    draw_arrow(24, 33, 32, 33)     # News -> NLP
    draw_arrow(52, 50, 60, 50)     # Indicators -> Normalize
    draw_arrow(52, 35, 60, 37)     # NLP -> FinBERT
    draw_arrow(52, 31, 60, 27)     # NLP -> MiniLM
    draw_arrow(78, 50, 85, 50)     # Normalize -> Feature Matrix
    draw_arrow(78, 37, 85, 42)     # FinBERT -> Feature Matrix
    draw_arrow(78, 26, 85, 36)     # MiniLM -> Feature Matrix
    draw_arrow(101, 45, 107, 45)   # Feature Matrix -> Env
    draw_arrow(116, 38, 116, 32)   # Env -> Agent (obs)
    draw_arrow(120, 32, 120, 38)   # Agent -> Env (action)

    # Labels on agent-env arrows
    ax.text(113, 35, "obs", fontsize=8, color="#666")
    ax.text(122, 35, "action", fontsize=8, color="#666")

    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig4_pipeline_architecture.png")
    plt.close(fig)
    print(f"  [OK] fig4_pipeline_architecture.png")


# ── Figure 5: Reward Function Visualization ──────────────────────────

def fig5_reward_function():
    """3D-like heatmap: reward as function of allocation and price return."""
    allocations = np.linspace(0, 1, 100)
    log_returns = np.linspace(-0.05, 0.05, 100)
    A, R = np.meshgrid(allocations, log_returns)

    tx_cost = 0.001
    # Assume prev_allocation = 0 for illustration
    reward = R * A - tx_cost * np.abs(A)

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.contourf(A, R * 100, reward * 100, levels=30, cmap="RdYlGn")
    ax.contour(A, R * 100, reward * 100, levels=[0], colors="black",
               linewidths=1.5, linestyles="--")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Reward (× 100)", fontsize=10)

    ax.set_xlabel("Allocation (a)", fontsize=11)
    ax.set_ylabel("Price Log-Return (%)", fontsize=11)
    ax.set_title("Рисунок 5 — Функция награды: reward = log_return × a − 0.001 × |Δa|",
                 fontsize=12)
    ax.grid(True, alpha=0.2)

    # Annotate key regions
    ax.annotate("Зона прибыли\n(рост цены +\nвысокая аллокация)",
                xy=(0.85, 3.5), fontsize=9, ha="center",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    ax.annotate("Зона убытка\n(падение цены +\nвысокая аллокация)",
                xy=(0.85, -3.5), fontsize=9, ha="center",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig5_reward_function.png")
    plt.close(fig)
    print(f"  [OK] fig5_reward_function.png")


# ── Figure 6: Trading Environment Cycle ──────────────────────────────

def fig6_env_cycle():
    """Diagram showing the RL agent-environment interaction loop."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)
    ax.axis("off")
    ax.set_title("Рисунок 6 — Цикл взаимодействия агента и торговой среды",
                 fontsize=13, pad=15)

    def draw_box(x, y, w, h, text, color, fontsize=9):
        rect = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.5",
            facecolor=color, alpha=0.2,
            edgecolor=color, lw=2
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text,
                ha="center", va="center", fontsize=fontsize,
                fontweight="bold", linespacing=1.4)

    # Agent box
    draw_box(5, 15, 30, 20, "PPO Agent\n(MLP [256, 256])\n\nπ(a|s)", "#3498db",
             fontsize=10)

    # Environment box
    draw_box(60, 15, 32, 20, "Trading\nEnvironment\n\nBinance OHLCV\n+ Indicators",
             "#1abc9c", fontsize=10)

    # Arrows with labels
    # Agent -> Env (action)
    ax.annotate("", xy=(60, 30), xytext=(35, 30),
                arrowprops=dict(arrowstyle="-|>", color="#e74c3c", lw=2))
    ax.text(47.5, 32, "action\na ∈ [0, 1]", ha="center", fontsize=9,
            color="#e74c3c", fontweight="bold")

    # Env -> Agent (observation, top arc)
    ax.annotate("", xy=(35, 37), xytext=(60, 37),
                arrowprops=dict(arrowstyle="-|>", color="#3498db", lw=2,
                                connectionstyle="arc3,rad=-0.3"))
    ax.text(47.5, 44, "observation\nobs ∈ ℝ⁷²⁰", ha="center", fontsize=9,
            color="#3498db", fontweight="bold")

    # Env -> Agent (reward, bottom arc)
    ax.annotate("", xy=(35, 20), xytext=(60, 20),
                arrowprops=dict(arrowstyle="-|>", color="#27ae60", lw=2,
                                connectionstyle="arc3,rad=0.3"))
    ax.text(47.5, 8, "reward\nr = log_ret × a − cost", ha="center",
            fontsize=9, color="#27ae60", fontweight="bold")

    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig6_env_cycle.png")
    plt.close(fig)
    print(f"  [OK] fig6_env_cycle.png")


# ── Main ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating report figures...")
    print(f"Output directory: {OUT_DIR.resolve()}\n")

    # 3 figures used in report:
    # Рисунок 1 — Архитектура конвейера (fig4_pipeline_architecture.png)
    # Рисунок 2 — Ценовой график с индикаторами (fig1_price_indicators.png)
    # Рисунок 3 — Эффект нормализации (fig2_normalization_effect.png)
    fig4_pipeline_architecture()
    fig1_price_with_indicators()
    fig2_normalization_effect()

    # Additional figures (generated but not included in report):
    fig3_agent_observation_spaces()
    fig5_reward_function()
    fig6_env_cycle()

    print(f"\nDone! {len(list(OUT_DIR.glob('*.png')))} figures saved to {OUT_DIR}/")
