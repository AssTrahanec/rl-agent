"""Visualization functions for ablation study results."""
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


def plot_equity_curves(
    results: List[Dict[str, Any]],
    save_path: Optional[str] = None,
    title: str = "Equity Curves",
) -> None:
    """Plot equity curves for multiple agents on one figure.

    Args:
        results: List of dicts with keys 'label' and 'equity_curve' (np.ndarray).
        save_path: If provided, save figure to this path.
        title: Plot title.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    for r in results:
        ax.plot(r["equity_curve"], label=r["label"], linewidth=1.5)
    ax.set_xlabel("Trading Day")
    ax.set_ylabel("Portfolio Value (normalized)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save_or_show(fig, save_path)


def plot_metrics_bar(
    results: List[Dict[str, Any]],
    metric: str = "sharpe_ratio",
    save_path: Optional[str] = None,
    title: Optional[str] = None,
) -> None:
    """Bar chart comparing a single metric across agents.

    Args:
        results: List of dicts with keys 'label' and 'metrics' (dict).
        metric: Metric key to plot.
        save_path: If provided, save figure to this path.
        title: Plot title (defaults to metric name).
    """
    labels = [r["label"] for r in results]
    values = [r["metrics"].get(metric, 0.0) for r in results]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = sns.color_palette("muted", len(labels))
    bars = ax.bar(labels, values, color=colors)
    ax.set_xlabel("Agent")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title or metric.replace("_", " ").title())
    ax.grid(True, axis="y", alpha=0.3)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    _save_or_show(fig, save_path)


def plot_metrics_heatmap(
    results: List[Dict[str, Any]],
    metrics: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    title: str = "Agent Metrics Heatmap",
) -> None:
    """Heatmap of agents × metrics.

    Args:
        results: List of dicts with keys 'label' and 'metrics' (dict).
        metrics: List of metric keys to include. Defaults to all from first result.
        save_path: If provided, save figure to this path.
        title: Plot title.
    """
    if metrics is None:
        metrics = list(results[0]["metrics"].keys()) if results else []

    labels = [r["label"] for r in results]
    data = np.array([[r["metrics"].get(m, 0.0) for m in metrics] for r in results])

    fig, ax = plt.subplots(figsize=(max(8, len(metrics) * 2), max(4, len(labels) * 1.2)))
    sns.heatmap(
        data,
        annot=True,
        fmt=".3f",
        xticklabels=[m.replace("_", " ").title() for m in metrics],
        yticklabels=labels,
        ax=ax,
        cmap="RdYlGn",
    )
    ax.set_title(title)
    fig.tight_layout()
    _save_or_show(fig, save_path)


def _save_or_show(fig: plt.Figure, save_path: Optional[str]) -> None:
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved plot to {save_path}")
    else:
        plt.show()
    plt.close(fig)
