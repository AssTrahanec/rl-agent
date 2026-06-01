"""Tests for visualization functions."""
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.eval.visualize import (
    plot_equity_curves,
    plot_metrics_bar,
    plot_metrics_heatmap,
)


def make_dummy_results():
    """Create dummy backtest results for testing."""
    np.random.seed(42)
    results = []
    for agent in ["baseline", "sentiment", "embeddings"]:
        n = 100
        returns = np.random.randn(n) * 0.01
        equity = np.cumprod(1 + returns)
        equity = np.insert(equity, 0, 1.0)
        results.append({
            "label": agent,
            "equity_curve": equity,
            "metrics": {
                "sharpe_ratio": np.random.uniform(-1, 2),
                "total_return": np.random.uniform(-0.2, 0.5),
                "max_drawdown": np.random.uniform(0.05, 0.3),
            },
        })
    return results


def test_plot_equity_curves_no_exception():
    results = make_dummy_results()
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = str(Path(tmpdir) / "equity_curves.png")
        plot_equity_curves(results, save_path=save_path)
        assert Path(save_path).exists()


def test_plot_metrics_bar_no_exception():
    results = make_dummy_results()
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = str(Path(tmpdir) / "metrics_bar.png")
        plot_metrics_bar(results, metric="sharpe_ratio", save_path=save_path)
        assert Path(save_path).exists()


def test_plot_metrics_heatmap_no_exception():
    results = make_dummy_results()
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = str(Path(tmpdir) / "heatmap.png")
        plot_metrics_heatmap(results, save_path=save_path)
        assert Path(save_path).exists()
