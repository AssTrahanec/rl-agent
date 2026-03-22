import tempfile
from pathlib import Path

import numpy as np

from src.eval.backtest import run_backtest


def test_backtest_returns_metrics():
    """Backtest on dummy model returns dict with expected keys."""
    n = 100
    n_features = 20
    np.random.seed(42)
    features = np.random.randn(n, n_features).astype(np.float32)
    prices = (100 + np.cumsum(np.random.randn(n) * 0.5)).astype(np.float64)
    prices = np.maximum(prices, 1.0)

    result = run_backtest(features=features, prices=prices, model_path=None, window=30)

    assert "metrics" in result
    assert "daily_returns" in result
    assert "allocations" in result
    assert "equity_curve" in result
    assert len(result["daily_returns"]) > 0


def test_backtest_equity_curve_starts_at_one():
    np.random.seed(42)
    n, n_f = 80, 10
    features = np.random.randn(n, n_f).astype(np.float32)
    prices = (100 + np.cumsum(np.random.randn(n) * 0.3)).astype(np.float64)
    prices = np.maximum(prices, 1.0)

    result = run_backtest(features=features, prices=prices, model_path=None, window=30)
    assert abs(result["equity_curve"][0] - 1.0) < 1e-8


def test_backtest_save_plot():
    np.random.seed(42)
    n, n_f = 80, 10
    features = np.random.randn(n, n_f).astype(np.float32)
    prices = (100 + np.cumsum(np.random.randn(n) * 0.3)).astype(np.float64)
    prices = np.maximum(prices, 1.0)

    with tempfile.TemporaryDirectory() as tmpdir:
        plot_path = Path(tmpdir) / "equity.png"
        run_backtest(
            features=features, prices=prices, model_path=None,
            window=30, plot_path=str(plot_path),
        )
        assert plot_path.exists()


def test_backtest_allow_short():
    """Backtest with allow_short passes through to TradingEnv."""
    np.random.seed(42)
    features = np.random.randn(100, 5).astype(np.float32)
    prices = (100 + np.cumsum(np.random.randn(100) * 0.5)).astype(np.float64)
    prices = np.maximum(prices, 1.0)
    result = run_backtest(features=features, prices=prices, model_path=None,
                          window=30, allow_short=True)
    assert "metrics" in result
