import numpy as np
from src.eval.metrics import compute_metrics


def test_sharpe_ratio_positive():
    returns = np.array([0.01, 0.02, -0.005, 0.015, 0.01])
    metrics = compute_metrics(returns)
    assert "sharpe_ratio" in metrics
    assert metrics["sharpe_ratio"] > 0


def test_sharpe_ratio_negative():
    returns = np.array([-0.01, -0.02, -0.005, -0.015, -0.01])
    metrics = compute_metrics(returns)
    assert metrics["sharpe_ratio"] < 0


def test_max_drawdown():
    returns = np.array([0.1, -0.2, 0.05, -0.1, 0.03])
    metrics = compute_metrics(returns)
    assert 0 <= metrics["max_drawdown"] <= 1


def test_total_return():
    returns = np.array([0.1, 0.1])  # (1.1 * 1.1 - 1) = 0.21
    metrics = compute_metrics(returns)
    assert abs(metrics["total_return"] - 0.21) < 1e-6


def test_sortino_ratio():
    returns = np.array([0.01, 0.02, -0.005, 0.015, 0.01])
    metrics = compute_metrics(returns)
    assert "sortino_ratio" in metrics
    assert metrics["sortino_ratio"] > 0


def test_calmar_ratio():
    returns = np.array([0.01, -0.05, 0.02, 0.03, -0.01])
    metrics = compute_metrics(returns)
    assert "calmar_ratio" in metrics


def test_all_keys_present():
    returns = np.array([0.01, -0.01, 0.02, -0.005])
    metrics = compute_metrics(returns)
    expected_keys = {"sharpe_ratio", "sortino_ratio", "max_drawdown", "calmar_ratio", "total_return"}
    assert expected_keys <= set(metrics.keys())
