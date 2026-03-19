"""Tests for Buy & Hold baseline."""
import numpy as np
import pytest

from src.eval.baselines import buy_and_hold


def test_total_return_simple():
    prices = np.array([100.0, 110.0])
    result = buy_and_hold(prices)
    assert abs(result["metrics"]["total_return"] - 0.10) < 1e-6


def test_total_return_flat():
    prices = np.array([50.0, 50.0, 50.0])
    result = buy_and_hold(prices)
    assert abs(result["metrics"]["total_return"]) < 1e-6


def test_returns_length():
    prices = np.array([100.0, 110.0, 121.0])
    result = buy_and_hold(prices)
    assert len(result["daily_returns"]) == len(prices) - 1


def test_equity_curve_starts_at_one():
    prices = np.array([100.0, 105.0, 110.0])
    result = buy_and_hold(prices)
    assert abs(result["equity_curve"][0] - 1.0) < 1e-9


def test_metrics_keys_present():
    prices = np.array([100.0, 90.0, 95.0, 110.0])
    result = buy_and_hold(prices)
    for key in ("total_return", "sharpe_ratio", "max_drawdown"):
        assert key in result["metrics"], f"Missing key: {key}"
