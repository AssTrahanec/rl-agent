"""Tests for ensemble backtest."""
import numpy as np
import pytest
from stable_baselines3 import PPO

from src.env.trading_env import TradingEnv
from src.eval.ensemble_backtest import run_ensemble_backtest


@pytest.fixture
def dummy_data():
    np.random.seed(42)
    n, n_feat = 100, 18
    features = np.random.randn(n, n_feat).astype(np.float32)
    prices = (100 + np.cumsum(np.random.randn(n) * 0.5)).astype(np.float64)
    prices = np.maximum(prices, 1.0)
    return features, prices


@pytest.fixture
def two_models(dummy_data, tmp_path):
    features, prices = dummy_data
    paths = []
    for seed in [42, 43]:
        env = TradingEnv(features=features, prices=prices, window=30, tx_cost=0.001)
        model = PPO("MlpPolicy", env, seed=seed, device="cpu")
        model.learn(total_timesteps=200)
        p = tmp_path / f"model_s{seed}"
        model.save(str(p))
        paths.append(str(p) + ".zip")
    return paths


def test_ensemble_returns_metrics(dummy_data, two_models):
    features, prices = dummy_data
    result = run_ensemble_backtest(
        model_paths=two_models,
        features=features, prices=prices,
        window=30, tx_cost=0.001,
    )
    assert "metrics" in result
    assert "sharpe_ratio" in result["metrics"]
    assert "daily_returns" in result
    assert "allocations" in result
    assert "equity_curve" in result
    assert len(result["daily_returns"]) > 0


def test_ensemble_allocations_bounded(dummy_data, two_models):
    """Ensemble allocations should be between 0 and 1."""
    features, prices = dummy_data
    result = run_ensemble_backtest(
        model_paths=two_models,
        features=features, prices=prices,
        window=30, tx_cost=0.001,
    )
    allocs = result["allocations"]
    assert np.all(allocs >= 0.0)
    assert np.all(allocs <= 1.0)


def test_ensemble_with_weights(dummy_data, two_models):
    """Weighted ensemble should accept custom weights."""
    features, prices = dummy_data
    result = run_ensemble_backtest(
        model_paths=two_models,
        features=features, prices=prices,
        window=30, tx_cost=0.001,
        weights=[0.7, 0.3],
    )
    assert "metrics" in result


def test_ensemble_single_model_matches_backtest(dummy_data, two_models):
    """Ensemble with 1 model should produce same result as regular backtest."""
    from src.eval.backtest import run_backtest
    features, prices = dummy_data
    single = run_backtest(
        features=features, prices=prices,
        model_path=two_models[0], window=30, tx_cost=0.001,
    )
    ensemble = run_ensemble_backtest(
        model_paths=[two_models[0]],
        features=features, prices=prices,
        window=30, tx_cost=0.001,
    )
    np.testing.assert_allclose(
        single["daily_returns"], ensemble["daily_returns"], atol=1e-6,
    )
