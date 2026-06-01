"""Test that backtest auto-detects algorithm type."""
import numpy as np
import pytest
from stable_baselines3 import A2C

from src.env.trading_env import TradingEnv
from src.eval.backtest import run_backtest


@pytest.fixture
def dummy_data():
    np.random.seed(42)
    n, n_feat = 100, 18
    features = np.random.randn(n, n_feat).astype(np.float32)
    prices = (100 + np.cumsum(np.random.randn(n) * 0.5)).astype(np.float64)
    prices = np.maximum(prices, 1.0)
    return features, prices


def test_backtest_loads_a2c_model(dummy_data, tmp_path):
    """run_backtest should load A2C models without error."""
    features, prices = dummy_data
    env = TradingEnv(features=features, prices=prices, window=30, tx_cost=0.001)
    model = A2C("MlpPolicy", env, seed=42, device="cpu")
    model.learn(total_timesteps=100)
    model_path = tmp_path / "a2c_model"
    model.save(str(model_path))

    result = run_backtest(
        features=features, prices=prices,
        model_path=str(model_path) + ".zip",
        window=30, tx_cost=0.001,
    )
    assert "metrics" in result
    assert "sharpe_ratio" in result["metrics"]
