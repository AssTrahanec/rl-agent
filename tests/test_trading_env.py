import numpy as np
import gymnasium as gym
from src.env.trading_env import TradingEnv


def make_dummy_features(n=100, n_features=20):
    np.random.seed(42)
    return np.random.randn(n, n_features)


def test_env_creation():
    features = make_dummy_features()
    prices = 100 + np.cumsum(np.random.randn(100))
    env = TradingEnv(features=features, prices=prices, window=30)
    assert isinstance(env.observation_space, gym.spaces.Box)
    assert isinstance(env.action_space, gym.spaces.Box)


def test_env_reset():
    features = make_dummy_features()
    prices = 100 + np.cumsum(np.random.randn(100))
    env = TradingEnv(features=features, prices=prices, window=30)
    obs, info = env.reset()
    assert obs.shape == (30 * 20,)  # flattened for SB3


def test_env_step():
    features = make_dummy_features()
    prices = 100 + np.cumsum(np.random.randn(100))
    env = TradingEnv(features=features, prices=prices, window=30)
    obs, info = env.reset()
    action = np.array([0.5])
    obs, reward, terminated, truncated, info = env.step(action)
    assert isinstance(reward, float)
    assert obs.shape == (30 * 20,)


def test_env_reward_calculation():
    """Reward = log(price_next/price_current) * allocation - tx_cost_if_changed."""
    np.random.seed(0)
    features = make_dummy_features(n=35, n_features=5)
    # After reset, current_step=30, so step() reads prices[29] and prices[30]
    prices = np.array([100.0] * 30 + [110.0] + [110.0] * 4)  # 10% jump at index 30
    env = TradingEnv(features=features, prices=prices, window=30, tx_cost=0.001)
    env.reset()
    action = np.array([1.0])  # full allocation
    _, reward, _, _, _ = env.step(action)
    expected_log_return = np.log(110.0 / 100.0)
    # First step: tx_cost applied (position changed from 0 to 1)
    assert abs(reward - (expected_log_return * 1.0 - 0.001)) < 1e-6


def test_env_terminates():
    features = make_dummy_features(n=40, n_features=5)
    prices = np.ones(40) * 100.0
    env = TradingEnv(features=features, prices=prices, window=30)
    env.reset()
    terminated = False
    steps = 0
    while not terminated:
        _, _, terminated, _, _ = env.step(np.array([0.5]))
        steps += 1
        assert steps < 100, "Environment never terminated"


def test_action_space_bounds():
    features = make_dummy_features()
    prices = 100 + np.cumsum(np.random.randn(100))
    env = TradingEnv(features=features, prices=prices, window=30)
    assert env.action_space.low[0] == 0.0
    assert env.action_space.high[0] == 1.0
    assert env.action_space.shape == (1,)


def test_check_env():
    """SB3 env_checker must not raise."""
    from stable_baselines3.common.env_checker import check_env
    features = make_dummy_features(n=100, n_features=5)
    prices = 100 + np.cumsum(np.random.randn(100))
    env = TradingEnv(features=features, prices=prices, window=30)
    check_env(env, warn=True)
