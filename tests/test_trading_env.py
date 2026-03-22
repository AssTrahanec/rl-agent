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


def test_reward_risk_adjusted():
    """risk_adjusted reward subtracts volatility_penalty * |delta| in addition to tx_cost."""
    np.random.seed(0)
    features = make_dummy_features(n=35, n_features=5)
    prices = np.array([100.0] * 30 + [110.0] + [110.0] * 4)
    env = TradingEnv(
        features=features, prices=prices, window=30,
        tx_cost=0.001, reward_type="risk_adjusted", volatility_penalty=0.5
    )
    env.reset()
    action = np.array([1.0])
    _, reward, _, _, _ = env.step(action)
    expected_log_return = np.log(110.0 / 100.0)
    delta = abs(1.0 - 0.0)  # prev_allocation starts at 0
    expected_reward = expected_log_return * 1.0 - 0.001 * delta - 0.5 * delta
    assert abs(reward - expected_reward) < 1e-6


def test_reward_risk_adjusted_no_change():
    """No volatility penalty when position doesn't change between steps."""
    np.random.seed(0)
    features = make_dummy_features(n=36, n_features=5)
    prices = np.array([100.0] * 30 + [110.0] + [120.0] + [120.0] * 4)
    env = TradingEnv(
        features=features, prices=prices, window=30,
        tx_cost=0.001, reward_type="risk_adjusted", volatility_penalty=0.5
    )
    env.reset()
    # First step: allocation changes from 0 -> 1
    env.step(np.array([1.0]))
    # Second step: allocation stays at 1 -> no delta, no penalty
    _, reward2, _, _, _ = env.step(np.array([1.0]))
    expected_log_return = np.log(120.0 / 110.0)
    expected_reward2 = expected_log_return * 1.0  # no tx_cost, no volatility_penalty
    assert abs(reward2 - expected_reward2) < 1e-6


def test_reward_basic_unchanged():
    """Default reward_type='basic' produces the same result as before."""
    np.random.seed(0)
    features = make_dummy_features(n=35, n_features=5)
    prices = np.array([100.0] * 30 + [110.0] + [110.0] * 4)
    env = TradingEnv(features=features, prices=prices, window=30, tx_cost=0.001)
    env.reset()
    _, reward, _, _, _ = env.step(np.array([1.0]))
    expected_log_return = np.log(110.0 / 100.0)
    assert abs(reward - (expected_log_return * 1.0 - 0.001)) < 1e-6


def test_allow_short():
    """allow_short=True sets action_space to [-1, 1] and short positions work."""
    features = make_dummy_features(n=100, n_features=5)
    prices = 100 + np.cumsum(np.random.randn(100))
    env = TradingEnv(features=features, prices=prices, window=30, allow_short=True)
    assert env.action_space.low[0] == -1.0
    assert env.action_space.high[0] == 1.0
    env.reset()
    # Short position: allocation = -0.5
    _, reward, _, _, info = env.step(np.array([-0.5]))
    assert info["allocation"] == -0.5
    assert isinstance(reward, float)
