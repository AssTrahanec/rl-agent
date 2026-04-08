import gymnasium as gym
import numpy as np
import logging

logger = logging.getLogger(__name__)


class TradingEnv(gym.Env):
    """Single-asset trading environment with continuous allocation action.

    Observation: flattened window of normalized feature rows + prev_allocation
                 shape: (window * n_features + 1,)
    Action:      allocation ∈ [0, 1] (or [-1, 1] if allow_short=True)
    Reward (basic):        log_return * allocation - tx_cost * |delta|
    Reward (risk_adjusted): log_return * allocation - tx_cost * |delta|
                             - volatility_penalty * |delta|
    Reward (dsr):          Differential Sharpe Ratio increment
    where delta = |allocation - prev_allocation|.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        features: np.ndarray,
        prices: np.ndarray,
        window: int = 30,
        tx_cost: float = 0.001,
        reward_type: str = "basic",
        allow_short: bool = False,
        volatility_penalty: float = 0.5,
        sentiment_signal: np.ndarray | None = None,
        sentiment_lambda: float = 0.1,
    ):
        super().__init__()
        assert len(features) == len(prices), "features and prices must have same length"
        assert len(features) > window, "Need more data rows than window size"
        assert reward_type in ("basic", "risk_adjusted", "dsr"), (
            f"reward_type must be 'basic', 'risk_adjusted', or 'dsr', got '{reward_type}'"
        )

        self.features = features.astype(np.float32)
        self.prices = prices.astype(np.float64)
        self.window = window
        self.tx_cost = tx_cost
        self.reward_type = reward_type
        self.allow_short = allow_short
        self.volatility_penalty = volatility_penalty
        self.sentiment_signal = sentiment_signal
        self.sentiment_lambda = sentiment_lambda

        n_features = features.shape[1]
        # +1 for prev_allocation appended to obs
        obs_dim = window * n_features + 1
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        action_low = -1.0 if allow_short else 0.0
        self.action_space = gym.spaces.Box(
            low=action_low, high=1.0, shape=(1,), dtype=np.float32
        )

        self.current_step: int = 0
        self.prev_allocation: float = 0.0
        # DSR running statistics
        self._dsr_A: float = 0.0
        self._dsr_B: float = 0.0
        self._dsr_eta: float = 0.01

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window
        self.prev_allocation = 0.0
        self._dsr_A = 0.0
        self._dsr_B = 0.0
        return self._get_obs(), {}

    def step(self, action):
        action_low = self.action_space.low[0]
        allocation = float(np.clip(action[0], action_low, 1.0))

        price_current = self.prices[self.current_step - 1]
        price_next = self.prices[self.current_step]
        log_return = float(np.log(price_next / price_current))

        delta = abs(allocation - self.prev_allocation)
        tx_penalty = self.tx_cost * delta

        R_t = log_return * allocation - tx_penalty

        if self.reward_type == "dsr":
            reward = self._compute_dsr(R_t)
        elif self.reward_type == "risk_adjusted":
            reward = float(R_t - self.volatility_penalty * delta)
        else:
            reward = float(R_t)

        # SAPPO-lite: sentiment reward modifier
        if self.sentiment_signal is not None and self.reward_type == "dsr":
            sent = float(self.sentiment_signal[self.current_step])
            reward += self.sentiment_lambda * sent * log_return

        self.prev_allocation = allocation
        self.current_step += 1

        terminated = self.current_step >= len(self.prices) - 1
        obs = self._get_obs()
        info = {"log_return": log_return, "allocation": allocation}
        return obs, reward, terminated, False, info

    def _compute_dsr(self, R_t: float) -> float:
        """Compute Differential Sharpe Ratio increment."""
        eta = self._dsr_eta
        dA = R_t - self._dsr_A
        dB = R_t ** 2 - self._dsr_B

        denominator = self._dsr_B - self._dsr_A ** 2
        if denominator < 1e-12:
            reward = float(R_t)
        else:
            reward = float(
                (self._dsr_B * dA - 0.5 * self._dsr_A * dB)
                / (denominator ** 1.5)
            )

        self._dsr_A += eta * dA
        self._dsr_B += eta * dB
        return reward

    def _get_obs(self) -> np.ndarray:
        start = self.current_step - self.window
        end = self.current_step
        window_obs = self.features[start:end].flatten()
        return np.append(window_obs, self.prev_allocation).astype(np.float32)
