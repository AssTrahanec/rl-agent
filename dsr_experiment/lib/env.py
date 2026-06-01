"""TradingEnv: DSR reward + sentiment bonus + transaction cost.

Action space:
    - discrete:   Discrete(3) — 0=Hold, 1=Buy all-in, 2=Sell all-out (DQN).
    - continuous: Box([0,1]) — allocation fraction (SAC).

Reward = DSR(R_t) + sentiment_lambda * sentiment_t * log_return_t,
where R_t = log_return * allocation − tx_cost * |Δallocation|.
"""
import logging

import gymnasium as gym
import numpy as np

logger = logging.getLogger(__name__)

_DSR_ETA = 0.01  # forgetting factor for the differential Sharpe ratio


class TradingEnv(gym.Env):
    """Single-asset long-only trading environment (Moody & Saffell DSR reward)."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        features: np.ndarray,
        prices: np.ndarray,
        window: int = 30,
        tx_cost: float = 0.001,
        sentiment_signal: "np.ndarray | None" = None,
        sentiment_lambda: float = 0.1,
        action_space_type: str = "discrete",
    ):
        super().__init__()
        assert len(features) == len(prices), "features and prices must have same length"
        assert len(features) > window, "Need more rows than window"
        assert action_space_type in ("continuous", "discrete"), action_space_type

        self.features = features.astype(np.float32)
        self.prices = prices.astype(np.float64)
        self.window = window
        self.tx_cost = tx_cost
        self.sentiment_signal = sentiment_signal
        self.sentiment_lambda = sentiment_lambda
        self.action_space_type = action_space_type

        n_features = features.shape[1]
        obs_dim = window * n_features + 1
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32,
        )
        if action_space_type == "discrete":
            self.action_space = gym.spaces.Discrete(3)   # 0=Hold, 1=Buy, 2=Sell
        else:
            self.action_space = gym.spaces.Box(
                low=0.0, high=1.0, shape=(1,), dtype=np.float32,
            )

        self.current_step = 0
        self.prev_allocation = 0.0
        self._dsr_A = 0.0
        self._dsr_B = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window
        self.prev_allocation = 0.0
        self._dsr_A = 0.0
        self._dsr_B = 0.0
        return self._get_obs(), {}

    def step(self, action):
        if self.action_space_type == "discrete":
            arr = np.asarray(action)
            a = int(arr.item()) if arr.ndim == 0 else int(arr.flatten()[0])
            if a == 1:
                allocation = 1.0
            elif a == 2:
                allocation = 0.0
            else:
                allocation = self.prev_allocation
        else:
            allocation = float(np.clip(action[0], 0.0, 1.0))

        price_curr = self.prices[self.current_step - 1]
        price_next = self.prices[self.current_step]
        log_return = float(np.log(price_next / price_curr))

        delta = abs(allocation - self.prev_allocation)
        R_t = log_return * allocation - self.tx_cost * delta

        reward = self._compute_dsr(R_t)
        if self.sentiment_signal is not None:
            sent = float(self.sentiment_signal[self.current_step])
            reward += self.sentiment_lambda * sent * log_return

        self.prev_allocation = allocation
        self.current_step += 1
        terminated = self.current_step >= len(self.prices) - 1
        info = {"log_return": log_return, "allocation": allocation}
        return self._get_obs(), reward, terminated, False, info

    def _compute_dsr(self, R_t: float) -> float:
        """Differential Sharpe Ratio (Moody & Saffell, 2001) via EMA of moments A, B."""
        dA = R_t - self._dsr_A
        dB = R_t ** 2 - self._dsr_B
        denom = self._dsr_B - self._dsr_A ** 2
        if denom < 1e-12:
            reward = float(R_t)
        else:
            reward = float((self._dsr_B * dA - 0.5 * self._dsr_A * dB) / (denom ** 1.5))
        self._dsr_A += _DSR_ETA * dA
        self._dsr_B += _DSR_ETA * dB
        return reward

    def _get_obs(self) -> np.ndarray:
        start = self.current_step - self.window
        window_obs = self.features[start:self.current_step].flatten()
        return np.append(window_obs, self.prev_allocation).astype(np.float32)
