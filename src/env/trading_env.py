import gymnasium as gym
import numpy as np
import logging

logger = logging.getLogger(__name__)


class TradingEnv(gym.Env):
    """Single-asset trading environment with continuous allocation action.

    Observation: flattened window of normalized feature rows (window * n_features,)
    Action:      allocation ∈ [0, 1]
    Reward:      log_return * allocation - tx_cost * |allocation - prev_allocation|
    """

    metadata = {"render_modes": []}

    def __init__(self, features: np.ndarray, prices: np.ndarray,
                 window: int = 30, tx_cost: float = 0.001):
        super().__init__()
        assert len(features) == len(prices), "features and prices must have same length"
        assert len(features) > window, "Need more data rows than window size"

        self.features = features.astype(np.float32)
        self.prices = prices.astype(np.float64)
        self.window = window
        self.tx_cost = tx_cost

        n_features = features.shape[1]
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(window * n_features,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(1,), dtype=np.float32
        )

        self.current_step: int = 0
        self.prev_allocation: float = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window
        self.prev_allocation = 0.0
        return self._get_obs(), {}

    def step(self, action):
        allocation = float(np.clip(action[0], 0.0, 1.0))

        price_current = self.prices[self.current_step - 1]
        price_next = self.prices[self.current_step]
        log_return = float(np.log(price_next / price_current))

        tx_penalty = self.tx_cost * abs(allocation - self.prev_allocation)
        reward = float(log_return * allocation - tx_penalty)

        self.prev_allocation = allocation
        self.current_step += 1

        terminated = self.current_step >= len(self.prices) - 1
        obs = self._get_obs()
        info = {"log_return": log_return, "allocation": allocation}
        return obs, reward, terminated, False, info

    def _get_obs(self) -> np.ndarray:
        start = self.current_step - self.window
        end = self.current_step
        return self.features[start:end].flatten()
