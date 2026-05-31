"""Run a trained model on OOS data; return metrics + equity curve."""
import logging
from pathlib import Path
from typing import Optional

import numpy as np

from lib.env import TradingEnv
from lib.metrics import compute_metrics

logger = logging.getLogger(__name__)


def load_model(model_path: str):
    from stable_baselines3 import PPO, SAC, DQN
    for cls in (PPO, SAC, DQN):
        try:
            return cls.load(model_path, device="cpu")
        except Exception:
            continue
    raise ValueError(f"Cannot load model from {model_path}")


def run_backtest(
    features: np.ndarray,
    prices: np.ndarray,
    model_path: Optional[str],
    window: int = 30,
    tx_cost: float = 0.001,
    allow_short: bool = False,
    vecnorm_path: Optional[str] = None,
    action_space_type: str = "continuous",
) -> dict:
    env = TradingEnv(
        features=features, prices=prices,
        window=window, tx_cost=tx_cost, allow_short=allow_short,
        action_space_type=action_space_type,
    )

    use_vecnorm = vecnorm_path is not None and Path(vecnorm_path).exists()
    if use_vecnorm:
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        vec_env = DummyVecEnv([lambda: env])
        vec_env = VecNormalize.load(vecnorm_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False

        model = load_model(model_path) if model_path else None
        obs = vec_env.reset()
        daily_returns, allocations = [], []
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True) if model else (vec_env.action_space.sample(), None)
            obs, _, done_arr, infos = vec_env.step(action)
            done = done_arr[0]
            info = infos[0]
            allocation = float(info.get("allocation", 0.0))
            log_ret = float(info.get("log_return", 0.0))
            daily_returns.append(np.exp(log_ret * allocation) - 1)
            allocations.append(allocation)
    else:
        model = load_model(model_path) if model_path else None
        obs, _ = env.reset()
        daily_returns, allocations = [], []
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True) if model else (env.action_space.sample(), None)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            allocation = float(info.get("allocation", 0.0))
            log_ret = float(info.get("log_return", 0.0))
            daily_returns.append(np.exp(log_ret * allocation) - 1)
            allocations.append(allocation)

    daily_returns = np.array(daily_returns)
    allocations_arr = np.array(allocations)
    equity_curve = np.insert(np.cumprod(1 + daily_returns), 0, 1.0)
    return {
        "metrics": compute_metrics(daily_returns, allocations=allocations_arr),
        "daily_returns": daily_returns,
        "allocations": allocations_arr,
        "equity_curve": equity_curve,
    }
