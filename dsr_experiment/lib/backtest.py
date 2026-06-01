"""Run a trained model on OOS data; return metrics + equity curve."""
import logging
from typing import Optional

import numpy as np

from lib.env import TradingEnv
from lib.metrics import compute_metrics

logger = logging.getLogger(__name__)


def load_model(model_path: str):
    from stable_baselines3 import SAC, DQN
    for cls in (SAC, DQN):
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
    action_space_type: str = "discrete",
) -> dict:
    """Deterministic rollout of a trained model; returns metrics + per-bar series.

    `step_returns` are per-4h-bar simple returns (the dict key stays "daily_returns"
    for backward compatibility with saved .npz artifacts).
    """
    env = TradingEnv(
        features=features, prices=prices,
        window=window, tx_cost=tx_cost,
        action_space_type=action_space_type,
    )

    model = load_model(model_path) if model_path else None
    obs, _ = env.reset()
    step_returns, allocations = [], []
    done = False
    while not done:
        if model:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = env.action_space.sample()
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        allocation = float(info.get("allocation", 0.0))
        log_ret = float(info.get("log_return", 0.0))
        step_returns.append(np.exp(log_ret * allocation) - 1)
        allocations.append(allocation)

    step_returns = np.array(step_returns)
    allocations_arr = np.array(allocations)
    equity_curve = np.insert(np.cumprod(1 + step_returns), 0, 1.0)
    return {
        "metrics": compute_metrics(step_returns, allocations=allocations_arr),
        "daily_returns": step_returns,
        "allocations": allocations_arr,
        "equity_curve": equity_curve,
    }
