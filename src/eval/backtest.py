"""Backtesting: run a trained model on test data, compute metrics, plot equity curve."""
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.env.trading_env import TradingEnv
from src.eval.metrics import compute_metrics

logger = logging.getLogger(__name__)


def run_backtest(
    features: np.ndarray,
    prices: np.ndarray,
    model_path: Optional[str],
    window: int = 30,
    tx_cost: float = 0.001,
    plot_path: Optional[str] = None,
) -> dict:
    """Run backtest on given data.

    Args:
        features: (T, n_features) array of normalized features.
        prices: (T,) array of close prices.
        model_path: Path to saved SB3 model .zip. If None, uses random actions.
        window: Observation window size.
        tx_cost: Transaction cost fraction.
        plot_path: If provided, save equity curve plot to this path.

    Returns:
        Dict with keys: metrics, daily_returns, allocations, equity_curve.
    """
    env = TradingEnv(features=features, prices=prices, window=window, tx_cost=tx_cost)

    model = None
    if model_path is not None:
        from stable_baselines3 import PPO
        model = PPO.load(model_path)

    obs, _ = env.reset()
    daily_returns = []
    allocations = []
    done = False

    while not done:
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        allocation = float(info.get("allocation", 0.0))
        log_ret = float(info.get("log_return", 0.0))
        daily_returns.append(np.exp(log_ret * allocation) - 1)
        allocations.append(allocation)

    daily_returns = np.array(daily_returns)
    equity_curve = np.cumprod(1 + daily_returns)
    equity_curve = np.insert(equity_curve, 0, 1.0)

    metrics = compute_metrics(daily_returns)

    if plot_path is not None:
        _plot_equity_curve(equity_curve, plot_path)

    return {
        "metrics": metrics,
        "daily_returns": daily_returns,
        "allocations": np.array(allocations),
        "equity_curve": equity_curve,
    }


def _plot_equity_curve(equity_curve: np.ndarray, save_path: str):
    """Save equity curve plot."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(equity_curve, linewidth=1.5)
    ax.set_xlabel("Trading Day")
    ax.set_ylabel("Portfolio Value")
    ax.set_title("Equity Curve")
    ax.grid(True, alpha=0.3)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved equity curve to {save_path}")
