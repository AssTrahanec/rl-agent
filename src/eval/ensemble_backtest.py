"""Ensemble backtest: run multiple models in lockstep, average allocations."""
import logging
from typing import Optional

import numpy as np

from src.eval.backtest import load_model, _plot_equity_curve
from src.env.trading_env import TradingEnv
from src.eval.metrics import compute_metrics

logger = logging.getLogger(__name__)


def run_ensemble_backtest(
    model_paths: list[str],
    features: np.ndarray,
    prices: np.ndarray,
    window: int = 30,
    tx_cost: float = 0.001,
    weights: Optional[list[float]] = None,
    plot_path: Optional[str] = None,
) -> dict:
    """Run ensemble backtest with multiple models.

    Each step, all models predict on the same observation. Their allocations
    are averaged (weighted or equal) and applied to the environment.

    Args:
        model_paths: List of paths to saved SB3 model .zip files.
        features: (T, n_features) array of normalized features.
        prices: (T,) array of close prices.
        window: Observation window size.
        tx_cost: Transaction cost fraction.
        weights: Optional weights for each model (must sum to ~1).
                 If None, equal weights are used.
        plot_path: If provided, save equity curve plot.

    Returns:
        Dict with keys: metrics, daily_returns, allocations, equity_curve,
                        individual_allocations.
    """
    n_models = len(model_paths)
    if n_models == 0:
        raise ValueError("At least one model path required")

    if weights is None:
        weights = [1.0 / n_models] * n_models
    else:
        if len(weights) != n_models:
            raise ValueError(f"weights length {len(weights)} != models {n_models}")
        w_sum = sum(weights)
        weights = [w / w_sum for w in weights]

    models = [load_model(mp) for mp in model_paths]
    logger.info(f"Loaded {n_models} models for ensemble")

    env = TradingEnv(features=features, prices=prices, window=window, tx_cost=tx_cost)
    obs, _ = env.reset()

    daily_returns = []
    allocations = []
    individual_allocs = []
    done = False

    while not done:
        model_actions = []
        for m in models:
            action, _ = m.predict(obs, deterministic=True)
            model_actions.append(float(action[0]))

        ensemble_alloc = sum(w * a for w, a in zip(weights, model_actions))
        ensemble_alloc = np.clip(ensemble_alloc, 0.0, 1.0)

        action = np.array([ensemble_alloc], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        allocation = float(info.get("allocation", 0.0))
        log_ret = float(info.get("log_return", 0.0))
        daily_returns.append(np.exp(log_ret * allocation) - 1)
        allocations.append(allocation)
        individual_allocs.append(model_actions)

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
        "individual_allocations": np.array(individual_allocs),
    }
