"""Naive baseline strategies for comparison."""
import logging
from typing import Dict, Any

import numpy as np

from src.eval.metrics import compute_metrics

logger = logging.getLogger(__name__)


def buy_and_hold(prices: np.ndarray) -> Dict[str, Any]:
    """Compute Buy & Hold strategy metrics.

    Buys on day 0 (full allocation), holds until the end.

    Args:
        prices: 1D array of daily close prices.

    Returns:
        Dict with keys: metrics, daily_returns, equity_curve.
    """
    prices = np.asarray(prices, dtype=np.float64)
    daily_returns = prices[1:] / prices[:-1] - 1.0
    equity_curve = np.cumprod(1 + daily_returns)
    equity_curve = np.insert(equity_curve, 0, 1.0)
    metrics = compute_metrics(daily_returns)

    logger.info(
        f"Buy & Hold: total_return={metrics['total_return']:.4f}, "
        f"sharpe={metrics['sharpe_ratio']:.4f}, "
        f"max_dd={metrics['max_drawdown']:.4f}"
    )

    return {
        "metrics": metrics,
        "daily_returns": daily_returns,
        "equity_curve": equity_curve,
    }
