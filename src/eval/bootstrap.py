"""Bootstrap confidence intervals for financial metrics across seeds."""
import logging
from typing import List

import numpy as np

from src.eval.metrics import compute_metrics

logger = logging.getLogger(__name__)


def bootstrap_ci(
    returns_list: List[np.ndarray],
    n_bootstrap: int = 5000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict:
    """Compute bootstrap CI for Sharpe Ratio and Total Return across seeds.

    Args:
        returns_list: List of daily return arrays, one per seed.
        n_bootstrap: Number of bootstrap samples.
        confidence: Confidence level (default 95%).
        seed: Random seed for reproducibility.

    Returns:
        Dict keyed by metric name, each containing mean, std, ci_lower, ci_upper.
    """
    rng = np.random.RandomState(seed)

    # Compute metric per seed
    seed_metrics = [compute_metrics(r) for r in returns_list]
    metric_names = ["sharpe_ratio", "total_return"]

    result = {}
    for metric in metric_names:
        values = np.array([m[metric] for m in seed_metrics])
        mean_val = float(np.mean(values))
        std_val = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0

        # Bootstrap: resample seeds with replacement
        boot_means = np.empty(n_bootstrap)
        for i in range(n_bootstrap):
            sample = rng.choice(values, size=len(values), replace=True)
            boot_means[i] = np.mean(sample)

        alpha = (1 - confidence) / 2
        ci_lower = float(np.percentile(boot_means, alpha * 100))
        ci_upper = float(np.percentile(boot_means, (1 - alpha) * 100))

        result[metric] = {
            "mean": mean_val,
            "std": std_val,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        }

    logger.info(f"Bootstrap CI ({confidence*100:.0f}%) computed over {len(returns_list)} seeds")
    return result
