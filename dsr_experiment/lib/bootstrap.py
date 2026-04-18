"""Bootstrap CI for Sharpe + total_return across seeds."""
import logging
from typing import List

import numpy as np

from lib.metrics import compute_metrics

logger = logging.getLogger(__name__)


def bootstrap_ci(
    returns_list: List[np.ndarray],
    n_bootstrap: int = 5000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict:
    rng = np.random.RandomState(seed)
    seed_metrics = [compute_metrics(r) for r in returns_list]
    metric_names = ["sharpe_ratio", "total_return"]
    out = {}
    for metric in metric_names:
        values = np.array([m[metric] for m in seed_metrics])
        mean_v = float(np.mean(values))
        std_v = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        boot = np.empty(n_bootstrap)
        for i in range(n_bootstrap):
            sample = rng.choice(values, size=len(values), replace=True)
            boot[i] = np.mean(sample)
        alpha = (1 - confidence) / 2
        out[metric] = {
            "mean": mean_v,
            "std": std_v,
            "ci_lower": float(np.percentile(boot, alpha * 100)),
            "ci_upper": float(np.percentile(boot, (1 - alpha) * 100)),
        }
    return out
