"""Permutation importance and feature-group analysis for trained DRL agents."""
import logging

import numpy as np

from lib.backtest import run_backtest

logger = logging.getLogger(__name__)

_GROUP_PREFIXES = {
    "sentiment": ("sentiment_",),
    "embeddings": ("emb_",),
    "news_count": ("news_count",),
}


def identify_feature_groups(feature_names):
    """Map feature column names to index groups: price / sentiment / embeddings / news_count.

    Also adds a composite 'news' group = sentiment ∪ embeddings ∪ news_count.
    """
    groups = {"price": [], "sentiment": [], "embeddings": [], "news_count": []}
    for idx, name in enumerate(feature_names):
        name_lc = name.lower()
        assigned = False
        for group_name, prefixes in _GROUP_PREFIXES.items():
            if any(name_lc.startswith(p) for p in prefixes):
                groups[group_name].append(idx)
                assigned = True
                break
        if not assigned:
            groups["price"].append(idx)
    groups["news"] = sorted(set(
        groups["sentiment"] + groups["embeddings"] + groups["news_count"]
    ))
    return groups


def permutation_importance(model_path, features, prices, feature_groups,
                           backtest_kwargs=None, n_repeats=5, mode="zero"):
    """Drop in OOS metrics when each feature group is zeroed/shuffled.

    Positive Sharpe `drop_mean` for a group means the group helps the agent.
    """
    if mode not in ("zero", "shuffle"):
        raise ValueError(f"mode must be 'zero' or 'shuffle', got '{mode}'")
    if n_repeats < 1:
        raise ValueError("n_repeats must be >= 1")

    backtest_kwargs = dict(backtest_kwargs or {})
    rng = np.random.RandomState(42)

    baseline = run_backtest(features=features, prices=prices,
                            model_path=model_path, **backtest_kwargs)
    baseline_metrics = baseline["metrics"]
    metric_names = list(baseline_metrics.keys())

    out_groups = {}
    for group_name, col_indices in feature_groups.items():
        col_indices = list(col_indices)
        if not col_indices:
            continue
        per_metric = {m: [] for m in metric_names}
        repeats = 1 if mode == "zero" else n_repeats
        for _ in range(repeats):
            perturbed = features.copy()
            if mode == "zero":
                perturbed[:, col_indices] = 0.0
            else:
                for c in col_indices:
                    perturbed[:, c] = perturbed[rng.permutation(perturbed.shape[0]), c]
            res = run_backtest(features=perturbed, prices=prices,
                               model_path=model_path, **backtest_kwargs)
            for m in metric_names:
                per_metric[m].append(float(res["metrics"][m]))

        group_out = {}
        for m in metric_names:
            vals = np.array(per_metric[m], dtype=np.float64)
            baseline_v = float(baseline_metrics[m])
            group_out[m] = {
                "perturbed_mean": float(vals.mean()),
                "perturbed_std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                "drop_mean": baseline_v - float(vals.mean()),
                "drop_std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                "n_repeats": repeats,
            }
        out_groups[group_name] = group_out
        logger.info(f"  group '{group_name}' ({len(col_indices)} cols): "
                    f"Sharpe drop = {group_out['sharpe_ratio']['drop_mean']:+.3f}")

    return {
        "baseline": {k: float(v) for k, v in baseline_metrics.items()},
        "groups": out_groups,
        "feature_count": int(features.shape[1]),
        "mode": mode,
    }
