"""Permutation importance and action analysis for trained DRL agents."""
import logging

import numpy as np
import pandas as pd

from lib.backtest import run_backtest

logger = logging.getLogger(__name__)

_GROUP_PREFIXES = {
    "sentiment": ("sentiment_",),
    "embeddings": ("emb_",),
    "news_count": ("news_count",),
}


def identify_feature_groups(feature_names):
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
                           backtest_kwargs=None, n_repeats=5, mode="zero", seed=42):
    if mode not in ("zero", "shuffle"):
        raise ValueError(f"mode must be 'zero' or 'shuffle', got '{mode}'")
    if n_repeats < 1:
        raise ValueError("n_repeats must be >= 1")

    backtest_kwargs = dict(backtest_kwargs or {})
    rng = np.random.RandomState(seed)

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


def action_feature_correlation(allocations, features, feature_names, method="spearman"):
    if method not in ("spearman", "pearson"):
        raise ValueError(f"method must be 'spearman' or 'pearson', got '{method}'")
    if features.shape[1] != len(feature_names):
        raise ValueError(f"features has {features.shape[1]} cols but "
                         f"{len(feature_names)} names")

    from scipy.stats import spearmanr, pearsonr

    T_a, T_f = len(allocations), features.shape[0]
    if T_a == 0:
        raise ValueError("Empty allocations array")
    if T_f >= T_a:
        features = features[T_f - T_a:]
    else:
        allocations = allocations[len(allocations) - T_f:]

    corr_fn = spearmanr if method == "spearman" else pearsonr
    rows = []
    for j, name in enumerate(feature_names):
        col = features[:, j]
        if np.allclose(col, col[0]) or np.allclose(allocations, allocations[0]):
            rho, p = float("nan"), float("nan")
        else:
            res = corr_fn(col, allocations)
            rho = float(res.correlation if hasattr(res, "correlation") else res[0])
            p = float(res.pvalue if hasattr(res, "pvalue") else res[1])
        rows.append({
            "feature": name,
            "correlation": rho,
            "p_value": p,
            "abs_corr": abs(rho) if not np.isnan(rho) else 0.0,
        })
    return pd.DataFrame(rows).sort_values("abs_corr", ascending=False).reset_index(drop=True)


def action_distribution_by_sentiment_regime(allocations, sentiment, n_bins=3, labels=None):
    if n_bins < 2:
        raise ValueError("n_bins must be >= 2")
    T_a, T_s = len(allocations), len(sentiment)
    if T_a == 0 or T_s == 0:
        raise ValueError("Empty allocations or sentiment array")

    if T_s >= T_a:
        sentiment = sentiment[T_s - T_a:]
    else:
        allocations = allocations[T_a - T_s:]

    if labels is None:
        labels = ["negative", "neutral", "positive"] if n_bins == 3 \
            else [f"bin_{i}" for i in range(n_bins)]
    if len(labels) != n_bins:
        raise ValueError(f"len(labels)={len(labels)} != n_bins={n_bins}")

    edges = np.unique(np.quantile(sentiment, np.linspace(0.0, 1.0, n_bins + 1)))
    if len(edges) < 2:
        return pd.DataFrame([{
            "bin": labels[0],
            "mean_action": float(allocations.mean()),
            "std_action": float(allocations.std(ddof=1)) if len(allocations) > 1 else 0.0,
            "n": int(len(allocations)),
            "sentiment_lower": float(sentiment.min()),
            "sentiment_upper": float(sentiment.max()),
        }]).set_index("bin")

    effective_bins = len(edges) - 1
    if effective_bins < n_bins:
        labels = list(labels)[:effective_bins]
        n_bins = effective_bins

    indices = np.clip(np.digitize(sentiment, edges[1:-1], right=False), 0, n_bins - 1)

    rows = []
    for i in range(n_bins):
        mask = indices == i
        slot = allocations[mask]
        sent_slot = sentiment[mask]
        if len(slot) == 0:
            rows.append({"bin": labels[i], "mean_action": float("nan"),
                         "std_action": float("nan"), "n": 0,
                         "sentiment_lower": float("nan"), "sentiment_upper": float("nan")})
        else:
            rows.append({
                "bin": labels[i],
                "mean_action": float(slot.mean()),
                "std_action": float(slot.std(ddof=1)) if len(slot) > 1 else 0.0,
                "n": int(len(slot)),
                "sentiment_lower": float(sent_slot.min()),
                "sentiment_upper": float(sent_slot.max()),
            })
    return pd.DataFrame(rows).set_index("bin")
