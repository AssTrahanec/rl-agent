"""News Analyzer logic: score news articles and measure their effect on the model."""
import numpy as np

_EMB_DIM = 64
_NEWS_STAT_COLUMNS = (
    "sentiment_mean", "sentiment_max", "sentiment_min",
    "sentiment_std", "sentiment_spread", "news_count",
)


def inject_news_features(features, feature_columns, stats, emb_64):
    """Return a copy of `features` with aggregated news stats injected into every row.

    features: (T, F) array. feature_columns: list of F column names.
    stats: dict with keys sentiment_mean/max/min/std/spread and news_count.
    emb_64: 64 PCA-compressed embedding values.
    The news signal is written into all rows so the agent sees a consistent news
    environment across its observation window — this matches the training
    distribution, where news was present in nearly every bar. Lag/rolling news
    columns are intentionally left untouched (need history).
    """
    if len(emb_64) != _EMB_DIM:
        raise ValueError(f"emb_64 must have {_EMB_DIM} values, got {len(emb_64)}")

    out = np.array(features, dtype=np.float32, copy=True)
    col_idx = {name: i for i, name in enumerate(feature_columns)}

    for name in _NEWS_STAT_COLUMNS:
        if name in col_idx and name in stats:
            out[:, col_idx[name]] = float(stats[name])
    for i in range(_EMB_DIM):
        name = f"emb_{i}"
        if name in col_idx:
            out[:, col_idx[name]] = float(emb_64[i])
    return out


def score_news_batch(texts):
    """Score a list of news texts: per-article sentiment, aggregate stats, embedding.

    Returns dict:
      per_news — list of (text, sentiment) for each article
      stats    — dict: sentiment_mean/max/min/std/spread, news_count
      emb_64   — 64-d PCA-compressed, sentiment-weighted embedding of all articles
    """
    from lib.features.sentiment import compute_sentiment_scores
    from lib.features.embeddings import compute_embeddings
    from dashboard.utils.model_loader import load_compressor

    if not texts:
        raise ValueError("texts is empty")

    scores = [float(s) for s in compute_sentiment_scores(texts)]
    per_news = list(zip(texts, scores))

    arr = np.array(scores, dtype=np.float64)
    stats = {
        "sentiment_mean": float(arr.mean()),
        "sentiment_max": float(arr.max()),
        "sentiment_min": float(arr.min()),
        "sentiment_std": float(arr.std()) if len(arr) > 1 else 0.0,
        "sentiment_spread": float(arr.max() - arr.min()),
        "news_count": float(len(texts)),
    }

    # Sentiment-weighted embedding: emotionally stronger news weigh more
    # (replicates the aggregation used when the training data was built).
    signed_w = [s + np.sign(s) * 0.1 if s != 0 else 0.1 for s in scores]
    raw_emb = compute_embeddings(texts, weights=signed_w)   # 768-d vector
    compressor = load_compressor()
    emb_64 = compressor.transform(raw_emb.reshape(1, -1))[0]

    return {"per_news": per_news, "stats": stats, "emb_64": emb_64}


def analyze_news_impact(texts):
    """Run the DQN ensemble with and without the given news articles injected.

    texts: list of news strings.
    Returns dict:
      per_news         — list of (text, sentiment)
      stats            — aggregate sentiment stats dict
      decision_without — {"allocation", "votes", "total_seeds"}
      decision_with    — same shape, with the news injected
    """
    from dashboard.utils.features_live import (
        build_live_features, build_live_obs, expected_feature_columns,
    )
    from dashboard.utils.feed_decisions import _ensemble_on_obs, _fetch_ohlcv_cached
    from dashboard.utils.model_catalog import list_model_entries
    from dashboard.utils import snapshot

    entries = list_model_entries()
    dqn_entry = next(e for e in entries if e.algo == "DQN")
    models = snapshot.discover_models(dqn_entry.snapshot).get("DQN", [])
    seeds_paths = [(m["seed"], m["path"]) for m in models]

    ohlcv = _fetch_ohlcv_cached()
    features, _prices, _debug = build_live_features(ohlcv)
    columns = expected_feature_columns()

    obs_base = build_live_obs(features, prev_allocation=0.0, window=30)
    decision_without = _ensemble_on_obs(obs_base, seeds_paths, "DQN", 0.0)

    scored = score_news_batch(texts)
    features_news = inject_news_features(
        features, columns, scored["stats"], scored["emb_64"]
    )
    obs_news = build_live_obs(features_news, prev_allocation=0.0, window=30)
    decision_with = _ensemble_on_obs(obs_news, seeds_paths, "DQN", 0.0)

    return {
        "per_news": scored["per_news"],
        "stats": scored["stats"],
        "decision_without": decision_without,
        "decision_with": decision_with,
    }
