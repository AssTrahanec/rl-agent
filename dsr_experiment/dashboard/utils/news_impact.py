"""News Analyzer logic: inject a news article into the model's state."""
import numpy as np

_SENTIMENT_DIRECT = ("sentiment_mean", "sentiment_max", "sentiment_min")
_SENTIMENT_ZERO = ("sentiment_std", "sentiment_spread")
_EMB_DIM = 64


def inject_news_features(features, feature_columns, sentiment, emb_64):
    """Return a copy of `features` with one news article in the last row.

    features: (T, F) array. feature_columns: list of F column names.
    sentiment: float in [-1, 1]. emb_64: 64 PCA-compressed embedding values.
    Lag/rolling news columns are intentionally left untouched (need history).
    """
    if len(emb_64) != _EMB_DIM:
        raise ValueError(f"emb_64 must have {_EMB_DIM} values, got {len(emb_64)}")

    out = np.array(features, dtype=np.float32, copy=True)
    last = out.shape[0] - 1
    col_idx = {name: i for i, name in enumerate(feature_columns)}

    for name in _SENTIMENT_DIRECT:
        if name in col_idx:
            out[last, col_idx[name]] = sentiment
    # a single article has no spread — std/spread set to 0 by design
    for name in _SENTIMENT_ZERO:
        if name in col_idx:
            out[last, col_idx[name]] = 0.0
    if "news_count" in col_idx:
        out[last, col_idx["news_count"]] = 1.0
    for i in range(_EMB_DIM):
        name = f"emb_{i}"
        if name in col_idx:
            out[last, col_idx[name]] = float(emb_64[i])
    return out


def score_news(text):
    """Return (sentiment_score, emb_64) for a news text.

    sentiment_score: float in [-1, 1] from FinBERT.
    emb_64: PCA-compressed FinLang embedding (64 values).
    """
    from lib.features.sentiment import compute_sentiment_scores
    from lib.features.embeddings import compute_embeddings
    from dashboard.utils.model_loader import load_compressor

    sentiment = float(compute_sentiment_scores([text])[0])
    raw_emb = compute_embeddings([text])                 # 768-d vector
    compressor = load_compressor()
    emb_64 = compressor.transform(raw_emb.reshape(1, -1))[0]
    return sentiment, emb_64


def analyze_news_impact(text):
    """Run the DQN ensemble with and without a news article injected.

    Returns dict:
      sentiment        — float in [-1, 1]
      emb_64           — 64-d compressed embedding
      decision_without — {"allocation", "votes", "total_seeds"}
      decision_with    — same shape, with the news injected
    """
    from dashboard.utils.features_live import (
        build_live_features, build_live_obs, expected_feature_columns,
    )
    from dashboard.utils.feed_decisions import _ensemble_on_obs, _fetch_ohlcv_cached
    from dashboard.utils.model_catalog import list_model_entries
    from dashboard.utils import snapshot

    # 1. DQN ensemble model paths
    entries = list_model_entries()
    dqn_entry = next(e for e in entries if e.algo == "DQN")
    models = snapshot.discover_models(dqn_entry.snapshot).get("DQN", [])
    seeds_paths = [(m["seed"], m["path"]) for m in models]

    # 2. Live features (news zero-filled)
    ohlcv = _fetch_ohlcv_cached()
    features, _prices, _debug = build_live_features(ohlcv)
    columns = expected_feature_columns()

    # 3. Decision WITHOUT the news
    obs_base = build_live_obs(features, prev_allocation=0.0, window=30)
    decision_without = _ensemble_on_obs(obs_base, seeds_paths, "DQN", 0.0)

    # 4. Inject the news, decision WITH the news
    sentiment, emb_64 = score_news(text)
    features_news = inject_news_features(features, columns, sentiment, emb_64)
    obs_news = build_live_obs(features_news, prev_allocation=0.0, window=30)
    decision_with = _ensemble_on_obs(obs_news, seeds_paths, "DQN", 0.0)

    return {
        "sentiment": sentiment,
        "emb_64": emb_64,
        "decision_without": decision_without,
        "decision_with": decision_with,
    }
