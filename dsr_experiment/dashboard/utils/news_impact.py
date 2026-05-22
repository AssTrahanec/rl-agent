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
