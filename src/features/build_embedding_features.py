"""Build daily embedding features by merging compressed embeddings with price features."""
import logging

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.features.embeddings import _get_model
from src.features.sentiment import compute_sentiment_scores
from src.features.lag_features import add_lag_features, add_rolling_features

logger = logging.getLogger(__name__)

COMPRESSED_DIM = 20   # reduced from 64 to prevent overfit
TOP_PCA_LAGS = 3      # lag top-3 PCA components


def _deduplicate_embeddings(
    texts: list,
    scores: list,
    embeddings: np.ndarray,
    threshold: float = 0.85,
) -> tuple:
    """Remove articles with cosine similarity > threshold.

    Keeps the first occurrence. Returns filtered texts, scores, embeddings.
    """
    if len(texts) <= 1:
        return texts, scores, embeddings

    sim_matrix = cosine_similarity(embeddings)
    keep = []
    for i in range(len(texts)):
        is_dup = False
        for j in keep:
            if sim_matrix[i, j] > threshold:
                is_dup = True
                break
        if not is_dup:
            keep.append(i)

    return (
        [texts[i] for i in keep],
        [scores[i] for i in keep],
        embeddings[keep],
    )


def _temporal_weights(
    timestamps: pd.DatetimeIndex,
    day_close: pd.Timestamp,
    alpha: float = 0.1,
) -> list:
    """Compute exponential decay weights based on time distance to day close.

    weight_i = exp(-alpha * hours_since_close).
    Normalized to sum=1.
    """
    hours = [(day_close - ts).total_seconds() / 3600.0 for ts in timestamps]
    raw = [np.exp(-alpha * max(h, 0.0)) for h in hours]
    total = sum(raw)
    if total < 1e-8:
        return [1.0 / len(raw)] * len(raw)
    return [w / total for w in raw]


def build_embedding_features(
    price_features: pd.DataFrame,
    news_by_day: pd.DataFrame,
    compressor,
) -> pd.DataFrame:
    """Merge compressed daily embeddings with price features.

    Adds:
    - 20 emb_0..emb_19 columns (PCA-compressed embeddings, weighted by |sentiment|)
    - news_count, news_count_lag1, news_count_lag2, news_count_roll7
    - sentiment_max, sentiment_min, sentiment_spread (per-day extremes)
    - emb_0_lag1..emb_2_lag2 (top-3 PCA lags for "news direction" memory)

    Days without news get zeros for all NLP features.

    Args:
        price_features: DataFrame indexed by date with normalized price features.
        news_by_day: DataFrame with columns [date, texts] from news_preprocessor.
        compressor: Fitted EmbeddingCompressor with transform() method (outputs 20d).

    Returns:
        price_features with emb columns, news_count, sentiment extremes, and lag/rolling features added.
    """
    result = price_features.copy()

    # Initialize embedding columns with zeros
    emb_cols = [f"emb_{i}" for i in range(COMPRESSED_DIM)]
    for col in emb_cols:
        result[col] = 0.0
    result["news_count"] = 0
    result["sentiment_max"] = 0.0
    result["sentiment_min"] = 0.0
    result["sentiment_spread"] = 0.0

    st_model = _get_model()

    for _, row in news_by_day.iterrows():
        day = row["date"].normalize()
        texts = row["texts"]
        if day not in result.index:
            continue

        n = len(texts)
        result.loc[day, "news_count"] = n

        # Compute per-article sentiment scores
        sentiment_scores = compute_sentiment_scores(texts)
        result.loc[day, "sentiment_max"] = max(sentiment_scores)
        result.loc[day, "sentiment_min"] = min(sentiment_scores)
        result.loc[day, "sentiment_spread"] = max(sentiment_scores) - min(sentiment_scores)

        # Compute raw embeddings for dedup
        raw_embs = st_model.encode(texts, show_progress_bar=False)

        # Deduplicate similar articles
        texts, sentiment_scores, raw_embs = _deduplicate_embeddings(
            texts, sentiment_scores, raw_embs, threshold=0.85
        )

        # Weighted pooling: |sentiment| * temporal_decay
        if "timestamps" in row and row["timestamps"] is not None:
            day_close = day + pd.Timedelta(hours=23, minutes=59, seconds=59)
            t_weights = _temporal_weights(
                pd.to_datetime(row["timestamps"], utc=True), day_close
            )
            weights = [(abs(s) + 0.1) * tw for s, tw in zip(sentiment_scores, t_weights)]
        else:
            weights = [abs(s) + 0.1 for s in sentiment_scores]

        # Mean pooling of deduplicated embeddings with weights
        w = np.array(weights, dtype=np.float32)
        w = w / w.sum()
        mean_emb = np.average(raw_embs, axis=0, weights=w).astype(np.float32)

        compressed = compressor.transform(mean_emb.reshape(1, -1))[0]
        result.loc[day, emb_cols] = compressed

    # Lag and rolling features for news_count
    result = add_lag_features(result, columns=["news_count"], lags=[1, 2])
    result = add_rolling_features(result, columns=["news_count"], window=7)

    # Lag top-3 PCA components (captures "news direction" from yesterday/day before)
    top_pca_cols = [f"emb_{i}" for i in range(TOP_PCA_LAGS)]
    result = add_lag_features(result, columns=top_pca_cols, lags=[1, 2])

    logger.info(
        f"Added {COMPRESSED_DIM} emb + 3 sentiment extremes + lags/rolling "
        f"to {len(result)} rows ({len(news_by_day)} days with news)"  # noqa: E501
    )
    return result
