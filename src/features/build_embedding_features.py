"""Build daily embedding features by merging compressed embeddings with price features."""
import logging

import numpy as np
import pandas as pd

from src.features.embeddings import compute_embeddings
from src.features.sentiment import compute_sentiment_scores
from src.features.lag_features import add_lag_features, add_rolling_features

logger = logging.getLogger(__name__)

COMPRESSED_DIM = 64  # increased from 32
TOP_PCA_LAGS = 3     # lag top-3 PCA components


def build_embedding_features(
    price_features: pd.DataFrame,
    news_by_day: pd.DataFrame,
    compressor,
) -> pd.DataFrame:
    """Merge compressed daily embeddings with price features.

    Adds:
    - 64 emb_0..emb_63 columns (PCA-compressed embeddings, weighted by |sentiment|)
    - news_count, news_count_lag1, news_count_lag2, news_count_roll7
    - sentiment_max, sentiment_min, sentiment_spread (per-day extremes)
    - emb_0_lag1..emb_2_lag2 (top-3 PCA lags for "news direction" memory)

    Days without news get zeros for all NLP features.

    Args:
        price_features: DataFrame indexed by date with normalized price features.
        news_by_day: DataFrame with columns [date, texts] from news_preprocessor.
        compressor: Fitted EmbeddingCompressor with transform() method (outputs 64d).

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

    for _, row in news_by_day.iterrows():
        day = row["date"].normalize()
        texts = row["texts"]
        if day not in result.index:
            continue

        n = len(texts)
        result.loc[day, "news_count"] = n

        # Compute per-article sentiment scores (single batched pipeline call)
        sentiment_scores = compute_sentiment_scores(texts)
        result.loc[day, "sentiment_max"] = max(sentiment_scores)
        result.loc[day, "sentiment_min"] = min(sentiment_scores)
        result.loc[day, "sentiment_spread"] = max(sentiment_scores) - min(sentiment_scores)

        # Weighted pooling: |sentiment| so extreme articles weigh more
        weights = [abs(s) + 0.1 for s in sentiment_scores]

        raw_emb = compute_embeddings(texts, weights=weights)
        compressed = compressor.transform(raw_emb.reshape(1, -1))[0]
        result.loc[day, emb_cols] = compressed

    # Lag and rolling features for news_count
    result = add_lag_features(result, columns=["news_count"], lags=[1, 2])
    result = add_rolling_features(result, columns=["news_count"], window=7)

    # Lag top-3 PCA components (captures "news direction" from yesterday/day before)
    top_pca_cols = [f"emb_{i}" for i in range(TOP_PCA_LAGS)]
    result = add_lag_features(result, columns=top_pca_cols, lags=[1, 2])

    logger.info(
        f"Added {COMPRESSED_DIM} emb + 3 sentiment extremes + lags/rolling "
        f"to {len(result)} rows ({len(news_by_day)} days with news)"
    )
    return result
