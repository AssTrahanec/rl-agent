"""Build daily embedding features by merging compressed embeddings with price features."""
import logging

import numpy as np
import pandas as pd

from src.features.embeddings import compute_embeddings
from src.features.sentiment import compute_sentiment

logger = logging.getLogger(__name__)

COMPRESSED_DIM = 64  # increased from 32


def build_embedding_features(
    price_features: pd.DataFrame,
    news_by_day: pd.DataFrame,
    compressor,
) -> pd.DataFrame:
    """Merge compressed daily embeddings with price features.

    Uses weighted pooling: articles weighted by |sentiment_score| so that
    strongly positive/negative articles contribute more than neutral ones.

    Args:
        price_features: DataFrame indexed by date with normalized price features.
        news_by_day: DataFrame with columns [date, texts] from news_preprocessor.
        compressor: Fitted EmbeddingCompressor with transform() method (outputs 64d).

    Returns:
        price_features with 64 emb_0..emb_63 columns and news_count column added.
        Days without news get zeros for embeddings and 0 for news_count.
    """
    result = price_features.copy()

    emb_cols = [f"emb_{i}" for i in range(COMPRESSED_DIM)]
    for col in emb_cols:
        result[col] = 0.0
    result["news_count"] = 0

    for _, row in news_by_day.iterrows():
        day = row["date"].normalize()
        texts = row["texts"]
        if day not in result.index:
            continue

        n = len(texts)
        result.loc[day, "news_count"] = n

        # Compute sentiment weights: |score| so extreme articles weigh more
        sentiment_scores = [compute_sentiment([t]) for t in texts]
        weights = [abs(s) + 0.1 for s in sentiment_scores]  # +0.1 ensures no zero weights

        raw_emb = compute_embeddings(texts, weights=weights)
        compressed = compressor.transform(raw_emb.reshape(1, -1))[0]
        result.loc[day, emb_cols] = compressed

    logger.info(
        f"Added {COMPRESSED_DIM} embedding features + news_count to {len(result)} rows "
        f"({len(news_by_day)} days with news)"
    )
    return result
