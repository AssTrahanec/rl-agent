"""Build daily embedding features by merging compressed embeddings with price features."""
import logging

import numpy as np
import pandas as pd

from src.features.embeddings import compute_embeddings

logger = logging.getLogger(__name__)

COMPRESSED_DIM = 32


def build_embedding_features(
    price_features: pd.DataFrame,
    news_by_day: pd.DataFrame,
    compressor,
) -> pd.DataFrame:
    """Merge compressed daily embeddings with price features.

    Args:
        price_features: DataFrame indexed by date with normalized price features.
        news_by_day: DataFrame with columns [date, texts] from news_preprocessor.
        compressor: Fitted EmbeddingCompressor with transform() method.

    Returns:
        price_features with 32 added emb_0..emb_31 columns.
        Days without news get zeros.
    """
    result = price_features.copy()

    # Initialize embedding columns with zeros
    emb_cols = [f"emb_{i}" for i in range(COMPRESSED_DIM)]
    for col in emb_cols:
        result[col] = 0.0

    for _, row in news_by_day.iterrows():
        day = row["date"].normalize()
        texts = row["texts"]
        if day not in result.index:
            continue

        raw_emb = compute_embeddings(texts)
        compressed = compressor.transform(raw_emb.reshape(1, -1))[0]
        result.loc[day, emb_cols] = compressed

    logger.info(
        f"Added {COMPRESSED_DIM} embedding features to {len(result)} rows "
        f"({len(news_by_day)} days with news)"
    )
    return result
