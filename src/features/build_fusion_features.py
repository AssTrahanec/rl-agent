"""Build Agent-4 fusion features: price + sentiment + compressed embeddings.

Combines the outputs of build_sentiment_features and build_embedding_features
in a single pipeline for the fusion agent.
"""
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

COMPRESSED_DIM = 32


def build_fusion_features(
    price_features: pd.DataFrame,
    news_by_day: pd.DataFrame,
    compressor=None,
    use_mock_nlp: bool = False,
) -> pd.DataFrame:
    """Build fusion features: price + sentiment + 32d compressed embeddings.

    Args:
        price_features: DataFrame indexed by date with normalized price features.
        news_by_day: DataFrame with columns [date, texts] from news_preprocessor.
        compressor: Fitted EmbeddingCompressor. If None and use_mock_nlp=False, raises.
        use_mock_nlp: If True, use random mock values instead of real NLP models
                      (for unit testing without GPU/model download).

    Returns:
        price_features with added 'sentiment' column and emb_0..emb_31 columns.
        Days without news get 0.0 fallback.
    """
    result = price_features.copy()

    # Initialize new columns with zeros (fallback)
    result["sentiment"] = 0.0
    emb_cols = [f"emb_{i}" for i in range(COMPRESSED_DIM)]
    for col in emb_cols:
        result[col] = 0.0

    if use_mock_nlp:
        _fill_mock(result, news_by_day, emb_cols)
    else:
        _fill_real(result, news_by_day, emb_cols, compressor)

    logger.info(
        f"Built fusion features: {result.shape[1]} total columns "
        f"({len(price_features.columns)} price + 1 sentiment + {COMPRESSED_DIM} embeddings)"
    )
    return result


def _fill_mock(result: pd.DataFrame, news_by_day: pd.DataFrame, emb_cols: list) -> None:
    """Fill sentiment and embedding columns with deterministic mock values (for tests)."""
    rng = np.random.default_rng(seed=0)
    for _, row in news_by_day.iterrows():
        day = pd.Timestamp(row["date"]).normalize()
        if hasattr(day, "tz_localize") and day.tzinfo is None:
            day = day.tz_localize("UTC")
        # Normalize index tz for comparison
        idx = result.index.normalize()
        mask = idx == day
        if mask.any():
            result.loc[mask, "sentiment"] = float(rng.uniform(-1.0, 1.0))
            result.loc[mask, emb_cols] = rng.standard_normal(len(emb_cols))


def _fill_real(
    result: pd.DataFrame,
    news_by_day: pd.DataFrame,
    emb_cols: list,
    compressor,
) -> None:
    """Fill using real FinBERT sentiment and embedding compressor."""
    from src.features.sentiment import compute_sentiment
    from src.features.embeddings import compute_embeddings

    if compressor is None:
        raise ValueError(
            "compressor must be provided when use_mock_nlp=False. "
            "Fit an EmbeddingCompressor on training embeddings first."
        )

    for _, row in news_by_day.iterrows():
        day = pd.Timestamp(row["date"]).normalize()
        if hasattr(day, "tz_localize") and day.tzinfo is None:
            day = day.tz_localize("UTC")
        idx = result.index.normalize()
        mask = idx == day
        if not mask.any():
            continue

        texts = row["texts"]
        sentiment = compute_sentiment(texts)
        result.loc[mask, "sentiment"] = sentiment

        raw_emb = compute_embeddings(texts)
        compressed = compressor.transform(raw_emb.reshape(1, -1))[0]
        result.loc[mask, emb_cols] = compressed
