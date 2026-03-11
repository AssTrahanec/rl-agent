"""Build daily sentiment features by merging news sentiment with price features.

Usage:
    python -m src.features.build_sentiment_features
"""
import logging

import pandas as pd

from src.features.sentiment import compute_sentiment

logger = logging.getLogger(__name__)


def build_sentiment_features(
    price_features: pd.DataFrame,
    news_by_day: pd.DataFrame,
) -> pd.DataFrame:
    """Merge daily sentiment scores with price features.

    Args:
        price_features: DataFrame indexed by date with normalized price features.
        news_by_day: DataFrame with columns [date, texts] from news_preprocessor.

    Returns:
        price_features with added 'sentiment' column. Days without news get 0.0.
    """
    result = price_features.copy()
    result["sentiment"] = 0.0

    # Build date -> sentiment mapping
    for _, row in news_by_day.iterrows():
        day = row["date"].normalize()
        texts = row["texts"]
        score = compute_sentiment(texts)
        if day in result.index:
            result.loc[day, "sentiment"] = score

    logger.info(
        f"Added sentiment to {len(result)} rows "
        f"({len(news_by_day)} days with news)"
    )
    return result
