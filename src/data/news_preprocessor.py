"""Preprocess raw news: deduplicate, group by trading day (UTC)."""
import logging

import pandas as pd

logger = logging.getLogger(__name__)


def preprocess_news(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate news and group by trading day.

    Args:
        df: Raw news DataFrame with columns [title, text, date].

    Returns:
        DataFrame with columns [date, texts] where texts is a list of
        article texts for that day. One row per unique trading day, sorted.
    """
    df = df.copy()

    # Deduplicate by title
    before = len(df)
    df = df.drop_duplicates(subset=["title"], keep="first")
    logger.debug(f"Deduplication: {before} -> {len(df)} articles")

    # Normalize date to day (strip time component)
    df["date"] = df["date"].dt.normalize()

    # Group by day, collect texts into lists
    grouped = (
        df.groupby("date")["text"]
        .apply(list)
        .reset_index()
        .rename(columns={"text": "texts"})
    )

    grouped = grouped.sort_values("date").reset_index(drop=True)
    logger.info(f"Preprocessed {len(df)} articles into {len(grouped)} trading days")
    return grouped
