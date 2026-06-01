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


def preprocess_news_4h(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate news and group by 4-hour window (UTC).

    4h windows: 00:00, 04:00, 08:00, 12:00, 16:00, 20:00

    Args:
        df: Raw news DataFrame with columns [title, text, date].

    Returns:
        DataFrame with columns [date, texts] where date is the 4h window start
        and texts is a list of article texts in that window. Sorted by date.
    """
    df = df.copy()

    # Deduplicate by title
    before = len(df)
    df = df.drop_duplicates(subset=["title"], keep="first")
    logger.debug(f"Deduplication: {before} -> {len(df)} articles")

    # Floor to 4h window
    df["date"] = df["date"].dt.floor("4h")

    # Group by 4h window, collect texts
    grouped = (
        df.groupby("date")["text"]
        .apply(list)
        .reset_index()
        .rename(columns={"text": "texts"})
    )

    grouped = grouped.sort_values("date").reset_index(drop=True)
    logger.info(f"Preprocessed {len(df)} articles into {len(grouped)} 4h windows")
    return grouped
