"""Load news datasets from HuggingFace for NLP pipeline.

Usage:
    from src.data.news_collector import load_news
    df = load_news("edaschau/bitcoin_news", start="2020-01-01", end="2024-12-31")
"""
import logging

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ["title", "text", "date"]

# Alias for mocking in tests
try:
    from datasets import load_dataset as hf_load_dataset
except ImportError:
    hf_load_dataset = None
    logger.warning("HuggingFace datasets library not installed. Install with: pip install datasets")


def load_news(
    dataset_name: str,
    start: str,
    end: str,
    split: str = "train",
    title_col: str = "title",
    text_col: str = "text",
    date_col: str = "date",
) -> pd.DataFrame:
    """Load and filter news dataset from HuggingFace.

    Args:
        dataset_name: HuggingFace dataset identifier (e.g., 'edaschau/bitcoin_news').
        start: Start date string (inclusive), e.g., '2020-01-01'.
        end: End date string (inclusive), e.g., '2024-12-31'.
        split: Dataset split to load.
        title_col: Column name for article title.
        text_col: Column name for article text/body.
        date_col: Column name for publication date.

    Returns:
        DataFrame with columns [title, text, date], filtered by date range.
    """
    if hf_load_dataset is None:
        raise RuntimeError("HuggingFace datasets library required. pip install datasets")

    logger.info(f"Loading dataset '{dataset_name}' split='{split}'")
    ds = hf_load_dataset(dataset_name)
    df = ds[split].to_pandas()

    # Rename columns to standard names if needed
    rename_map = {}
    if title_col != "title":
        rename_map[title_col] = "title"
    if text_col != "text":
        rename_map[text_col] = "text"
    if date_col != "date":
        rename_map[date_col] = "date"
    if rename_map:
        df = df.rename(columns=rename_map)

    # Parse dates
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df = df.dropna(subset=["date"])

    # Filter by date range
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")
    df = df[(df["date"] >= start_ts) & (df["date"] <= end_ts)]

    # Keep only required columns + any extras
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Dataset missing required column: '{col}'")

    df = df.reset_index(drop=True)
    logger.info(f"Loaded {len(df)} news articles from {start} to {end}")
    return df


def save_news(dataset_name: str, start: str, end: str, output_path: str, **kwargs):
    """Load news and save to parquet."""
    df = load_news(dataset_name, start, end, **kwargs)
    df.to_parquet(output_path)
    logger.info(f"Saved {len(df)} articles to {output_path}")
