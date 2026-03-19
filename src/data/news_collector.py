"""Load news datasets from HuggingFace for NLP pipeline.

Usage:
    from src.data.news_collector import load_news
    df = load_news("edaschau/bitcoin_news", start="2020-01-01", end="2024-12-31")
"""
import logging
from pathlib import Path

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
    text_col: str = "article_text",
    date_col: str = "date_time",
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
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path)
    logger.info(f"Saved {len(df)} articles to {output_path}")


def main():
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Download news from HuggingFace and save to parquet")
    parser.add_argument("--dataset", default="edaschau/bitcoin_news")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--output", default="data/raw/bitcoin_news.parquet")
    parser.add_argument("--title-col", default="title")
    parser.add_argument("--text-col", default="article_text")
    parser.add_argument("--date-col", default="date_time")
    args = parser.parse_args()

    save_news(
        args.dataset,
        args.start,
        args.end,
        args.output,
        title_col=args.title_col,
        text_col=args.text_col,
        date_col=args.date_col,
    )


if __name__ == "__main__":
    main()
