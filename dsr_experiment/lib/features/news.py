"""News loading (HuggingFace) + preprocessing (dedup, 4h grouping)."""
import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


_TEXT_COL = "article_text"
_DATE_COL = "date_time"
_SPLIT = "train"


def load_news_from_hf(dataset_name: str, start: str, end: str) -> pd.DataFrame:
    """Load news from the HuggingFace dataset, filter by date range.

    Returns DataFrame with columns [text, date] (date as tz-aware UTC).
    """
    from datasets import load_dataset
    logger.info(f"Loading HF dataset '{dataset_name}'")
    ds = load_dataset(dataset_name)
    df = ds[_SPLIT].to_pandas()
    df = df.rename(columns={_TEXT_COL: "text", _DATE_COL: "date"})

    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df = df.dropna(subset=["date"])
    if "title" in df.columns:
        df = df.drop_duplicates(subset=["title"], keep="first")

    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")
    df = df[(df["date"] >= start_ts) & (df["date"] <= end_ts)].reset_index(drop=True)
    logger.info(f"Filtered to {len(df)} articles for {start} to {end}")
    return df[["text", "date"] + (["title"] if "title" in df.columns else [])]


def preprocess_news_4h(df: pd.DataFrame) -> pd.DataFrame:
    """Group news into 4h windows (UTC).

    Returns DataFrame with [date, texts] where date is the 4h window start
    and texts is list of article strings.
    """
    df = df.copy()
    df["date"] = df["date"].dt.floor("4h")
    grouped = (
        df.groupby("date")["text"]
        .apply(list)
        .reset_index()
        .rename(columns={"text": "texts"})
        .sort_values("date")
        .reset_index(drop=True)
    )
    logger.info(f"Grouped {len(df)} articles into {len(grouped)} 4h windows")
    return grouped


def deduplicate_embeddings(
    texts: list,
    scores: list,
    embeddings: np.ndarray,
    threshold: float = 0.85,
) -> tuple:
    """Drop duplicates with cosine similarity > threshold (keep first)."""
    if len(texts) <= 1:
        return texts, scores, embeddings
    sim = cosine_similarity(embeddings)
    keep = []
    for i in range(len(texts)):
        if not any(sim[i, j] > threshold for j in keep):
            keep.append(i)
    return [texts[i] for i in keep], [scores[i] for i in keep], embeddings[keep]
