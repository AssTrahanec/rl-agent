"""Build BTC 4h embedding features for 2025-01-01 to 2025-04-10.
Uses existing PCA compressor — NO refit.
Output: data/processed/btc_4h_2025_embedding_features.parquet
"""
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.price_collector import fetch_ohlcv
from src.features.technical import add_technical_indicators
from src.features.normalizer import rolling_zscore_normalize
from src.features.embedding_compressor import EmbeddingCompressor
from src.features.embeddings import compute_embeddings
from src.features.sentiment import compute_sentiment_scores
from src.features.lag_features import add_lag_features, add_rolling_features
from src.data.news_preprocessor import preprocess_news_4h

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

START = "2025-01-01"
END = "2025-04-10"
ASSET = "BTC/USDT"
TIMEFRAME = "4h"
COMPRESSOR_PATH = "data/processed/btc_4h_compressor.pkl"
OUTPUT_PATH = "data/processed/btc_4h_2025_embedding_features.parquet"
COMPRESSED_DIM = 64
TOP_PCA_LAGS = 3


def load_news_from_hf(start: str, end: str) -> pd.DataFrame:
    """Load Bitcoin news from HuggingFace dataset, filter by date range."""
    from datasets import load_dataset
    logger.info("Loading dataset from HuggingFace: edaschau/bitcoin_news")
    ds = load_dataset("edaschau/bitcoin_news")
    df = ds["train"].to_pandas()

    # Rename columns to match expected format
    col_map = {}
    if "article_text" in df.columns:
        col_map["article_text"] = "text"
    if "date_time" in df.columns:
        col_map["date_time"] = "date"
    if col_map:
        df = df.rename(columns=col_map)

    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df = df.dropna(subset=["date"])

    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")
    df = df[(df["date"] >= start_ts) & (df["date"] <= end_ts)]
    df = df.reset_index(drop=True)
    logger.info(f"Filtered to {len(df)} news articles for {start} to {end}")
    return df


def main():
    logger.info(f"Fetching {ASSET} {TIMEFRAME} prices {START} to {END}")
    prices_df = fetch_ohlcv(ASSET, start=START, end=END, timeframe=TIMEFRAME)
    logger.info(f"Prices shape: {prices_df.shape}")

    logger.info("Adding technical indicators")
    prices_df = add_technical_indicators(prices_df)
    prices_df["raw_close"] = prices_df["close"].copy()

    logger.info("Normalizing features")
    cols_to_norm = [c for c in prices_df.columns if c != "raw_close"]
    prices_df[cols_to_norm] = rolling_zscore_normalize(prices_df[cols_to_norm])

    logger.info("Loading news for 2025")
    raw_news = load_news_from_hf(START, END)
    logger.info(f"Loaded {len(raw_news)} news articles")

    if len(raw_news) > 0:
        logger.info("Preprocessing news into 4h windows")
        news_4h = preprocess_news_4h(raw_news)
        logger.info(f"4h news windows: {len(news_4h)}")
    else:
        logger.warning("No news found for 2025, using NLP zeros")
        news_4h = pd.DataFrame(columns=["date", "texts"])

    logger.info(f"Loading compressor from {COMPRESSOR_PATH}")
    compressor = EmbeddingCompressor.load(COMPRESSOR_PATH)

    # Initialize NLP columns
    result = prices_df.copy()
    emb_cols = [f"emb_{i}" for i in range(COMPRESSED_DIM)]
    for col in emb_cols:
        result[col] = 0.0
    result["news_count"] = 0
    result["sentiment_max"] = 0.0
    result["sentiment_min"] = 0.0
    result["sentiment_spread"] = 0.0

    matched = 0
    for _, row in news_4h.iterrows():
        window_start = row["date"]
        texts = row["texts"]
        if window_start not in result.index:
            continue

        matched += 1
        n = len(texts)
        result.loc[window_start, "news_count"] = n

        sentiment_scores = compute_sentiment_scores(texts)
        if sentiment_scores:
            result.loc[window_start, "sentiment_max"] = max(sentiment_scores)
            result.loc[window_start, "sentiment_min"] = min(sentiment_scores)
            result.loc[window_start, "sentiment_spread"] = max(sentiment_scores) - min(sentiment_scores)

            weights = [abs(s) + 0.1 for s in sentiment_scores]
            raw_emb = compute_embeddings(texts, weights=weights)
            compressed = compressor.transform(raw_emb.reshape(1, -1))[0]
            result.loc[window_start, emb_cols] = compressed

    logger.info(f"Matched {matched}/{len(news_4h)} news windows to OHLCV index")

    # Add lag and rolling features
    result = add_lag_features(result, columns=["news_count"], lags=[1, 2])
    result = add_rolling_features(result, columns=["news_count"], window=7)
    top_pca_cols = [f"emb_{i}" for i in range(TOP_PCA_LAGS)]
    result = add_lag_features(result, columns=top_pca_cols, lags=[1, 2])

    feat_cols = [c for c in result.columns
                 if c.lower() not in ("open", "high", "low", "close", "volume", "raw_close")]
    logger.info(f"Feature columns: {len(feat_cols)}")
    assert len(feat_cols) == 95, f"Expected 95 feature cols, got {len(feat_cols)}: {feat_cols}"

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(OUTPUT_PATH)
    logger.info(f"Saved to {OUTPUT_PATH}")
    logger.info(f"Date range: {result.index.min()} to {result.index.max()}")
    logger.info(f"Shape: {result.shape}")
    logger.info(f"News windows with data: {(result['news_count'] > 0).sum()}")


if __name__ == "__main__":
    main()
