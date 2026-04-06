# scripts/build_4h_dataset.py
"""Build complete 4h dataset: OHLCV + indicators + NLP features.

Usage:
    PYTHONPATH=. python scripts/build_4h_dataset.py
    PYTHONPATH=. python scripts/build_4h_dataset.py --skip-download  # если OHLCV уже скачан
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.price_collector import fetch_ohlcv
from src.features.technical import add_technical_indicators
from src.features.normalizer import rolling_zscore_normalize
from src.data.news_preprocessor import preprocess_news_4h
from src.features.embeddings import compute_embeddings, EMBEDDING_DIM
from src.features.embedding_compressor import EmbeddingCompressor
from src.features.sentiment import compute_sentiment_scores
from src.features.lag_features import add_lag_features, add_rolling_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

RAW_OHLCV_PATH = "data/raw/btc_4h_ohlcv.parquet"
RAW_NEWS_PATH = "data/raw/bitcoin_news.parquet"
OUTPUT_BASELINE = "data/processed/btc_4h_features.parquet"
OUTPUT_EMBEDDINGS = "data/processed/btc_4h_embedding_features.parquet"
COMPRESSOR_PATH = "data/processed/btc_4h_compressor.pkl"

COMPRESSED_DIM = 64
TOP_PCA_LAGS = 3


def step1_download_ohlcv():
    """Download 4h OHLCV from Binance."""
    logger.info("Step 1: Downloading 4h OHLCV from Binance...")
    df = fetch_ohlcv("BTC/USDT", "2020-01-01", "2024-12-31", timeframe="4h")
    Path(RAW_OHLCV_PATH).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(RAW_OHLCV_PATH)
    logger.info(f"  Saved {len(df)} candles to {RAW_OHLCV_PATH}")
    return df


def step2_baseline_features(ohlcv: pd.DataFrame):
    """Add technical indicators + normalize → baseline parquet."""
    logger.info("Step 2: Building baseline features...")
    df = add_technical_indicators(ohlcv)

    # Save raw_close before normalization (needed for backtest prices)
    df["raw_close"] = df["close"].copy()
    cols_to_normalize = [c for c in df.columns if c != "raw_close"]
    df[cols_to_normalize] = rolling_zscore_normalize(df[cols_to_normalize], window=30)

    Path(OUTPUT_BASELINE).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_BASELINE)
    logger.info(f"  Baseline: {df.shape} saved to {OUTPUT_BASELINE}")
    return df


def step3_embedding_features(baseline_df: pd.DataFrame):
    """Add NLP features (sentiment extremes + embeddings + lags) → embedding parquet."""
    logger.info("Step 3: Loading and preprocessing news for 4h windows...")
    raw_news = pd.read_parquet(RAW_NEWS_PATH)

    # Rename columns if needed
    col_map = {}
    if "article_text" in raw_news.columns:
        col_map["article_text"] = "text"
    if "date_time" in raw_news.columns:
        col_map["date_time"] = "date"
    if col_map:
        raw_news = raw_news.rename(columns=col_map)
    raw_news["date"] = pd.to_datetime(raw_news["date"], utc=True)

    news_4h = preprocess_news_4h(raw_news)
    logger.info(f"  {len(news_4h)} 4h windows with news")

    # 3a. Compute all raw embeddings for PCA fitting
    logger.info("Step 3a: Computing raw embeddings for PCA...")
    all_embeddings = []
    for idx, row in news_4h.iterrows():
        for text in row["texts"]:
            emb = compute_embeddings([text])
            all_embeddings.append(emb)
        if (idx + 1) % 500 == 0:
            logger.info(f"  PCA embeddings: {idx + 1}/{len(news_4h)} windows...")

    all_embeddings = np.array(all_embeddings, dtype=np.float32)
    logger.info(f"  Total embeddings for PCA: {all_embeddings.shape}")

    # 3b. Fit compressor
    logger.info("Step 3b: Fitting PCA compressor...")
    compressor = EmbeddingCompressor(input_dim=EMBEDDING_DIM, output_dim=COMPRESSED_DIM)
    compressor.fit(all_embeddings)
    compressor.save(COMPRESSOR_PATH)
    logger.info(f"  Explained variance: {compressor.explained_variance_ratio():.4f}")

    # 3c. Build features per 4h window
    logger.info("Step 3c: Building embedding features per 4h window...")
    result = baseline_df.copy()

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

        # Per-article sentiment
        sentiment_scores = compute_sentiment_scores(texts)
        if sentiment_scores:
            result.loc[window_start, "sentiment_max"] = max(sentiment_scores)
            result.loc[window_start, "sentiment_min"] = min(sentiment_scores)
            result.loc[window_start, "sentiment_spread"] = max(sentiment_scores) - min(sentiment_scores)

            # Weighted embeddings
            weights = [abs(s) + 0.1 for s in sentiment_scores]
            raw_emb = compute_embeddings(texts, weights=weights)
            compressed = compressor.transform(raw_emb.reshape(1, -1))[0]
            result.loc[window_start, emb_cols] = compressed

    logger.info(f"  Matched {matched}/{len(news_4h)} windows to OHLCV index")

    # 3d. Add lag and rolling features
    result = add_lag_features(result, columns=["news_count"], lags=[1, 2])
    result = add_rolling_features(result, columns=["news_count"], window=7)
    top_pca_cols = [f"emb_{i}" for i in range(TOP_PCA_LAGS)]
    result = add_lag_features(result, columns=top_pca_cols, lags=[1, 2])

    result.to_parquet(OUTPUT_EMBEDDINGS)
    feat_cols = [c for c in result.columns if c.lower() not in {"open", "high", "low", "close", "volume", "raw_close"}]
    logger.info(f"  Embeddings: {result.shape}, {len(feat_cols)} features saved to {OUTPUT_EMBEDDINGS}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Build 4h BTC dataset")
    parser.add_argument("--skip-download", action="store_true", help="Skip OHLCV download (use cached)")
    args = parser.parse_args()

    if args.skip_download and Path(RAW_OHLCV_PATH).exists():
        logger.info("Using cached OHLCV...")
        ohlcv = pd.read_parquet(RAW_OHLCV_PATH)
    else:
        ohlcv = step1_download_ohlcv()

    baseline = step2_baseline_features(ohlcv)
    step3_embedding_features(baseline)
    logger.info("Done! Files saved to data/processed/btc_4h_*.parquet")


if __name__ == "__main__":
    main()
