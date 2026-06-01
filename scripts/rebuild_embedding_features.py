"""Rebuild btc_embedding_features.parquet with FinLang 768d + PCA 64d + weighted pooling.

Usage:
    PYTHONPATH=. python scripts/rebuild_embedding_features.py
"""
import logging
import sys

import numpy as np
import pandas as pd

from src.data.news_preprocessor import preprocess_news
from src.features.embeddings import compute_embeddings, EMBEDDING_DIM
from src.features.embedding_compressor import EmbeddingCompressor
from src.features.build_embedding_features import build_embedding_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

RAW_NEWS = "data/raw/bitcoin_news.parquet"
PRICE_FEATURES = "data/processed/btc_features.parquet"
OUTPUT_PATH = "data/processed/btc_embedding_features.parquet"
COMPRESSOR_PATH = "data/processed/embedding_compressor.pkl"


def main():
    # 1. Load price features
    logger.info("Loading price features...")
    price_features = pd.read_parquet(PRICE_FEATURES)
    logger.info(f"  Price features: {price_features.shape}")

    # 2. Load and preprocess news
    logger.info("Loading raw news...")
    raw_news = pd.read_parquet(RAW_NEWS)
    logger.info(f"  Raw news: {len(raw_news)} articles")

    # Rename columns to match expected format
    col_map = {}
    if "article_text" in raw_news.columns:
        col_map["article_text"] = "text"
    if "date_time" in raw_news.columns:
        col_map["date_time"] = "date"
    if col_map:
        raw_news = raw_news.rename(columns=col_map)

    # Ensure date is datetime with UTC
    raw_news["date"] = pd.to_datetime(raw_news["date"], utc=True)

    news_by_day = preprocess_news(raw_news)
    logger.info(f"  Grouped into {len(news_by_day)} trading days")

    # 3. Compute all raw embeddings for PCA fitting
    logger.info(f"Computing raw {EMBEDDING_DIM}d embeddings for PCA fitting...")
    all_embeddings = []
    for idx, row in news_by_day.iterrows():
        texts = row["texts"]
        for text in texts:
            emb = compute_embeddings([text])
            all_embeddings.append(emb)
        if (idx + 1) % 100 == 0:
            logger.info(f"  Processed {idx + 1}/{len(news_by_day)} days for PCA...")

    all_embeddings = np.array(all_embeddings, dtype=np.float32)
    logger.info(f"  Total embeddings for PCA: {all_embeddings.shape}")

    # 4. Fit compressor
    logger.info("Fitting PCA compressor (768 -> 64)...")
    compressor = EmbeddingCompressor(input_dim=EMBEDDING_DIM, output_dim=64)
    compressor.fit(all_embeddings)
    logger.info(f"  Explained variance: {compressor.explained_variance_ratio():.4f}")

    # Save compressor
    compressor.save(COMPRESSOR_PATH)

    # 5. Build embedding features with weighted pooling
    logger.info("Building embedding features with weighted pooling...")
    result = build_embedding_features(price_features, news_by_day, compressor)
    logger.info(f"  Result shape: {result.shape}")

    # 6. Save
    result.to_parquet(OUTPUT_PATH)
    logger.info(f"Saved to {OUTPUT_PATH}")

    # Verify
    emb_cols = [c for c in result.columns if c.startswith("emb_")]
    logger.info(f"  Embedding columns: {len(emb_cols)}")
    logger.info(f"  Has news_count: {'news_count' in result.columns}")
    logger.info(f"  Non-zero emb days: {(result[emb_cols].abs().sum(axis=1) > 0).sum()}")


if __name__ == "__main__":
    main()
