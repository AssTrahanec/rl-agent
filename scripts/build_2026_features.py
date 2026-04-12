"""Build BTC 4h features for 2026 YTD (2026-01-01 to 2026-04-12).
No news data available — NLP columns filled with zeros.
Output: data/processed/btc_4h_2026_embedding_features.parquet
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

START = "2026-01-01"
END = "2026-04-12"
ASSET = "BTC/USDT"
TIMEFRAME = "4h"
OUTPUT_PATH = "data/processed/btc_4h_2026_embedding_features.parquet"

COMPRESSED_DIM = 64
TOP_PCA_LAGS = 3

# All NLP-derived columns — filled with zeros
NLP_ZERO_COLS = (
    [f"emb_{i}" for i in range(COMPRESSED_DIM)] +
    ["news_count", "sentiment_max", "sentiment_min", "sentiment_spread",
     "news_count_lag1", "news_count_lag2", "news_count_roll7"] +
    [f"emb_{i}_lag1" for i in range(TOP_PCA_LAGS)] +
    [f"emb_{i}_lag2" for i in range(TOP_PCA_LAGS)]
)


def main():
    logger.info(f"Fetching {ASSET} {TIMEFRAME} prices {START} to {END}")
    prices_df = fetch_ohlcv(ASSET, start=START, end=END, timeframe=TIMEFRAME)
    logger.info(f"Prices shape: {prices_df.shape}")

    logger.info("Adding technical indicators")
    prices_df = add_technical_indicators(prices_df)
    prices_df["raw_close"] = prices_df["close"].copy()

    logger.info("Normalizing")
    cols_to_norm = [c for c in prices_df.columns if c != "raw_close"]
    prices_df[cols_to_norm] = rolling_zscore_normalize(prices_df[cols_to_norm])

    # Fill all NLP columns with zeros
    for col in NLP_ZERO_COLS:
        prices_df[col] = 0.0

    feat_cols = [c for c in prices_df.columns
                 if c.lower() not in ("open", "high", "low", "close", "volume", "raw_close")]
    logger.info(f"Feature columns: {len(feat_cols)}")
    assert len(feat_cols) == 95, f"Expected 95 feature cols, got {len(feat_cols)}: {feat_cols}"

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    prices_df.to_parquet(OUTPUT_PATH)
    logger.info(f"Saved to {OUTPUT_PATH}")
    logger.info(f"Date range: {prices_df.index.min()} to {prices_df.index.max()}")
    logger.info(f"Shape: {prices_df.shape}")


if __name__ == "__main__":
    main()
