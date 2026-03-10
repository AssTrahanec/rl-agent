"""Pipeline: fetch OHLCV -> add indicators -> normalize -> save parquet.

Usage:
    python src/data/build_price_features.py
    python src/data/build_price_features.py --start 2020-01-01 --end 2024-12-31
"""
import argparse
import logging
import os
from pathlib import Path

import pandas as pd

from src.data.price_collector import fetch_ohlcv
from src.features.technical import add_technical_indicators
from src.features.normalizer import rolling_zscore_normalize

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ASSETS = {
    "BTC/USDT": "btc",
    "ETH/USDT": "eth",
}


def build_features(symbol: str, start: str, end: str, output_dir: str, window: int = 30) -> Path:
    logger.info(f"Fetching {symbol} from {start} to {end}")
    df = fetch_ohlcv(symbol, start, end)

    logger.info(f"Adding technical indicators ({len(df)} rows)")
    df = add_technical_indicators(df)

    logger.info(f"Normalizing with rolling z-score window={window}")
    df = rolling_zscore_normalize(df, window=window)

    prefix = ASSETS.get(symbol, symbol.replace("/", "").lower())
    out_path = Path(output_dir) / f"{prefix}_features.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)
    logger.info(f"Saved {len(df)} rows to {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Build price feature datasets")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--window", type=int, default=30)
    args = parser.parse_args()

    for symbol in ASSETS:
        build_features(symbol, args.start, args.end, args.output_dir, args.window)


if __name__ == "__main__":
    main()
