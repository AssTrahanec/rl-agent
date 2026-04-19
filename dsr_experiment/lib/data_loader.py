"""Load train / OOS feature parquet files."""
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_PRICE_COLUMNS_EXT = {"open", "high", "low", "close", "volume", "raw_close"}


def _load_features_parquet(path: str, period_start: str, period_end: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Feature file not found: {p}\n"
            f"Run: python build_data.py --build <key>"
        )
    df = pd.read_parquet(p)
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"{p}: expected DatetimeIndex")
    start_ts = pd.Timestamp(period_start, tz="UTC") if df.index.tz else pd.Timestamp(period_start)
    end_ts = pd.Timestamp(period_end, tz="UTC") if df.index.tz else pd.Timestamp(period_end)
    df = df.loc[start_ts:end_ts]
    if df.empty:
        raise ValueError(f"{p}: empty for {period_start}..{period_end}")

    if "raw_close" in df.columns:
        prices = df["raw_close"].to_numpy(dtype=np.float64)
    else:
        prices = df["close"].to_numpy(dtype=np.float64)

    feature_cols = [c for c in df.columns if c.lower() not in _PRICE_COLUMNS_EXT]
    if not feature_cols:
        raise ValueError(f"{p}: no feature columns")

    features = df[feature_cols].to_numpy(dtype=np.float32)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    sentiment = (
        df["sentiment_max"].to_numpy(dtype=np.float32)
        if "sentiment_max" in df.columns
        else np.zeros(len(df), dtype=np.float32)
    )
    sentiment = np.nan_to_num(sentiment, nan=0.0)

    logger.info(f"Loaded {len(df)} rows × {features.shape[1]} features from {p.name}")
    return features, prices, sentiment


def load_train(cfg) -> tuple:
    """Load train feature matrix + prices + sentiment from cfg.data.paths.train_features."""
    train = cfg.periods["train"]
    return _load_features_parquet(cfg.data.paths.train_features, train.start, train.end)


def load_oos(cfg, period_key: str) -> tuple:
    """Load OOS feature matrix + prices + sentiment for a given period key."""
    if period_key not in cfg.periods:
        raise ValueError(f"Unknown period: '{period_key}'")
    period = cfg.periods[period_key]
    path = cfg.oos_features_path(period_key)
    return _load_features_parquet(path, period.start, period.end)
