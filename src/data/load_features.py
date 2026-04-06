"""Load preprocessed feature files for RL agent training.

Conventions (set by build_price_features.py / build_sentiment_features /
build_embedding_features):

    data/processed/
        btc_features.parquet          # baseline (price + tech indicators)
        eth_features.parquet
        btc_sentiment_features.parquet
        eth_sentiment_features.parquet
        btc_embedding_features.parquet
        eth_embedding_features.parquet
        btc_fusion_features.parquet   # sentiment + embeddings
        eth_fusion_features.parquet

Each parquet has a DatetimeIndex (UTC) and normalised feature columns.
The 'close' column (raw price) must be present for the trading env.

Usage:
    from src.data.load_features import load_features_for_agent
    features, prices = load_features_for_agent(
        agent_type="sentiment", asset="BTC/USDT",
        train_start="2020-01-01", train_end="2023-12-31",
    )
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Map (asset, agent_type) -> parquet filename stem
_ASSET_PREFIX = {
    "BTC/USDT": "btc",
    "ETH/USDT": "eth",
    # also accept short form
    "BTC": "btc",
    "ETH": "eth",
}

_AGENT_SUFFIX = {
    "baseline": "features",
    "sentiment": "sentiment_features",
    "embeddings": "embedding_features",
    "fusion": "fusion_features",
}

# Columns that are NOT features (raw price data kept only for prices array)
_PRICE_COLUMNS = {"open", "high", "low", "close", "volume"}


def _parquet_path(agent_type: str, asset: str, data_dir: str, timeframe: str = "1d") -> Path:
    prefix = _ASSET_PREFIX.get(asset)
    if prefix is None:
        raise ValueError(f"Unknown asset '{asset}'. Known: {list(_ASSET_PREFIX)}")
    suffix = _AGENT_SUFFIX.get(agent_type)
    if suffix is None:
        raise ValueError(f"Unknown agent_type '{agent_type}'. Known: {list(_AGENT_SUFFIX)}")
    tf_prefix = f"{prefix}_4h" if timeframe == "4h" else prefix
    return Path(data_dir) / f"{tf_prefix}_{suffix}.parquet"


def load_features_for_agent(
    agent_type: str,
    asset: str,
    train_start: str,
    train_end: str,
    data_dir: str = "data/processed",
    timeframe: str = "1d",
) -> tuple[np.ndarray, np.ndarray]:
    """Load feature matrix and price series for a train/test split.

    Args:
        agent_type: One of 'baseline', 'sentiment', 'embeddings', 'fusion'.
        asset: Asset symbol, e.g. 'BTC/USDT'.
        train_start: Inclusive start date string, e.g. '2020-01-01'.
        train_end: Inclusive end date string, e.g. '2023-12-31'.
        data_dir: Directory containing parquet files (default: 'data/processed').
        timeframe: Candle timeframe, '1d' (default) or '4h'.

    Returns:
        Tuple (features, prices):
            features: float32 numpy array of shape (T, n_features).
            prices:   float64 numpy array of shape (T,) — close prices.

    Raises:
        FileNotFoundError: If the parquet file does not exist.
        ValueError:        If the date slice produces an empty DataFrame.
    """
    path = _parquet_path(agent_type, asset, data_dir, timeframe=timeframe)
    if not path.exists():
        raise FileNotFoundError(
            f"Feature file not found: {path}\n"
            f"Run the appropriate build_*_features script first:\n"
            f"  python -m src.data.build_price_features\n"
            f"  python -m src.features.build_sentiment_features  (for sentiment/fusion)\n"
            f"  python -m src.features.build_embedding_features  (for embeddings/fusion)"
        )

    logger.info(f"Loading {path}")
    df = pd.read_parquet(path)

    # Ensure DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"{path}: expected DatetimeIndex, got {type(df.index)}")

    # Slice date range
    start_ts = pd.Timestamp(train_start)
    end_ts = pd.Timestamp(train_end)
    # Handle tz-aware index
    if df.index.tz is not None:
        start_ts = start_ts.tz_localize("UTC") if start_ts.tzinfo is None else start_ts
        end_ts = end_ts.tz_localize("UTC") if end_ts.tzinfo is None else end_ts

    df = df.loc[start_ts:end_ts]

    if df.empty:
        raise ValueError(
            f"No data in {path} for period {train_start} – {train_end}. "
            "Check that the parquet covers this range."
        )

    # Extract close prices (required by TradingEnv) — use raw_close if available,
    # otherwise fall back to 'close' (which may be normalised in legacy files).
    if "raw_close" in df.columns:
        prices = df["raw_close"].to_numpy(dtype=np.float64)
    elif "close" in df.columns:
        prices = df["close"].to_numpy(dtype=np.float64)
    else:
        raise ValueError(f"{path}: missing 'close' / 'raw_close' column (needed for prices array)")

    # Feature columns: drop raw OHLCV + raw_close helper, keep normalised indicators + NLP features
    _PRICE_COLUMNS_EXT = _PRICE_COLUMNS | {"raw_close"}
    feature_cols = [c for c in df.columns if c.lower() not in _PRICE_COLUMNS_EXT]
    if not feature_cols:
        raise ValueError(f"{path}: no feature columns found after dropping OHLCV")

    features = df[feature_cols].to_numpy(dtype=np.float32)

    # Replace any NaN/Inf (from rolling normalisation at start of series) with 0
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    logger.info(
        f"Loaded {len(df)} rows | {features.shape[1]} features | "
        f"{train_start} – {train_end} | asset={asset}"
    )
    return features, prices
