"""Build observation from live OHLCV (zero-filled NLP columns).

Replicates build_data._build_baseline + zero-filled _attach_nlp_features
so that live inference produces features with the same shape as training.
"""
from typing import Tuple

import numpy as np
import pandas as pd
import streamlit as st

from dashboard.utils.paths import TRAIN_FEATURES, ensure_lib_on_path

ensure_lib_on_path()
from lib.data_loader import _PRICE_COLUMNS_EXT

EMB_DIM = 64
NORMALIZE_WINDOW = 30
NEWS_LAGS = [1, 2]
NEWS_ROLL = 7
TOP_PCA_LAGS = 3


@st.cache_data
def expected_feature_count() -> int:
    """Count feature columns in training parquet (excluding price columns)."""
    if not TRAIN_FEATURES.exists():
        raise FileNotFoundError(f"Training features parquet missing: {TRAIN_FEATURES}")
    df = pd.read_parquet(TRAIN_FEATURES)
    return sum(1 for c in df.columns if c.lower() not in _PRICE_COLUMNS_EXT)


@st.cache_data
def expected_feature_columns() -> list[str]:
    df = pd.read_parquet(TRAIN_FEATURES)
    return [c for c in df.columns if c.lower() not in _PRICE_COLUMNS_EXT]


def build_live_features(ohlcv: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Return (features_f32[T,F], prices_f64[T], debug_df).

    Matches the schema of data/oos/*_features.parquet (minus news actually
    computed — here zero-filled).
    """
    from lib.features.price import add_technical_indicators, rolling_zscore_normalize
    from lib.features.lag import add_lag_features, add_rolling_features

    # 1. Technical indicators
    df = add_technical_indicators(ohlcv)

    # 2. Preserve raw close for downstream price access
    df["raw_close"] = df["close"].copy()

    # 3. Rolling z-score normalization on every non-raw_close column
    cols_norm = [c for c in df.columns if c != "raw_close"]
    df[cols_norm] = rolling_zscore_normalize(df[cols_norm], window=NORMALIZE_WINDOW)

    # 4. Zero-fill NLP columns
    for i in range(EMB_DIM):
        df[f"emb_{i}"] = 0.0
    df["news_count"] = 0
    for col in ("sentiment_max", "sentiment_min", "sentiment_mean",
                "sentiment_std", "sentiment_spread"):
        df[col] = 0.0

    # 5. news_count lags
    df = add_lag_features(df, columns=["news_count"], lags=NEWS_LAGS)

    # 6. news_count + sentiment_mean rolling
    df = add_rolling_features(
        df, columns=["news_count", "sentiment_mean"], window=NEWS_ROLL,
    )

    # 7. Top-3 embedding lags
    top_cols = [f"emb_{i}" for i in range(TOP_PCA_LAGS)]
    df = add_lag_features(df, columns=top_cols, lags=NEWS_LAGS)

    # 8. Extract prices + features
    prices = df["raw_close"].to_numpy(dtype=np.float64)
    feature_cols = [c for c in df.columns if c.lower() not in _PRICE_COLUMNS_EXT]
    features = df[feature_cols].to_numpy(dtype=np.float32)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    # Sanity check shape vs training parquet
    exp_count = expected_feature_count()
    if features.shape[1] != exp_count:
        exp_cols = set(expected_feature_columns())
        got_cols = set(feature_cols)
        missing = sorted(exp_cols - got_cols)
        extra = sorted(got_cols - exp_cols)
        raise ValueError(
            f"Feature schema mismatch: expected {exp_count} columns, got {features.shape[1]}.\n"
            f"  Missing: {missing[:5]}{'...' if len(missing) > 5 else ''}\n"
            f"  Extra: {extra[:5]}{'...' if len(extra) > 5 else ''}"
        )

    return features, prices, df


def build_live_obs(features: np.ndarray,
                   prev_allocation: float = 0.0,
                   window: int = 30) -> np.ndarray:
    """Reproduce TradingEnv._get_obs: flatten last `window` rows + prev_allocation."""
    if len(features) < window:
        raise ValueError(f"Need >= {window} rows, got {len(features)}")
    return np.append(features[-window:].flatten(), prev_allocation).astype(np.float32)
