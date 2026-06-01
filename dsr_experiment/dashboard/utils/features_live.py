"""Build observation from live OHLCV (zero-filled NLP columns).

Replicates build_data._build_baseline + the minimal NLP attach so that live
inference produces features with the same 41-column schema as training
(8 indicators + sentiment_mean + 32 PCA embeddings).
"""
from typing import Tuple

import numpy as np
import pandas as pd
import streamlit as st

from dashboard.utils.paths import TRAIN_FEATURES, ensure_lib_on_path

ensure_lib_on_path()
from lib.data_loader import _PRICE_COLUMNS_EXT

EMB_DIM = 32
NORMALIZE_WINDOW = 30


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

    Matches the 41-column schema of data/train/features.parquet; news columns are
    zero-filled here (live news is injected separately by feed_decisions/news_impact).
    """
    from lib.features.price import add_technical_indicators_minimal, rolling_zscore_normalize

    # 1. Technical indicators (8, same as training)
    df = add_technical_indicators_minimal(ohlcv)

    # 2. Preserve raw close for downstream price access
    df["raw_close"] = df["close"].copy()

    # 3. Rolling z-score normalization on every non-raw_close column
    cols_norm = [c for c in df.columns if c != "raw_close"]
    df[cols_norm] = rolling_zscore_normalize(df[cols_norm], window=NORMALIZE_WINDOW)

    # 4. Zero-fill NLP columns (no news on plain historical bars)
    for i in range(EMB_DIM):
        df[f"emb_{i}"] = 0.0
    df["sentiment_mean"] = 0.0

    # 5. Extract prices + features in the exact training column order
    prices = df["raw_close"].to_numpy(dtype=np.float64)
    target_cols = expected_feature_columns()
    missing = [c for c in target_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Live features missing training columns: {missing[:5]}")
    features = df.reindex(columns=target_cols).to_numpy(dtype=np.float32)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return features, prices, df


def build_live_obs(features: np.ndarray,
                   prev_allocation: float = 0.0,
                   window: int = 30) -> np.ndarray:
    """Reproduce TradingEnv._get_obs: flatten last `window` rows + prev_allocation."""
    if len(features) < window:
        raise ValueError(f"Need >= {window} rows, got {len(features)}")
    return np.append(features[-window:].flatten(), prev_allocation).astype(np.float32)
