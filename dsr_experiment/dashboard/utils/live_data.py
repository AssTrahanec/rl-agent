"""Live OHLCV fetcher with cached fallback."""
from typing import Tuple

import pandas as pd
import streamlit as st

from dashboard.utils.paths import RAW_OHLCV, ensure_lib_on_path


@st.cache_data(ttl=900)
def fetch_live_ohlcv(
    symbol: str = "BTC/USDT",
    timeframe: str = "4h",
    lookback_bars: int = 120,
    use_live: bool = True,
) -> Tuple[pd.DataFrame, str]:
    """Return (df, source_tag).

    source_tag ∈ {"live", "cached_raw"}.
    DataFrame has columns [open, high, low, close, volume] with DatetimeIndex UTC.
    """
    if use_live:
        try:
            ensure_lib_on_path()
            from lib.features.price import fetch_ohlcv

            now = pd.Timestamp.utcnow().tz_convert("UTC") if pd.Timestamp.utcnow().tzinfo else pd.Timestamp.utcnow().tz_localize("UTC")
            days_back = max(lookback_bars // 6 + 5, 10)
            start = (now - pd.Timedelta(days=days_back)).strftime("%Y-%m-%d")
            end = (now + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            df = fetch_ohlcv(symbol, start, end, timeframe=timeframe)
            if len(df) > lookback_bars:
                df = df.tail(lookback_bars)
            if len(df) >= 31:
                return df, "live"
        except Exception:  # noqa: BLE001
            pass

    if not RAW_OHLCV.exists():
        raise FileNotFoundError(f"No cached OHLCV at {RAW_OHLCV}")
    df = pd.read_parquet(RAW_OHLCV).tail(lookback_bars)
    return df, "cached_raw"


def latest_bar_timestamp(df: pd.DataFrame) -> pd.Timestamp:
    return df.index[-1]
