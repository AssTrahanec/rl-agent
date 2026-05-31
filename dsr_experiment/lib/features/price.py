"""Price-side features: OHLCV download (ccxt), technical indicators, rolling z-score normalization."""
import logging

import ccxt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_CANDLE_MS = {
    "1d": 86400000,
    "4h": 14400000,
    "1h": 3600000,
}


def fetch_ohlcv(
    symbol: str,
    start: str,
    end: str,
    exchange_id: str = "binance",
    timeframe: str = "1d",
) -> pd.DataFrame:
    """Fetch OHLCV from exchange via ccxt.

    Returns DataFrame with [open, high, low, close, volume], DatetimeIndex (UTC).
    """
    exchange = getattr(ccxt, exchange_id)()
    since = exchange.parse8601(f"{start}T00:00:00Z")
    end_ts = exchange.parse8601(f"{end}T00:00:00Z")
    candle_ms = _CANDLE_MS.get(timeframe, 86400000)

    all_data = []
    while since < end_ts:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=500)
        if not ohlcv:
            break
        all_data.extend(ohlcv)
        since = ohlcv[-1][0] + candle_ms

    df = pd.DataFrame(all_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index()
    df = df[df.index <= pd.Timestamp(end, tz="UTC")]
    df = df[~df.index.duplicated(keep="first")]
    return df


def add_technical_indicators_minimal(df: pd.DataFrame) -> pd.DataFrame:
    """Минимальный набор из 8 индикаторов по литературному стандарту.

    Покрывает основные категории рынка по одному индикатору на категорию:
    тренд (EMA-26), momentum (MACD), mean-reversion (RSI-14),
    расширение волатильности (BB width), модуль волатильности (ATR-14),
    объём (OBV), позиция в диапазоне (Stoch %K), доходность (return_1d).

    Соответствует рекомендациям FinRL (Liu et al., 2021) и Delft TU (2025)
    о том, что избыточные индикаторы в категориях momentum и volatility
    приводят к переобучению DQN на временных рядах.
    """
    df = df.copy()
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    df["ema_26"] = close.ewm(span=26, adjust=False).mean()

    ema_12 = close.ewm(span=12, adjust=False).mean()
    df["macd"] = ema_12 - df["ema_26"]

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    sma_20 = close.rolling(20).mean()
    std_20 = close.rolling(20).std()
    df["bb_width"] = (4 * std_20) / sma_20

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr_14"] = tr.ewm(com=13, adjust=False).mean()

    direction = np.sign(close.diff()).fillna(0)
    df["obv"] = (direction * volume).cumsum()

    lowest_low = low.rolling(14).min()
    highest_high = high.rolling(14).max()
    denom = (highest_high - lowest_low).replace(0, np.nan)
    df["stoch_k"] = 100 * (close - lowest_low) / denom

    df["return_1d"] = close.pct_change()

    return df


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute ~20 technical indicators."""
    df = df.copy()
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    df["sma_7"] = close.rolling(7).mean()
    df["sma_25"] = close.rolling(25).mean()
    df["ema_12"] = close.ewm(span=12, adjust=False).mean()
    df["ema_26"] = close.ewm(span=26, adjust=False).mean()

    df["macd"] = df["ema_12"] - df["ema_26"]
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    sma_20 = close.rolling(20).mean()
    std_20 = close.rolling(20).std()
    df["bb_upper"] = sma_20 + 2 * std_20
    df["bb_lower"] = sma_20 - 2 * std_20
    df["bb_mid"] = sma_20
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr_14"] = tr.ewm(com=13, adjust=False).mean()

    direction = np.sign(close.diff()).fillna(0)
    df["obv"] = (direction * volume).cumsum()

    lowest_low = low.rolling(14).min()
    highest_high = high.rolling(14).max()
    denom = (highest_high - lowest_low).replace(0, np.nan)
    df["stoch_k"] = 100 * (close - lowest_low) / denom
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()

    df["return_1d"] = close.pct_change()
    df["return_5d"] = close.pct_change(5)

    return df


def rolling_zscore_normalize(df: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    """Rolling z-score normalization of all columns. NaN -> 0."""
    rolling_mean = df.rolling(window=window, min_periods=1).mean()
    rolling_std = df.rolling(window=window, min_periods=1).std().replace(0, 1)
    normalized = (df - rolling_mean) / rolling_std
    return normalized.fillna(0)
