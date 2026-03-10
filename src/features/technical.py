import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical indicators from OHLCV DataFrame.

    Input:  DataFrame with columns [open, high, low, close, volume]
    Output: DataFrame with ~20 additional indicator columns
    """
    df = df.copy()
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # Moving averages
    df["sma_7"] = close.rolling(7).mean()
    df["sma_25"] = close.rolling(25).mean()
    df["ema_12"] = close.ewm(span=12, adjust=False).mean()
    df["ema_26"] = close.ewm(span=26, adjust=False).mean()

    # MACD
    df["macd"] = df["ema_12"] - df["ema_26"]
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # RSI (14)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # Bollinger Bands (20, 2)
    sma_20 = close.rolling(20).mean()
    std_20 = close.rolling(20).std()
    df["bb_upper"] = sma_20 + 2 * std_20
    df["bb_lower"] = sma_20 - 2 * std_20
    df["bb_mid"] = sma_20
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]

    # ATR (14)
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr_14"] = tr.ewm(com=13, adjust=False).mean()

    # OBV
    direction = np.sign(close.diff()).fillna(0)
    df["obv"] = (direction * volume).cumsum()

    # Stochastic %K, %D (14, 3)
    lowest_low = low.rolling(14).min()
    highest_high = high.rolling(14).max()
    denom = (highest_high - lowest_low).replace(0, np.nan)
    df["stoch_k"] = 100 * (close - lowest_low) / denom
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()

    # Returns
    df["return_1d"] = close.pct_change()
    df["return_5d"] = close.pct_change(5)

    logger.debug(f"Added {len(df.columns) - 5} technical indicators")
    return df
