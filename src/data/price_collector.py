import ccxt
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# Candle duration in milliseconds for pagination
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
    """Fetch OHLCV data from exchange via ccxt.

    Args:
        symbol: Trading pair, e.g. 'BTC/USDT'.
        start: Start date string, e.g. '2020-01-01'.
        end: End date string, e.g. '2024-12-31'.
        exchange_id: ccxt exchange id (default: 'binance').
        timeframe: Candle interval — '1d', '4h', or '1h' (default: '1d').

    Returns:
        DataFrame with columns [open, high, low, close, volume], DatetimeIndex (UTC).
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


def save_prices(symbol: str, start: str, end: str, output_path: str, timeframe: str = "1d"):
    df = fetch_ohlcv(symbol, start, end, timeframe=timeframe)
    df.to_parquet(output_path)
    logger.info(f"Saved {len(df)} rows ({timeframe}) to {output_path}")
