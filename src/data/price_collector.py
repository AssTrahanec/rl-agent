import ccxt
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def fetch_ohlcv(symbol: str, start: str, end: str, exchange_id: str = "binance") -> pd.DataFrame:
    """Fetch daily OHLCV data from exchange via ccxt."""
    exchange = getattr(ccxt, exchange_id)()
    since = exchange.parse8601(f"{start}T00:00:00Z")
    end_ts = exchange.parse8601(f"{end}T00:00:00Z")

    all_data = []
    while since < end_ts:
        ohlcv = exchange.fetch_ohlcv(symbol, "1d", since=since, limit=500)
        if not ohlcv:
            break
        all_data.extend(ohlcv)
        since = ohlcv[-1][0] + 86400000  # next day

    df = pd.DataFrame(all_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index()
    df = df[df.index <= pd.Timestamp(end, tz="UTC")]
    df = df[~df.index.duplicated(keep="first")]
    return df


def save_prices(symbol: str, start: str, end: str, output_path: str):
    df = fetch_ohlcv(symbol, start, end)
    df.to_parquet(output_path)
    logger.info(f"Saved {len(df)} rows to {output_path}")
