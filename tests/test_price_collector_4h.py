# tests/test_price_collector_4h.py
import pandas as pd
from unittest.mock import patch, MagicMock
from src.data.price_collector import fetch_ohlcv


def test_fetch_4h_passes_timeframe():
    """fetch_ohlcv with timeframe='4h' passes '4h' to exchange."""
    mock_exchange = MagicMock()
    mock_exchange.parse8601.side_effect = [1000, 9999999]
    mock_exchange.fetch_ohlcv.return_value = [
        [1000, 100, 105, 95, 102, 500],
        [1000 + 14400000, 102, 108, 100, 106, 600],  # +4h in ms
    ]

    with patch("ccxt.binance", return_value=mock_exchange):
        df = fetch_ohlcv("BTC/USDT", "2024-01-01", "2024-01-02", timeframe="4h")

    mock_exchange.fetch_ohlcv.assert_called_once()
    call_args = mock_exchange.fetch_ohlcv.call_args
    assert call_args[0][1] == "4h"  # timeframe argument
    assert len(df) == 2


def test_fetch_4h_returns_correct_index():
    """4h data should have multiple rows per day."""
    mock_exchange = MagicMock()
    base_ts = 1704067200000  # 2024-01-01 00:00 UTC
    mock_exchange.parse8601.side_effect = [base_ts, base_ts + 86400000]
    mock_exchange.fetch_ohlcv.return_value = [
        [base_ts + i * 14400000, 100 + i, 105, 95, 102, 500]
        for i in range(6)  # 6 candles per day
    ]

    with patch("ccxt.binance", return_value=mock_exchange):
        df = fetch_ohlcv("BTC/USDT", "2024-01-01", "2024-01-02", timeframe="4h")

    assert len(df) == 6
    assert isinstance(df.index, pd.DatetimeIndex)
