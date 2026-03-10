import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from src.data.price_collector import fetch_ohlcv


MOCK_OHLCV = [
    [1704067200000, 42000.0, 43000.0, 41500.0, 42500.0, 1000.0],
    [1704153600000, 42500.0, 44000.0, 42000.0, 43800.0, 1200.0],
    [1704240000000, 43800.0, 44500.0, 43000.0, 44200.0, 900.0],
]


def make_mock_exchange(ohlcv_data=None):
    exchange = MagicMock()
    exchange.parse8601.side_effect = lambda s: int(
        pd.Timestamp(s).timestamp() * 1000
    )
    exchange.fetch_ohlcv.return_value = ohlcv_data if ohlcv_data is not None else MOCK_OHLCV
    return exchange


def test_fetch_ohlcv_returns_dataframe():
    exchange = make_mock_exchange()
    exchange.fetch_ohlcv.side_effect = [MOCK_OHLCV, []]

    with patch("src.data.price_collector.ccxt") as mock_ccxt:
        mock_ccxt.binance.return_value = exchange
        df = fetch_ohlcv("BTC/USDT", "2024-01-01", "2024-01-31")

    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) >= {"open", "high", "low", "close", "volume"}
    assert len(df) > 0
    assert df.index.is_monotonic_increasing


def test_fetch_ohlcv_no_nulls():
    exchange = make_mock_exchange()
    exchange.fetch_ohlcv.side_effect = [MOCK_OHLCV, []]

    with patch("src.data.price_collector.ccxt") as mock_ccxt:
        mock_ccxt.binance.return_value = exchange
        df = fetch_ohlcv("BTC/USDT", "2024-01-01", "2024-01-31")

    assert df.isnull().sum().sum() == 0


def test_fetch_ohlcv_correct_columns():
    exchange = make_mock_exchange()
    exchange.fetch_ohlcv.side_effect = [MOCK_OHLCV, []]

    with patch("src.data.price_collector.ccxt") as mock_ccxt:
        mock_ccxt.binance.return_value = exchange
        df = fetch_ohlcv("BTC/USDT", "2024-01-01", "2024-01-31")

    assert list(df.columns) == ["open", "high", "low", "close", "volume"]


def test_fetch_ohlcv_index_is_datetime():
    exchange = make_mock_exchange()
    exchange.fetch_ohlcv.side_effect = [MOCK_OHLCV, []]

    with patch("src.data.price_collector.ccxt") as mock_ccxt:
        mock_ccxt.binance.return_value = exchange
        df = fetch_ohlcv("BTC/USDT", "2024-01-01", "2024-01-31")

    assert isinstance(df.index, pd.DatetimeIndex)


@pytest.mark.integration
def test_fetch_ohlcv_live_binance():
    """Integration test — requires internet access to Binance API."""
    df = fetch_ohlcv("BTC/USDT", "2024-01-01", "2024-01-31")
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert df.isnull().sum().sum() == 0
