import pandas as pd
import numpy as np
from src.features.technical import add_technical_indicators


def make_dummy_ohlcv(n=100):
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(n))
    return pd.DataFrame({
        "open": close + np.random.randn(n) * 0.5,
        "high": close + abs(np.random.randn(n)),
        "low": close - abs(np.random.randn(n)),
        "close": close,
        "volume": np.random.randint(1000, 10000, n).astype(float),
    })


def test_indicators_added():
    df = make_dummy_ohlcv()
    result = add_technical_indicators(df)
    expected_cols = ["sma_7", "sma_25", "rsi_14", "macd", "bb_upper", "atr_14", "obv"]
    for col in expected_cols:
        assert col in result.columns, f"Missing {col}"


def test_rsi_range():
    df = make_dummy_ohlcv(200)
    result = add_technical_indicators(df).dropna()
    assert result["rsi_14"].between(0, 100).all()


def test_original_columns_preserved():
    df = make_dummy_ohlcv()
    result = add_technical_indicators(df)
    assert set(["open", "high", "low", "close", "volume"]).issubset(result.columns)


def test_output_has_more_columns_than_input():
    df = make_dummy_ohlcv()
    result = add_technical_indicators(df)
    assert len(result.columns) > len(df.columns)


def test_obv_is_cumulative():
    df = make_dummy_ohlcv(50)
    result = add_technical_indicators(df)
    # OBV should not be constant if prices and volume vary
    assert result["obv"].nunique() > 1
