import pandas as pd
import numpy as np
from src.features.lag_features import add_lag_features, add_rolling_features


def test_lag_adds_correct_columns():
    """add_lag_features creates _lag1 and _lag2 columns."""
    idx = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
    df = pd.DataFrame({"sentiment": [0.1, 0.5, -0.3, 0.8, 0.0]}, index=idx)
    result = add_lag_features(df, columns=["sentiment"], lags=[1, 2])
    assert "sentiment_lag1" in result.columns
    assert "sentiment_lag2" in result.columns


def test_lag_values_are_shifted():
    """Lag-1 should equal previous day's value."""
    idx = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
    df = pd.DataFrame({"sentiment": [0.1, 0.5, -0.3, 0.8, 0.0]}, index=idx)
    result = add_lag_features(df, columns=["sentiment"], lags=[1, 2])
    assert result["sentiment_lag1"].iloc[2] == 0.5
    assert result["sentiment_lag2"].iloc[2] == 0.1


def test_lag_fills_nan_with_zero():
    """First rows where lag is unavailable should be filled with 0."""
    idx = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
    df = pd.DataFrame({"sentiment": [0.1, 0.5, -0.3, 0.8, 0.0]}, index=idx)
    result = add_lag_features(df, columns=["sentiment"], lags=[1, 2])
    assert result["sentiment_lag1"].iloc[0] == 0.0
    assert result["sentiment_lag2"].iloc[0] == 0.0
    assert result["sentiment_lag2"].iloc[1] == 0.0


def test_lag_preserves_original_columns():
    """Original columns should remain unchanged."""
    idx = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
    df = pd.DataFrame({
        "price": [100, 101, 102, 103, 104],
        "sentiment": [0.1, 0.5, -0.3, 0.8, 0.0],
    }, index=idx)
    result = add_lag_features(df, columns=["sentiment"], lags=[1, 2])
    pd.testing.assert_series_equal(result["price"], df["price"])
    pd.testing.assert_series_equal(result["sentiment"], df["sentiment"])


def test_lag_multiple_columns():
    """Can lag multiple columns at once."""
    idx = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
    df = pd.DataFrame({
        "sentiment": [0.1, 0.5, -0.3, 0.8, 0.0],
        "news_count": [3, 5, 0, 2, 7],
    }, index=idx)
    result = add_lag_features(df, columns=["sentiment", "news_count"], lags=[1, 2])
    assert "sentiment_lag1" in result.columns
    assert "news_count_lag1" in result.columns
    assert "news_count_lag2" in result.columns
    assert result["news_count_lag1"].iloc[2] == 5


def test_rolling_adds_column():
    """add_rolling_features creates rolling mean column."""
    idx = pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC")
    df = pd.DataFrame({"news_count": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}, index=idx)
    result = add_rolling_features(df, columns=["news_count"], window=3)
    assert "news_count_roll3" in result.columns


def test_rolling_values_correct():
    """Rolling mean should average over window."""
    idx = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
    df = pd.DataFrame({"val": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=idx)
    result = add_rolling_features(df, columns=["val"], window=3)
    # Row 3 (index 2): mean(1, 2, 3) = 2.0
    assert result["val_roll3"].iloc[2] == 2.0
    # Row 4 (index 3): mean(2, 3, 4) = 3.0
    assert result["val_roll3"].iloc[3] == 3.0


def test_rolling_fills_nan_with_zero():
    """First rows where window is incomplete should be 0."""
    idx = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
    df = pd.DataFrame({"val": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=idx)
    result = add_rolling_features(df, columns=["val"], window=3)
    assert result["val_roll3"].iloc[0] == 0.0
    assert result["val_roll3"].iloc[1] == 0.0
