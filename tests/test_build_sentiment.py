import pandas as pd
import numpy as np
from unittest.mock import patch
from src.features.build_sentiment_features import build_sentiment_features


def _make_price_features():
    """Dummy price features with date index."""
    dates = pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC")
    return pd.DataFrame(
        np.random.randn(10, 5),
        index=dates,
        columns=["f1", "f2", "f3", "f4", "f5"],
    )


def _make_news_by_day():
    """Dummy preprocessed news (grouped by day)."""
    return pd.DataFrame({
        "date": pd.to_datetime(["2024-01-02", "2024-01-05", "2024-01-08"], utc=True),
        "texts": [
            ["Bitcoin surges past 50k"],
            ["Market crash", "Investors panic"],
            ["Ethereum upgrade announced"],
        ],
    })


def test_sentiment_column_exists():
    prices = _make_price_features()
    news = _make_news_by_day()
    with patch("src.features.build_sentiment_features.compute_sentiment") as mock_sent:
        mock_sent.return_value = 0.5
        result = build_sentiment_features(prices, news)
    assert "sentiment" in result.columns


def test_fallback_zero_for_no_news_days():
    prices = _make_price_features()
    news = _make_news_by_day()
    with patch("src.features.build_sentiment_features.compute_sentiment") as mock_sent:
        mock_sent.return_value = 0.8
        result = build_sentiment_features(prices, news)
    # Days without news should have sentiment = 0.0
    no_news_days = result.index.difference(news["date"].dt.normalize())
    assert (result.loc[no_news_days, "sentiment"] == 0.0).all()


def test_output_same_length_as_prices():
    prices = _make_price_features()
    news = _make_news_by_day()
    with patch("src.features.build_sentiment_features.compute_sentiment") as mock_sent:
        mock_sent.return_value = 0.3
        result = build_sentiment_features(prices, news)
    assert len(result) == len(prices)


def test_original_columns_preserved():
    prices = _make_price_features()
    news = _make_news_by_day()
    with patch("src.features.build_sentiment_features.compute_sentiment") as mock_sent:
        mock_sent.return_value = 0.1
        result = build_sentiment_features(prices, news)
    for col in prices.columns:
        assert col in result.columns
