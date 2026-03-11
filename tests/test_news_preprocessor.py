import pandas as pd
import numpy as np
from src.data.news_preprocessor import preprocess_news


def _make_raw_news():
    """Create raw news DataFrame with duplicates and multiple articles per day."""
    return pd.DataFrame({
        "title": [
            "Bitcoin surges",
            "Bitcoin surges",  # duplicate
            "ETH upgrade",
            "Market crash",
            "BTC rally continues",
        ],
        "text": [
            "Bitcoin surges past 50k.",
            "Bitcoin surges past 50k.",  # duplicate
            "Ethereum devs announce upgrade.",
            "The crypto market crashed.",
            "BTC rally continues today.",
        ],
        "date": pd.to_datetime([
            "2024-01-05",
            "2024-01-05",
            "2024-01-05",
            "2024-01-06",
            "2024-01-07",
        ], utc=True),
    })


def test_no_duplicate_titles():
    df = _make_raw_news()
    result = preprocess_news(df)
    # After dedup, each day's texts list should not contain duplicates
    for texts in result["texts"]:
        assert len(texts) == len(set(texts))


def test_grouped_by_day():
    df = _make_raw_news()
    result = preprocess_news(df)
    # Should have 3 unique days
    assert len(result) == 3
    assert "date" in result.columns
    assert "texts" in result.columns


def test_texts_is_list():
    df = _make_raw_news()
    result = preprocess_news(df)
    for texts in result["texts"]:
        assert isinstance(texts, list)
        assert all(isinstance(t, str) for t in texts)


def test_date_range_filter():
    df = _make_raw_news()
    result = preprocess_news(df)
    # All dates should be valid
    assert result["date"].is_monotonic_increasing


def test_day_with_multiple_articles():
    df = _make_raw_news()
    result = preprocess_news(df)
    # Jan 5 has 2 unique articles (after dedup from 3 rows)
    jan5 = result[result["date"] == pd.Timestamp("2024-01-05", tz="UTC")]
    assert len(jan5) == 1
    assert len(jan5.iloc[0]["texts"]) == 2
