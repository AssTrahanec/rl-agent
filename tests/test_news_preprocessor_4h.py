# tests/test_news_preprocessor_4h.py
import pandas as pd
from src.data.news_preprocessor import preprocess_news_4h


def test_groups_by_4h_window():
    """News at 10:00 and 11:00 should be in the same 08:00-12:00 window."""
    news = pd.DataFrame({
        "title": ["A", "B", "C"],
        "text": ["text A", "text B", "text C"],
        "date": pd.to_datetime([
            "2024-01-01 10:00:00",
            "2024-01-01 11:30:00",
            "2024-01-01 14:00:00",
        ], utc=True),
    })
    result = preprocess_news_4h(news)
    # 10:00 and 11:30 -> window 08:00; 14:00 -> window 12:00
    assert len(result) == 2
    assert len(result.iloc[0]["texts"]) == 2
    assert len(result.iloc[1]["texts"]) == 1


def test_4h_window_column_name():
    """Result should have 'date' column with 4h-floored timestamps."""
    news = pd.DataFrame({
        "title": ["A"],
        "text": ["text"],
        "date": pd.to_datetime(["2024-01-01 10:30:00"], utc=True),
    })
    result = preprocess_news_4h(news)
    assert "date" in result.columns
    assert "texts" in result.columns
    # 10:30 floors to 08:00
    assert result.iloc[0]["date"] == pd.Timestamp("2024-01-01 08:00:00", tz="UTC")


def test_4h_deduplicates():
    """Duplicate titles should be removed."""
    news = pd.DataFrame({
        "title": ["Same", "Same", "Different"],
        "text": ["text1", "text1", "text2"],
        "date": pd.to_datetime([
            "2024-01-01 10:00:00",
            "2024-01-01 10:05:00",
            "2024-01-01 10:10:00",
        ], utc=True),
    })
    result = preprocess_news_4h(news)
    assert len(result) == 1
    assert len(result.iloc[0]["texts"]) == 2  # 2 unique articles


def test_4h_sorted():
    """Result should be sorted by date."""
    news = pd.DataFrame({
        "title": ["B", "A"],
        "text": ["text B", "text A"],
        "date": pd.to_datetime([
            "2024-01-02 10:00:00",
            "2024-01-01 10:00:00",
        ], utc=True),
    })
    result = preprocess_news_4h(news)
    assert result.iloc[0]["date"] < result.iloc[1]["date"]


def test_4h_preserves_all_windows():
    """6 different 4h windows in one day should produce 6 rows."""
    news = pd.DataFrame({
        "title": [f"News {i}" for i in range(6)],
        "text": [f"text {i}" for i in range(6)],
        "date": pd.to_datetime([
            "2024-01-01 01:00:00",  # 00:00 window
            "2024-01-01 05:00:00",  # 04:00 window
            "2024-01-01 09:00:00",  # 08:00 window
            "2024-01-01 13:00:00",  # 12:00 window
            "2024-01-01 17:00:00",  # 16:00 window
            "2024-01-01 21:00:00",  # 20:00 window
        ], utc=True),
    })
    result = preprocess_news_4h(news)
    assert len(result) == 6
