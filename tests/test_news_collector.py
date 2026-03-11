import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from src.data.news_collector import load_news, REQUIRED_COLUMNS


MOCK_NEWS_DATA = {
    "title": [
        "Bitcoin surges past 50k",
        "Ethereum upgrade announced",
        "Crypto market crashes",
        "BTC hits new all-time high",
    ],
    "text": [
        "Bitcoin has surged past the 50k mark today.",
        "Ethereum devs announce major upgrade.",
        "The crypto market has crashed by 20%.",
        "Bitcoin has reached a new all-time high of 69k.",
    ],
    "date": [
        "2024-01-05",
        "2024-01-10",
        "2024-01-15",
        "2024-02-01",
    ],
}


def _make_mock_dataset():
    """Create a mock HF dataset that behaves like datasets.Dataset."""
    ds = MagicMock()
    ds.to_pandas.return_value = pd.DataFrame(MOCK_NEWS_DATA)
    return ds


def test_load_news_returns_dataframe():
    mock_ds = _make_mock_dataset()
    with patch("src.data.news_collector.hf_load_dataset") as mock_load:
        mock_load.return_value = {"train": mock_ds}
        df = load_news("fake/dataset", start="2024-01-01", end="2024-12-31")

    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_load_news_has_required_columns():
    mock_ds = _make_mock_dataset()
    with patch("src.data.news_collector.hf_load_dataset") as mock_load:
        mock_load.return_value = {"train": mock_ds}
        df = load_news("fake/dataset", start="2024-01-01", end="2024-12-31")

    for col in REQUIRED_COLUMNS:
        assert col in df.columns, f"Missing column: {col}"


def test_load_news_filters_by_date():
    mock_ds = _make_mock_dataset()
    with patch("src.data.news_collector.hf_load_dataset") as mock_load:
        mock_load.return_value = {"train": mock_ds}
        df = load_news("fake/dataset", start="2024-01-01", end="2024-01-20")

    # Only 3 of 4 rows should survive (2024-02-01 is outside range)
    assert len(df) == 3
    assert (df["date"] <= pd.Timestamp("2024-01-20", tz="UTC")).all()


def test_load_news_date_is_datetime():
    mock_ds = _make_mock_dataset()
    with patch("src.data.news_collector.hf_load_dataset") as mock_load:
        mock_load.return_value = {"train": mock_ds}
        df = load_news("fake/dataset", start="2024-01-01", end="2024-12-31")

    assert pd.api.types.is_datetime64_any_dtype(df["date"])
