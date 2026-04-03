import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from src.features.build_embedding_features import build_embedding_features


def _make_price_features():
    dates = pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC")
    return pd.DataFrame(
        np.random.randn(10, 5),
        index=dates,
        columns=["f1", "f2", "f3", "f4", "f5"],
    )


def _make_news_by_day():
    return pd.DataFrame({
        "date": pd.to_datetime(["2024-01-02", "2024-01-05", "2024-01-08"], utc=True),
        "texts": [
            ["Bitcoin surges past 50k"],
            ["Market crash", "Investors panic"],
            ["Ethereum upgrade announced"],
        ],
    })


def _mock_compressor():
    compressor = MagicMock()
    compressor.transform.return_value = np.random.randn(1, 64).astype(np.float32)
    return compressor


def test_embedding_columns_added():
    prices = _make_price_features()
    news = _make_news_by_day()
    comp = _mock_compressor()
    with patch("src.features.build_embedding_features.compute_embeddings") as mock_emb, \
         patch("src.features.build_embedding_features.compute_sentiment", return_value=0.5):
        mock_emb.return_value = np.random.randn(768).astype(np.float32)
        result = build_embedding_features(prices, news, compressor=comp)
    emb_cols = [c for c in result.columns if c.startswith("emb_")]
    assert len(emb_cols) == 64


def test_fallback_zeros_for_no_news_days():
    prices = _make_price_features()
    news = _make_news_by_day()
    comp = _mock_compressor()
    with patch("src.features.build_embedding_features.compute_embeddings") as mock_emb, \
         patch("src.features.build_embedding_features.compute_sentiment", return_value=0.5):
        mock_emb.return_value = np.ones(768, dtype=np.float32)
        result = build_embedding_features(prices, news, compressor=comp)
    # Days without news should have emb_ columns = 0.0
    no_news_day = pd.Timestamp("2024-01-03", tz="UTC")
    emb_cols = [c for c in result.columns if c.startswith("emb_")]
    assert (result.loc[no_news_day, emb_cols] == 0.0).all()


def test_output_same_length():
    prices = _make_price_features()
    news = _make_news_by_day()
    comp = _mock_compressor()
    with patch("src.features.build_embedding_features.compute_embeddings") as mock_emb, \
         patch("src.features.build_embedding_features.compute_sentiment", return_value=0.5):
        mock_emb.return_value = np.random.randn(768).astype(np.float32)
        result = build_embedding_features(prices, news, compressor=comp)
    assert len(result) == len(prices)


def test_original_columns_preserved():
    prices = _make_price_features()
    news = _make_news_by_day()
    comp = _mock_compressor()
    with patch("src.features.build_embedding_features.compute_embeddings") as mock_emb, \
         patch("src.features.build_embedding_features.compute_sentiment", return_value=0.5):
        mock_emb.return_value = np.random.randn(768).astype(np.float32)
        result = build_embedding_features(prices, news, compressor=comp)
    for col in prices.columns:
        assert col in result.columns


def test_news_count_column_added():
    """build_embedding_features adds news_count column."""
    prices = _make_price_features()
    news = _make_news_by_day()
    comp = _mock_compressor()
    with patch("src.features.build_embedding_features.compute_embeddings") as mock_emb, \
         patch("src.features.build_embedding_features.compute_sentiment", return_value=0.5):
        mock_emb.return_value = np.zeros(768, dtype=np.float32)
        result = build_embedding_features(prices, news, compressor=comp)
    assert "news_count" in result.columns
    # 2024-01-02 has 1 article, 2024-01-05 has 2 articles
    assert result.loc[pd.Timestamp("2024-01-02", tz="UTC"), "news_count"] == 1
    assert result.loc[pd.Timestamp("2024-01-05", tz="UTC"), "news_count"] == 2
    # No news day
    assert result.loc[pd.Timestamp("2024-01-03", tz="UTC"), "news_count"] == 0


def test_embedding_columns_are_64d():
    """build_embedding_features produces 64 emb columns."""
    prices = _make_price_features()
    news = _make_news_by_day()
    comp = _mock_compressor()
    with patch("src.features.build_embedding_features.compute_embeddings") as mock_emb, \
         patch("src.features.build_embedding_features.compute_sentiment", return_value=0.5):
        mock_emb.return_value = np.zeros(768, dtype=np.float32)
        result = build_embedding_features(prices, news, compressor=comp)
    emb_cols = [c for c in result.columns if c.startswith("emb_")]
    assert len(emb_cols) == 64
