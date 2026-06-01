import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from src.features.build_embedding_features import (
    build_embedding_features,
    _deduplicate_embeddings,
    _temporal_weights,
)


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
    compressor.transform.return_value = np.random.randn(1, 20).astype(np.float32)
    return compressor


def _make_mock_model():
    """Return a mock ST model whose encode() returns (n_texts, 768) dynamically."""
    mock_model = MagicMock()
    mock_model.encode.side_effect = lambda texts, **kwargs: np.random.randn(
        len(texts), 768
    ).astype(np.float32)
    return mock_model


def test_embedding_columns_added():
    prices = _make_price_features()
    news = _make_news_by_day()
    comp = _mock_compressor()
    with patch("src.features.build_embedding_features._get_model") as mock_get_model, \
         patch("src.features.build_embedding_features.compute_sentiment_scores", side_effect=lambda texts: [0.5] * len(texts)):
        mock_get_model.return_value = _make_mock_model()
        result = build_embedding_features(prices, news, compressor=comp)
    # Only base emb columns (not lag columns like emb_0_lag1)
    emb_cols = [c for c in result.columns if c.startswith("emb_") and "_lag" not in c]
    assert len(emb_cols) == 20


def test_fallback_zeros_for_no_news_days():
    prices = _make_price_features()
    news = _make_news_by_day()
    comp = _mock_compressor()
    with patch("src.features.build_embedding_features._get_model") as mock_get_model, \
         patch("src.features.build_embedding_features.compute_sentiment_scores", side_effect=lambda texts: [0.5] * len(texts)):
        mock_get_model.return_value = _make_mock_model()
        result = build_embedding_features(prices, news, compressor=comp)
    # Days without news should have base emb_ columns = 0.0
    no_news_day = pd.Timestamp("2024-01-03", tz="UTC")
    emb_cols = [c for c in result.columns if c.startswith("emb_") and "_lag" not in c]
    assert (result.loc[no_news_day, emb_cols] == 0.0).all()


def test_output_same_length():
    prices = _make_price_features()
    news = _make_news_by_day()
    comp = _mock_compressor()
    with patch("src.features.build_embedding_features._get_model") as mock_get_model, \
         patch("src.features.build_embedding_features.compute_sentiment_scores", side_effect=lambda texts: [0.5] * len(texts)):
        mock_get_model.return_value = _make_mock_model()
        result = build_embedding_features(prices, news, compressor=comp)
    assert len(result) == len(prices)


def test_original_columns_preserved():
    prices = _make_price_features()
    news = _make_news_by_day()
    comp = _mock_compressor()
    with patch("src.features.build_embedding_features._get_model") as mock_get_model, \
         patch("src.features.build_embedding_features.compute_sentiment_scores", side_effect=lambda texts: [0.5] * len(texts)):
        mock_get_model.return_value = _make_mock_model()
        result = build_embedding_features(prices, news, compressor=comp)
    for col in prices.columns:
        assert col in result.columns


def test_news_count_column_added():
    """build_embedding_features adds news_count column."""
    prices = _make_price_features()
    news = _make_news_by_day()
    comp = _mock_compressor()
    with patch("src.features.build_embedding_features._get_model") as mock_get_model, \
         patch("src.features.build_embedding_features.compute_sentiment_scores", side_effect=lambda texts: [0.5] * len(texts)):
        mock_get_model.return_value = _make_mock_model()
        result = build_embedding_features(prices, news, compressor=comp)
    assert "news_count" in result.columns
    # 2024-01-02 has 1 article, 2024-01-05 has 2 articles
    assert result.loc[pd.Timestamp("2024-01-02", tz="UTC"), "news_count"] == 1
    assert result.loc[pd.Timestamp("2024-01-05", tz="UTC"), "news_count"] == 2
    # No news day
    assert result.loc[pd.Timestamp("2024-01-03", tz="UTC"), "news_count"] == 0


def test_embedding_columns_are_20d():
    """build_embedding_features produces 20 emb columns."""
    prices = _make_price_features()
    news = _make_news_by_day()
    comp = _mock_compressor()
    with patch("src.features.build_embedding_features._get_model") as mock_get_model, \
         patch("src.features.build_embedding_features.compute_sentiment_scores", side_effect=lambda texts: [0.5] * len(texts)):
        mock_get_model.return_value = _make_mock_model()
        result = build_embedding_features(prices, news, compressor=comp)
    # Only base emb columns (not lag columns like emb_0_lag1)
    emb_cols = [c for c in result.columns if c.startswith("emb_") and "_lag" not in c]
    assert len(emb_cols) == 20


def test_sentiment_extreme_columns_exist():
    """build_embedding_features adds sentiment_max, sentiment_min, sentiment_spread."""
    prices = _make_price_features()
    news = _make_news_by_day()
    comp = _mock_compressor()
    with patch("src.features.build_embedding_features._get_model") as mock_get_model, \
         patch("src.features.build_embedding_features.compute_sentiment_scores") as mock_sent:
        mock_get_model.return_value = _make_mock_model()
        mock_sent.side_effect = lambda texts: [0.5] * len(texts) if "surges" in texts[0] else [-0.3] * len(texts)
        result = build_embedding_features(prices, news, compressor=comp)
    assert "sentiment_max" in result.columns
    assert "sentiment_min" in result.columns
    assert "sentiment_spread" in result.columns


def test_sentiment_extremes_values():
    """sentiment_max/min capture per-article extremes, not mean."""
    prices = _make_price_features()
    # Day with 2 articles: one positive, one negative
    news = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-02"], utc=True),
        "texts": [["Great news for BTC", "Terrible crash coming"]],
    })
    comp = _mock_compressor()
    with patch("src.features.build_embedding_features._get_model") as mock_get_model, \
         patch("src.features.build_embedding_features.compute_sentiment_scores") as mock_sent:
        mock_get_model.return_value = _make_mock_model()
        mock_sent.side_effect = lambda texts: [0.8, -0.6]
        result = build_embedding_features(prices, news, compressor=comp)
    day = pd.Timestamp("2024-01-02", tz="UTC")
    assert result.loc[day, "sentiment_max"] == 0.8
    assert result.loc[day, "sentiment_min"] == -0.6
    assert abs(result.loc[day, "sentiment_spread"] - 1.4) < 1e-6


def test_pca_lag_columns_exist():
    """build_embedding_features adds emb_0_lag1..emb_2_lag2 (top-3 PCA lags)."""
    prices = _make_price_features()
    news = _make_news_by_day()
    comp = _mock_compressor()
    with patch("src.features.build_embedding_features._get_model") as mock_get_model, \
         patch("src.features.build_embedding_features.compute_sentiment_scores", side_effect=lambda texts: [0.5] * len(texts)):
        mock_get_model.return_value = _make_mock_model()
        result = build_embedding_features(prices, news, compressor=comp)
    for i in range(3):
        assert f"emb_{i}_lag1" in result.columns
        assert f"emb_{i}_lag2" in result.columns


def test_news_count_lag_columns_exist():
    """build_embedding_features adds news_count_lag1, news_count_lag2, news_count_roll7."""
    prices = _make_price_features()
    news = _make_news_by_day()
    comp = _mock_compressor()
    with patch("src.features.build_embedding_features._get_model") as mock_get_model, \
         patch("src.features.build_embedding_features.compute_sentiment_scores", side_effect=lambda texts: [0.5] * len(texts)):
        mock_get_model.return_value = _make_mock_model()
        result = build_embedding_features(prices, news, compressor=comp)
    assert "news_count_lag1" in result.columns
    assert "news_count_lag2" in result.columns
    assert "news_count_roll7" in result.columns


def test_news_count_lag_values_correct():
    """news_count_lag1 should be previous day's news_count."""
    prices = _make_price_features()
    news = _make_news_by_day()
    comp = _mock_compressor()
    with patch("src.features.build_embedding_features._get_model") as mock_get_model, \
         patch("src.features.build_embedding_features.compute_sentiment_scores", side_effect=lambda texts: [0.5] * len(texts)):
        mock_get_model.return_value = _make_mock_model()
        result = build_embedding_features(prices, news, compressor=comp)
    day_after = pd.Timestamp("2024-01-03", tz="UTC")
    assert result.loc[day_after, "news_count_lag1"] == 1


# ── Task 2: Cosine similarity dedup ──────────────────────────────────────────

def test_cosine_dedup_removes_similar():
    """Embeddings with cosine sim > 0.85 are deduplicated."""
    base = np.random.randn(768).astype(np.float32)
    similar = base + np.random.randn(768).astype(np.float32) * 0.01
    different = np.random.randn(768).astype(np.float32)
    embeddings = np.stack([base, similar, different])
    texts = ["text A", "text A repost", "text B"]
    scores = [0.5, 0.4, -0.3]

    deduped_texts, deduped_scores, deduped_embs = _deduplicate_embeddings(
        texts, scores, embeddings, threshold=0.85
    )
    assert len(deduped_texts) == 2
    assert len(deduped_scores) == 2
    assert deduped_embs.shape[0] == 2


def test_cosine_dedup_keeps_all_when_different():
    """No dedup when all embeddings are different."""
    embeddings = np.random.randn(3, 768).astype(np.float32)
    texts = ["a", "b", "c"]
    scores = [0.1, 0.2, 0.3]
    deduped_texts, deduped_scores, deduped_embs = _deduplicate_embeddings(
        texts, scores, embeddings, threshold=0.85
    )
    assert len(deduped_texts) == 3


# ── Task 3: Temporal decay weighting ─────────────────────────────────────────

def test_temporal_weights_recent_higher():
    """More recent articles (closer to midnight) get higher weight."""
    timestamps = pd.to_datetime([
        "2024-01-05 06:00:00",
        "2024-01-05 12:00:00",
        "2024-01-05 23:00:00",
    ], utc=True)
    day_close = pd.Timestamp("2024-01-05 23:59:59", tz="UTC")
    weights = _temporal_weights(timestamps, day_close, alpha=0.1)
    assert len(weights) == 3
    assert weights[2] > weights[1] > weights[0]


def test_temporal_weights_single_article():
    """Single article gets weight 1.0."""
    timestamps = pd.to_datetime(["2024-01-05 12:00:00"], utc=True)
    day_close = pd.Timestamp("2024-01-05 23:59:59", tz="UTC")
    weights = _temporal_weights(timestamps, day_close, alpha=0.1)
    assert len(weights) == 1
    assert abs(weights[0] - 1.0) < 1e-6
