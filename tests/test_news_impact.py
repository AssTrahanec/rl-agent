"""Tests for the News Analyzer logic (dashboard/utils/news_impact.py)."""
import sys
from pathlib import Path

import numpy as np
import pytest

# Add dsr_experiment/ to sys.path so `dashboard...` and `lib...` resolve.
_DSR_EXP = Path(__file__).resolve().parent.parent / "dsr_experiment"
if str(_DSR_EXP) not in sys.path:
    sys.path.insert(0, str(_DSR_EXP))

from dashboard.utils.news_impact import inject_news_features, score_news_batch  # noqa: E402


def _columns():
    cols = ["rsi_14", "macd",
            "sentiment_mean", "sentiment_max", "sentiment_min",
            "sentiment_std", "sentiment_spread", "news_count"]
    cols += [f"emb_{i}" for i in range(64)]
    return cols


def _stats(mean=0.5, mx=0.8, mn=0.2, std=0.3, spread=0.6, count=3):
    return {
        "sentiment_mean": mean, "sentiment_max": mx, "sentiment_min": mn,
        "sentiment_std": std, "sentiment_spread": spread, "news_count": count,
    }


# ----- inject_news_features ---------------------------------------------------


def test_inject_writes_stats_into_all_rows():
    cols = _columns()
    features = np.zeros((10, len(cols)), dtype=np.float32)
    stats = _stats()
    out = inject_news_features(features, cols, stats, np.zeros(64, dtype=np.float32))
    for name, val in stats.items():
        assert np.all(out[:, cols.index(name)] == pytest.approx(val))


def test_inject_writes_embeddings_into_all_rows():
    cols = _columns()
    features = np.zeros((10, len(cols)), dtype=np.float32)
    emb = np.arange(64, dtype=np.float32)
    out = inject_news_features(features, cols, _stats(), emb)
    for i in range(64):
        assert np.all(out[:, cols.index(f"emb_{i}")] == pytest.approx(float(i)))


def test_inject_leaves_price_columns_untouched():
    cols = _columns()
    features = np.ones((10, len(cols)), dtype=np.float32)
    out = inject_news_features(features, cols, _stats(), np.zeros(64, dtype=np.float32))
    assert np.all(out[:, cols.index("rsi_14")] == 1.0)
    assert np.all(out[:, cols.index("macd")] == 1.0)


def test_inject_returns_copy():
    cols = _columns()
    features = np.zeros((10, len(cols)), dtype=np.float32)
    out = inject_news_features(features, cols, _stats(), np.zeros(64, dtype=np.float32))
    assert np.all(features == 0.0)
    assert out is not features


def test_inject_rejects_wrong_embedding_length():
    cols = _columns()
    features = np.zeros((10, len(cols)), dtype=np.float32)
    with pytest.raises(ValueError):
        inject_news_features(features, cols, _stats(), np.zeros(32, dtype=np.float32))


def test_inject_missing_stat_key_is_skipped():
    # stats with only one key — present key lands, others stay at zero.
    cols = _columns()
    features = np.zeros((10, len(cols)), dtype=np.float32)
    out = inject_news_features(features, cols, {"sentiment_mean": 0.7},
                               np.zeros(64, dtype=np.float32))
    assert np.all(out[:, cols.index("sentiment_mean")] == pytest.approx(0.7))
    assert np.all(out[:, cols.index("news_count")] == 0.0)


# ----- score_news_batch (integration — loads FinBERT + FinLang) ---------------


@pytest.mark.integration
def test_score_news_batch_aggregates():
    """A positive + negative pair: stats are aggregated, embedding has shape (64,)."""
    result = score_news_batch([
        "Bitcoin surges to a new all-time high as institutional demand soars.",
        "Regulators warn of a crackdown that could crash the crypto market.",
    ])
    assert len(result["per_news"]) == 2
    assert result["stats"]["news_count"] == 2.0
    s = result["stats"]
    assert s["sentiment_min"] <= s["sentiment_mean"] <= s["sentiment_max"]
    assert result["emb_64"].shape == (64,)
