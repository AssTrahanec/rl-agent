"""Tests for the News Analyzer logic (dashboard/utils/news_impact.py)."""
import sys
from pathlib import Path

import numpy as np
import pytest

# Add dsr_experiment/ to sys.path so `dashboard...` and `lib...` resolve.
_DSR_EXP = Path(__file__).resolve().parent.parent / "dsr_experiment"
if str(_DSR_EXP) not in sys.path:
    sys.path.insert(0, str(_DSR_EXP))

from dashboard.utils.news_impact import inject_news_features  # noqa: E402
from dashboard.utils.news_impact import score_news  # noqa: E402


def _columns():
    # Minimal column layout: 2 price/tech + news columns the function touches.
    cols = ["rsi_14", "macd",
            "sentiment_mean", "sentiment_max", "sentiment_min",
            "sentiment_std", "sentiment_spread", "news_count"]
    cols += [f"emb_{i}" for i in range(64)]
    return cols


def test_inject_sets_last_row_sentiment():
    cols = _columns()
    features = np.zeros((10, len(cols)), dtype=np.float32)
    emb = np.arange(64, dtype=np.float32)
    out = inject_news_features(features, cols, sentiment=0.8, emb_64=emb)
    last = out.shape[0] - 1
    for name in ("sentiment_mean", "sentiment_max", "sentiment_min"):
        assert out[last, cols.index(name)] == pytest.approx(0.8)
    assert out[last, cols.index("news_count")] == 1.0


def test_inject_sets_embeddings():
    cols = _columns()
    features = np.zeros((10, len(cols)), dtype=np.float32)
    emb = np.arange(64, dtype=np.float32)
    out = inject_news_features(features, cols, sentiment=0.0, emb_64=emb)
    last = out.shape[0] - 1
    for i in range(64):
        assert out[last, cols.index(f"emb_{i}")] == pytest.approx(float(i))


def test_inject_does_not_touch_other_rows():
    cols = _columns()
    features = np.zeros((10, len(cols)), dtype=np.float32)
    out = inject_news_features(features, cols, sentiment=0.9,
                               emb_64=np.ones(64, dtype=np.float32))
    # Every row except the last stays all-zero.
    assert np.all(out[:-1] == 0.0)


def test_inject_returns_copy():
    cols = _columns()
    features = np.zeros((10, len(cols)), dtype=np.float32)
    out = inject_news_features(features, cols, sentiment=0.5,
                               emb_64=np.zeros(64, dtype=np.float32))
    # Original must be unchanged.
    assert np.all(features == 0.0)
    assert out is not features


def test_inject_zeroes_sentiment_std_and_spread():
    cols = _columns()
    features = np.zeros((10, len(cols)), dtype=np.float32)
    out = inject_news_features(features, cols, sentiment=0.8,
                               emb_64=np.ones(64, dtype=np.float32))
    last = out.shape[0] - 1
    assert out[last, cols.index("sentiment_std")] == 0.0
    assert out[last, cols.index("sentiment_spread")] == 0.0


def test_inject_rejects_wrong_embedding_length():
    cols = _columns()
    features = np.zeros((10, len(cols)), dtype=np.float32)
    with pytest.raises(ValueError):
        inject_news_features(features, cols, sentiment=0.5,
                             emb_64=np.zeros(32, dtype=np.float32))


@pytest.mark.integration
def test_score_news_positive_text():
    """A clearly positive headline scores positive; embedding has shape (64,)."""
    sentiment, emb = score_news(
        "Bitcoin surges to a new all-time high as institutional demand soars."
    )
    assert -1.0 <= sentiment <= 1.0
    assert sentiment > 0.0
    assert emb.shape == (64,)
