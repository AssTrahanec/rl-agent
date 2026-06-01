"""Unit tests for dsr_experiment/lib/interpretability.py (identify_feature_groups).

permutation_importance() is exercised via integration (it calls run_backtest with a
trained model), not here.
"""
from __future__ import annotations

import sys
from pathlib import Path

_DSR_EXP = Path(__file__).resolve().parent.parent / "dsr_experiment"
if str(_DSR_EXP) not in sys.path:
    sys.path.insert(0, str(_DSR_EXP))

from lib.interpretability import identify_feature_groups  # noqa: E402


def test_identify_feature_groups_basic():
    names = [
        "close", "rsi_14", "macd",
        "sentiment_mean", "sentiment_std",
        "news_count", "news_count_lag1",
        "emb_0", "emb_1", "emb_0_lag1",
    ]
    g = identify_feature_groups(names)
    assert g["price"] == [0, 1, 2]
    assert g["sentiment"] == [3, 4]
    assert g["news_count"] == [5, 6]
    assert g["embeddings"] == [7, 8, 9]
    # Composite news = union of sentiment, embeddings, news_count.
    assert g["news"] == sorted([3, 4, 5, 6, 7, 8, 9])


def test_identify_feature_groups_case_insensitive():
    names = ["Close", "Sentiment_Mean", "EMB_5", "NEWS_COUNT"]
    g = identify_feature_groups(names)
    assert 0 in g["price"]
    assert 1 in g["sentiment"]
    assert 2 in g["embeddings"]
    assert 3 in g["news_count"]


def test_identify_feature_groups_empty():
    g = identify_feature_groups([])
    assert all(len(v) == 0 for v in g.values())
