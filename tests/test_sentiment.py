import pytest
from src.features.sentiment import compute_sentiment


@pytest.mark.integration
def test_positive_sentiment():
    texts = ["Bitcoin surges to all-time high, investors celebrate massive gains"]
    score = compute_sentiment(texts)
    assert score > 0.0


@pytest.mark.integration
def test_negative_sentiment():
    texts = ["Crypto market crashes, billions wiped out in massive sell-off"]
    score = compute_sentiment(texts)
    assert score < 0.0


def test_empty_texts_fallback():
    score = compute_sentiment([])
    assert score == 0.0


def test_returns_float():
    # Even with empty input, should return float
    score = compute_sentiment([])
    assert isinstance(score, float)
