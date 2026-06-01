"""FinBERT sentiment scoring for news texts."""
import logging
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_pipeline = None


def _get_pipeline():
    """Lazy-load the FinBERT sentiment pipeline."""
    global _pipeline
    if _pipeline is None:
        from transformers import pipeline
        logger.info("Loading ProsusAI/finbert sentiment pipeline...")
        import torch
        device = 0 if torch.cuda.is_available() else -1
        _pipeline = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            device=device,
        )
    return _pipeline


def compute_sentiment(texts: List[str], model: Optional[object] = None) -> float:
    """Compute mean sentiment score from list of texts.

    Uses ProsusAI/finbert: positive -> +1, negative -> -1, neutral -> 0.
    Returns mean score across all texts.

    Args:
        texts: List of news article texts.
        model: Optional pre-loaded pipeline (for testing).

    Returns:
        float in [-1, +1]. Returns 0.0 if texts is empty.
    """
    if not texts:
        return 0.0

    pipe = model if model is not None else _get_pipeline()

    # FinBERT has 512 token limit, truncate long texts
    truncated = [t[:512] for t in texts]
    results = pipe(truncated, truncation=True, max_length=512)

    label_map = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
    scores = []
    for r in results:
        label = r["label"].lower()
        score = r["score"] * label_map.get(label, 0.0)
        scores.append(score)

    return float(np.mean(scores))


def compute_sentiment_scores(texts: List[str], model: Optional[object] = None) -> List[float]:
    """Compute per-article sentiment scores from list of texts.

    Uses ProsusAI/finbert: positive -> +1, negative -> -1, neutral -> 0.
    Returns list of scores, one per article.

    Args:
        texts: List of news article texts.
        model: Optional pre-loaded pipeline (for testing).

    Returns:
        List of floats in [-1, +1]. Returns empty list if texts is empty.
    """
    if not texts:
        return []

    pipe = model if model is not None else _get_pipeline()

    truncated = [t[:512] for t in texts]
    results = pipe(truncated, truncation=True, max_length=512)

    label_map = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
    scores = []
    for r in results:
        label = r["label"].lower()
        score = r["score"] * label_map.get(label, 0.0)
        scores.append(score)

    return [float(s) for s in scores]
