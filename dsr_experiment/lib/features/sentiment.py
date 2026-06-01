"""FinBERT sentiment scoring."""
import logging
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_pipeline = None


def _get_pipeline(model_name: str = "ProsusAI/finbert"):
    global _pipeline
    if _pipeline is None:
        from transformers import pipeline
        import torch
        device = 0 if torch.cuda.is_available() else -1
        logger.info(f"Loading FinBERT pipeline ({model_name}) on device={device}")
        _pipeline = pipeline(
            "sentiment-analysis",
            model=model_name,
            tokenizer=model_name,
            device=device,
        )
    return _pipeline


def compute_sentiment_scores(texts: List[str], model: Optional[object] = None) -> List[float]:
    """Per-article sentiment in [-1, +1]."""
    if not texts:
        return []
    pipe = model if model is not None else _get_pipeline()
    results = pipe(texts, truncation=True, max_length=512)
    label_map = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
    return [float(r["score"] * label_map.get(r["label"].lower(), 0.0)) for r in results]
