"""Tests for FinBERT fine-tuning utilities."""
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.features.finetune_finbert import (
    prepare_finetune_dataset,
    FinBERTFinetuner,
)


def make_dummy_sentiment_df():
    """Create a minimal sentiment-labeled DataFrame."""
    return pd.DataFrame({
        "text": [
            "Bitcoin surges to record high",
            "Crypto market collapses in massive sell-off",
            "Ethereum price remains stable today",
            "Investors optimistic about blockchain future",
            "Major exchange hacked, millions lost",
        ],
        "label": ["positive", "negative", "neutral", "positive", "negative"],
    })


def test_prepare_finetune_dataset_structure():
    df = make_dummy_sentiment_df()
    dataset = prepare_finetune_dataset(df)
    assert hasattr(dataset, "__len__")
    assert len(dataset) == len(df)


def test_prepare_finetune_dataset_label_encoding():
    df = make_dummy_sentiment_df()
    dataset = prepare_finetune_dataset(df)
    item = dataset[0]
    assert "input_ids" in item
    assert "attention_mask" in item
    assert "labels" in item
    assert item["labels"] in (0, 1, 2)


def test_finetuner_init():
    finetuner = FinBERTFinetuner(model_name="ProsusAI/finbert")
    assert finetuner.model_name == "ProsusAI/finbert"


def test_finetuner_train_smoke():
    """Smoke test: fine-tune for 1 step on 4 samples, model saves."""
    df = make_dummy_sentiment_df()
    dataset = prepare_finetune_dataset(df)
    with tempfile.TemporaryDirectory() as tmpdir:
        finetuner = FinBERTFinetuner(model_name="ProsusAI/finbert")
        save_path = finetuner.train(
            dataset,
            num_epochs=1,
            batch_size=2,
            max_steps=1,
            output_dir=tmpdir,
        )
        assert Path(save_path).exists()


def test_finetuner_predict_smoke():
    """Smoke test: predict sentiment scores on a few texts."""
    finetuner = FinBERTFinetuner(model_name="ProsusAI/finbert")
    texts = ["Bitcoin is booming", "Markets are crashing"]
    scores = finetuner.predict(texts)
    assert len(scores) == 2
    assert all(-1.0 <= s <= 1.0 for s in scores)
