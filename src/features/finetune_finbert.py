"""Fine-tune ProsusAI/finbert on crypto sentiment data.

Usage:
    python -m src.features.finetune_finbert --data data/raw/kaggle_sentiment.csv
"""
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

logger = logging.getLogger(__name__)

LABEL2ID = {"positive": 0, "negative": 1, "neutral": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
LABEL_SCORE = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}


class SentimentDataset(Dataset):
    """PyTorch Dataset wrapping tokenized sentiment samples."""

    def __init__(self, encodings: dict, labels: List[int]):
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def prepare_finetune_dataset(
    df: pd.DataFrame,
    text_col: str = "text",
    label_col: str = "label",
    model_name: str = "ProsusAI/finbert",
    max_length: int = 128,
) -> SentimentDataset:
    """Tokenize a labeled DataFrame and return a SentimentDataset.

    Args:
        df: DataFrame with text and label columns.
        text_col: Column name for news texts.
        label_col: Column name for sentiment labels ('positive', 'negative', 'neutral').
        model_name: HuggingFace model name for tokenizer.
        max_length: Max token length for truncation.

    Returns:
        SentimentDataset ready for Trainer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    texts = df[text_col].tolist()
    labels = [LABEL2ID[lbl.lower()] for lbl in df[label_col].tolist()]

    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors=None,  # return plain lists for Dataset
    )
    return SentimentDataset(encodings, labels)


class FinBERTFinetuner:
    """Wrapper around HuggingFace Trainer for fine-tuning FinBERT."""

    def __init__(self, model_name: str = "ProsusAI/finbert"):
        self.model_name = model_name
        self._pipeline = None

    def train(
        self,
        dataset: SentimentDataset,
        num_epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        output_dir: str = "experiments/finbert_finetuned",
        max_steps: int = -1,
    ) -> str:
        """Fine-tune FinBERT and save to output_dir.

        Args:
            dataset: Tokenized SentimentDataset.
            num_epochs: Number of training epochs.
            batch_size: Per-device training batch size.
            learning_rate: Learning rate for AdamW.
            output_dir: Directory to save fine-tuned model.
            max_steps: If > 0, override num_epochs (used for smoke testing).

        Returns:
            Path to saved model directory.
        """
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=3,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            ignore_mismatched_sizes=True,
        )

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            logging_steps=10,
            save_strategy="epoch",
            use_cpu=not torch.cuda.is_available(),
            report_to="none",
            max_steps=max_steps,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
        )

        logger.info(f"Fine-tuning {self.model_name} for {num_epochs} epoch(s)...")
        trainer.train()
        trainer.save_model(output_dir)
        logger.info(f"Fine-tuned model saved to {output_dir}")

        # Invalidate cached pipeline so next predict() loads the new model
        self._pipeline = None

        return output_dir

    def predict(
        self,
        texts: List[str],
        model_dir: Optional[str] = None,
    ) -> List[float]:
        """Predict sentiment scores for a list of texts.

        Args:
            texts: List of news texts.
            model_dir: Path to fine-tuned model directory. If None, uses base model.

        Returns:
            List of floats in [-1, +1] — one per text.
        """
        if not texts:
            return []

        from transformers import pipeline as hf_pipeline

        model_path = model_dir if model_dir is not None else self.model_name
        if self._pipeline is None or model_dir is not None:
            self._pipeline = hf_pipeline(
                "sentiment-analysis",
                model=model_path,
                tokenizer=self.model_name,
                device=-1,
            )

        truncated = [t[:512] for t in texts]
        results = self._pipeline(truncated, truncation=True, max_length=512)

        scores = []
        for r in results:
            label = r["label"].lower()
            score = r["score"] * LABEL_SCORE.get(label, 0.0)
            scores.append(float(score))
        return scores


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Fine-tune FinBERT on crypto sentiment")
    parser.add_argument("--data", type=str, required=True, help="CSV with text,label columns")
    parser.add_argument("--output-dir", type=str, default="experiments/finbert_finetuned")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    dataset = prepare_finetune_dataset(df)
    finetuner = FinBERTFinetuner()
    finetuner.train(
        dataset,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
    )
