"""Tests for walk-forward validation logic."""
import pytest
from scripts.run_walkforward import generate_splits


def test_generate_splits_default():
    splits = generate_splits(
        start_year=2020, end_year=2024,
        train_years=2, test_years=1,
    )
    assert len(splits) == 3
    assert splits[0] == {
        "train_start": "2020-01-01", "train_end": "2021-12-31",
        "test_start": "2022-01-01", "test_end": "2022-12-31",
    }
    assert splits[2] == {
        "train_start": "2022-01-01", "train_end": "2023-12-31",
        "test_start": "2024-01-01", "test_end": "2024-12-31",
    }


def test_generate_splits_single():
    splits = generate_splits(
        start_year=2022, end_year=2024,
        train_years=2, test_years=1,
    )
    assert len(splits) == 1
    assert splits[0]["train_start"] == "2022-01-01"
    assert splits[0]["test_end"] == "2024-12-31"


def test_generate_splits_no_room():
    splits = generate_splits(
        start_year=2023, end_year=2024,
        train_years=2, test_years=1,
    )
    assert len(splits) == 0
