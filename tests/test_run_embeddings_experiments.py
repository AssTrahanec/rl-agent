from pathlib import Path
from unittest.mock import patch
import pandas as pd


def test_run_embeddings_experiments_dispatches_all_combos(tmp_path):
    from src.agents import run_embeddings_experiments as rex

    calls = []

    def fake_train(config, **kwargs):
        calls.append((config.algorithm, config.seed))
        model_path = tmp_path / f"{config.algorithm}_{config.seed}" / "model.zip"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_bytes(b"stub")
        return model_path

    csv_path = tmp_path / "runs.csv"
    with patch.object(rex, "train_agent", side_effect=fake_train):
        rex.run_embeddings_experiments(
            seeds=[42, 123, 7, 2024, 99],
            algos=["PPO", "A2C", "SAC"],
            output_csv=csv_path,
            dummy=True,
        )

    assert len(calls) == 15
    df = pd.read_csv(csv_path)
    assert set(df["algorithm"]) == {"PPO", "A2C", "SAC"}
    assert len(df) == 15
    assert "model_path" in df.columns
    assert "vecnorm_path" in df.columns
    assert "seed" in df.columns
