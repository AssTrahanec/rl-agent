import tempfile
from pathlib import Path

from src.agents.train import train_agent
from src.agents.config import AgentConfig


def test_train_smoke():
    """Smoke test: train 1000 steps on dummy data, model file is saved."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = AgentConfig(
            total_timesteps=1000,
            save_dir=tmpdir,
            seed=42,
        )
        model_path = train_agent(config, dummy=True)
        assert model_path.exists()
        assert model_path.suffix == ".zip"


def test_train_returns_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = AgentConfig(
            total_timesteps=512,
            save_dir=tmpdir,
            seed=0,
        )
        model_path = train_agent(config, dummy=True)
        assert isinstance(model_path, Path)
