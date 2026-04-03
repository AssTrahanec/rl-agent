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


def test_train_sentiment_agent():
    """Agent-2: train with sentiment features (1 extra feature)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = AgentConfig(
            total_timesteps=512,
            save_dir=tmpdir,
            seed=42,
            agent_type="sentiment",
        )
        model_path = train_agent(config, dummy=True)
        assert model_path.exists()


def test_train_embeddings_agent():
    """Agent-3: train with embedding features (32 extra features)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = AgentConfig(
            total_timesteps=512,
            save_dir=tmpdir,
            seed=42,
            agent_type="embeddings",
        )
        model_path = train_agent(config, dummy=True)
        assert model_path.exists()


def test_train_a2c():
    """Smoke test: train A2C for 512 steps."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = AgentConfig(
            total_timesteps=512,
            save_dir=tmpdir,
            seed=42,
            algorithm="A2C",
        )
        model_path = train_agent(config, dummy=True)
        assert model_path.exists()


def test_train_sac():
    """Smoke test: train SAC for 512 steps."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = AgentConfig(
            total_timesteps=512,
            save_dir=tmpdir,
            seed=42,
            algorithm="SAC",
        )
        model_path = train_agent(config, dummy=True)
        assert model_path.exists()


def test_feature_counts_embeddings_updated():
    """embeddings agent feature count reflects 18 baseline + 64 emb + 1 news_count."""
    from src.agents.train import FEATURE_COUNTS
    assert FEATURE_COUNTS["embeddings"] == 83  # 18 + 64 + 1


def test_feature_counts_fusion_updated():
    """fusion agent feature count reflects 18 baseline + 1 sentiment + 64 emb + 1 news_count."""
    from src.agents.train import FEATURE_COUNTS
    assert FEATURE_COUNTS["fusion"] == 84  # 18 + 1 + 64 + 1


def test_train_fusion_agent():
    """Agent-4: train with fusion features (sentiment + embeddings, 53 features total)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = AgentConfig(
            total_timesteps=512,
            save_dir=tmpdir,
            seed=42,
            agent_type="fusion",
        )
        model_path = train_agent(config, dummy=True)
        assert model_path.exists()
