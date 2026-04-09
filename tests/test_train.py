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
    """embeddings = 51 features total."""
    from src.agents.train import FEATURE_COUNTS
    assert FEATURE_COUNTS["embeddings"] == 51


def test_feature_counts_fusion_updated():
    """fusion = 54 features total."""
    from src.agents.train import FEATURE_COUNTS
    assert FEATURE_COUNTS["fusion"] == 54


def test_train_fusion_agent():
    """Agent-4: train with fusion features (price + sentiment + embeddings, 98 features total)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = AgentConfig(
            total_timesteps=512,
            save_dir=tmpdir,
            seed=42,
            agent_type="fusion",
        )
        model_path = train_agent(config, dummy=True)
        assert model_path.exists()


def test_train_embeddings_with_linear_lr():
    """Embeddings agent trains with linear LR schedule."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = AgentConfig(
            total_timesteps=512,
            save_dir=tmpdir,
            seed=42,
            agent_type="embeddings",
            lr_schedule="linear",
        )
        model_path = train_agent(config, dummy=True)
        assert model_path.exists()


def test_embeddings_uses_larger_network():
    """Embeddings agent should use [128, 64] network."""
    from src.agents.train import EMBEDDINGS_NET_ARCH
    assert EMBEDDINGS_NET_ARCH == [128, 64]


def test_train_saves_vecnormalize():
    """Training saves VecNormalize stats alongside model."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = AgentConfig(total_timesteps=256, seed=42, algorithm="PPO", save_dir=tmpdir)
        model_path = train_agent(config, dummy=True)
        vecnorm_path = model_path.parent / "vecnormalize.pkl"
        assert vecnorm_path.exists(), "VecNormalize stats not saved"


def test_warmup_linear_schedule():
    """Warmup+linear schedule: LR rises then decays."""
    from src.agents.train import _warmup_linear_schedule
    schedule = _warmup_linear_schedule(3e-4, warmup_frac=0.1)
    # At start (progress_remaining=1.0): warmup phase, LR near 0
    assert schedule(1.0) < 1e-5
    # At 90% remaining (10% done = end of warmup): LR at peak
    assert abs(schedule(0.9) - 3e-4) < 1e-5
    # At 0% remaining: LR at 0
    assert schedule(0.0) == 0.0
    # At 50% remaining: LR should be ~half of peak (linear decay)
    mid = schedule(0.5)
    assert 1e-4 < mid < 2.5e-4
