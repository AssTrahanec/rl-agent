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


def test_algo_specific_kwargs_sac_reads_config():
    from src.agents.config import sac_embeddings_config
    from src.agents.train import _algo_specific_kwargs
    cfg = sac_embeddings_config(seed=42)
    kw = _algo_specific_kwargs(cfg)
    assert kw["buffer_size"] == 300_000
    assert kw["learning_starts"] == 10_000
    assert kw["batch_size"] == 256
    assert kw["tau"] == 0.02
    assert kw["train_freq"] == 8
    assert kw["gradient_steps"] == 8
    assert kw["ent_coef"] == "auto"
    assert kw["gamma"] == 0.99
    assert kw["use_sde"] is True
    assert kw["optimize_memory_usage"] is True


def test_algo_specific_kwargs_ppo_reads_config():
    from src.agents.config import ppo_embeddings_config
    from src.agents.train import _algo_specific_kwargs
    cfg = ppo_embeddings_config(seed=42)
    kw = _algo_specific_kwargs(cfg)
    assert kw["n_steps"] == 2048
    assert kw["batch_size"] == 64
    assert kw["n_epochs"] == 10
    assert kw["clip_range"] == 0.2
    assert kw["ent_coef"] == 0.01
    assert kw["max_grad_norm"] == 0.5
    assert kw["use_sde"] is True


def test_algo_specific_kwargs_a2c_reads_config():
    from src.agents.config import a2c_embeddings_config
    from src.agents.train import _algo_specific_kwargs
    cfg = a2c_embeddings_config(seed=42)
    kw = _algo_specific_kwargs(cfg)
    assert kw["n_steps"] == 16
    assert kw["gae_lambda"] == 1.0
    assert kw["ent_coef"] == 0.01
    assert kw["max_grad_norm"] == 0.5
    assert kw["normalize_advantage"] is True
    assert kw["use_sde"] is True


def test_train_uses_config_device(monkeypatch):
    """train_agent must pass config.device to the SB3 constructor."""
    from src.agents.config import AgentConfig
    from src.agents import train as train_mod

    captured = {}
    real_ppo = train_mod.ALGO_MAP["PPO"]

    class SpyPPO(real_ppo):
        def __init__(self, *args, **kwargs):
            captured["device"] = kwargs.get("device")
            super().__init__(*args, **kwargs)

    monkeypatch.setitem(train_mod.ALGO_MAP, "PPO", SpyPPO)

    cfg = AgentConfig(
        algorithm="PPO", agent_type="baseline",
        total_timesteps=64, n_steps=32, batch_size=16,
        device="cpu",
    )
    train_mod.train_agent(cfg, dummy=True)
    assert captured["device"] == "cpu"


def test_train_saves_vecnormalize():
    """Training saves VecNormalize stats alongside model."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = AgentConfig(total_timesteps=256, seed=42, algorithm="PPO", save_dir=tmpdir)
        model_path = train_agent(config, dummy=True)
        vecnorm_path = model_path.parent / "vecnormalize.pkl"
        assert vecnorm_path.exists(), "VecNormalize stats not saved"
