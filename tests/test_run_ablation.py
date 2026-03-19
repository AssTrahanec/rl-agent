"""Smoke tests for the ablation study runner."""
import tempfile
from pathlib import Path

import pytest

from src.agents.run_ablation import run_ablation, AblationConfig


def test_ablation_config_defaults():
    cfg = AblationConfig()
    assert cfg.agent_types == ["baseline", "sentiment", "embeddings"]
    assert cfg.assets == ["BTC/USDT", "ETH/USDT"]
    assert cfg.seeds == [42, 43, 44]
    assert cfg.total_timesteps > 0


def test_run_ablation_smoke():
    """Smoke test: run ablation on 1 agent type, 1 asset, 1 seed for 1000 steps."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = AblationConfig(
            agent_types=["baseline"],
            assets=["BTC/USDT"],
            seeds=[42],
            total_timesteps=1000,
            save_dir=tmpdir,
            dummy=True,
        )
        results = run_ablation(cfg)
        assert isinstance(results, list)
        assert len(results) == 1
        run = results[0]
        assert run["agent_type"] == "baseline"
        assert run["asset"] == "BTC/USDT"
        assert run["seed"] == 42
        assert Path(run["model_path"]).exists()


def test_run_ablation_multiple_configs():
    """Run 2 agent types × 1 asset × 1 seed = 2 runs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = AblationConfig(
            agent_types=["baseline", "sentiment"],
            assets=["BTC/USDT"],
            seeds=[42],
            total_timesteps=1000,
            save_dir=tmpdir,
            dummy=True,
        )
        results = run_ablation(cfg)
        assert len(results) == 2
        agent_types = {r["agent_type"] for r in results}
        assert agent_types == {"baseline", "sentiment"}
