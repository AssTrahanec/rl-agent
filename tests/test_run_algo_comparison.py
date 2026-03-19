"""Smoke tests for the algorithm comparison runner."""
import tempfile
from pathlib import Path

from src.agents.run_algo_comparison import run_algo_comparison, AlgoComparisonConfig


def test_algo_comparison_config_defaults():
    cfg = AlgoComparisonConfig()
    assert set(cfg.algorithms) == {"PPO", "A2C", "SAC"}
    assert cfg.agent_type in ("baseline", "sentiment", "embeddings")
    assert len(cfg.seeds) >= 1
    assert cfg.total_timesteps > 0


def test_run_algo_comparison_smoke():
    """Smoke test: run 1 algorithm × 1 seed for 1000 steps."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = AlgoComparisonConfig(
            algorithms=["PPO"],
            agent_type="baseline",
            seeds=[42],
            total_timesteps=1000,
            save_dir=tmpdir,
            dummy=True,
        )
        results = run_algo_comparison(cfg)
        assert isinstance(results, list)
        assert len(results) == 1
        run = results[0]
        assert run["algorithm"] == "PPO"
        assert run["seed"] == 42
        assert Path(run["model_path"]).exists()


def test_run_algo_comparison_all_algos():
    """Run PPO + A2C + SAC × 1 seed = 3 runs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = AlgoComparisonConfig(
            algorithms=["PPO", "A2C", "SAC"],
            agent_type="baseline",
            seeds=[42],
            total_timesteps=1000,
            save_dir=tmpdir,
            dummy=True,
        )
        results = run_algo_comparison(cfg)
        assert len(results) == 3
        algos = {r["algorithm"] for r in results}
        assert algos == {"PPO", "A2C", "SAC"}
