"""Ablation study runner: trains all agent/asset/seed combinations.

Usage:
    python -m src.agents.run_ablation
    python -m src.agents.run_ablation --total-timesteps 500000
"""
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any

from src.agents.config import AgentConfig
from src.agents.train import train_agent

logger = logging.getLogger(__name__)


@dataclass
class AblationConfig:
    """Configuration for the ablation study."""

    agent_types: List[str] = field(default_factory=lambda: ["baseline", "sentiment", "embeddings"])
    assets: List[str] = field(default_factory=lambda: ["BTC/USDT", "ETH/USDT"])
    seeds: List[int] = field(default_factory=lambda: [42, 43, 44])
    total_timesteps: int = 500_000
    save_dir: str = "experiments"
    algorithm: str = "PPO"
    dummy: bool = False


def run_ablation(cfg: AblationConfig) -> List[Dict[str, Any]]:
    """Run all agent/asset/seed combinations and return results.

    Args:
        cfg: Ablation configuration.

    Returns:
        List of dicts with keys: agent_type, asset, seed, model_path.
    """
    results = []
    total_runs = len(cfg.agent_types) * len(cfg.assets) * len(cfg.seeds)
    run_idx = 0

    for agent_type in cfg.agent_types:
        for asset in cfg.assets:
            for seed in cfg.seeds:
                run_idx += 1
                logger.info(
                    f"[{run_idx}/{total_runs}] Training {agent_type} on {asset}, seed={seed}"
                )
                agent_config = AgentConfig(
                    agent_type=agent_type,
                    algorithm=cfg.algorithm,
                    total_timesteps=cfg.total_timesteps,
                    seed=seed,
                    save_dir=cfg.save_dir,
                )
                model_path = train_agent(agent_config, dummy=cfg.dummy)
                results.append(
                    {
                        "agent_type": agent_type,
                        "asset": asset,
                        "seed": seed,
                        "model_path": str(model_path),
                    }
                )
                logger.info(f"  -> Saved to {model_path}")

    logger.info(f"Ablation complete: {len(results)} runs.")
    return results


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Run ablation study")
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--save-dir", type=str, default="experiments")
    parser.add_argument("--algorithm", type=str, default="PPO")
    args = parser.parse_args()

    cfg = AblationConfig(
        total_timesteps=args.total_timesteps,
        save_dir=args.save_dir,
        algorithm=args.algorithm,
    )
    run_ablation(cfg)
