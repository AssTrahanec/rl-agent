"""Algorithm comparison runner: PPO vs A2C vs SAC on the best agent type.

Usage:
    python -m src.agents.run_algo_comparison
    python -m src.agents.run_algo_comparison --agent-type sentiment --total-timesteps 500000
"""
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any

from src.agents.config import AgentConfig
from src.agents.train import train_agent

logger = logging.getLogger(__name__)


@dataclass
class AlgoComparisonConfig:
    """Configuration for the algorithm comparison experiment."""

    algorithms: List[str] = field(default_factory=lambda: ["PPO", "A2C", "SAC"])
    agent_type: str = "baseline"
    seeds: List[int] = field(default_factory=lambda: [42, 43, 44])
    total_timesteps: int = 500_000
    save_dir: str = "experiments/algo_comparison"
    dummy: bool = False


def run_algo_comparison(cfg: AlgoComparisonConfig) -> List[Dict[str, Any]]:
    """Run PPO/A2C/SAC × seeds on one agent type and return results.

    Args:
        cfg: Algorithm comparison configuration.

    Returns:
        List of dicts with keys: algorithm, seed, agent_type, model_path.
    """
    results = []
    total_runs = len(cfg.algorithms) * len(cfg.seeds)
    run_idx = 0

    for algorithm in cfg.algorithms:
        for seed in cfg.seeds:
            run_idx += 1
            logger.info(
                f"[{run_idx}/{total_runs}] {algorithm} ({cfg.agent_type}), seed={seed}"
            )
            agent_config = AgentConfig(
                agent_type=cfg.agent_type,
                algorithm=algorithm,
                total_timesteps=cfg.total_timesteps,
                seed=seed,
                save_dir=cfg.save_dir,
            )
            model_path = train_agent(agent_config, dummy=cfg.dummy)
            results.append(
                {
                    "algorithm": algorithm,
                    "agent_type": cfg.agent_type,
                    "seed": seed,
                    "model_path": str(model_path),
                }
            )
            logger.info(f"  -> Saved to {model_path}")

    logger.info(f"Algorithm comparison complete: {len(results)} runs.")
    return results


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Run algorithm comparison experiment")
    parser.add_argument("--agent-type", type=str, default="baseline")
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--save-dir", type=str, default="experiments/algo_comparison")
    args = parser.parse_args()

    cfg = AlgoComparisonConfig(
        agent_type=args.agent_type,
        total_timesteps=args.total_timesteps,
        save_dir=args.save_dir,
    )
    run_algo_comparison(cfg)
