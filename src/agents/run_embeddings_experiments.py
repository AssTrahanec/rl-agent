"""Orchestrate 3 algos × N seeds embeddings training sweep.

Writes a CSV manifest of (algorithm, seed, model_path, vecnorm_path, timestamp)
for downstream OOS backtest.
"""
from __future__ import annotations

import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import List

from src.agents.config import (
    AgentConfig,
    a2c_embeddings_config,
    ppo_embeddings_config,
    sac_embeddings_config,
)
from src.agents.train import train_agent

logger = logging.getLogger(__name__)

DEFAULT_SEEDS = [42, 123, 7, 2024, 99]
DEFAULT_ALGOS = ["PPO", "A2C", "SAC"]

FACTORIES = {
    "PPO": ppo_embeddings_config,
    "A2C": a2c_embeddings_config,
    "SAC": sac_embeddings_config,
}


def run_embeddings_experiments(
    seeds: List[int] = None,
    algos: List[str] = None,
    asset: str = "BTC/USDT",
    timeframe: str = "4h",
    train_start: str = "2020-01-01",
    train_end: str = "2023-12-31",
    output_csv: Path = Path("results/embeddings_runs_v4.csv"),
    dummy: bool = False,
) -> Path:
    seeds = seeds or DEFAULT_SEEDS
    algos = algos or DEFAULT_ALGOS
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for algo in algos:
        factory = FACTORIES[algo]
        for seed in seeds:
            cfg: AgentConfig = factory(seed)
            logger.info(f"=== Training {algo} seed={seed} ===")
            if dummy:
                cfg.total_timesteps = 64
                if algo in ("PPO", "A2C"):
                    cfg.n_steps = 32
                    cfg.batch_size = 16
                else:
                    cfg.learning_starts = 4
                    cfg.buffer_size = 128
                    cfg.batch_size = 16
                    cfg.train_freq = 1
                    cfg.gradient_steps = 1
            model_path = train_agent(
                cfg,
                dummy=dummy,
                asset=asset,
                timeframe=timeframe,
                train_start=train_start,
                train_end=train_end,
            )
            vecnorm_path = Path(model_path).parent / "vecnormalize.pkl"
            rows.append({
                "algorithm": algo,
                "seed": seed,
                "model_path": str(model_path),
                "vecnorm_path": str(vecnorm_path) if vecnorm_path.exists() else "",
                "timestamp": datetime.now().isoformat(),
            })

    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["algorithm", "seed", "model_path", "vecnorm_path", "timestamp"],
        )
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"Wrote {len(rows)} runs to {output_csv}")
    return output_csv


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_embeddings_experiments()
