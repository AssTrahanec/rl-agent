"""Train RL agents on trading environment.

Usage:
    python -m src.agents.train
    python -m src.agents.train --agent-type sentiment --asset BTC/USDT
"""
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO, A2C, SAC

from src.agents.config import AgentConfig
from src.env.trading_env import TradingEnv

logger = logging.getLogger(__name__)

ALGO_MAP = {
    "PPO": PPO,
    "A2C": A2C,
    "SAC": SAC,
}


FEATURE_COUNTS = {
    "baseline": 20,
    "sentiment": 21,     # baseline + 1 sentiment score
    "embeddings": 52,    # baseline + 32 compressed embedding dims
}


def _make_dummy_env(config: AgentConfig) -> TradingEnv:
    """Create a small environment with random data for smoke testing."""
    n = 100
    n_features = FEATURE_COUNTS.get(config.agent_type, 20)
    np.random.seed(config.seed)
    features = np.random.randn(n, n_features).astype(np.float32)
    prices = (100 + np.cumsum(np.random.randn(n) * 0.5)).astype(np.float64)
    prices = np.maximum(prices, 1.0)  # avoid zero/negative prices
    return TradingEnv(
        features=features, prices=prices,
        window=config.window, tx_cost=config.tx_cost,
    )


def train_agent(config: AgentConfig, dummy: bool = False) -> Path:
    """Train an RL agent and save the model.

    Args:
        config: Agent configuration.
        dummy: If True, use random dummy data instead of loading real features.

    Returns:
        Path to saved model .zip file.
    """
    if dummy:
        env = _make_dummy_env(config)
    else:
        raise NotImplementedError("Real data loading not yet implemented")

    algo_cls = ALGO_MAP[config.algorithm]

    model = algo_cls(
        "MlpPolicy",
        env,
        learning_rate=config.learning_rate,
        seed=config.seed,
        verbose=0,
        policy_kwargs=config.policy_kwargs(),
        **_algo_specific_kwargs(config),
    )

    logger.info(
        f"Training {config.algorithm} ({config.agent_type}) "
        f"for {config.total_timesteps} steps, seed={config.seed}"
    )
    model.learn(total_timesteps=config.total_timesteps)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(config.save_dir) / config.agent_type / timestamp
    save_dir.mkdir(parents=True, exist_ok=True)
    model_path = save_dir / "model.zip"
    model.save(str(model_path.with_suffix("")))  # SB3 adds .zip automatically
    logger.info(f"Saved model to {model_path}")

    return model_path


def _algo_specific_kwargs(config: AgentConfig) -> dict:
    """Return algorithm-specific kwargs for SB3 constructor."""
    if config.algorithm == "PPO":
        return {
            "n_steps": config.n_steps,
            "batch_size": config.batch_size,
            "n_epochs": config.n_epochs,
            "gamma": config.gamma,
            "gae_lambda": config.gae_lambda,
            "clip_range": config.clip_range,
        }
    elif config.algorithm == "A2C":
        return {
            "n_steps": config.n_steps,
            "gamma": config.gamma,
            "gae_lambda": config.gae_lambda,
        }
    elif config.algorithm == "SAC":
        return {
            "gamma": config.gamma,
            "batch_size": config.batch_size,
        }
    return {}
