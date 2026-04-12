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
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.agents.config import AgentConfig
from src.data.load_features import load_features_for_agent
from src.env.trading_env import TradingEnv

logger = logging.getLogger(__name__)


class ProgressCallback(BaseCallback):
    """Prints training progress every N steps."""

    def __init__(self, total_timesteps: int, print_every: int = 50_000):
        super().__init__()
        self.total_timesteps = total_timesteps
        self.print_every = print_every
        self._last_print = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_print >= self.print_every:
            pct = 100 * self.num_timesteps / self.total_timesteps
            print(f"  [{pct:5.1f}%] {self.num_timesteps:>7}/{self.total_timesteps} steps", flush=True)
            self._last_print = self.num_timesteps
        return True

ALGO_MAP = {
    "PPO": PPO,
    "A2C": A2C,
    "SAC": SAC,
}


FEATURE_COUNTS = {
    "baseline": 18,
    "sentiment": 19,     # baseline + 1 sentiment score
    "embeddings": 51,    # 18 base + 20 emb + news_count(3: raw+lag1+lag2) + roll7 + 3 sent extremes + 6 PCA lags
    "fusion": 54,        # embeddings(51) + 3 sentiment (raw + lag1 + lag2)
}

# Smaller network for lower-dimensional agents (20d PCA)
EMBEDDINGS_NET_ARCH = [128, 64]


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
        reward_type=config.reward_type, allow_short=config.allow_short,
    )


def _linear_schedule(initial_lr: float):
    """Linear LR decay from initial_lr to 0."""
    def schedule(progress_remaining: float) -> float:
        return progress_remaining * initial_lr
    return schedule


def train_agent(
    config: AgentConfig,
    dummy: bool = False,
    train_start: str = "2020-01-01",
    train_end: str = "2023-12-31",
    asset: str = "BTC/USDT",
    data_dir: str = "data/processed",
    timeframe: str = "1d",
) -> Path:
    """Train an RL agent and save the model.

    Args:
        config: Agent configuration.
        dummy: If True, use random dummy data instead of loading real features.
        train_start: Training period start date (used when dummy=False).
        train_end: Training period end date (used when dummy=False).
        asset: Asset symbol, e.g. 'BTC/USDT' (used when dummy=False).
        data_dir: Directory with preprocessed parquet files (used when dummy=False).
        timeframe: Candle timeframe, e.g. '1d' or '4h' (used when dummy=False).

    Returns:
        Path to saved model .zip file.
    """
    if dummy:
        env = _make_dummy_env(config)
        sentiment = None
    else:
        if config.agent_type in ("embeddings", "fusion"):
            features, prices, sentiment = load_features_for_agent(
                agent_type=config.agent_type,
                asset=asset,
                train_start=train_start,
                train_end=train_end,
                data_dir=data_dir,
                timeframe=timeframe,
                return_sentiment=True,
            )
        else:
            features, prices = load_features_for_agent(
                agent_type=config.agent_type,
                asset=asset,
                train_start=train_start,
                train_end=train_end,
                data_dir=data_dir,
                timeframe=timeframe,
            )
            sentiment = None
        env = TradingEnv(
            features=features,
            prices=prices,
            window=config.window,
            tx_cost=config.tx_cost,
            reward_type=config.reward_type,
            allow_short=config.allow_short,
            sentiment_signal=sentiment if config.agent_type in ("embeddings", "fusion") else None,
            sentiment_lambda=config.sentiment_lambda,
        )

    # Wrap in VecNormalize (skip for off-policy SAC — causes replay buffer issues)
    vec_env = DummyVecEnv([lambda: env])
    if config.algorithm != "SAC":
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=5.0)

    algo_cls = ALGO_MAP[config.algorithm]

    # Learning rate: constant or linear decay
    lr = config.learning_rate
    if config.lr_schedule == "linear":
        lr = _linear_schedule(config.learning_rate)

    # Override net_arch for high-dimensional embeddings
    policy_kwargs = config.policy_kwargs()
    if config.agent_type in ("embeddings", "fusion") and config.net_arch == [256, 256]:
        policy_kwargs["net_arch"] = EMBEDDINGS_NET_ARCH

    model = algo_cls(
        "MlpPolicy",
        vec_env,
        learning_rate=lr,
        seed=config.seed,
        verbose=0,
        device=config.device,
        policy_kwargs=policy_kwargs,
        **_algo_specific_kwargs(config),
    )

    logger.info(
        f"Training {config.algorithm} ({config.agent_type}) "
        f"for {config.total_timesteps} steps, seed={config.seed}"
    )
    callback = ProgressCallback(total_timesteps=config.total_timesteps)
    model.learn(total_timesteps=config.total_timesteps, callback=callback)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(config.save_dir) / config.agent_type / timestamp
    save_dir.mkdir(parents=True, exist_ok=True)
    model_path = save_dir / "model.zip"
    model.save(str(model_path.with_suffix("")))  # SB3 adds .zip automatically
    logger.info(f"Saved model to {model_path}")

    if isinstance(vec_env, VecNormalize):
        vecnorm_path = save_dir / "vecnormalize.pkl"
        vec_env.save(str(vecnorm_path))
        logger.info(f"Saved VecNormalize stats to {vecnorm_path}")

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
            "ent_coef": config.ent_coef,
        }
    elif config.algorithm == "A2C":
        return {
            "n_steps": config.n_steps,
            "gamma": config.gamma,
            "gae_lambda": config.gae_lambda,
            "ent_coef": config.ent_coef,
        }
    elif config.algorithm == "SAC":
        return {
            "gamma": config.gamma,
            "batch_size": 256,
            "buffer_size": 10_000,
            "ent_coef": "auto",
            "learning_starts": 1000,
            "train_freq": 4,
            "gradient_steps": 2,
        }
    return {}
