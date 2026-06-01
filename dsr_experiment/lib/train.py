"""Train SAC / DQN agent using a TradingEnv built from train features."""
import logging
from datetime import datetime
from pathlib import Path
from typing import Union

import numpy as np
import torch
from stable_baselines3 import SAC, DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from lib.env import TradingEnv

logger = logging.getLogger(__name__)

ALGO_MAP = {"SAC": SAC, "DQN": DQN}
ACTIVATION_MAP = {"tanh": torch.nn.Tanh, "relu": torch.nn.ReLU}


class ProgressCallback(BaseCallback):
    def __init__(self, total_timesteps: int, print_every: int = 50_000):
        super().__init__()
        self.total = total_timesteps
        self.every = print_every
        self._last = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last >= self.every:
            pct = 100 * self.num_timesteps / self.total
            print(f"  [{pct:5.1f}%] {self.num_timesteps:>7}/{self.total} steps", flush=True)
            self._last = self.num_timesteps
        return True


def _linear_schedule(initial_lr: float):
    def schedule(progress_remaining: float) -> float:
        return progress_remaining * initial_lr
    return schedule


def _build_env(cfg, features: np.ndarray, prices: np.ndarray, sentiment: np.ndarray) -> TradingEnv:
    env_cfg = cfg.env
    return TradingEnv(
        features=features,
        prices=prices,
        window=env_cfg.window,
        tx_cost=env_cfg.tx_cost,
        sentiment_signal=sentiment,
        sentiment_lambda=env_cfg.sentiment_lambda,
        action_space_type=getattr(env_cfg, "action_space_type", "discrete"),
    )


def _algo_kwargs(algo: str, agent_cfg) -> dict:
    if algo == "SAC":
        return dict(
            gamma=agent_cfg.gamma,
            batch_size=agent_cfg.batch_size,
            buffer_size=agent_cfg.buffer_size,
            ent_coef=agent_cfg.ent_coef,
            learning_starts=agent_cfg.learning_starts,
            train_freq=agent_cfg.train_freq,
            gradient_steps=agent_cfg.gradient_steps,
            tau=agent_cfg.tau,
            use_sde=agent_cfg.use_sde,
        )
    if algo == "DQN":
        return dict(
            gamma=agent_cfg.gamma,
            batch_size=agent_cfg.batch_size,
            buffer_size=agent_cfg.buffer_size,
            learning_starts=agent_cfg.learning_starts,
            train_freq=agent_cfg.train_freq,
            gradient_steps=agent_cfg.gradient_steps,
            target_update_interval=agent_cfg.target_update_interval,
            exploration_fraction=agent_cfg.exploration_fraction,
            exploration_final_eps=agent_cfg.exploration_final_eps,
            tau=agent_cfg.tau,
        )
    raise ValueError(algo)


def train_agent(
    cfg,
    algo: str,
    seed: int,
    features: np.ndarray,
    prices: np.ndarray,
    sentiment: np.ndarray,
    models_dir: str = "models",
) -> Path:
    """Train one algo (SAC or DQN) with one seed; return path to saved model."""
    assert algo in ALGO_MAP, f"unsupported algo: {algo}"
    agent_cfg = cfg.agent_sac if algo == "SAC" else cfg.agent_dqn

    env = _build_env(cfg, features, prices, sentiment)
    vec_env = DummyVecEnv([lambda: env])

    lr: Union[float, callable] = agent_cfg.learning_rate
    if agent_cfg.lr_schedule == "linear":
        lr = _linear_schedule(agent_cfg.learning_rate)

    policy_kwargs = {
        "net_arch": agent_cfg.net_arch,
        "activation_fn": ACTIVATION_MAP[agent_cfg.activation_fn],
    }

    algo_cls = ALGO_MAP[algo]
    model = algo_cls(
        "MlpPolicy",
        vec_env,
        learning_rate=lr,
        seed=seed,
        verbose=0,
        device=agent_cfg.device,
        policy_kwargs=policy_kwargs,
        **_algo_kwargs(algo, agent_cfg),
    )

    logger.info(f"Training {algo} seed={seed} for {agent_cfg.total_timesteps} steps")
    model.learn(
        total_timesteps=agent_cfg.total_timesteps,
        callback=ProgressCallback(agent_cfg.total_timesteps),
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(models_dir) / algo / f"{algo}_seed{seed}_{ts}"
    save_dir.mkdir(parents=True, exist_ok=True)
    model_path = save_dir / "model.zip"
    model.save(str(model_path.with_suffix("")))
    logger.info(f"Saved model to {model_path}")

    return model_path
