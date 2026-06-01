from dataclasses import dataclass, field
from typing import List, Union

import torch


@dataclass
class AgentConfig:
    """Configuration for RL agent training."""

    # PPO hyperparameters
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: Union[float, str] = 0.0
    sentiment_lambda: float = 0.1
    net_arch: List[int] = field(default_factory=lambda: [256, 256])
    activation_fn: str = "tanh"
    lr_schedule: str = "constant"    # "constant" or "linear" (decay to 0)

    # Shared on-policy / off-policy
    max_grad_norm: float = 0.5
    use_sde: bool = False
    normalize_advantage: bool = True

    # SAC off-policy
    buffer_size: int = 1_000_000
    tau: float = 0.005
    train_freq: int = 1
    gradient_steps: int = 1
    learning_starts: int = 100
    optimize_memory_usage: bool = False

    # Device
    device: str = "cpu"

    # Training
    total_timesteps: int = 500_000
    seed: int = 42
    algorithm: str = "PPO"

    # Environment
    window: int = 30
    tx_cost: float = 0.001

    # Reward
    reward_type: str = "basic"       # "basic" or "risk_adjusted"
    allow_short: bool = False        # allow allocation in [-1, 1]

    # Agent type
    agent_type: str = "baseline"

    # Paths
    save_dir: str = "experiments"

    def policy_kwargs(self) -> dict:
        """Return policy_kwargs dict for SB3."""
        activation_map = {
            "tanh": torch.nn.Tanh,
            "relu": torch.nn.ReLU,
        }
        return {
            "net_arch": self.net_arch,
            "activation_fn": activation_map[self.activation_fn],
        }


def sac_embeddings_config(seed: int) -> AgentConfig:
    """SAC tuned for embeddings 4h per RL Zoo BipedalWalker (short-episode env)."""
    return AgentConfig(
        algorithm="SAC",
        agent_type="embeddings",
        seed=seed,
        learning_rate=7.3e-4,
        lr_schedule="constant",
        buffer_size=300_000,
        learning_starts=10_000,
        batch_size=256,
        tau=0.02,
        gamma=0.99,
        ent_coef="auto",
        train_freq=8,
        gradient_steps=8,
        use_sde=True,
        net_arch=[128, 128],
        activation_fn="relu",
        total_timesteps=200_000,
        optimize_memory_usage=False,
        device="cuda",
        window=30,
        tx_cost=0.001,
        reward_type="basic",
        allow_short=False,
    )


def a2c_embeddings_config(seed: int) -> AgentConfig:
    """A2C tuned for embeddings 4h per RL Zoo defaults."""
    return AgentConfig(
        algorithm="A2C",
        agent_type="embeddings",
        seed=seed,
        learning_rate=7e-4,
        lr_schedule="linear",
        n_steps=16,
        gamma=0.99,
        gae_lambda=1.0,
        ent_coef=0.01,
        max_grad_norm=0.5,
        use_sde=True,
        normalize_advantage=True,
        net_arch=[128, 128],
        activation_fn="tanh",
        total_timesteps=500_000,
        device="cpu",
        window=30,
        tx_cost=0.001,
        reward_type="basic",
        allow_short=False,
    )


def ppo_embeddings_config(seed: int) -> AgentConfig:
    """PPO tuned for embeddings 4h per RL Zoo / SB3 best practices."""
    return AgentConfig(
        algorithm="PPO",
        agent_type="embeddings",
        seed=seed,
        learning_rate=3e-4,
        lr_schedule="linear",
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        max_grad_norm=0.5,
        use_sde=True,
        net_arch=[128, 128],
        activation_fn="tanh",
        total_timesteps=500_000,
        device="cpu",
        window=30,
        tx_cost=0.001,
        reward_type="basic",
        allow_short=False,
    )


def sac_embeddings_dsr_config(seed: int) -> AgentConfig:
    """SAC with DSR reward + sentiment modifier for embeddings 4h."""
    return AgentConfig(
        algorithm="SAC",
        agent_type="embeddings",
        seed=seed,
        learning_rate=7.3e-4,
        lr_schedule="constant",
        buffer_size=300_000,
        learning_starts=10_000,
        batch_size=256,
        tau=0.02,
        gamma=0.99,
        ent_coef="auto",
        train_freq=8,
        gradient_steps=8,
        use_sde=True,
        net_arch=[128, 128],
        activation_fn="relu",
        total_timesteps=200_000,
        optimize_memory_usage=False,
        device="cuda",
        window=30,
        tx_cost=0.001,
        reward_type="dsr",
        sentiment_lambda=0.1,
        allow_short=False,
    )


def ppo_embeddings_dsr_config(seed: int) -> AgentConfig:
    """PPO with DSR reward + sentiment modifier for embeddings 4h."""
    return AgentConfig(
        algorithm="PPO",
        agent_type="embeddings",
        seed=seed,
        learning_rate=3e-4,
        lr_schedule="linear",
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        max_grad_norm=0.5,
        use_sde=True,
        net_arch=[128, 128],
        activation_fn="tanh",
        total_timesteps=500_000,
        device="cpu",
        window=30,
        tx_cost=0.001,
        reward_type="dsr",
        sentiment_lambda=0.1,
        allow_short=False,
    )
