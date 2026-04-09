from dataclasses import dataclass, field
from typing import List

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
    ent_coef: float = 0.0
    sentiment_lambda: float = 0.1
    net_arch: List[int] = field(default_factory=lambda: [256, 256])
    activation_fn: str = "tanh"
    lr_schedule: str = "constant"    # "constant", "linear", or "warmup_linear"

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
