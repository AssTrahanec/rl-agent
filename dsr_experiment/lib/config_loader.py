"""Load config.yaml into typed dataclasses with validation."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Union

import yaml


@dataclass
class DataPaths:
    raw_ohlcv: str
    raw_news: str
    train_features: str
    compressor: str
    oos_dir: str


@dataclass
class DataConfig:
    asset: str
    timeframe: str
    paths: DataPaths


@dataclass
class Period:
    start: str
    end: str


@dataclass
class NewsConfig:
    hf_dataset: str
    text_col: str
    date_col: str
    dedup_threshold: float


@dataclass
class EmbeddingsConfig:
    model_name: str
    raw_dim: int
    compressed_dim: int
    top_pca_lags: int
    temporal_decay_alpha: float


@dataclass
class SentimentConfig:
    model_name: str


@dataclass
class FeaturesConfig:
    normalize_window: int
    news_count_lags: List[int]
    news_count_roll: int


@dataclass
class EnvConfig:
    window: int
    tx_cost: float
    reward_type: str
    allow_short: bool
    volatility_penalty: float
    dsr_eta: float
    sentiment_lambda: float


@dataclass
class ExperimentConfig:
    seeds: List[int]
    algos: List[str]
    oos_periods: List[str]


@dataclass
class SACConfig:
    device: str
    total_timesteps: int
    learning_rate: float
    lr_schedule: str
    buffer_size: int
    batch_size: int
    learning_starts: int
    tau: float
    gamma: float
    ent_coef: Union[float, str]
    train_freq: int
    gradient_steps: int
    use_sde: bool
    net_arch: List[int]
    activation_fn: str


@dataclass
class PPOConfig:
    device: str
    total_timesteps: int
    learning_rate: float
    lr_schedule: str
    n_steps: int
    batch_size: int
    n_epochs: int
    gamma: float
    gae_lambda: float
    clip_range: float
    ent_coef: Union[float, str]
    max_grad_norm: float
    use_sde: bool
    net_arch: List[int]
    activation_fn: str


@dataclass
class Config:
    data: DataConfig
    periods: Dict[str, Period]
    news: NewsConfig
    embeddings: EmbeddingsConfig
    sentiment: SentimentConfig
    features: FeaturesConfig
    env: EnvConfig
    experiment: ExperimentConfig
    agent_sac: SACConfig
    agent_ppo: PPOConfig

    def oos_features_path(self, period_key: str) -> str:
        return f"{self.data.paths.oos_dir}/{period_key}_features.parquet"


def load_config(path: str) -> Config:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    cfg = Config(
        data=DataConfig(
            asset=raw["data"]["asset"],
            timeframe=raw["data"]["timeframe"],
            paths=DataPaths(**raw["data"]["paths"]),
        ),
        periods={k: Period(**v) for k, v in raw["periods"].items()},
        news=NewsConfig(**raw["news"]),
        embeddings=EmbeddingsConfig(**raw["embeddings"]),
        sentiment=SentimentConfig(**raw["sentiment"]),
        features=FeaturesConfig(**raw["features"]),
        env=EnvConfig(**raw["env"]),
        experiment=ExperimentConfig(**raw["experiment"]),
        agent_sac=SACConfig(**raw["agent_sac"]),
        agent_ppo=PPOConfig(**raw["agent_ppo"]),
    )
    _validate(cfg)
    return cfg


def _validate(cfg: Config) -> None:
    if "train" not in cfg.periods:
        raise ValueError("config.periods must include 'train'")
    if "train" in cfg.experiment.oos_periods:
        raise ValueError("'train' cannot be used as OOS period")
    for k in cfg.experiment.oos_periods:
        if k not in cfg.periods:
            raise ValueError(f"oos_period '{k}' not found in config.periods")
    if cfg.env.reward_type not in {"basic", "risk_adjusted", "dsr"}:
        raise ValueError(f"env.reward_type invalid: {cfg.env.reward_type}")
    for algo in cfg.experiment.algos:
        if algo not in {"SAC", "PPO"}:
            raise ValueError(f"experiment.algos: only SAC/PPO supported, got '{algo}'")
    if cfg.embeddings.compressed_dim > cfg.embeddings.raw_dim:
        raise ValueError("embeddings.compressed_dim must be <= raw_dim")
