"""Load config.yaml into typed dataclasses with validation.

Loading is tolerant of unknown/legacy keys (filtered out via `_only_known`), so
older config files with retired fields (reward_type, allow_short, agent_ppo, ...)
still load against the current, slimmer dataclasses.
"""
from dataclasses import dataclass, fields
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
    sentiment_lambda: float
    action_space_type: str = "discrete"


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
class DQNConfig:
    device: str
    total_timesteps: int
    learning_rate: float
    lr_schedule: str
    buffer_size: int
    batch_size: int
    learning_starts: int
    tau: float
    gamma: float
    train_freq: int
    gradient_steps: int
    target_update_interval: int
    exploration_fraction: float
    exploration_final_eps: float
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
    agent_dqn: "DQNConfig | None" = None

    def oos_features_path(self, period_key: str) -> str:
        return f"{self.data.paths.oos_dir}/{period_key}_features.parquet"


def _only_known(cls, d: dict) -> dict:
    """Keep only keys that are fields of dataclass `cls` (drop legacy/extra keys)."""
    names = {f.name for f in fields(cls)}
    return {k: v for k, v in d.items() if k in names}


def load_config(path: str) -> Config:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    cfg = Config(
        data=DataConfig(
            asset=raw["data"]["asset"],
            timeframe=raw["data"]["timeframe"],
            paths=DataPaths(**_only_known(DataPaths, raw["data"]["paths"])),
        ),
        periods={k: Period(**_only_known(Period, v)) for k, v in raw["periods"].items()},
        news=NewsConfig(**_only_known(NewsConfig, raw["news"])),
        embeddings=EmbeddingsConfig(**_only_known(EmbeddingsConfig, raw["embeddings"])),
        sentiment=SentimentConfig(**_only_known(SentimentConfig, raw["sentiment"])),
        features=FeaturesConfig(**_only_known(FeaturesConfig, raw["features"])),
        env=EnvConfig(**_only_known(EnvConfig, raw["env"])),
        experiment=ExperimentConfig(**_only_known(ExperimentConfig, raw["experiment"])),
        agent_sac=SACConfig(**_only_known(SACConfig, raw["agent_sac"])),
        agent_dqn=DQNConfig(**_only_known(DQNConfig, raw["agent_dqn"])) if "agent_dqn" in raw else None,
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
    for algo in cfg.experiment.algos:
        if algo not in {"SAC", "DQN"}:
            raise ValueError(f"experiment.algos: only SAC/DQN supported, got '{algo}'")
        if algo == "DQN" and cfg.agent_dqn is None:
            raise ValueError("algos includes DQN but agent_dqn config missing")
        if algo == "DQN" and cfg.env.action_space_type != "discrete":
            raise ValueError("DQN requires env.action_space_type='discrete'")
    if cfg.embeddings.compressed_dim > cfg.embeddings.raw_dim:
        raise ValueError("embeddings.compressed_dim must be <= raw_dim")
