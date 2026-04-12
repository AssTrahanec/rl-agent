from src.agents.config import AgentConfig


def test_default_values():
    config = AgentConfig()
    assert config.learning_rate == 3e-4
    assert config.n_steps == 2048
    assert config.batch_size == 64
    assert config.n_epochs == 10
    assert config.gamma == 0.99
    assert config.gae_lambda == 0.95
    assert config.clip_range == 0.2
    assert config.net_arch == [256, 256]
    assert config.activation_fn == "tanh"
    assert config.total_timesteps == 500_000
    assert config.seed == 42
    assert config.window == 30
    assert config.tx_cost == 0.001


def test_custom_values():
    config = AgentConfig(learning_rate=1e-3, total_timesteps=1000, seed=0)
    assert config.learning_rate == 1e-3
    assert config.total_timesteps == 1000
    assert config.seed == 0


def test_agent_type_default():
    config = AgentConfig()
    assert config.agent_type == "baseline"
    assert config.algorithm == "PPO"


def test_policy_kwargs():
    config = AgentConfig()
    kwargs = config.policy_kwargs()
    assert kwargs["net_arch"] == [256, 256]
    assert "activation_fn" in kwargs


def test_save_dir_default():
    config = AgentConfig()
    assert config.save_dir == "experiments"


def test_config_reward_type_default():
    config = AgentConfig()
    assert config.reward_type == "basic"
    assert config.allow_short is False


def test_config_reward_type_custom():
    config = AgentConfig(reward_type="risk_adjusted", allow_short=True)
    assert config.reward_type == "risk_adjusted"
    assert config.allow_short is True


def test_config_new_fields_defaults():
    """New fields ent_coef and sentiment_lambda have correct defaults."""
    config = AgentConfig()
    assert config.ent_coef == 0.0
    assert config.sentiment_lambda == 0.1


def test_config_embeddings_overrides():
    """Embeddings agent can use custom ent_coef."""
    config = AgentConfig(agent_type="embeddings", ent_coef=0.01, gamma=0.95)
    assert config.ent_coef == 0.01
    assert config.gamma == 0.95


def test_ppo_embeddings_config():
    from src.agents.config import ppo_embeddings_config
    cfg = ppo_embeddings_config(seed=42)
    assert cfg.algorithm == "PPO"
    assert cfg.agent_type == "embeddings"
    assert cfg.seed == 42
    assert cfg.learning_rate == 3e-4
    assert cfg.lr_schedule == "linear"
    assert cfg.n_steps == 2048
    assert cfg.batch_size == 64
    assert cfg.n_epochs == 10
    assert cfg.gamma == 0.99
    assert cfg.gae_lambda == 0.95
    assert cfg.clip_range == 0.2
    assert cfg.ent_coef == 0.01
    assert cfg.max_grad_norm == 0.5
    assert cfg.use_sde is True
    assert cfg.net_arch == [128, 128]
    assert cfg.activation_fn == "tanh"
    assert cfg.total_timesteps == 500_000
    assert cfg.device == "cpu"
    assert cfg.window == 30
    assert cfg.tx_cost == 0.001


def test_config_new_fields_defaults():
    from src.agents.config import AgentConfig
    cfg = AgentConfig()
    assert cfg.max_grad_norm == 0.5
    assert cfg.use_sde is False
    assert cfg.buffer_size == 1_000_000
    assert cfg.tau == 0.005
    assert cfg.train_freq == 1
    assert cfg.gradient_steps == 1
    assert cfg.learning_starts == 100
    assert cfg.device == "cpu"
    assert cfg.normalize_advantage is True
    assert cfg.optimize_memory_usage is False
