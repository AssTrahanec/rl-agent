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
