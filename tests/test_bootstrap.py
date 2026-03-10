import numpy as np
from src.eval.bootstrap import bootstrap_ci


def test_bootstrap_ci_keys():
    """Result contains expected keys for each metric."""
    returns_list = [np.random.randn(50) * 0.01 for _ in range(3)]
    result = bootstrap_ci(returns_list)
    for metric in ["sharpe_ratio", "total_return"]:
        assert metric in result
        assert "mean" in result[metric]
        assert "std" in result[metric]
        assert "ci_lower" in result[metric]
        assert "ci_upper" in result[metric]


def test_bootstrap_ci_contains_mean():
    """95% CI should contain the sample mean for well-behaved data."""
    np.random.seed(42)
    returns_list = [np.random.randn(100) * 0.01 + 0.001 for _ in range(5)]
    result = bootstrap_ci(returns_list, n_bootstrap=1000)
    for metric in ["sharpe_ratio", "total_return"]:
        ci_low = result[metric]["ci_lower"]
        ci_high = result[metric]["ci_upper"]
        mean_val = result[metric]["mean"]
        assert ci_low <= mean_val <= ci_high


def test_bootstrap_ci_lower_less_than_upper():
    np.random.seed(0)
    returns_list = [np.random.randn(30) * 0.02 for _ in range(3)]
    result = bootstrap_ci(returns_list)
    for metric in ["sharpe_ratio", "total_return"]:
        assert result[metric]["ci_lower"] <= result[metric]["ci_upper"]


def test_bootstrap_single_seed():
    """Should work with a single seed (degenerate case)."""
    np.random.seed(42)
    returns_list = [np.random.randn(50) * 0.01]
    result = bootstrap_ci(returns_list)
    assert "sharpe_ratio" in result
