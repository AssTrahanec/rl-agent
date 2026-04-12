from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd


def test_run_oos_backtest_produces_csv(tmp_path):
    from src.eval import run_oos_backtest as oos

    runs_csv = tmp_path / "runs.csv"
    pd.DataFrame([
        {"algorithm": "PPO", "seed": 42, "model_path": "a.zip", "vecnorm_path": "a.pkl", "timestamp": "t"},
        {"algorithm": "SAC", "seed": 42, "model_path": "b.zip", "vecnorm_path": "", "timestamp": "t"},
    ]).to_csv(runs_csv, index=False)

    def fake_backtest(*args, **kwargs):
        return {
            "equity_curve": np.linspace(1.0, 1.2, 100),
            "allocations": np.zeros(100),
            "daily_returns": np.random.RandomState(0).randn(100) * 0.01,
        }

    def fake_metrics(returns):
        return {
            "sharpe": 1.5,
            "sortino": 2.0,
            "max_drawdown": -0.1,
            "calmar": 0.5,
            "total_return": 0.2,
        }

    def fake_bootstrap(returns, statistic, n_iter=1000, ci=0.95, seed=0):
        return (0.9, 2.1)

    out_csv = tmp_path / "oos.csv"
    with patch.object(oos, "run_backtest", side_effect=fake_backtest), \
         patch.object(oos, "compute_metrics", side_effect=fake_metrics), \
         patch.object(oos, "bootstrap_ci", side_effect=fake_bootstrap), \
         patch.object(oos, "_load_test_features", return_value=(np.zeros((200, 95), dtype=np.float32), np.ones(200))):
        oos.run_oos_backtest(
            runs_csv=runs_csv,
            output_csv=out_csv,
            asset="BTC/USDT",
            timeframe="4h",
            test_start="2024-01-01",
            test_end="2024-12-31",
        )

    df = pd.read_csv(out_csv)
    assert len(df) == 2
    assert "sharpe" in df.columns
    assert "sharpe_ci_low" in df.columns
    assert "sharpe_ci_high" in df.columns
    assert "total_return" in df.columns
