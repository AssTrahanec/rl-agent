import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.compare_results import load_metrics, compute_deltas, run_ttest


def _make_csv(path, rows):
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def test_load_metrics():
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "metrics.csv"
        _make_csv(csv_path, [
            {"agent_type": "baseline", "asset": "BTC/USDT", "seed": 42,
             "total_return": 0.1, "sharpe_ratio": 0.5, "sortino_ratio": 0.4,
             "max_drawdown": 0.1, "calmar_ratio": 0.3},
        ])
        df = load_metrics(str(csv_path))
        assert len(df) == 1
        assert df.iloc[0]["sharpe_ratio"] == 0.5


def test_compute_deltas():
    old = pd.DataFrame([
        {"agent_type": "embeddings", "seed": 42, "sharpe_ratio": 0.5, "total_return": 0.1},
        {"agent_type": "embeddings", "seed": 43, "sharpe_ratio": 0.1, "total_return": -0.01},
    ])
    new = pd.DataFrame([
        {"agent_type": "embeddings", "seed": 42, "sharpe_ratio": 0.8, "total_return": 0.2},
        {"agent_type": "embeddings", "seed": 43, "sharpe_ratio": 0.6, "total_return": 0.15},
    ])
    deltas = compute_deltas(old, new, agent_type="embeddings")
    assert len(deltas) == 2
    assert abs(deltas.iloc[0]["sharpe_delta"] - 0.3) < 1e-6


def test_run_ttest():
    old_sharpes = np.array([0.1, 0.2, 0.15, 0.12, 0.18])
    new_sharpes = np.array([0.8, 0.9, 0.85, 0.82, 0.88])
    t_stat, p_value = run_ttest(old_sharpes, new_sharpes)
    assert p_value < 0.05
