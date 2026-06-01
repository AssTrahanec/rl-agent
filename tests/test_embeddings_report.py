from pathlib import Path
import numpy as np
import pandas as pd


def test_embeddings_report_generates_artifacts(tmp_path):
    from src.eval import embeddings_report as rep

    rng = np.random.RandomState(0)
    rows = []
    for algo in ("PPO", "A2C", "SAC"):
        for seed in (42, 123, 7, 2024, 99):
            rows.append({
                "algorithm": algo,
                "seed": seed,
                "sharpe": rng.randn(),
                "sharpe_ci_low": 0.0,
                "sharpe_ci_high": 1.0,
                "sortino": rng.randn(),
                "max_drawdown": -abs(rng.randn()) * 0.1,
                "calmar": rng.randn(),
                "total_return": rng.randn() * 0.2,
                "total_return_ci_low": 0.0,
                "total_return_ci_high": 0.3,
            })
    oos_csv = tmp_path / "oos.csv"
    pd.DataFrame(rows).to_csv(oos_csv, index=False)

    out_md = tmp_path / "report.md"
    fig_dir = tmp_path / "figures"
    rep.generate_embeddings_report(
        oos_csv=oos_csv,
        report_md=out_md,
        figures_dir=fig_dir,
        bh_sharpe=0.5,
        bh_total_return=0.1,
    )
    assert out_md.exists()
    md = out_md.read_text()
    assert "PPO" in md and "A2C" in md and "SAC" in md
    assert "Mann-Whitney" in md
    assert (fig_dir / "sharpe_bar.png").exists()
