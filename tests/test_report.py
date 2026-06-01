"""Tests for the report generator."""
import tempfile
from pathlib import Path

import numpy as np

from src.eval.report import generate_report


def make_dummy_results():
    """Create dummy results dicts for testing."""
    np.random.seed(42)
    results = []
    for agent in ["baseline", "sentiment", "embeddings"]:
        results.append({
            "label": agent,
            "metrics": {
                "total_return": round(np.random.uniform(-0.2, 0.5), 4),
                "sharpe_ratio": round(np.random.uniform(-1.0, 2.0), 4),
                "sortino_ratio": round(np.random.uniform(-1.0, 2.5), 4),
                "max_drawdown": round(np.random.uniform(0.05, 0.3), 4),
                "calmar_ratio": round(np.random.uniform(0.0, 3.0), 4),
            },
        })
    return results


def test_generate_report_creates_markdown():
    results = make_dummy_results()
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = str(Path(tmpdir) / "report.md")
        generate_report(results, output_path=out_path, fmt="markdown")
        assert Path(out_path).exists()
        content = Path(out_path).read_text()
        assert "baseline" in content
        assert "sharpe" in content.lower()


def test_generate_report_creates_latex():
    results = make_dummy_results()
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = str(Path(tmpdir) / "report.tex")
        generate_report(results, output_path=out_path, fmt="latex")
        assert Path(out_path).exists()
        content = Path(out_path).read_text()
        assert "tabular" in content or "begin" in content


def test_generate_report_all_agents_present():
    results = make_dummy_results()
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = str(Path(tmpdir) / "report.md")
        generate_report(results, output_path=out_path, fmt="markdown")
        content = Path(out_path).read_text()
        for agent in ["baseline", "sentiment", "embeddings"]:
            assert agent in content
