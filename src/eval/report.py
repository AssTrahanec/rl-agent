"""Generate summary tables from backtest results.

Produces markdown and LaTeX tables for the ablation study and algorithm comparison.
"""
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

METRIC_DISPLAY = {
    "total_return": "Total Return",
    "sharpe_ratio": "Sharpe Ratio",
    "sortino_ratio": "Sortino Ratio",
    "max_drawdown": "Max Drawdown",
    "calmar_ratio": "Calmar Ratio",
}


def generate_report(
    results: List[Dict[str, Any]],
    output_path: str,
    fmt: str = "markdown",
    metrics: Optional[List[str]] = None,
    title: str = "Results Summary",
) -> None:
    """Generate a summary table and write to file.

    Args:
        results: List of dicts with keys 'label' and 'metrics' (dict).
        output_path: Path to write the output file.
        fmt: Output format — 'markdown' or 'latex'.
        metrics: Metric keys to include. Defaults to all from first result.
        title: Table title / caption.
    """
    if metrics is None:
        metrics = list(results[0]["metrics"].keys()) if results else []

    if fmt == "markdown":
        content = _build_markdown(results, metrics, title)
    elif fmt == "latex":
        content = _build_latex(results, metrics, title)
    else:
        raise ValueError(f"Unsupported format: {fmt!r}. Use 'markdown' or 'latex'.")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(content, encoding="utf-8")
    logger.info(f"Saved {fmt} report to {out}")


def _build_markdown(
    results: List[Dict[str, Any]],
    metrics: List[str],
    title: str,
) -> str:
    headers = ["Agent"] + [METRIC_DISPLAY.get(m, m) for m in metrics]
    header_row = "| " + " | ".join(headers) + " |"
    sep_row = "| " + " | ".join(["---"] * len(headers)) + " |"

    rows = [f"# {title}\n", header_row, sep_row]
    for r in results:
        vals = [r["label"]] + [f"{r['metrics'].get(m, float('nan')):.4f}" for m in metrics]
        rows.append("| " + " | ".join(vals) + " |")

    return "\n".join(rows) + "\n"


def _build_latex(
    results: List[Dict[str, Any]],
    metrics: List[str],
    title: str,
) -> str:
    col_spec = "l" + "r" * len(metrics)
    headers = ["Agent"] + [METRIC_DISPLAY.get(m, m) for m in metrics]

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        f"\\caption{{{title}}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    for r in results:
        vals = [r["label"]] + [f"{r['metrics'].get(m, float('nan')):.4f}" for m in metrics]
        lines.append(" & ".join(vals) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"
