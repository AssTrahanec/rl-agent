"""Small styling helpers: plotly colors + metric formatters."""

ALGO_COLORS = {
    "SAC": "#1f77b4",
    "DQN": "#2ca02c",
    "PPO": "#ff7f0e",
}

BH_COLOR = "#000000"


def fmt_pct(x: float, sign: bool = True) -> str:
    """Format fraction as percent. 0.05 -> '+5.0%'."""
    s = f"{x * 100:+.1f}%" if sign else f"{x * 100:.1f}%"
    return s


def fmt_sharpe(x: float) -> str:
    """Format Sharpe with 3 decimals."""
    return f"{x:+.3f}"


def fmt_ci(low: float, high: float) -> str:
    """Format a confidence interval as '[+0.66, +0.91]'."""
    return f"[{low:+.3f}, {high:+.3f}]"


def hex_to_rgba(hex_color: str, alpha: float = 0.2) -> str:
    """Convert '#rrggbb' to 'rgba(r,g,b,alpha)' for plotly fill colors."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"
