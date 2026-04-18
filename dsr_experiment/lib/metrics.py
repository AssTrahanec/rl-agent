"""Financial metrics."""
import numpy as np

TRADING_DAYS_PER_YEAR = 365


def compute_metrics(returns: np.ndarray) -> dict:
    """Compute Sharpe, Sortino, MaxDD, Calmar, total return from daily returns."""
    returns = np.asarray(returns, dtype=np.float64)
    total_return = float(np.prod(1 + returns) - 1)

    mean_r = np.mean(returns)
    std_r = np.std(returns, ddof=1) if len(returns) > 1 else 1.0
    sharpe_ratio = float((mean_r / std_r) * np.sqrt(TRADING_DAYS_PER_YEAR)) if std_r > 0 else 0.0

    downside = returns[returns < 0]
    downside_std = float(np.std(downside, ddof=1)) if len(downside) > 1 else 1.0
    sortino_ratio = float((mean_r / downside_std) * np.sqrt(TRADING_DAYS_PER_YEAR)) if downside_std > 0 else 0.0

    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = 1 - cumulative / running_max
    max_drawdown = float(np.max(drawdowns))

    annual_return = (1 + total_return) ** (TRADING_DAYS_PER_YEAR / max(len(returns), 1)) - 1
    calmar_ratio = float(annual_return / max_drawdown) if max_drawdown > 0 else 0.0

    return {
        "total_return": total_return,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "max_drawdown": max_drawdown,
        "calmar_ratio": calmar_ratio,
    }
