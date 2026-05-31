import numpy as np

TRADING_DAYS_PER_YEAR = 365
PERIODS_PER_YEAR_4H = 365 * 6  # 2190

_PROFIT_FACTOR_CAP = 1000.0


def compute_metrics(returns, allocations=None):
    returns = np.asarray(returns, dtype=np.float64)
    n = len(returns)

    total_return = float(np.prod(1 + returns) - 1)

    mean_r = np.mean(returns) if n > 0 else 0.0
    std_r = np.std(returns, ddof=1) if n > 1 else 1.0
    sharpe = float(mean_r / std_r * np.sqrt(TRADING_DAYS_PER_YEAR)) if std_r > 0 else 0.0

    downside = returns[returns < 0]
    d_std = np.std(downside, ddof=1) if len(downside) > 1 else 1.0
    sortino = float(mean_r / d_std * np.sqrt(TRADING_DAYS_PER_YEAR)) if d_std > 0 else 0.0

    cumulative = np.cumprod(1 + returns) if n > 0 else np.array([1.0])
    drawdowns = 1 - cumulative / np.maximum.accumulate(cumulative)
    max_dd = float(np.max(drawdowns)) if n > 0 else 0.0

    annual_return = (1 + total_return) ** (TRADING_DAYS_PER_YEAR / max(n, 1)) - 1
    calmar = float(annual_return / max_dd) if max_dd > 0 else 0.0

    annualized_return = float((1 + total_return) ** (PERIODS_PER_YEAR_4H / max(n, 1)) - 1) if n > 0 else 0.0

    win_rate = float((returns > 0).sum() / n) if n > 0 else 0.0

    gross_profit = float(returns[returns > 0].sum())
    gross_loss = float(-returns[returns < 0].sum())
    if gross_loss > 0:
        profit_factor = min(gross_profit / gross_loss, _PROFIT_FACTOR_CAP)
    elif gross_profit > 0:
        profit_factor = _PROFIT_FACTOR_CAP
    else:
        profit_factor = 0.0

    out = {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_dd,
        "calmar_ratio": calmar,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
    }
    if allocations is not None:
        a = np.asarray(allocations)
        out["time_in_market"] = float((a > 0).sum() / len(a)) if len(a) > 0 else 0.0
    return out
