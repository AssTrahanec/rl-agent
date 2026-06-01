"""Unit tests for win_rate and profit_factor in dsr_experiment/lib/metrics.py.

These cover only the two new trading metrics added on top of the existing
Sharpe / Sortino / Calmar / MaxDD / TotalReturn (which already have coverage
in tests/test_metrics.py for the parallel src/eval/metrics module).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Add dsr_experiment/ to sys.path so `from lib.metrics import ...` resolves the
# same way it does when running scripts from inside that folder.
_DSR_EXP = Path(__file__).resolve().parent.parent / "dsr_experiment"
if str(_DSR_EXP) not in sys.path:
    sys.path.insert(0, str(_DSR_EXP))

from lib.metrics import compute_metrics  # noqa: E402


# ----- win_rate ----------------------------------------------------------------


def test_win_rate_all_positive():
    """All positive returns -> win_rate is 1.0."""
    m = compute_metrics(np.array([0.01, 0.02, 0.03]))
    assert m["win_rate"] == 1.0


def test_win_rate_all_negative():
    """All negative returns -> win_rate is 0.0."""
    m = compute_metrics(np.array([-0.01, -0.02, -0.03]))
    assert m["win_rate"] == 0.0


def test_win_rate_mixed():
    """3 of 5 strictly positive -> 0.6."""
    m = compute_metrics(np.array([0.01, -0.02, 0.03, -0.01, 0.02]))
    assert m["win_rate"] == 0.6


def test_win_rate_zero_returns_not_counted_as_win():
    """Strictly > 0 — exact zeros are not wins."""
    m = compute_metrics(np.array([0.0, 0.0, 0.01]))
    # Only one strictly positive out of three.
    assert m["win_rate"] == 1 / 3


# ----- profit_factor -----------------------------------------------------------


def test_profit_factor_balanced():
    """PF = sum(positive) / abs(sum(negative)). [+0.02, -0.01] -> 2.0."""
    m = compute_metrics(np.array([0.02, -0.01]))
    assert abs(m["profit_factor"] - 2.0) < 1e-9


def test_profit_factor_break_even():
    """Equal gross profit and loss -> PF = 1.0."""
    m = compute_metrics(np.array([0.02, -0.02, 0.03, -0.03]))
    assert abs(m["profit_factor"] - 1.0) < 1e-9


def test_profit_factor_no_losses_is_capped():
    """No negative returns -> PF cap (1000), not inf."""
    m = compute_metrics(np.array([0.01, 0.02, 0.03]))
    pf = m["profit_factor"]
    assert np.isfinite(pf)  # Not inf — crucial for bootstrap CI
    assert pf == 1000.0     # Exactly at cap


def test_profit_factor_all_zeros_returns_zero():
    """Pathological case: all-zero returns -> PF = 0.0 (no edge to measure)."""
    m = compute_metrics(np.array([0.0, 0.0, 0.0]))
    assert m["profit_factor"] == 0.0


def test_profit_factor_uneven_split():
    """Single small loss against many small wins -> high PF, still finite."""
    m = compute_metrics(np.array([0.05, 0.05, 0.05, -0.01]))
    # gross_profit = 0.15, gross_loss = 0.01 -> PF = 15
    assert abs(m["profit_factor"] - 15.0) < 1e-9


# ----- annualized_return ------------------------------------------------------


def test_annualized_return_full_4h_year():
    """Exactly 2190 4-hour bars (= 1 year) with no profit -> annualized = 0."""
    r = np.zeros(2190)
    m = compute_metrics(r)
    assert abs(m["annualized_return"] - 0.0) < 1e-9


def test_annualized_return_matches_total_for_full_year():
    """For exactly one 4h-year (2190 bars), annualized == total_return."""
    # Single positive bar at the end, so total return ≈ 0.05
    r = np.zeros(2190)
    r[-1] = 0.05
    m = compute_metrics(r)
    # Period length exactly matches PERIODS_PER_YEAR_4H, so exponent = 1.
    assert abs(m["annualized_return"] - m["total_return"]) < 1e-9


def test_annualized_return_half_year_extrapolates_upward():
    """Half year with positive total -> annualized > total (compounding)."""
    r = np.zeros(1095)  # half a 4h-year
    r[-1] = 0.20  # +20% over half a year
    m = compute_metrics(r)
    # Annualized = 1.20^2 - 1 = 0.44 ≈ 44 %.
    assert m["annualized_return"] > m["total_return"]
    assert abs(m["annualized_return"] - 0.44) < 0.01


# ----- time_in_market ---------------------------------------------------------


def test_time_in_market_all_in_position():
    """Every step has positive allocation -> 1.0."""
    r = np.zeros(5)
    a = np.array([0.5, 1.0, 0.5, 0.7, 0.9])
    m = compute_metrics(r, allocations=a)
    assert m["time_in_market"] == 1.0


def test_time_in_market_partial():
    """Two of four steps in position -> 0.5."""
    r = np.zeros(4)
    a = np.array([0.0, 0.5, 0.0, 0.5])
    m = compute_metrics(r, allocations=a)
    assert m["time_in_market"] == 0.5


def test_time_in_market_all_cash():
    """Agent never enters position -> 0.0."""
    r = np.zeros(3)
    a = np.array([0.0, 0.0, 0.0])
    m = compute_metrics(r, allocations=a)
    assert m["time_in_market"] == 0.0


def test_time_in_market_optional_field():
    """Without allocations -> field absent (backwards compatible)."""
    m = compute_metrics(np.array([0.01, -0.02]))
    assert "time_in_market" not in m
    # Other metrics still present.
    assert "annualized_return" in m
    assert "win_rate" in m


# ----- contract check ----------------------------------------------------------


def test_new_metrics_in_returned_dict():
    """Both new keys must be present alongside the legacy ones."""
    m = compute_metrics(np.array([0.01, -0.02, 0.03]))
    assert "win_rate" in m
    assert "profit_factor" in m
    assert "annualized_return" in m
    # Legacy keys still there for backwards compatibility.
    for k in ("total_return", "sharpe_ratio", "sortino_ratio",
              "max_drawdown", "calmar_ratio"):
        assert k in m


# ----- Sharpe/Sortino annualization (must use sqrt(2190), not sqrt(365)) -------


def test_sharpe_annualized_with_4h_periods():
    """Sharpe = (mean/std) * sqrt(PERIODS_PER_YEAR_4H); returns are per 4h bar."""
    from lib.metrics import PERIODS_PER_YEAR_4H
    rng = np.random.RandomState(0)
    r = 0.001 + 0.0001 * rng.randn(500)
    m = compute_metrics(r)
    per_period = r.mean() / r.std(ddof=1)
    expected = per_period * np.sqrt(PERIODS_PER_YEAR_4H)
    assert abs(m["sharpe_ratio"] - expected) < 1e-6, (m["sharpe_ratio"], expected)


def test_no_legacy_trading_days_constant():
    """The old sqrt(365) convention is gone — only one annualization constant remains."""
    import lib.metrics as M
    assert not hasattr(M, "TRADING_DAYS_PER_YEAR")
