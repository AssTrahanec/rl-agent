# Ensemble Agents + Walk-Forward Validation Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add ensemble backtest (averaging allocations from multiple models) and walk-forward validation to demonstrate variance reduction and robustness — key improvements for thesis defense.

**Architecture:** Ensemble backtest runs multiple SB3 models on the same TradingEnv in lockstep, averaging their allocation signals each step. Walk-forward splits data into rolling train/test windows and trains+backtests on each split. Both produce metrics comparable to existing single-model results.

**Tech Stack:** stable-baselines3 (PPO/A2C), numpy, existing TradingEnv, existing metrics/backtest infrastructure.

---

## File Structure

| File | Responsibility |
|------|---------------|
| `src/eval/ensemble_backtest.py` (CREATE) | Run N models in lockstep on one env, average allocations, return metrics |
| `src/eval/backtest.py` (MODIFY) | Fix hardcoded `PPO.load()` → auto-detect algorithm from model metadata |
| `scripts/run_ensemble.py` (CREATE) | CLI script: select top models, run ensemble backtest on 3 periods |
| `scripts/run_walkforward.py` (CREATE) | CLI script: rolling train/test splits, train+backtest per split |
| `tests/test_ensemble_backtest.py` (CREATE) | Unit tests for ensemble_backtest |
| `tests/test_walkforward.py` (CREATE) | Unit tests for walk-forward logic |

---

## Chunk 1: Fix Algorithm Auto-Detection in backtest.py

### Task 1: Fix hardcoded PPO.load() in backtest.py

Currently `backtest.py` line 42-43 hardcodes `PPO.load()`. A2C models from algo comparison will fail. Fix: try PPO first, fallback to A2C, then SAC.

**Files:**
- Modify: `src/eval/backtest.py:41-43`

- [ ] **Step 1: Write the failing test**

Create `tests/test_backtest_algo_detect.py`:

```python
"""Test that backtest auto-detects algorithm type."""
import numpy as np
import pytest
from pathlib import Path
from stable_baselines3 import A2C

from src.env.trading_env import TradingEnv
from src.eval.backtest import run_backtest


@pytest.fixture
def dummy_data():
    np.random.seed(42)
    n, n_feat = 100, 18
    features = np.random.randn(n, n_feat).astype(np.float32)
    prices = (100 + np.cumsum(np.random.randn(n) * 0.5)).astype(np.float64)
    prices = np.maximum(prices, 1.0)
    return features, prices


def test_backtest_loads_a2c_model(dummy_data, tmp_path):
    """run_backtest should load A2C models without error."""
    features, prices = dummy_data
    env = TradingEnv(features=features, prices=prices, window=30, tx_cost=0.001)
    model = A2C("MlpPolicy", env, seed=42, device="cpu")
    model.learn(total_timesteps=100)
    model_path = tmp_path / "a2c_model"
    model.save(str(model_path))

    result = run_backtest(
        features=features, prices=prices,
        model_path=str(model_path) + ".zip",
        window=30, tx_cost=0.001,
    )
    assert "metrics" in result
    assert "sharpe_ratio" in result["metrics"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd c:/Users/ilya/Desktop/rl/rl-agent && venv/Scripts/python.exe -m pytest tests/test_backtest_algo_detect.py -v`
Expected: FAIL — PPO.load() cannot load A2C model.

- [ ] **Step 3: Implement auto-detection in backtest.py**

Replace lines 41-43 in `src/eval/backtest.py` with:

```python
    model = None
    if model_path is not None:
        model = _load_model(model_path)
```

Add helper function before `run_backtest`:

```python
def _load_model(model_path: str):
    """Load SB3 model, auto-detecting algorithm type."""
    from stable_baselines3 import PPO, A2C, SAC
    for algo_cls in [PPO, A2C, SAC]:
        try:
            return algo_cls.load(model_path, device="cpu")
        except Exception:
            continue
    raise ValueError(f"Cannot load model from {model_path} (tried PPO, A2C, SAC)")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd c:/Users/ilya/Desktop/rl/rl-agent && venv/Scripts/python.exe -m pytest tests/test_backtest_algo_detect.py -v`
Expected: PASS

- [ ] **Step 5: Run existing backtest tests**

Run: `cd c:/Users/ilya/Desktop/rl/rl-agent && venv/Scripts/python.exe -m pytest tests/test_backtest.py -v`
Expected: All existing tests PASS (no regression).

- [ ] **Step 6: Commit**

```bash
git add src/eval/backtest.py tests/test_backtest_algo_detect.py
git commit -m "fix: auto-detect algorithm type in backtest model loading"
```

---

## Chunk 2: Ensemble Backtest Core

### Task 2: Create ensemble_backtest.py

The ensemble runs N models on the same environment in lockstep. Each step:
1. Each model gets the same observation
2. Each model predicts an allocation
3. Allocations are averaged (simple mean or weighted)
4. Averaged allocation is applied to the environment

**Files:**
- Create: `src/eval/ensemble_backtest.py`
- Create: `tests/test_ensemble_backtest.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_ensemble_backtest.py`:

```python
"""Tests for ensemble backtest."""
import numpy as np
import pytest
from pathlib import Path
from stable_baselines3 import PPO

from src.env.trading_env import TradingEnv
from src.eval.ensemble_backtest import run_ensemble_backtest


@pytest.fixture
def dummy_data():
    np.random.seed(42)
    n, n_feat = 100, 18
    features = np.random.randn(n, n_feat).astype(np.float32)
    prices = (100 + np.cumsum(np.random.randn(n) * 0.5)).astype(np.float64)
    prices = np.maximum(prices, 1.0)
    return features, prices


@pytest.fixture
def two_models(dummy_data, tmp_path):
    features, prices = dummy_data
    paths = []
    for seed in [42, 43]:
        env = TradingEnv(features=features, prices=prices, window=30, tx_cost=0.001)
        model = PPO("MlpPolicy", env, seed=seed, device="cpu")
        model.learn(total_timesteps=200)
        p = tmp_path / f"model_s{seed}"
        model.save(str(p))
        paths.append(str(p) + ".zip")
    return paths


def test_ensemble_returns_metrics(dummy_data, two_models):
    features, prices = dummy_data
    result = run_ensemble_backtest(
        model_paths=two_models,
        features=features,
        prices=prices,
        window=30,
        tx_cost=0.001,
    )
    assert "metrics" in result
    assert "sharpe_ratio" in result["metrics"]
    assert "daily_returns" in result
    assert "allocations" in result
    assert "equity_curve" in result
    assert len(result["daily_returns"]) > 0


def test_ensemble_allocations_are_averaged(dummy_data, two_models):
    """Ensemble allocations should be between 0 and 1 (averaged)."""
    features, prices = dummy_data
    result = run_ensemble_backtest(
        model_paths=two_models,
        features=features,
        prices=prices,
        window=30,
        tx_cost=0.001,
    )
    allocs = result["allocations"]
    assert np.all(allocs >= 0.0)
    assert np.all(allocs <= 1.0)


def test_ensemble_with_weights(dummy_data, two_models):
    """Weighted ensemble should accept custom weights."""
    features, prices = dummy_data
    result = run_ensemble_backtest(
        model_paths=two_models,
        features=features,
        prices=prices,
        window=30,
        tx_cost=0.001,
        weights=[0.7, 0.3],
    )
    assert "metrics" in result


def test_ensemble_single_model_matches_backtest(dummy_data, two_models):
    """Ensemble with 1 model should produce same result as regular backtest."""
    from src.eval.backtest import run_backtest
    features, prices = dummy_data
    single = run_backtest(
        features=features, prices=prices,
        model_path=two_models[0], window=30, tx_cost=0.001,
    )
    ensemble = run_ensemble_backtest(
        model_paths=[two_models[0]],
        features=features, prices=prices,
        window=30, tx_cost=0.001,
    )
    np.testing.assert_allclose(
        single["daily_returns"], ensemble["daily_returns"], atol=1e-10
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd c:/Users/ilya/Desktop/rl/rl-agent && venv/Scripts/python.exe -m pytest tests/test_ensemble_backtest.py -v`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Implement ensemble_backtest.py**

Create `src/eval/ensemble_backtest.py`:

```python
"""Ensemble backtest: run multiple models in lockstep, average allocations."""
import logging
from typing import Optional

import numpy as np

from src.eval.backtest import _load_model
from src.env.trading_env import TradingEnv
from src.eval.metrics import compute_metrics

logger = logging.getLogger(__name__)


def run_ensemble_backtest(
    model_paths: list[str],
    features: np.ndarray,
    prices: np.ndarray,
    window: int = 30,
    tx_cost: float = 0.001,
    weights: Optional[list[float]] = None,
    plot_path: Optional[str] = None,
) -> dict:
    """Run ensemble backtest with multiple models.

    Each step, all models predict on the same observation. Their allocations
    are averaged (weighted or equal) and applied to the environment.

    Args:
        model_paths: List of paths to saved SB3 model .zip files.
        features: (T, n_features) array of normalized features.
        prices: (T,) array of close prices.
        window: Observation window size.
        tx_cost: Transaction cost fraction.
        weights: Optional weights for each model (must sum to 1).
                 If None, equal weights are used.
        plot_path: If provided, save equity curve plot.

    Returns:
        Dict with keys: metrics, daily_returns, allocations, equity_curve,
                        individual_allocations.
    """
    n_models = len(model_paths)
    if n_models == 0:
        raise ValueError("At least one model path required")

    if weights is None:
        weights = [1.0 / n_models] * n_models
    else:
        if len(weights) != n_models:
            raise ValueError(f"weights length {len(weights)} != models {n_models}")
        w_sum = sum(weights)
        weights = [w / w_sum for w in weights]

    # Load all models
    models = []
    for mp in model_paths:
        models.append(_load_model(mp))
    logger.info(f"Loaded {n_models} models for ensemble")

    # Create environment
    env = TradingEnv(features=features, prices=prices, window=window, tx_cost=tx_cost)
    obs, _ = env.reset()

    daily_returns = []
    allocations = []
    individual_allocs = []
    done = False

    while not done:
        # Each model predicts
        model_actions = []
        for m in models:
            action, _ = m.predict(obs, deterministic=True)
            model_actions.append(float(action[0]))

        # Weighted average
        ensemble_alloc = sum(w * a for w, a in zip(weights, model_actions))
        ensemble_alloc = np.clip(ensemble_alloc, 0.0, 1.0)

        action = np.array([ensemble_alloc], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        allocation = float(info.get("allocation", 0.0))
        log_ret = float(info.get("log_return", 0.0))
        daily_returns.append(np.exp(log_ret * allocation) - 1)
        allocations.append(allocation)
        individual_allocs.append(model_actions)

    daily_returns = np.array(daily_returns)
    equity_curve = np.cumprod(1 + daily_returns)
    equity_curve = np.insert(equity_curve, 0, 1.0)
    metrics = compute_metrics(daily_returns)

    if plot_path is not None:
        from src.eval.backtest import _plot_equity_curve
        _plot_equity_curve(equity_curve, plot_path)

    return {
        "metrics": metrics,
        "daily_returns": daily_returns,
        "allocations": np.array(allocations),
        "equity_curve": equity_curve,
        "individual_allocations": np.array(individual_allocs),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd c:/Users/ilya/Desktop/rl/rl-agent && venv/Scripts/python.exe -m pytest tests/test_ensemble_backtest.py -v`
Expected: All 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/eval/ensemble_backtest.py tests/test_ensemble_backtest.py
git commit -m "feat: add ensemble backtest with weighted model averaging"
```

---

## Chunk 3: Ensemble Runner Script

### Task 3: Create scripts/run_ensemble.py

This script:
1. Finds top models from `results/all_experiments.csv`
2. Runs 3 ensemble strategies: Top-3 same-type, Cross-type (best of each), Top-5 all
3. Backtests on 3 periods: 2024 (in-sample test), 2022 (bear), 2025 Q1 (OOS)
4. Compares to single best model and Buy & Hold

**Files:**
- Create: `scripts/run_ensemble.py`

- [ ] **Step 1: Create the script**

```python
"""Run ensemble backtests on multiple market periods.

Usage:
    python -m scripts.run_ensemble
    python -m scripts.run_ensemble --period 2024
"""
import argparse
import csv
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.load_features import load_features_for_agent
from src.eval.ensemble_backtest import run_ensemble_backtest
from src.eval.backtest import run_backtest
from src.eval.metrics import compute_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_CSV = Path("results/all_experiments.csv")
PERIODS = {
    "2024": ("2024-01-01", "2024-12-31"),
    "2022": ("2022-01-01", "2022-12-31"),
    "2025q1": ("2025-01-01", "2025-03-19"),
}


def load_experiment_results() -> pd.DataFrame:
    """Load all experiment results and return sorted by Sharpe."""
    df = pd.read_csv(RESULTS_CSV)
    return df.sort_values("sharpe_ratio", ascending=False)


def select_top_models(df: pd.DataFrame, n: int = 3, agent_type: str = None) -> list[str]:
    """Select top N model paths by Sharpe ratio."""
    if agent_type:
        df = df[df["agent_type"] == agent_type]
    # Extract model paths from label column (format: "btc/type/timestamp")
    paths = []
    for _, row in df.head(n).iterrows():
        label = row["label"]
        parts = label.split("/")
        # label format: "btc/baseline/20260315_123456"
        if len(parts) >= 3:
            agent_t = parts[1]
            timestamp = parts[2]
            mp = Path("experiments") / agent_t / timestamp / "model.zip"
            if mp.exists():
                paths.append(str(mp))
    return paths


def select_cross_type_models(df: pd.DataFrame) -> list[str]:
    """Select best model from each agent type."""
    paths = []
    for at in ["baseline", "sentiment", "embeddings"]:
        sub = df[df["agent_type"] == at]
        if sub.empty:
            continue
        best = sub.iloc[0]
        label = best["label"]
        parts = label.split("/")
        if len(parts) >= 3:
            mp = Path("experiments") / parts[1] / parts[2] / "model.zip"
            if mp.exists():
                paths.append(str(mp))
    return paths


def buy_and_hold(prices):
    dr = np.diff(prices) / prices[:-1]
    return compute_metrics(dr)


def run_period(period_name: str, start: str, end: str, df: pd.DataFrame):
    """Run all ensemble strategies on one period."""
    print(f"\n{'#'*80}")
    print(f"#  ENSEMBLE BACKTEST: {period_name} ({start} -> {end})")
    print(f"{'#'*80}")

    # Load features for baseline (works for all models if padded)
    try:
        feat_bl, prices_bl = load_features_for_agent("baseline", "BTC/USDT", start, end)
    except Exception as e:
        logger.error(f"Cannot load features for {period_name}: {e}")
        logger.info("Trying OOS backtest approach (fetch fresh data)...")
        return []

    # Buy & Hold
    bh = buy_and_hold(prices_bl)
    print(f"\n  Buy & Hold: Sharpe={bh['sharpe_ratio']:.3f}  "
          f"Return={bh['total_return']*100:.1f}%  MaxDD={bh['max_drawdown']*100:.1f}%")

    results = []

    # Strategy 1: Top-3 baseline models
    top3_bl = select_top_models(df, n=3, agent_type="baseline")
    if len(top3_bl) >= 2:
        r = run_ensemble_backtest(top3_bl, feat_bl, prices_bl)
        m = r["metrics"]
        label = f"Top-{len(top3_bl)} Baseline"
        print(f"  {label}: Sharpe={m['sharpe_ratio']:.3f}  "
              f"Return={m['total_return']*100:.1f}%  MaxDD={m['max_drawdown']*100:.1f}%")
        results.append({"period": period_name, "strategy": label, **m})

    # Strategy 2: Top-3 embeddings models
    top3_emb = select_top_models(df, n=3, agent_type="embeddings")
    if len(top3_emb) >= 2:
        # Need embeddings features (50d)
        try:
            feat_emb, prices_emb = load_features_for_agent("embeddings", "BTC/USDT", start, end)
            r = run_ensemble_backtest(top3_emb, feat_emb, prices_emb)
            m = r["metrics"]
            label = f"Top-{len(top3_emb)} Embeddings"
            print(f"  {label}: Sharpe={m['sharpe_ratio']:.3f}  "
                  f"Return={m['total_return']*100:.1f}%  MaxDD={m['max_drawdown']*100:.1f}%")
            results.append({"period": period_name, "strategy": label, **m})
        except Exception as e:
            logger.warning(f"Cannot run embeddings ensemble: {e}")

    # Strategy 3: Cross-type (best of each type) — all use baseline features
    # (sentiment/embeddings models with baseline features = degraded but comparable)
    cross = select_cross_type_models(df)
    if len(cross) >= 2:
        r = run_ensemble_backtest(cross, feat_bl, prices_bl)
        m = r["metrics"]
        label = f"Cross-Type ({len(cross)} models)"
        print(f"  {label}: Sharpe={m['sharpe_ratio']:.3f}  "
              f"Return={m['total_return']*100:.1f}%  MaxDD={m['max_drawdown']*100:.1f}%")
        results.append({"period": period_name, "strategy": label, **m})

    # Strategy 4: Top-5 all models (baseline features)
    top5_all = select_top_models(df, n=5)
    if len(top5_all) >= 2:
        r = run_ensemble_backtest(top5_all, feat_bl, prices_bl)
        m = r["metrics"]
        label = f"Top-{len(top5_all)} All"
        print(f"  {label}: Sharpe={m['sharpe_ratio']:.3f}  "
              f"Return={m['total_return']*100:.1f}%  MaxDD={m['max_drawdown']*100:.1f}%")
        results.append({"period": period_name, "strategy": label, **m})

    # Single best model for comparison
    best_path = select_top_models(df, n=1)
    if best_path:
        r = run_backtest(feat_bl, prices_bl, best_path[0])
        m = r["metrics"]
        print(f"  Single Best: Sharpe={m['sharpe_ratio']:.3f}  "
              f"Return={m['total_return']*100:.1f}%  MaxDD={m['max_drawdown']*100:.1f}%")
        results.append({"period": period_name, "strategy": "Single Best", **m})

    # Buy & Hold row
    results.append({"period": period_name, "strategy": "Buy & Hold", **bh})

    return results


def main():
    parser = argparse.ArgumentParser(description="Run ensemble backtests")
    parser.add_argument("--period", type=str, default=None,
                        help="Period to test (2024, 2022, 2025q1). Default: all")
    args = parser.parse_args()

    if not RESULTS_CSV.exists():
        logger.error(f"{RESULTS_CSV} not found. Run scripts/analyze_all.py first.")
        return

    df = load_experiment_results()
    logger.info(f"Loaded {len(df)} experiment results")

    periods = {args.period: PERIODS[args.period]} if args.period else PERIODS
    all_results = []

    for name, (start, end) in periods.items():
        results = run_period(name, start, end, df)
        all_results.extend(results)

    # Save CSV
    if all_results:
        csv_path = Path("results/ensemble_results.csv")
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        fields = list(all_results[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nResults saved to {csv_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke test**

Run: `cd c:/Users/ilya/Desktop/rl/rl-agent && venv/Scripts/python.exe -m scripts.run_ensemble --period 2024`
Expected: Ensemble results printed, CSV saved.

- [ ] **Step 3: Commit**

```bash
git add scripts/run_ensemble.py
git commit -m "feat: add ensemble backtest runner with multiple strategies"
```

---

## Chunk 4: Walk-Forward Validation

### Task 4: Create walk-forward validation script

Walk-forward: train on rolling windows, test on next period.
Splits:
- Split 1: Train 2020-2021, Test 2022
- Split 2: Train 2021-2022, Test 2023
- Split 3: Train 2022-2023, Test 2024

This shows the agent can adapt to different market regimes.

**Files:**
- Create: `scripts/run_walkforward.py`
- Create: `tests/test_walkforward.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_walkforward.py`:

```python
"""Tests for walk-forward validation logic."""
import pytest
from scripts.run_walkforward import generate_splits


def test_generate_splits_default():
    splits = generate_splits(
        start_year=2020, end_year=2024,
        train_years=2, test_years=1,
    )
    assert len(splits) == 3
    assert splits[0] == {
        "train_start": "2020-01-01", "train_end": "2021-12-31",
        "test_start": "2022-01-01", "test_end": "2022-12-31",
    }
    assert splits[2] == {
        "train_start": "2022-01-01", "train_end": "2023-12-31",
        "test_start": "2024-01-01", "test_end": "2024-12-31",
    }


def test_generate_splits_single():
    splits = generate_splits(
        start_year=2022, end_year=2024,
        train_years=2, test_years=1,
    )
    assert len(splits) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd c:/Users/ilya/Desktop/rl/rl-agent && venv/Scripts/python.exe -m pytest tests/test_walkforward.py -v`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Implement walk-forward script**

Create `scripts/run_walkforward.py`:

```python
"""Walk-forward validation: rolling train/test splits.

Usage:
    python -m scripts.run_walkforward --dummy
    python -m scripts.run_walkforward --agent-type baseline --total-timesteps 500000
"""
import argparse
import csv
import logging
from pathlib import Path

import numpy as np

from src.agents.config import AgentConfig
from src.agents.train import train_agent
from src.data.load_features import load_features_for_agent
from src.eval.backtest import run_backtest
from src.eval.metrics import compute_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def generate_splits(
    start_year: int = 2020,
    end_year: int = 2024,
    train_years: int = 2,
    test_years: int = 1,
) -> list[dict]:
    """Generate rolling train/test date splits.

    Args:
        start_year: First year of data.
        end_year: Last year (inclusive for test).
        train_years: Number of years in each training window.
        test_years: Number of years in each test window.

    Returns:
        List of dicts with train_start, train_end, test_start, test_end.
    """
    splits = []
    year = start_year
    while year + train_years + test_years - 1 <= end_year:
        train_start = f"{year}-01-01"
        train_end = f"{year + train_years - 1}-12-31"
        test_start = f"{year + train_years}-01-01"
        test_end = f"{year + train_years + test_years - 1}-12-31"
        splits.append({
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
        })
        year += 1
    return splits


def run_walkforward(
    agent_type: str = "baseline",
    total_timesteps: int = 500_000,
    seed: int = 42,
    asset: str = "BTC/USDT",
    dummy: bool = False,
) -> list[dict]:
    """Run walk-forward validation.

    Returns:
        List of result dicts per split.
    """
    splits = generate_splits()
    results = []

    for i, split in enumerate(splits):
        print(f"\n{'='*60}")
        print(f"  Walk-Forward Split {i+1}/{len(splits)}")
        print(f"  Train: {split['train_start']} -> {split['train_end']}")
        print(f"  Test:  {split['test_start']} -> {split['test_end']}")
        print(f"{'='*60}")

        config = AgentConfig(
            agent_type=agent_type,
            algorithm="PPO",
            total_timesteps=total_timesteps,
            seed=seed,
            save_dir=f"experiments/walkforward/{agent_type}",
        )

        # Train
        model_path = train_agent(
            config, dummy=dummy,
            train_start=split["train_start"],
            train_end=split["train_end"],
            asset=asset,
        )

        # Test
        if dummy:
            from src.agents.train import _make_dummy_env, FEATURE_COUNTS
            rng = np.random.default_rng(seed + 1000)
            n_feat = FEATURE_COUNTS.get(agent_type, 18)
            test_feat = rng.standard_normal((150, n_feat)).astype(np.float32)
            test_prices = (100 + np.cumsum(rng.standard_normal(150) * 0.5)).astype(np.float64)
            test_prices = np.maximum(test_prices, 1.0)
        else:
            test_feat, test_prices = load_features_for_agent(
                agent_type, asset,
                split["test_start"], split["test_end"],
            )

        backtest = run_backtest(
            features=test_feat, prices=test_prices,
            model_path=str(model_path),
            window=config.window, tx_cost=config.tx_cost,
        )
        m = backtest["metrics"]

        # Buy & Hold for same period
        bh_returns = np.diff(test_prices) / test_prices[:-1]
        bh = compute_metrics(bh_returns)

        result = {
            "split": i + 1,
            "train_start": split["train_start"],
            "train_end": split["train_end"],
            "test_start": split["test_start"],
            "test_end": split["test_end"],
            "agent_type": agent_type,
            "agent_sharpe": m["sharpe_ratio"],
            "agent_return": m["total_return"],
            "agent_maxdd": m["max_drawdown"],
            "bh_sharpe": bh["sharpe_ratio"],
            "bh_return": bh["total_return"],
            "bh_maxdd": bh["max_drawdown"],
        }
        results.append(result)

        print(f"  Agent:    Sharpe={m['sharpe_ratio']:.3f}  "
              f"Return={m['total_return']*100:.1f}%  MaxDD={m['max_drawdown']*100:.1f}%")
        print(f"  Buy&Hold: Sharpe={bh['sharpe_ratio']:.3f}  "
              f"Return={bh['total_return']*100:.1f}%  MaxDD={bh['max_drawdown']*100:.1f}%")

    return results


def main():
    parser = argparse.ArgumentParser(description="Walk-forward validation")
    parser.add_argument("--agent-type", type=str, default="baseline")
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--asset", type=str, default="BTC/USDT")
    parser.add_argument("--dummy", action="store_true")
    args = parser.parse_args()

    results = run_walkforward(
        agent_type=args.agent_type,
        total_timesteps=args.total_timesteps,
        seed=args.seed,
        asset=args.asset,
        dummy=args.dummy,
    )

    # Summary
    print(f"\n{'='*70}")
    print("WALK-FORWARD SUMMARY")
    print(f"{'='*70}")
    print(f"{'Split':<8} {'Test Period':<25} {'Agent Sharpe':>13} {'B&H Sharpe':>12} {'Agent MaxDD':>13}")
    print(f"{'-'*70}")
    for r in results:
        print(f"{r['split']:<8} {r['test_start']}->{r['test_end']:<14} "
              f"{r['agent_sharpe']:>13.3f} {r['bh_sharpe']:>12.3f} "
              f"{r['agent_maxdd']*100:>12.1f}%")

    # Save
    csv_path = Path(f"results/walkforward_{args.agent_type}.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(results[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved to {csv_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests**

Run: `cd c:/Users/ilya/Desktop/rl/rl-agent && venv/Scripts/python.exe -m pytest tests/test_walkforward.py -v`
Expected: PASS

- [ ] **Step 5: Smoke test with dummy data**

Run: `cd c:/Users/ilya/Desktop/rl/rl-agent && PYTHONPATH=. venv/Scripts/python.exe -m scripts.run_walkforward --dummy --total-timesteps 1000`
Expected: 3 splits trained and tested, summary table printed.

- [ ] **Step 6: Commit**

```bash
git add scripts/run_walkforward.py tests/test_walkforward.py
git commit -m "feat: add walk-forward validation with rolling train/test splits"
```

---

## Chunk 5: Run Real Experiments

### Task 5: Execute ensemble + walk-forward on real data

- [ ] **Step 1: Run ensemble on 2024 period**

```bash
cd c:/Users/ilya/Desktop/rl/rl-agent
PYTHONPATH=. venv/Scripts/python.exe -m scripts.run_ensemble --period 2024
```

- [ ] **Step 2: Run walk-forward for baseline**

```bash
cd c:/Users/ilya/Desktop/rl/rl-agent
PYTHONPATH=. venv/Scripts/python.exe -m scripts.run_walkforward --agent-type baseline --total-timesteps 500000
```

Note: Walk-forward trains 3 new models (~15 min each at 500k steps). Total ~45 min.

- [ ] **Step 3: Run walk-forward for embeddings**

```bash
cd c:/Users/ilya/Desktop/rl/rl-agent
PYTHONPATH=. venv/Scripts/python.exe -m scripts.run_walkforward --agent-type embeddings --total-timesteps 500000
```

- [ ] **Step 4: Collect and compare all results**

Check `results/ensemble_results.csv` and `results/walkforward_*.csv`.

- [ ] **Step 5: Commit results**

```bash
git add results/ensemble_results.csv results/walkforward_*.csv
git commit -m "results: ensemble and walk-forward experiment outputs"
```

---

## Expected Outcomes

Based on literature (Yang et al. 2020, Soun et al. 2022):

| Strategy | Expected Sharpe Change | Why |
|----------|----------------------|-----|
| Top-3 Ensemble | +15-25% vs single best | Variance reduction from averaging |
| Cross-Type Ensemble | +10-20% | Diversity of feature perspectives |
| Walk-Forward | Consistent across splits | Shows adaptability, not overfitting |

**Key thesis arguments this enables:**
1. Ensemble reduces variance (lower std across seeds)
2. Walk-forward shows agents aren't overfit to one time period
3. Even if absolute return < Buy & Hold, risk-adjusted metrics (Sharpe, MaxDD) improve
4. RL's value is in **risk management**, not return maximization
