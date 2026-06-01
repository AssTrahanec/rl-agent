# Simplify Experiment (Core) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut the RL feature set from 100 → 41 columns, retrain DQN×5 behind a verification gate, then remove dead code (PPO, unused reward/action branches, never-varied knobs) and fix the Sharpe annualization — so `dsr_experiment/` is small, correct, and explainable in 30 minutes.

**Architecture:** Activate the already-written but dormant minimal feature functions (`add_technical_indicators_minimal`, `_attach_nlp_features_minimal`), set `compressed_dim: 32`, rebuild data, retrain DQN×5. **Phase 1 is a hard gate** — if the smaller features kill the OOS-2025 result, stop before touching anything else. Phases 2–4 are pure code cleanup that the trained DQN models still load and run.

**Tech Stack:** Python 3.10, Stable-Baselines3, gymnasium, pandas, pytest. Source in `dsr_experiment/`.

**Scope:** This plan covers the **core** (lib/, config, build_data, metrics, lib-facing docs). The **dashboard** is a separate plan (Plan B), written *after* Phase 1 establishes the new 41-feature schema + new model snapshot it must target. This plan **supersedes** [`2026-06-01-simplify-code.md`](2026-06-01-simplify-code.md) (whose code-only cleanup is absorbed into Phase 2).

**Design spec:** [`docs/superpowers/specs/2026-06-01-simplify-experiment-design.md`](../specs/2026-06-01-simplify-experiment-design.md).

**Conventions for all commands below:**
- Run from the repo root `C:/Users/ilya/Desktop/rl/rl-agent` unless stated. Use the Bash tool (git-bash) so `&&`, `PYTHONPATH=`, and `cd` behave as written.
- Python is the project venv: `venv/Scripts/python.exe`.
- Tests: `venv/Scripts/python.exe -m pytest tests/ ...` from repo root.
- `build_data.py` / `run.py` run from `dsr_experiment/` with `PYTHONPATH=.`.

---

## Readability bar (applies to EVERY task — this is the point)

The goal is code a thesis advisor can explain **on sight**, без замудрёностей. For every edit:
- **Delete before adding.** Prefer removing code to introducing a new helper/abstraction/branch.
- **Plain names, no cleverness.** No dense one-liners, no implicit magic. `step_returns`, not `dr`.
- **One-sentence docstring per function:** what it does + why.
- **Comment non-obvious math even if it stays** (1–2 lines, plain language): the DSR update in `env._compute_dsr` (Moody & Saffell differential Sharpe) and the sentiment-weighted embedding pooling in `_attach_nlp_features_minimal`.
- **No new config knobs.** A value that never varies becomes a named constant, not a parameter.
- **Stay in scope, but flag tangles.** If you notice a remaining "why is this here?" knot while editing a file (e.g. the CSV-merge in `run.py::oos_phase`), note it for the user — don't silently rewrite outside the task.
- A task isn't done if the diff is *correct but harder to read* than what it replaced.

---

## Testing strategy (AUTHORITATIVE — supersedes any per-task `Test:` / `pytest` line below)

Discovered at execution time: the repo-root `tests/` are **mixed**. Only **two** existing tests target `dsr_experiment` (they self-inject the folder onto `sys.path` and `import from lib`): `tests/test_dsr_metrics.py` and `tests/test_interpretability.py`. **Every other `tests/*` imports `from src…` and tests the OLD `src/` pipeline — irrelevant to this plan.** `tests/test_walkforward.py` (src) is **pre-broken** (imports a non-existent `scripts.run_walkforward`) and breaks whole-suite collection — exclude it; fixing it is out of scope (flag to user).

**Every NEW dsr_experiment test starts with this exact header** (copied from `test_dsr_metrics.py`):
```python
import sys
from pathlib import Path
_DSR_EXP = Path(__file__).resolve().parent.parent / "dsr_experiment"
if str(_DSR_EXP) not in sys.path:
    sys.path.insert(0, str(_DSR_EXP))
from lib.something import ...  # noqa: E402
```

**The dsr_experiment test set (run THIS, never the whole suite):**

| File | Status | Covers |
|---|---|---|
| `tests/test_dsr_metrics.py` | exists | lib.metrics; **Task 10 adds** the √2190 annualization test here (do **not** create `tests/test_metrics.py` — that name is a src test) |
| `tests/test_interpretability.py` | exists | lib.interpretability; **Task 9** removes the deleted-fn tests **and the top-level import of them** |
| `tests/test_minimal_features.py` | NEW (Task 1/2) | 8-indicator set + built 41-col schema |
| `tests/test_dsr_smoke.py` | NEW (Task 4) | end-to-end: TradingEnv → DQN train (~300 steps, synthetic) → backtest → metrics. **Regression guard** kept green through Phases 2–3 |
| `tests/test_dsr_config.py` | NEW (Task 7) | `load_config` → `algos==['DQN']`, `compressed_dim==32`, no `reward_type`/`agent_ppo` |

**Canonical run command (the dsr_experiment subset):**
```
venv/Scripts/python.exe -m pytest tests/test_dsr_metrics.py tests/test_interpretability.py tests/test_minimal_features.py tests/test_dsr_smoke.py tests/test_dsr_config.py -q -p no:cacheprovider
```

**Per-task verification (replaces wrong `Test:` references):** Task 1/2 → `test_minimal_features.py`; Task 4 → **create** `test_dsr_smoke.py` (TDD) + env import smoke; Task 5 → smoke stays green + `assert 'PPO' not in ALGO_MAP`; Task 6 → smoke green + `run.py --help` lists only SAC/DQN; Task 7 → **create** `test_dsr_config.py`; Task 8 → `import build_data` smoke; Task 9 → `test_interpretability.py`; Task 10 → add to `test_dsr_metrics.py`.

---

## Expected minimal feature schema (the contract for the whole plan)

After Phase 1, the model's feature matrix (what `lib/data_loader.py` returns, i.e. all columns except `open/high/low/close/volume/raw_close`) must be **exactly these 41 columns**:

```
# 8 technical indicators (add_technical_indicators_minimal)
ema_26, macd, rsi_14, bb_width, atr_14, obv, stoch_k, return_1d
# 1 sentiment scalar
sentiment_mean
# 32 PCA embedding dims
emb_0, emb_1, ..., emb_31
```

Observation dim = `window * n_features + 1 = 30 * 41 + 1 = 1231`.

---

## Phase 0 — Baseline

### Task 0: Record the pre-change baseline

**Files:** none modified (recording only).

- [ ] **Step 1: Run the dsr_experiment test subset, record the pass count**

Run: `venv/Scripts/python.exe -m pytest tests/test_dsr_metrics.py tests/test_interpretability.py -q -p no:cacheprovider 2>&1 | tail -15`
Expected: all pass. **Write down the number.** (The full `tests/` suite is mostly old-`src/` tests + a pre-broken `test_walkforward.py`; it is out of scope — see Testing strategy.)

- [ ] **Step 2: Snapshot the current defended OOS metrics**

Run: `cp dsr_experiment/results/oos_oos_2024.csv /tmp/baseline_oos_2024.csv 2>/dev/null; cp dsr_experiment/results/oos_oos_2025.csv /tmp/baseline_oos_2025.csv 2>/dev/null; echo done`
Expected: `done`. These are the √365-era numbers we compare the retrain against (Task 3 gate). If the files don't exist, note the defended figures from memory instead: DQN OOS-2025 Sharpe 0.79±0.22, BH 0.50, BH MaxDD 30.6%.

- [ ] **Step 3: Commit a marker (optional, keeps a clean point to roll back to)**

```bash
git -C C:/Users/ilya/Desktop/rl/rl-agent commit --allow-empty -m "chore: baseline before experiment simplification"
```

---

## Phase 1 — Minimal features + retrain gate

### Task 1: Wire in the minimal feature pipeline + schema test

**Files:**
- Modify: `dsr_experiment/build_data.py` (`_build_baseline`, `build_train`, `build_oos`)
- Modify: `dsr_experiment/config.yaml` (`embeddings.compressed_dim`)
- Create: `tests/test_minimal_features.py`

- [ ] **Step 1: Write the failing unit test for the 8-indicator minimal set**

Create `tests/test_minimal_features.py`:

```python
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_DSR_EXP = Path(__file__).resolve().parent.parent / "dsr_experiment"
if str(_DSR_EXP) not in sys.path:
    sys.path.insert(0, str(_DSR_EXP))

from lib.features.price import add_technical_indicators_minimal  # noqa: E402

EXPECTED_INDICATORS = [
    "ema_26", "macd", "rsi_14", "bb_width", "atr_14", "obv", "stoch_k", "return_1d",
]


def _toy_ohlcv(n=60):
    idx = pd.date_range("2020-01-01", periods=n, freq="4h", tz="UTC")
    rng = np.random.RandomState(0)
    close = 100 + np.cumsum(rng.randn(n))
    return pd.DataFrame(
        {"open": close, "high": close + 1, "low": close - 1, "close": close,
         "volume": rng.rand(n) * 10},
        index=idx,
    )


def test_minimal_indicators_are_exactly_eight():
    df = add_technical_indicators_minimal(_toy_ohlcv())
    added = [c for c in df.columns if c not in ("open", "high", "low", "close", "volume")]
    assert sorted(added) == sorted(EXPECTED_INDICATORS), added
```

- [ ] **Step 2: Run it — expect PASS already (function exists)**

Run: `venv/Scripts/python.exe -m pytest tests/test_minimal_features.py -v 2>&1 | tail -15`
Expected: PASS. (This pins the contract; the function already exists. If it FAILS, the indicator set drifted — reconcile `EXPECTED_INDICATORS` with `add_technical_indicators_minimal` before continuing.)

- [ ] **Step 3: Point `_build_baseline` at the minimal indicators**

In `dsr_experiment/build_data.py`, change the import on line 16:

```python
from lib.features.price import fetch_ohlcv, add_technical_indicators, rolling_zscore_normalize
```
to:
```python
from lib.features.price import (
    fetch_ohlcv, add_technical_indicators_minimal, rolling_zscore_normalize,
)
```

Then in `_build_baseline` (around line 76–81) change:
```python
    df = add_technical_indicators(ohlcv_slice)
```
to:
```python
    df = add_technical_indicators_minimal(ohlcv_slice)
```

- [ ] **Step 4: Point `build_train` / `build_oos` at the minimal NLP attach**

In `dsr_experiment/build_data.py`, line 243 (inside `build_train`):
```python
    features = _attach_nlp_features(baseline, news_train, cfg, compressor)
```
to:
```python
    features = _attach_nlp_features_minimal(baseline, news_train, cfg, compressor)
```

And line 265 (inside `build_oos`):
```python
    features = _attach_nlp_features(baseline, news_p, cfg, compressor)
```
to:
```python
    features = _attach_nlp_features_minimal(baseline, news_p, cfg, compressor)
```

- [ ] **Step 5: Delete the now-unused full NLP attach + its lag/rolling import**

In `dsr_experiment/build_data.py`:
- Delete the entire `_attach_nlp_features` function (lines ~84–142). `build_data` is its only caller (just rewired), so this is safe.
- Delete the now-unused import on line 20: `from lib.features.lag import add_lag_features, add_rolling_features`.

Do **not** delete `add_technical_indicators` (full) yet — the dashboard still imports it; it's removed in Plan B.

- [ ] **Step 6: Set the embedding dim to 32**

In `dsr_experiment/config.yaml`, change `embeddings.compressed_dim`:
```yaml
  compressed_dim: 64
```
to:
```yaml
  compressed_dim: 32
```

- [ ] **Step 7: Verify build_data still imports cleanly**

Run: `cd dsr_experiment && PYTHONPATH=. ../venv/Scripts/python.exe -c "import build_data; print('import OK')" && cd ..`
Expected: `import OK` (no NameError from the deleted function / import).

- [ ] **Step 8: Commit**

```bash
git -C C:/Users/ilya/Desktop/rl/rl-agent add dsr_experiment/build_data.py dsr_experiment/config.yaml tests/test_minimal_features.py
git -C C:/Users/ilya/Desktop/rl/rl-agent commit -m "feat(features): activate minimal feature pipeline (41 cols), compressed_dim 32

Wire build_data to add_technical_indicators_minimal (8 ind) and
_attach_nlp_features_minimal (sentiment_mean + 32 PCA emb, no lags/rolling).
Delete the full _attach_nlp_features. Observation 3001 -> 1231."
```

---

### Task 2: Rebuild train + OOS datasets and assert the schema

**Files:** none modified (`data/` is gitignored). Produces `dsr_experiment/data/train/features.parquet` + `dsr_experiment/data/oos/*`.

- [ ] **Step 1: Rebuild all datasets with the minimal pipeline**

Run: `cd dsr_experiment && PYTHONPATH=. ../venv/Scripts/python.exe build_data.py --build all 2>&1 | tail -20 && cd ..`
Expected: logs ending with `Train features saved: (NNNN, 47) -> ...` and one OOS line per period. (47 = 41 feature cols + open/high/low/close/volume + raw_close.) This step is slow (recomputes embeddings + refits PCA to 32).

- [ ] **Step 2: Assert the exact 41-column feature schema on the built parquet**

Append this integration test to `tests/test_minimal_features.py`:

```python
import os
import pytest

_TRAIN_PARQUET = os.path.join(
    os.path.dirname(__file__), "..", "dsr_experiment", "data", "train", "features.parquet"
)

EXPECTED_FEATURE_COLUMNS = (
    ["ema_26", "macd", "rsi_14", "bb_width", "atr_14", "obv", "stoch_k", "return_1d"]
    + ["sentiment_mean"]
    + [f"emb_{i}" for i in range(32)]
)
_PRICE_COLS = {"open", "high", "low", "close", "volume", "raw_close"}


@pytest.mark.skipif(not os.path.exists(_TRAIN_PARQUET), reason="run build_data --build train first")
def test_built_train_features_match_minimal_schema():
    df = pd.read_parquet(_TRAIN_PARQUET)
    feats = [c for c in df.columns if c.lower() not in _PRICE_COLS]
    assert sorted(feats) == sorted(EXPECTED_FEATURE_COLUMNS), feats
    assert len(feats) == 41
```

- [ ] **Step 3: Run the schema test**

Run: `venv/Scripts/python.exe -m pytest tests/test_minimal_features.py -v 2>&1 | tail -20`
Expected: both tests PASS (the skip is gone now that the parquet exists).

- [ ] **Step 4: Commit the test (data itself is gitignored)**

```bash
git -C C:/Users/ilya/Desktop/rl/rl-agent add tests/test_minimal_features.py
git -C C:/Users/ilya/Desktop/rl/rl-agent commit -m "test(features): assert built train parquet has exactly the 41 minimal feature cols"
```

---

### Task 3: 🚪 GATE — retrain DQN×5 and verify the result holds

**Files:** none modified. Produces models under `dsr_experiment/models/DQN/` and `dsr_experiment/results/oos_*.csv`.

> This is the decision point. If the result does **not** hold, STOP and report — do not start Phase 2. Fallback options are in the spec (§7): bump emb 32→64 keeping no-lags, re-add `news_count`, or revert to a code-only simplification.

- [ ] **Step 1: Retrain DQN on 5 seeds + run OOS 2024 + 2025**

Run: `cd dsr_experiment && PYTHONPATH=. ../venv/Scripts/python.exe run.py --algo DQN --seeds 42 123 7 11 99 2>&1 | tail -40 && cd ..`
Expected: training logs for 5 seeds, then two `=== oos_2024 ...` / `=== oos_2025 ...` summaries printing `DQN: Sharpe ... | Return ... | MaxDD ...`.

- [ ] **Step 2: Print the gate comparison**

Run:
```bash
venv/Scripts/python.exe - <<'PY'
import pandas as pd
df = pd.read_csv("dsr_experiment/results/oos_oos_2025.csv")
d = df[df.algorithm == "DQN"]
print("OOS 2025 DQN  n=", len(d))
print("  Sharpe mean ", round(d.sharpe_ratio.mean(), 3), "± std", round(d.sharpe_ratio.std(), 3))
print("  MaxDD  mean ", round(d.max_drawdown.mean(), 3))
print("Gate (vs BH sqrt365: Sharpe 0.50, MaxDD 0.306):")
print("  Sharpe > 0.50 ?", d.sharpe_ratio.mean() > 0.50)
print("  MaxDD  < 0.306?", d.max_drawdown.mean() < 0.306)
PY
```
Expected: a `Sharpe mean` clearly above 0.50 and `MaxDD mean` clearly below 0.306. (Metrics are still in √365 units here — that's fine, the comparison vs BH 0.50 is unit-consistent.)

- [ ] **Step 3: Record the decision**

- [ ] If **both** gate conditions hold → proceed to Phase 2. Update memory `vkr_state.md` "Защищаемые метрики" with the new DQN×5 numbers (note: 5 seeds, √365 units pre-fix).
- [ ] If they do **not** hold → STOP. Report the numbers and the fallback options from spec §7. Do not start Phase 2.

- [ ] **Step 4: Commit a marker**

```bash
git -C C:/Users/ilya/Desktop/rl/rl-agent commit --allow-empty -m "chore: phase-1 gate passed — DQN x5 on minimal features holds vs Buy & Hold"
```

---

## Phase 2 — Remove dead code (only after the gate passes)

### Task 4: Simplify `TradingEnv` to DSR-only, discrete-or-continuous

**Files:**
- Modify: `dsr_experiment/lib/env.py` (replace whole file)
- Test: `tests/test_trading_env.py`, `tests/test_dsr_metrics.py`

- [ ] **Step 1: Find env tests that use the soon-removed kwargs**

Run: `venv/Scripts/python.exe -m pytest tests/test_trading_env.py tests/test_dsr_metrics.py -q 2>&1 | tail -10`
Expected: current pass count (record it). Then:
Run: `grep -rn "reward_type\|allow_short\|volatility_penalty\|dsr_eta" tests/ 2>&1 | head -20`
Expected: a list of tests to update in Step 3 if they pass these kwargs.

- [ ] **Step 2: Replace `dsr_experiment/lib/env.py` with the simplified version**

```python
"""TradingEnv: DSR reward + sentiment bonus + transaction cost.

Action space:
    - discrete:   Discrete(3) — 0=Hold, 1=Buy all-in, 2=Sell all-out (DQN).
    - continuous: Box([0,1]) — allocation fraction (SAC).

Reward = DSR(R_t) + sentiment_lambda * sentiment_t * log_return_t,
where R_t = log_return * allocation − tx_cost * |Δallocation|.
"""
import logging

import gymnasium as gym
import numpy as np

logger = logging.getLogger(__name__)

_DSR_ETA = 0.01  # forgetting factor for the differential Sharpe ratio


class TradingEnv(gym.Env):
    """Single-asset long-only trading environment (Moody & Saffell DSR reward)."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        features: np.ndarray,
        prices: np.ndarray,
        window: int = 30,
        tx_cost: float = 0.001,
        sentiment_signal: "np.ndarray | None" = None,
        sentiment_lambda: float = 0.1,
        action_space_type: str = "discrete",
    ):
        super().__init__()
        assert len(features) == len(prices), "features and prices must have same length"
        assert len(features) > window, "Need more rows than window"
        assert action_space_type in ("continuous", "discrete"), action_space_type

        self.features = features.astype(np.float32)
        self.prices = prices.astype(np.float64)
        self.window = window
        self.tx_cost = tx_cost
        self.sentiment_signal = sentiment_signal
        self.sentiment_lambda = sentiment_lambda
        self.action_space_type = action_space_type

        n_features = features.shape[1]
        obs_dim = window * n_features + 1
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32,
        )
        if action_space_type == "discrete":
            self.action_space = gym.spaces.Discrete(3)   # 0=Hold, 1=Buy, 2=Sell
        else:
            self.action_space = gym.spaces.Box(
                low=0.0, high=1.0, shape=(1,), dtype=np.float32,
            )

        self.current_step = 0
        self.prev_allocation = 0.0
        self._dsr_A = 0.0
        self._dsr_B = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window
        self.prev_allocation = 0.0
        self._dsr_A = 0.0
        self._dsr_B = 0.0
        return self._get_obs(), {}

    def step(self, action):
        if self.action_space_type == "discrete":
            arr = np.asarray(action)
            a = int(arr.item()) if arr.ndim == 0 else int(arr.flatten()[0])
            if a == 1:
                allocation = 1.0
            elif a == 2:
                allocation = 0.0
            else:
                allocation = self.prev_allocation
        else:
            allocation = float(np.clip(action[0], 0.0, 1.0))

        price_curr = self.prices[self.current_step - 1]
        price_next = self.prices[self.current_step]
        log_return = float(np.log(price_next / price_curr))

        delta = abs(allocation - self.prev_allocation)
        R_t = log_return * allocation - self.tx_cost * delta

        reward = self._compute_dsr(R_t)
        if self.sentiment_signal is not None:
            sent = float(self.sentiment_signal[self.current_step])
            reward += self.sentiment_lambda * sent * log_return

        self.prev_allocation = allocation
        self.current_step += 1
        terminated = self.current_step >= len(self.prices) - 1
        info = {"log_return": log_return, "allocation": allocation}
        return self._get_obs(), reward, terminated, False, info

    def _compute_dsr(self, R_t: float) -> float:
        dA = R_t - self._dsr_A
        dB = R_t ** 2 - self._dsr_B
        denom = self._dsr_B - self._dsr_A ** 2
        if denom < 1e-12:
            reward = float(R_t)
        else:
            reward = float((self._dsr_B * dA - 0.5 * self._dsr_A * dB) / (denom ** 1.5))
        self._dsr_A += _DSR_ETA * dA
        self._dsr_B += _DSR_ETA * dB
        return reward

    def _get_obs(self) -> np.ndarray:
        start = self.current_step - self.window
        window_obs = self.features[start:self.current_step].flatten()
        return np.append(window_obs, self.prev_allocation).astype(np.float32)
```

- [ ] **Step 3: Update any tests that passed removed kwargs**

For each test surfaced in Step 1 that passes `reward_type=`, `allow_short=`, `volatility_penalty=`, or `dsr_eta=` to `TradingEnv`, delete that kwarg. Tests that asserted `reward_type="dsr"` behaviour now get DSR by default (it's the only path). Delete any test specifically for `reward_type="basic"`/`"risk_adjusted"` or `allow_short=True`.

- [ ] **Step 4: Run env tests**

Run: `venv/Scripts/python.exe -m pytest tests/test_trading_env.py tests/test_dsr_metrics.py -v 2>&1 | tail -25`
Expected: PASS (same count as Step 1 minus any intentionally-deleted basic/short tests).

- [ ] **Step 5: Commit**

```bash
git -C C:/Users/ilya/Desktop/rl/rl-agent add dsr_experiment/lib/env.py tests/
git -C C:/Users/ilya/Desktop/rl/rl-agent commit -m "refactor(env): DSR-only reward, drop basic/risk_adjusted/allow_short/continuous-short

env.py 137 -> ~95 lines. Hardcode dsr_eta=0.01. Long-only [0,1]."
```

---

### Task 5: Drop PPO + dead env kwargs from `train.py`

**Files:**
- Modify: `dsr_experiment/lib/train.py`
- Test: `tests/test_train.py`

- [ ] **Step 1: Remove PPO import + map**

Line 9: `from stable_baselines3 import PPO, SAC, DQN` → `from stable_baselines3 import SAC, DQN`
Line 17: `ALGO_MAP = {"PPO": PPO, "SAC": SAC, "DQN": DQN}` → `ALGO_MAP = {"SAC": SAC, "DQN": DQN}`

- [ ] **Step 2: Simplify `_build_env` to live kwargs only**

Replace `_build_env` (lines 42–56) with:
```python
def _build_env(cfg, features: np.ndarray, prices: np.ndarray, sentiment: np.ndarray) -> TradingEnv:
    env_cfg = cfg.env
    return TradingEnv(
        features=features,
        prices=prices,
        window=env_cfg.window,
        tx_cost=env_cfg.tx_cost,
        sentiment_signal=sentiment,
        sentiment_lambda=env_cfg.sentiment_lambda,
        action_space_type=getattr(env_cfg, "action_space_type", "discrete"),
    )
```

- [ ] **Step 3: Delete the PPO branch in `_algo_kwargs`**

In `_algo_kwargs` (lines 59–97) delete the entire `if algo == "PPO": return dict(...)` block (lines 60–71).

- [ ] **Step 4: Delete PPO branches in `train_agent`**

In `train_agent`, change the agent-cfg selection (lines 111–116):
```python
    if algo == "SAC":
        agent_cfg = cfg.agent_sac
    elif algo == "DQN":
        agent_cfg = cfg.agent_dqn
    else:
        agent_cfg = cfg.agent_ppo
```
to:
```python
    agent_cfg = cfg.agent_sac if algo == "SAC" else cfg.agent_dqn
```
And delete the VecNormalize-for-PPO lines 120–121:
```python
    if algo == "PPO":
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=5.0)
```
Then the `VecNormalize` import (line 11) and the `isinstance(vec_env, VecNormalize)` save block (lines 157–158) become dead — remove the import `, VecNormalize` and the save block (SAC/DQN never normalize). Keep `DummyVecEnv`.

- [ ] **Step 5: Run train tests**

Run: `venv/Scripts/python.exe -m pytest tests/test_train.py -v 2>&1 | tail -25`
Expected: PASS. Delete/adjust any test that explicitly trains PPO.

- [ ] **Step 6: Commit**

```bash
git -C C:/Users/ilya/Desktop/rl/rl-agent add dsr_experiment/lib/train.py tests/test_train.py
git -C C:/Users/ilya/Desktop/rl/rl-agent commit -m "refactor(train): drop PPO + VecNormalize + dead env kwargs (DQN/SAC only)"
```

---

### Task 6: Drop PPO + dead kwargs from `run.py` and `backtest.py`

**Files:**
- Modify: `dsr_experiment/run.py`
- Modify: `dsr_experiment/lib/backtest.py`
- Test: `tests/test_backtest.py` (if present)

- [ ] **Step 1: `run.py` — restrict algo choices**

Line 163: `ap.add_argument("--algo", choices=["SAC", "PPO", "DQN"], help="Train only this algo")`
→ `ap.add_argument("--algo", choices=["SAC", "DQN"], help="Train only this algo")`

- [ ] **Step 2: `run.py` — drop dead kwargs in the backtest call**

In `oos_phase`, change the `run_backtest(...)` call (lines 89–97) to drop `allow_short=` and pass discrete explicitly:
```python
                    res = run_backtest(
                        features=features, prices=prices,
                        model_path=str(path),
                        window=cfg.env.window,
                        tx_cost=cfg.env.tx_cost,
                        vecnorm_path=str(vecnorm) if vecnorm.exists() else None,
                        action_space_type=getattr(cfg.env, "action_space_type", "discrete"),
                    )
```
(The `vecnorm` lookup can stay — it just never finds a file now.)

- [ ] **Step 3: `backtest.py` — drop allow_short, default discrete, drop PPO from loader**

In `dsr_experiment/lib/backtest.py`:
- `load_model` (lines 14–21): `for cls in (PPO, SAC, DQN):` → `for cls in (SAC, DQN):` and the import `from stable_baselines3 import PPO, SAC, DQN` → `from stable_baselines3 import SAC, DQN`.
- `run_backtest` signature (lines 24–33): remove `allow_short: bool = False,` and change `action_space_type: str = "continuous",` → `action_space_type: str = "discrete",`.
- In the `TradingEnv(...)` construction (lines 34–38) remove `allow_short=allow_short,`.

- [ ] **Step 4: Run backtest + run-related tests**

Run: `venv/Scripts/python.exe -m pytest tests/test_backtest.py tests/test_run_ablation.py tests/test_run_algo_comparison.py -q 2>&1 | tail -20`
Expected: PASS (skip any file that doesn't exist). Update tests that referenced PPO or `allow_short`.

- [ ] **Step 5: Commit**

```bash
git -C C:/Users/ilya/Desktop/rl/rl-agent add dsr_experiment/run.py dsr_experiment/lib/backtest.py tests/
git -C C:/Users/ilya/Desktop/rl/rl-agent commit -m "refactor(run,backtest): drop PPO + allow_short, default discrete actions"
```

---

### Task 7: Clean `config.yaml` + `config_loader.py`

**Files:**
- Modify: `dsr_experiment/config.yaml`
- Modify: `dsr_experiment/lib/config_loader.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Trim `config.yaml`**

- Delete the entire `agent_ppo:` section (lines ~78–93).
- In `env:` delete `reward_type`, `allow_short`, `volatility_penalty`, `dsr_eta` (keep `window`, `tx_cost`, `sentiment_lambda`, `action_space_type`).
- In `agent_sac:` delete `use_sde: true`.
- In `news:` delete `text_col` and `date_col`.
- In `embeddings:` delete `top_pca_lags`.
- In `features:` delete `news_count_lags` and `news_count_roll` (keep `normalize_window`).
- In `experiment:` set `algos: ["DQN"]` and `seeds: [42, 123, 7, 11, 99]`.

- [ ] **Step 2: Trim `config_loader.py` dataclasses + validation**

- `NewsConfig` (lines 31–36): remove `text_col` and `date_col` fields.
- `EmbeddingsConfig` (lines 39–45): remove `top_pca_lags`.
- `FeaturesConfig` (lines 53–57): remove `news_count_lags`, `news_count_roll`.
- `EnvConfig` (lines 60–69): remove `reward_type`, `allow_short`, `volatility_penalty`, `dsr_eta`.
- `SACConfig` (lines 79–95): remove `use_sde`.
- Delete the whole `PPOConfig` dataclass (lines 98–114).
- `Config` (lines 137–149): remove the `agent_ppo: PPOConfig` field.
- `load_config` (lines 159–175): remove the `agent_ppo=PPOConfig(**raw["agent_ppo"]),` line.
- `_validate` (lines 180–198): delete the `reward_type` check (lines 188–189); in the algos loop change `{"SAC", "PPO", "DQN"}` → `{"SAC", "DQN"}`.

- [ ] **Step 3: Manual config smoke-check**

Run: `cd dsr_experiment && PYTHONPATH=. ../venv/Scripts/python.exe -c "from lib.config_loader import load_config; c=load_config('config.yaml'); print('algos', c.experiment.algos, '| emb', c.embeddings.compressed_dim, '| window', c.env.window)" && cd ..`
Expected: `algos ['DQN'] | emb 32 | window 30` with no error.

- [ ] **Step 4: Run config tests**

Run: `venv/Scripts/python.exe -m pytest tests/test_config.py -v 2>&1 | tail -20`
Expected: PASS. Update any test asserting `reward_type`, `agent_ppo`, or the removed fields.

- [ ] **Step 5: Commit**

```bash
git -C C:/Users/ilya/Desktop/rl/rl-agent add dsr_experiment/config.yaml dsr_experiment/lib/config_loader.py tests/test_config.py
git -C C:/Users/ilya/Desktop/rl/rl-agent commit -m "refactor(config): drop PPO, dead env knobs, news/emb/feature lag knobs; algos=[DQN]"
```

---

### Task 8: Hardcode never-varied knobs in `embeddings.py` + `news.py`

**Files:**
- Modify: `dsr_experiment/lib/features/embeddings.py`
- Modify: `dsr_experiment/lib/features/news.py`
- Modify: `dsr_experiment/build_data.py` (call sites)
- Test: `tests/test_embeddings.py`, `tests/test_news_collector.py`

- [ ] **Step 1: Read the two files to get exact current signatures**

Run: `grep -n "def \|model_name\|text_col\|date_col\|split" dsr_experiment/lib/features/embeddings.py dsr_experiment/lib/features/news.py 2>&1`
Expected: the signatures to edit (the spec's intent: `_get_model` drops the `model_name` param and hardcodes `FinLang/finance-embeddings-investopedia`; `load_news_from_hf` drops `text_col`/`date_col`/`split` and hardcodes `article_text`/`date_time`/`train`).

- [ ] **Step 2: Hardcode the embedding model name**

In `lib/features/embeddings.py`, replace the `_get_model(model_name=...)` parameter with a module constant `_MODEL_NAME = "FinLang/finance-embeddings-investopedia"` and make `_get_model()` take no args (load `_MODEL_NAME`). Update the singleton cache to drop the per-name tracking. Then update every `_get_model(cfg.embeddings.model_name)` call in `build_data.py` (lines 103, 183, 224) to `_get_model()`.

- [ ] **Step 3: Hardcode the news columns**

In `lib/features/news.py`, change `load_news_from_hf(dataset_name, start, end, text_col=..., date_col=..., split=...)` to `load_news_from_hf(dataset_name, start, end)` with module constants `_TEXT_COL="article_text"`, `_DATE_COL="date_time"`, `_SPLIT="train"` used internally. Update the call in `build_data.py` `ensure_raw_news` (lines 56–59) to `load_news_from_hf(cfg.news.hf_dataset, start, end)`.

- [ ] **Step 4: Run feature tests**

Run: `venv/Scripts/python.exe -m pytest tests/test_embeddings.py tests/test_news_collector.py tests/test_news_preprocessor.py -q 2>&1 | tail -20`
Expected: PASS. Update tests that pass the removed kwargs.

- [ ] **Step 5: Verify build_data still imports**

Run: `cd dsr_experiment && PYTHONPATH=. ../venv/Scripts/python.exe -c "import build_data; print('OK')" && cd ..`
Expected: `OK`.

- [ ] **Step 6: Commit**

```bash
git -C C:/Users/ilya/Desktop/rl/rl-agent add dsr_experiment/lib/features/embeddings.py dsr_experiment/lib/features/news.py dsr_experiment/build_data.py tests/
git -C C:/Users/ilya/Desktop/rl/rl-agent commit -m "refactor(features): hardcode FinLang model + article_text/date_time/train news cols"
```

---

### Task 9: Remove unused interpretability helpers

**Files:**
- Modify: `dsr_experiment/lib/interpretability.py`
- Modify: `dsr_experiment/scripts/run_permutation.py`
- Test: `tests/test_interpretability.py`

- [ ] **Step 1: Confirm the two helpers are only used by run_permutation**

Run: `grep -rn "action_feature_correlation\|action_distribution_by_sentiment_regime" dsr_experiment/ --include=*.py 2>&1`
Expected: definitions in `lib/interpretability.py` + calls in `scripts/run_permutation.py` (+ maybe tests). If anything else references them, stop and reassess.

- [ ] **Step 2: Delete the two functions + their callers**

- In `lib/interpretability.py` delete `action_feature_correlation` and `action_distribution_by_sentiment_regime` (keep `identify_feature_groups` + `permutation_importance`). Remove a now-unused `import pandas as pd` if nothing else uses it.
- In `scripts/run_permutation.py` remove the two functions from the import, delete the blocks that call them and write their CSVs.
- In `tests/test_interpretability.py` delete tests named `test_action_feature_correlation_*` and `test_action_distribution_by_sentiment_regime_*`.

- [ ] **Step 3: Run interpretability tests**

Run: `venv/Scripts/python.exe -m pytest tests/test_interpretability.py -v 2>&1 | tail -20`
Expected: remaining tests PASS.

- [ ] **Step 4: Commit**

```bash
git -C C:/Users/ilya/Desktop/rl/rl-agent add dsr_experiment/lib/interpretability.py dsr_experiment/scripts/run_permutation.py tests/test_interpretability.py
git -C C:/Users/ilya/Desktop/rl/rl-agent commit -m "refactor(interpretability): drop unused action-correlation/regime helpers"
```

---

## Phase 3 — Fix the Sharpe annualization (P1) + rename (P2)

### Task 10: Unify metrics annualization on √2190

**Files:**
- Modify: `dsr_experiment/lib/metrics.py`
- Modify: `dsr_experiment/lib/backtest.py` (rename only)
- Test: `tests/test_metrics.py` (or `tests/test_dsr_metrics.py`)

- [ ] **Step 1: Write the failing test for √2190 annualization**

Add to `tests/test_metrics.py` (create the file if absent):
```python
import numpy as np
from dsr_experiment.lib.metrics import compute_metrics, PERIODS_PER_YEAR_4H


def test_sharpe_annualized_with_4h_periods():
    # constant-ish positive returns; check the annualization factor is sqrt(2190)
    rng = np.random.RandomState(0)
    r = 0.001 + 0.0001 * rng.randn(500)
    m = compute_metrics(r)
    per_period = r.mean() / r.std(ddof=1)
    expected = per_period * np.sqrt(PERIODS_PER_YEAR_4H)
    assert abs(m["sharpe_ratio"] - expected) < 1e-6, (m["sharpe_ratio"], expected)


def test_no_trading_days_constant_leaks():
    import dsr_experiment.lib.metrics as M
    assert not hasattr(M, "TRADING_DAYS_PER_YEAR")
```

- [ ] **Step 2: Run it — expect FAIL**

Run: `venv/Scripts/python.exe -m pytest tests/test_metrics.py -v 2>&1 | tail -15`
Expected: FAIL (currently √365, and `TRADING_DAYS_PER_YEAR` still exists).

- [ ] **Step 3: Rewrite `dsr_experiment/lib/metrics.py` onto one annualization constant**

```python
import numpy as np

PERIODS_PER_YEAR_4H = 365 * 6  # 2190 four-hour bars per year (24/7 crypto)

_PROFIT_FACTOR_CAP = 1000.0


def compute_metrics(returns, allocations=None):
    """Per-4h-bar returns -> annualized performance metrics (annualization = sqrt/exp 2190)."""
    returns = np.asarray(returns, dtype=np.float64)
    n = len(returns)

    total_return = float(np.prod(1 + returns) - 1)

    mean_r = np.mean(returns) if n > 0 else 0.0
    std_r = np.std(returns, ddof=1) if n > 1 else 1.0
    sharpe = float(mean_r / std_r * np.sqrt(PERIODS_PER_YEAR_4H)) if std_r > 0 else 0.0

    downside = returns[returns < 0]
    d_std = np.std(downside, ddof=1) if len(downside) > 1 else 1.0
    sortino = float(mean_r / d_std * np.sqrt(PERIODS_PER_YEAR_4H)) if d_std > 0 else 0.0

    cumulative = np.cumprod(1 + returns) if n > 0 else np.array([1.0])
    drawdowns = 1 - cumulative / np.maximum.accumulate(cumulative)
    max_dd = float(np.max(drawdowns)) if n > 0 else 0.0

    annualized_return = float((1 + total_return) ** (PERIODS_PER_YEAR_4H / max(n, 1)) - 1) if n > 0 else 0.0
    calmar = float(annualized_return / max_dd) if max_dd > 0 else 0.0

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
```

(All 9 metric keys preserved; only the annualization is unified — `sharpe`/`sortino` now ×√2190, `calmar` derived from the single 2190-based `annualized_return`, and the duplicate 365-based `annual_return` + `TRADING_DAYS_PER_YEAR` are gone.)

- [ ] **Step 4: Run the metrics tests**

Run: `venv/Scripts/python.exe -m pytest tests/test_metrics.py -v 2>&1 | tail -15`
Expected: PASS. Update any existing metrics test whose hardcoded Sharpe/Sortino/Calmar expectation assumed √365.

- [ ] **Step 5: Rename `daily_returns` → `step_returns` in `backtest.py` (P2)**

In `dsr_experiment/lib/backtest.py`, rename the local `daily_returns` to `step_returns` everywhere it appears (lines 50, 59, 64, 72, 73, 75, 77, 79). Keep the **dict key** `"daily_returns"` in the returned dict unchanged so `run.py`'s `np.savez(daily_returns=...)` and any consumer stay working — rename only the local variable. (Or, if you also rename the key, update `run.py` line 105 and any reader; default to renaming the local only to minimize blast radius.)

Run: `venv/Scripts/python.exe -m pytest tests/test_backtest.py -q 2>&1 | tail -10`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git -C C:/Users/ilya/Desktop/rl/rl-agent add dsr_experiment/lib/metrics.py dsr_experiment/lib/backtest.py tests/test_metrics.py
git -C C:/Users/ilya/Desktop/rl/rl-agent commit -m "fix(metrics): annualize Sharpe/Sortino with sqrt(2190) (4h bars), unify annualization

Sharpe/Sortino were sqrt(365) on per-4h-bar returns -> ~2.45x too low and
inconsistent with annualized_return's 2190. Now one annualization constant.
All 9 metrics kept; relative comparisons/CIs unchanged. Rename backtest local
daily_returns -> step_returns (it is per-4h-bar)."
```

- [ ] **Step 7: Recompute OOS metrics under the fixed annualization (no retrain)**

Run: `cd dsr_experiment && PYTHONPATH=. ../venv/Scripts/python.exe run.py --algo DQN --skip-train 2>&1 | tail -20 && cd ..`
Expected: OOS summaries now show the √2190 Sharpe (DQN ~1.9 region). This re-runs only the backtest on the Task-3 models. Record the new numbers for the defense.

---

## Phase 4 — Sync lib-facing docs

### Task 11: Update `code_walkthrough.md` + `README.md`

**Files:**
- Modify: `dsr_experiment/vkr_defense/code_walkthrough.md`
- Modify: `dsr_experiment/README.md`

- [ ] **Step 1: Read both docs**

Run: `grep -n "PPO\|reward_type\|basic\|risk_adjusted\|allow_short\|64\|sentiment_max\|action_feature\|√365\|sqrt(365)\|365" dsr_experiment/vkr_defense/code_walkthrough.md dsr_experiment/README.md 2>&1 | head -40`
Expected: the lines to fix.

- [ ] **Step 2: Apply these factual edits (both files)**

Apply each concrete change:
1. Algorithms: any "SAC и PPO" / "PPO/SAC/DQN" → **"DQN (+ SAC на старых фичах)"**; remove PPO sections.
2. Reward: drop `basic`/`risk_adjusted` descriptions; keep **DSR + sentiment bonus** only. Note `dsr_eta=0.01` is now a module constant.
3. Features: "20 indicators + 64d emb + 5 sentiment + lags" → **"8 indicators + 32d emb + 1 sentiment_mean = 41 features; observation 30×41+1 = 1231"**.
4. Sentiment signal: "sentiment_max" → **"sentiment_mean"** (matches `data_loader`).
5. Interpretability: drop `action_feature_correlation` / `action_distribution_by_sentiment_regime`; keep `permutation_importance`.
6. Metrics: state annualization is **√2190 (2190 four-hour bars/year)**; update any quoted Sharpe to the √2190 convention.
7. **Add the no-leakage note (P3):** one paragraph — "Нормализация — строго трейлинг rolling z-score (окно [t−29,t], без будущего); PCA-компрессор фитится только на train и применяется к OOS. Look-ahead отсутствует."

- [ ] **Step 3: Commit**

```bash
git -C C:/Users/ilya/Desktop/rl/rl-agent add dsr_experiment/vkr_defense/code_walkthrough.md dsr_experiment/README.md
git -C C:/Users/ilya/Desktop/rl/rl-agent commit -m "docs: sync walkthrough+README to DQN-only, 41 features, sqrt(2190), no-leakage note"
```

---

## Phase 5 — Final verification

### Task 12: Whole-suite green + import smokes

**Files:** none modified.

- [ ] **Step 1: Full test suite**

Run: `venv/Scripts/python.exe -m pytest tests/ -q --ignore=tests/test_finetune_finbert.py 2>&1 | tail -15`
Expected: all pass (Task 0's count minus any intentionally-deleted PPO/basic/short tests, plus the new minimal-feature + metrics tests).

- [ ] **Step 2: Import + config + build smokes**

Run:
```bash
cd dsr_experiment && \
PYTHONPATH=. ../venv/Scripts/python.exe -c "import build_data, run; from lib.config_loader import load_config; from lib.env import TradingEnv; from lib.metrics import compute_metrics, PERIODS_PER_YEAR_4H; c=load_config('config.yaml'); print('algos', c.experiment.algos, 'emb', c.embeddings.compressed_dim, 'periods/yr', PERIODS_PER_YEAR_4H)" && cd ..
```
Expected: `algos ['DQN'] emb 32 periods/yr 2190`.

- [ ] **Step 3: Line-count summary for the defense**

Run: `wc -l dsr_experiment/lib/env.py dsr_experiment/lib/train.py dsr_experiment/lib/metrics.py dsr_experiment/lib/config_loader.py dsr_experiment/config.yaml dsr_experiment/build_data.py 2>&1`
Expected: env ~95 (was 137), config_loader smaller (was 199), config.yaml smaller (was 112), build_data smaller (was 297). Record before/after for the report.

- [ ] **Step 4: Commit summary**

```bash
git -C C:/Users/ilya/Desktop/rl/rl-agent commit --allow-empty -m "chore: core experiment simplification complete (features 100->41, DQN-only, sqrt2190 Sharpe)"
```

---

## Self-Review

**Spec coverage:**
- D1/D2/D3/D4/D5/D6 (minimal features, emb 32, DQN×5, 1 sentiment, no lags, window 30) → Tasks 1–3.
- D7 (keep all metrics) → Task 10 keeps all 9 keys.
- D8 (bootstrap unchanged) → no task touches `bootstrap.py`; it auto-benefits.
- D9 (dashboard) → **deferred to Plan B** (documented in Scope + Execution Handoff).
- D10 (SAC not retrained) → Task 3 trains DQN only.
- D11 + P1 (√2190 fix) → Task 10. P2 (rename) → Task 10 Step 5. P3 (no-leakage note) → Task 11 Step 2.7.
- Phase-2 cleanup (PPO, env branches, knobs, interpretability) → Tasks 4–9.

**Placeholder scan:** env.py and metrics.py given in full; all other edits show exact old→new blocks or exact field lists. Doc-prose task (11) is an explicit factual checklist, not "update docs". No "TBD"/"similar to Task N".

**Type/name consistency:** `TradingEnv.__init__` (Task 4) takes `features, prices, window, tx_cost, sentiment_signal, sentiment_lambda, action_space_type` — matches `_build_env` (Task 5) and `run_backtest` (Task 6). `PERIODS_PER_YEAR_4H` defined in Task 10 metrics.py is imported in Task 10 tests and Task 12 smoke. The 41-column contract (top of plan) matches `EXPECTED_FEATURE_COLUMNS` (Task 2) and `add_technical_indicators_minimal` output (Task 1).

**Known intentional residue:** full `add_technical_indicators` stays in `price.py` until Plan B migrates the dashboard (avoids breaking `features_live.py`).

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-06-01-simplify-experiment-plan.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration. The Phase-1 gate (Task 3) is a natural human checkpoint.

**2. Inline Execution** — execute tasks in this session via executing-plans, batch with checkpoints (stop at the Task-3 gate for your go/no-go).

**Note:** the **dashboard (Plan B)** will be written *after* Task 3, once the new 41-feature schema + DQN snapshot exist — so its `features_live.py` rewrite targets a real schema, not a guess.

**Which approach?**
