# Укрепление результатов: reward shaping + шорты + walk-forward

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Улучшить доходность RL-агента за счёт трёх изменений: (1) функция награды с штрафом за волатильность (risk-adjusted), (2) разрешение коротких позиций, (3) walk-forward валидация на реальных данных.

**Architecture:** Добавляем новый параметр `reward_type` в TradingEnv (default="basic" — обратная совместимость). Расширяем action_space до [-1, 1] через параметр `allow_short`. Модифицируем `backtest.py` для проброса новых параметров. Существующий `scripts/run_walkforward.py` запускаем на реальных данных. Новые эксперименты сохраняются в отдельные папки, не затрагивая уже обученные модели.

**Tech Stack:** stable-baselines3, gymnasium, numpy, pytest

---

## Файловая структура

| Действие | Файл | Ответственность |
|----------|------|-----------------|
| Modify | `src/env/trading_env.py` | Добавить `reward_type` и `allow_short` параметры |
| Modify | `src/agents/config.py` | Добавить `reward_type` и `allow_short` в AgentConfig |
| Modify | `src/agents/train.py` | Пробросить новые параметры в TradingEnv |
| Modify | `src/eval/backtest.py` | Пробросить `allow_short` в TradingEnv при бэктесте |
| Modify | `tests/test_trading_env.py` | Тесты для новых режимов награды и шортов |
| Create | `experiments/run_improved.py` | Скрипт запуска экспериментов с новыми настройками |
| Create | `tests/test_run_improved.py` | Smoke test для run_improved |

---

### Task 1: Reward type — risk-adjusted return

Формула: `reward = log_return * allocation - lambda * (allocation - prev_allocation)^2 - tx_penalty`

Штраф за резкие изменения позиции (`lambda=0.5`) заставляет агента торговать плавнее и снижает оборот. Это отличается от tx_cost — тот штрафует за сам факт изменения, а `lambda` штрафует за величину изменения квадратично.

Для варианта с шортами добавляется: `allow_short=True`, action_space [-1, 1]. Агент может шортить — зарабатывать на падении.

**Files:**
- Modify: `src/env/trading_env.py:18-56`
- Modify: `tests/test_trading_env.py`

- [ ] **Step 1: Написать тесты для новых режимов**

В `tests/test_trading_env.py` добавить:

```python
def test_reward_risk_adjusted():
    """Risk-adjusted reward includes quadratic penalty for position changes."""
    features = make_dummy_features(n=35, n_features=5)
    prices = np.array([100.0] * 30 + [110.0] + [110.0] * 4)
    env = TradingEnv(features=features, prices=prices, window=30, tx_cost=0.001,
                     reward_type="risk_adjusted")
    env.reset()
    action = np.array([1.0])
    _, reward, _, _, _ = env.step(action)
    log_ret = np.log(110.0 / 100.0)
    # prev_allocation=0, allocation=1, delta=1
    # reward = log_ret * 1.0 - 0.5 * 1.0^2 - 0.001 * 1.0
    expected = log_ret * 1.0 - 0.5 * 1.0 - 0.001 * 1.0
    assert abs(reward - expected) < 1e-6


def test_reward_risk_adjusted_no_change():
    """No position change -> no quadratic penalty."""
    features = make_dummy_features(n=35, n_features=5)
    prices = np.array([100.0] * 30 + [110.0, 120.0] + [120.0] * 3)
    env = TradingEnv(features=features, prices=prices, window=30, tx_cost=0.001,
                     reward_type="risk_adjusted")
    env.reset()
    # First step: alloc=0.5 from 0 -> delta=0.5
    env.step(np.array([0.5]))
    # Second step: alloc=0.5, same -> delta=0
    _, reward, _, _, _ = env.step(np.array([0.5]))
    log_ret = np.log(120.0 / 110.0)
    # delta=0 -> no quad penalty, no tx penalty
    expected = log_ret * 0.5
    assert abs(reward - expected) < 1e-6


def test_reward_basic_unchanged():
    """Default reward_type='basic' produces same result as before."""
    features = make_dummy_features(n=35, n_features=5)
    prices = np.array([100.0] * 30 + [110.0] + [110.0] * 4)
    env = TradingEnv(features=features, prices=prices, window=30, tx_cost=0.001)
    env.reset()
    action = np.array([1.0])
    _, reward, _, _, _ = env.step(action)
    expected = np.log(110.0 / 100.0) * 1.0 - 0.001
    assert abs(reward - expected) < 1e-6


def test_allow_short():
    """With allow_short=True, action space is [-1, 1]."""
    features = make_dummy_features(n=35, n_features=5)
    prices = np.array([100.0] * 30 + [90.0] + [90.0] * 4)  # price drops
    env = TradingEnv(features=features, prices=prices, window=30, tx_cost=0.0,
                     allow_short=True)
    assert env.action_space.low[0] == -1.0
    assert env.action_space.high[0] == 1.0
    env.reset()
    # Short: allocation=-1, price drops -> positive return for agent
    _, reward, _, _, _ = env.step(np.array([-1.0]))
    log_ret = np.log(90.0 / 100.0)  # negative
    expected = log_ret * (-1.0)  # positive
    assert reward > 0
    assert abs(reward - expected) < 1e-6
```

- [ ] **Step 2: Запустить тесты, убедиться что они падают**

Run: `python -m pytest tests/test_trading_env.py::test_reward_risk_adjusted tests/test_trading_env.py::test_allow_short -v`
Expected: FAIL — TradingEnv не принимает reward_type/allow_short

- [ ] **Step 3: Реализовать reward_type и allow_short в TradingEnv**

В `src/env/trading_env.py` заменить `__init__` и `step`:

```python
def __init__(self, features: np.ndarray, prices: np.ndarray,
             window: int = 30, tx_cost: float = 0.001,
             reward_type: str = "basic", allow_short: bool = False,
             volatility_penalty: float = 0.5):
    super().__init__()
    assert len(features) == len(prices), "features and prices must have same length"
    assert len(features) > window, "Need more data rows than window size"
    assert reward_type in ("basic", "risk_adjusted"), f"Unknown reward_type: {reward_type}"

    self.features = features.astype(np.float32)
    self.prices = prices.astype(np.float64)
    self.window = window
    self.tx_cost = tx_cost
    self.reward_type = reward_type
    self.volatility_penalty = volatility_penalty

    n_features = features.shape[1]
    self.observation_space = gym.spaces.Box(
        low=-np.inf, high=np.inf,
        shape=(window * n_features,),
        dtype=np.float32,
    )

    low_action = -1.0 if allow_short else 0.0
    self.action_space = gym.spaces.Box(
        low=low_action, high=1.0, shape=(1,), dtype=np.float32
    )

    self.current_step: int = 0
    self.prev_allocation: float = 0.0
```

```python
def step(self, action):
    allocation = float(np.clip(action[0], self.action_space.low[0], self.action_space.high[0]))

    price_current = self.prices[self.current_step - 1]
    price_next = self.prices[self.current_step]
    log_return = float(np.log(price_next / price_current))

    delta = abs(allocation - self.prev_allocation)
    tx_penalty = self.tx_cost * delta

    if self.reward_type == "risk_adjusted":
        quad_penalty = self.volatility_penalty * delta
        reward = float(log_return * allocation - quad_penalty - tx_penalty)
    else:
        reward = float(log_return * allocation - tx_penalty)

    self.prev_allocation = allocation
    self.current_step += 1

    terminated = self.current_step >= len(self.prices) - 1
    obs = self._get_obs()
    info = {"log_return": log_return, "allocation": allocation}
    return obs, reward, terminated, False, info
```

Обновить docstring класса:

```python
"""Single-asset trading environment with continuous allocation action.

Observation: flattened window of normalized feature rows (window * n_features,)
Action:      allocation ∈ [0, 1] (or [-1, 1] if allow_short=True)
Reward:
  basic:         log_return * allocation - tx_cost * |delta|
  risk_adjusted: log_return * allocation - volatility_penalty * |delta| - tx_cost * |delta|
"""
```

- [ ] **Step 4: Запустить все тесты trading_env**

Run: `python -m pytest tests/test_trading_env.py -v`
Expected: все PASS (включая старые тесты — обратная совместимость)

- [ ] **Step 5: Коммит**

```bash
git add src/env/trading_env.py tests/test_trading_env.py
git commit -m "feat: add risk_adjusted reward_type and allow_short to TradingEnv"
```

---

### Task 2: Добавить параметры в AgentConfig и train_agent

**Files:**
- Modify: `src/agents/config.py`
- Modify: `src/agents/train.py`
- Modify: `tests/test_config.py`

- [ ] **Step 1: Написать тест**

В `tests/test_config.py` добавить:

```python
def test_config_reward_type_default():
    config = AgentConfig()
    assert config.reward_type == "basic"
    assert config.allow_short is False


def test_config_reward_type_custom():
    config = AgentConfig(reward_type="risk_adjusted", allow_short=True)
    assert config.reward_type == "risk_adjusted"
    assert config.allow_short is True
```

- [ ] **Step 2: Запустить, убедиться что падает**

Run: `python -m pytest tests/test_config.py::test_config_reward_type_default -v`
Expected: FAIL

- [ ] **Step 3: Добавить поля в AgentConfig**

В `src/agents/config.py` после строки `tx_cost: float = 0.001`:

```python
    # Reward
    reward_type: str = "basic"       # "basic" or "risk_adjusted"
    allow_short: bool = False        # allow allocation in [-1, 1]
```

- [ ] **Step 4: Пробросить параметры в train_agent**

В `src/agents/train.py`, в функции `train_agent`, изменить создание TradingEnv (строка 98-103):

```python
env = TradingEnv(
    features=features, prices=prices,
    window=config.window, tx_cost=config.tx_cost,
    reward_type=config.reward_type, allow_short=config.allow_short,
)
```

Также в `_make_dummy_env` (строка 61-64):

```python
return TradingEnv(
    features=features, prices=prices,
    window=config.window, tx_cost=config.tx_cost,
    reward_type=config.reward_type, allow_short=config.allow_short,
)
```

- [ ] **Step 5: Запустить все тесты config и train**

Run: `python -m pytest tests/test_config.py tests/test_train.py -v`
Expected: все PASS

- [ ] **Step 6: Коммит**

```bash
git add src/agents/config.py src/agents/train.py tests/test_config.py
git commit -m "feat: add reward_type and allow_short to AgentConfig and train_agent"
```

---

### Task 3: Пробросить allow_short в backtest.py

**Files:**
- Modify: `src/eval/backtest.py:28-49`
- Modify: `tests/test_backtest.py`

Без этого шага модели, обученные с `allow_short=True`, будут тестироваться в среде с `action_space=[0,1]`, и их отрицательные аллокации будут обрезаны до 0.

- [ ] **Step 1: Написать тест**

В `tests/test_backtest.py` добавить:

```python
def test_backtest_allow_short():
    """Backtest with allow_short passes through to TradingEnv."""
    from src.eval.backtest import run_backtest
    import numpy as np
    features = np.random.randn(100, 5).astype(np.float32)
    prices = (100 + np.cumsum(np.random.randn(100) * 0.5)).astype(np.float64)
    prices = np.maximum(prices, 1.0)
    result = run_backtest(features=features, prices=prices, model_path=None,
                          window=30, allow_short=True)
    assert "metrics" in result
```

- [ ] **Step 2: Запустить, убедиться что падает**

Run: `python -m pytest tests/test_backtest.py::test_backtest_allow_short -v`
Expected: FAIL — `run_backtest` не принимает `allow_short`

- [ ] **Step 3: Добавить allow_short в run_backtest**

В `src/eval/backtest.py`, изменить сигнатуру `run_backtest`:

```python
def run_backtest(
    features: np.ndarray,
    prices: np.ndarray,
    model_path: Optional[str],
    window: int = 30,
    tx_cost: float = 0.001,
    plot_path: Optional[str] = None,
    allow_short: bool = False,
) -> dict:
```

Изменить создание среды (строка 49):

```python
env = TradingEnv(features=features, prices=prices, window=window, tx_cost=tx_cost,
                 allow_short=allow_short)
```

- [ ] **Step 4: Запустить все тесты backtest**

Run: `python -m pytest tests/test_backtest.py -v`
Expected: все PASS

- [ ] **Step 5: Коммит**

```bash
git add src/eval/backtest.py tests/test_backtest.py
git commit -m "feat: add allow_short parameter to run_backtest"
```

---

### Task 4: Скрипт запуска улучшенных экспериментов

**Files:**
- Create: `experiments/run_improved.py`
- Create: `tests/test_run_improved.py`

- [ ] **Step 1: Написать smoke test**

Создать `tests/test_run_improved.py`:

```python
import subprocess
import sys


def test_run_improved_smoke():
    """Smoke test: run_improved with dummy data."""
    result = subprocess.run(
        [sys.executable, "experiments/run_improved.py",
         "--dummy", "--total-timesteps", "1000",
         "--seeds", "42"],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, f"STDERR: {result.stderr}"
    assert "DONE" in result.stdout
```

- [ ] **Step 2: Запустить, убедиться что падает**

Run: `python -m pytest tests/test_run_improved.py -v`
Expected: FAIL — файл не существует

- [ ] **Step 3: Создать experiments/run_improved.py**

```python
"""Run improved experiments: risk-adjusted reward + short positions.

Usage:
    python experiments/run_improved.py --dummy
    python experiments/run_improved.py --total-timesteps 500000 --seeds 42,43,44
"""
import argparse
import csv
import logging
from pathlib import Path

import numpy as np

from src.agents.config import AgentConfig
from src.agents.train import train_agent, FEATURE_COUNTS
from src.eval.backtest import run_backtest
from src.data.load_features import load_features_for_agent

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_CSV = Path("results/improved_metrics.csv")
CSV_FIELDS = [
    "experiment", "agent_type", "reward_type", "allow_short",
    "asset", "seed",
    "total_return", "sharpe_ratio", "sortino_ratio", "max_drawdown", "calmar_ratio",
]

EXPERIMENTS = [
    {"label": "risk_adjusted", "reward_type": "risk_adjusted", "allow_short": False},
    {"label": "risk_adjusted_short", "reward_type": "risk_adjusted", "allow_short": True},
]


def _make_dummy_data(agent_type, seed, n=150):
    rng = np.random.default_rng(seed + 2000)
    n_features = FEATURE_COUNTS.get(agent_type, 18)
    features = rng.standard_normal((n, n_features)).astype(np.float32)
    prices = (100 + np.cumsum(rng.standard_normal(n) * 0.5)).astype(np.float64)
    prices = np.maximum(prices, 1.0)
    return features, prices


def main():
    parser = argparse.ArgumentParser(description="Run improved experiments")
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--agent-type", type=str, default="baseline")
    parser.add_argument("--asset", type=str, default="BTC/USDT")
    parser.add_argument("--dummy", action="store_true")
    parser.add_argument("--train-start", type=str, default="2020-01-01")
    parser.add_argument("--train-end", type=str, default="2023-12-31")
    parser.add_argument("--test-start", type=str, default="2024-01-01")
    parser.add_argument("--test-end", type=str, default="2024-12-31")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    all_results = []

    for exp in EXPERIMENTS:
        for seed in seeds:
            print(f"\n{'='*60}", flush=True)
            print(f"  {exp['label']}  seed={seed}  agent={args.agent_type}", flush=True)
            print(f"{'='*60}", flush=True)

            config = AgentConfig(
                agent_type=args.agent_type,
                algorithm="PPO",
                total_timesteps=args.total_timesteps,
                seed=seed,
                save_dir=f"experiments/improved/{exp['label']}",
                reward_type=exp["reward_type"],
                allow_short=exp["allow_short"],
            )

            model_path = train_agent(
                config, dummy=args.dummy,
                train_start=args.train_start,
                train_end=args.train_end,
                asset=args.asset,
            )

            if args.dummy:
                test_feat, test_prices = _make_dummy_data(args.agent_type, seed)
            else:
                test_feat, test_prices = load_features_for_agent(
                    args.agent_type, args.asset,
                    args.test_start, args.test_end,
                )

            backtest = run_backtest(
                features=test_feat, prices=test_prices,
                model_path=str(model_path),
                window=config.window, tx_cost=config.tx_cost,
                allow_short=config.allow_short,
            )
            m = backtest["metrics"]

            row = {
                "experiment": exp["label"],
                "agent_type": args.agent_type,
                "reward_type": exp["reward_type"],
                "allow_short": exp["allow_short"],
                "asset": args.asset,
                "seed": seed,
                **{k: m[k] for k in ["total_return", "sharpe_ratio", "sortino_ratio",
                                      "max_drawdown", "calmar_ratio"]},
            }
            all_results.append(row)

            print(f"  DONE  Sharpe={m['sharpe_ratio']:.3f}  "
                  f"Return={m['total_return']*100:.1f}%  "
                  f"MaxDD={m['max_drawdown']*100:.1f}%", flush=True)

    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for exp in EXPERIMENTS:
        rows = [r for r in all_results if r["experiment"] == exp["label"]]
        sharpes = [r["sharpe_ratio"] for r in rows]
        returns = [r["total_return"] for r in rows]
        maxdds = [r["max_drawdown"] for r in rows]
        print(f"{exp['label']:<25} Sharpe={np.mean(sharpes):.3f}+-{np.std(sharpes):.2f}  "
              f"Return={np.mean(returns)*100:.1f}%  MaxDD={np.mean(maxdds)*100:.1f}%")
    print(f"\nSaved to {RESULTS_CSV}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Запустить smoke test**

Run: `python -m pytest tests/test_run_improved.py -v`
Expected: PASS

- [ ] **Step 5: Коммит**

```bash
git add experiments/run_improved.py tests/test_run_improved.py
git commit -m "feat: add run_improved script for risk-adjusted reward + short experiments"
```

---

### Task 5: Запуск экспериментов на реальных данных

**Files:**
- Run: `experiments/run_improved.py`
- Run: `scripts/run_walkforward.py`

- [ ] **Step 1: Запустить улучшенные эксперименты (baseline, 3 seeds)**

Run: `PYTHONPATH=. python experiments/run_improved.py --total-timesteps 500000 --seeds 42,43,44 --agent-type baseline --asset BTC/USDT`
Expected: CSV в `results/improved_metrics.csv` (~45 мин: 2 эксперимента × 3 seeds × ~7 мин)

- [ ] **Step 2: Запустить walk-forward валидацию**

Run: `PYTHONPATH=. python -m scripts.run_walkforward --agent-type baseline --total-timesteps 500000 --seed 42 --asset BTC/USDT`
Expected: CSV в `results/walkforward_baseline.csv` (~21 мин: 3 сплита × ~7 мин)

- [ ] **Step 3: Проанализировать результаты**

Прочитать `results/improved_metrics.csv` и `results/walkforward_baseline.csv`. Сравнить:
- risk_adjusted vs basic reward (по Sharpe, Return, MaxDD)
- risk_adjusted + short vs risk_adjusted only (по Sharpe, Return, MaxDD)
- walk-forward: стабилен ли Sharpe по сплитам

- [ ] **Step 4: Обновить таблицу в тезисе**

Добавить новые строки в таблицу `docs/thesis_tezis.md` с реальными результатами.

- [ ] **Step 5: Коммит**

```bash
git add results/improved_metrics.csv results/walkforward_baseline.csv docs/thesis_tezis.md
git commit -m "feat: add improved experiment results and walk-forward validation"
```

---

### Task 6 (если есть время): Ансамбль на улучшенных моделях

**Files:**
- Run: `scripts/run_ensemble.py` (уже существует)

- [ ] **Step 1: Запустить ансамбль из улучшенных моделей**

Собрать топ-3 модели из `experiments/improved/` по Sharpe, запустить ансамбль. Нужно передать `allow_short=True` если модели обучены с шортами.

- [ ] **Step 2: Сравнить с текущим ансамблем**

Текущий ансамбль (basic reward): Sharpe 2,63, MaxDD 12,2%.

- [ ] **Step 3: Обновить тезис и коммит**

---

## Что это даст для тезиса

После выполнения плана в таблице результатов будут:
1. **Базовый агент** (текущий) — Sharpe 1,80, Return +64%
2. **Агент с risk-adjusted reward** — ожидаемо ниже оборот, стабильнее equity curve
3. **Агент с risk-adjusted + шорт** — возможность зарабатывать на падении
4. **Walk-forward** — подтверждение адаптивности модели на 3 скользящих сплитах
5. **Ансамбль улучшенных моделей** — лучший общий результат

Каждый пункт — отдельный эксперимент и абзац в работе.
