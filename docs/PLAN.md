# RL + NLP Trading Agent: Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Построить и сравнить 3 PPO-агента (baseline, +sentiment, +embeddings) для торговли BTC/ETH в рамках 3-way ablation study.

**Architecture:** Vertical slices — каждый слайс даёт рабочий результат от данных до оценки. Shared-backbone подход: Agent-2 и Agent-3 расширяют observation space Agent-1. Gymnasium environment + Stable-Baselines3 PPO.

**Tech Stack:** Python 3.10+, stable-baselines3, gymnasium, transformers (FinBERT), sentence-transformers, ccxt, pandas, mlflow, pytest.

---

## Slice 1: Price Data Pipeline (~3h)

**Цель:** Скачать OHLCV данные BTC/ETH, вычислить технические индикаторы, нормализовать, сохранить.

### Task 1.1: Сбор OHLCV данных

| | |
|---|---|
| **Файл** | `src/data/price_collector.py` |
| **Что делает** | Скачивает daily OHLCV с Binance через ccxt (yfinance — резерв) |
| **Input** | Тикер (BTC/USDT, ETH/USDT), период (2020-01-01 — 2024-12-31) |
| **Output** | `data/raw/btc_ohlcv.parquet`, `data/raw/eth_ohlcv.parquet` |
| **Тест** | `tests/test_price_collector.py` — проверка колонок, типов, дат, отсутствия NaN |

**Step 1: Write failing test**

```python
# tests/test_price_collector.py
import pandas as pd
from src.data.price_collector import fetch_ohlcv

def test_fetch_ohlcv_returns_dataframe():
    df = fetch_ohlcv("BTC/USDT", "2024-01-01", "2024-01-31")
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) >= {"open", "high", "low", "close", "volume"}
    assert len(df) > 0
    assert df.index.is_monotonic_increasing

def test_fetch_ohlcv_no_nulls():
    df = fetch_ohlcv("BTC/USDT", "2024-01-01", "2024-01-31")
    assert df.isnull().sum().sum() == 0
```

**Step 2:** Run `pytest tests/test_price_collector.py -v` — expect FAIL.

**Step 3: Implement**

```python
# src/data/price_collector.py
import ccxt
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def fetch_ohlcv(symbol: str, start: str, end: str, exchange_id: str = "binance") -> pd.DataFrame:
    """Fetch daily OHLCV data from exchange via ccxt."""
    exchange = getattr(ccxt, exchange_id)()
    since = exchange.parse8601(f"{start}T00:00:00Z")
    end_ts = exchange.parse8601(f"{end}T00:00:00Z")

    all_data = []
    while since < end_ts:
        ohlcv = exchange.fetch_ohlcv(symbol, "1d", since=since, limit=500)
        if not ohlcv:
            break
        all_data.extend(ohlcv)
        since = ohlcv[-1][0] + 86400000  # next day

    df = pd.DataFrame(all_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index()
    df = df[df.index <= pd.Timestamp(end, tz="UTC")]
    df = df[~df.index.duplicated(keep="first")]
    return df

def save_prices(symbol: str, start: str, end: str, output_path: str):
    df = fetch_ohlcv(symbol, start, end)
    df.to_parquet(output_path)
    logger.info(f"Saved {len(df)} rows to {output_path}")
```

**Step 4:** Run `pytest tests/test_price_collector.py -v` — expect PASS.

**Step 5:** Commit: `feat(data): add OHLCV price collector with ccxt`

### Task 1.2: Технические индикаторы

| | |
|---|---|
| **Файл** | `src/features/technical.py` |
| **Что делает** | Вычисляет SMA, EMA, RSI, MACD, Bollinger, ATR, OBV, Stochastic из OHLCV |
| **Input** | DataFrame с OHLCV колонками |
| **Output** | DataFrame с ~20 дополнительными колонками индикаторов |
| **Тест** | `tests/test_technical.py` — проверка наличия колонок, диапазонов значений |

**Step 1: Write failing test**

```python
# tests/test_technical.py
import pandas as pd
import numpy as np
from src.features.technical import add_technical_indicators

def make_dummy_ohlcv(n=100):
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(n))
    return pd.DataFrame({
        "open": close + np.random.randn(n) * 0.5,
        "high": close + abs(np.random.randn(n)),
        "low": close - abs(np.random.randn(n)),
        "close": close,
        "volume": np.random.randint(1000, 10000, n).astype(float),
    })

def test_indicators_added():
    df = make_dummy_ohlcv()
    result = add_technical_indicators(df)
    expected_cols = ["sma_7", "sma_25", "rsi_14", "macd", "bb_upper", "atr_14", "obv"]
    for col in expected_cols:
        assert col in result.columns, f"Missing {col}"

def test_rsi_range():
    df = make_dummy_ohlcv(200)
    result = add_technical_indicators(df).dropna()
    assert result["rsi_14"].between(0, 100).all()
```

**Step 2:** Run test — expect FAIL.

**Step 3:** Implement using `ta` library (или вручную через pandas).

**Step 4:** Run test — expect PASS.

**Step 5:** Commit: `feat(features): add technical indicators computation`

### Task 1.3: Нормализация (rolling z-score)

| | |
|---|---|
| **Файл** | `src/features/normalizer.py` |
| **Что делает** | Rolling z-score нормализация всех фичей (window=30) |
| **Input** | DataFrame с фичами |
| **Output** | DataFrame с нормализованными значениями, NaN заполнены 0 |
| **Тест** | `tests/test_normalizer.py` — проверка mean~0, std~1 после нормализации |

**Step 1-5:** TDD цикл аналогично.

**Step 6:** Commit: `feat(features): add rolling z-score normalization`

### Task 1.4: Pipeline скрипт для полного прогона

| | |
|---|---|
| **Файл** | `src/data/build_price_features.py` |
| **Что делает** | Оркестрирует: fetch OHLCV -> indicators -> normalize -> save |
| **Input** | CLI args или config |
| **Output** | `data/processed/btc_features.parquet`, `data/processed/eth_features.parquet` |

Commit: `feat(data): add price feature pipeline script`

---

## Slice 2: Trading Environment (~4h)

**Цель:** Gymnasium environment для торговли одним активом с continuous action space.

### Task 2.1: Базовый Trading Environment

| | |
|---|---|
| **Файл** | `src/env/trading_env.py` |
| **Что делает** | Gymnasium env: observation = window 30 дней фичей, action = [0,1] allocation, reward = log return * allocation - tx cost |
| **Input** | DataFrame с нормализованными фичами, параметры (window, tx_cost) |
| **Output** | Gymnasium-совместимый environment |
| **Тест** | `tests/test_trading_env.py` — проверка spaces, reset, step, reward calculation |

**Step 1: Write failing test**

```python
# tests/test_trading_env.py
import numpy as np
import gymnasium as gym
from src.env.trading_env import TradingEnv

def make_dummy_features(n=100, n_features=20):
    np.random.seed(42)
    return np.random.randn(n, n_features)

def test_env_creation():
    features = make_dummy_features()
    prices = 100 + np.cumsum(np.random.randn(100))
    env = TradingEnv(features=features, prices=prices, window=30)
    assert isinstance(env.observation_space, gym.spaces.Box)
    assert isinstance(env.action_space, gym.spaces.Box)

def test_env_reset():
    features = make_dummy_features()
    prices = 100 + np.cumsum(np.random.randn(100))
    env = TradingEnv(features=features, prices=prices, window=30)
    obs, info = env.reset()
    assert obs.shape == (30, 20)

def test_env_step():
    features = make_dummy_features()
    prices = 100 + np.cumsum(np.random.randn(100))
    env = TradingEnv(features=features, prices=prices, window=30)
    obs, info = env.reset()
    action = np.array([0.5])
    obs, reward, terminated, truncated, info = env.step(action)
    assert isinstance(reward, float)
    assert obs.shape == (30, 20)

def test_env_reward_calculation():
    """Reward = log(price_next/price_current) * allocation - tx_cost_if_changed."""
    features = make_dummy_features(n=35, n_features=5)
    prices = np.array([100.0] * 33 + [110.0, 110.0])  # 10% jump at step 33
    env = TradingEnv(features=features, prices=prices, window=30, tx_cost=0.001)
    env.reset()
    action = np.array([1.0])  # full allocation
    _, reward, _, _, _ = env.step(action)
    expected_log_return = np.log(110.0 / 100.0)
    # First step: tx_cost applied (position changed from 0 to 1)
    assert abs(reward - (expected_log_return * 1.0 - 0.001)) < 1e-6
```

**Step 2:** Run test — expect FAIL.

**Step 3: Implement**

```python
# src/env/trading_env.py
import gymnasium as gym
import numpy as np
import logging

logger = logging.getLogger(__name__)

class TradingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, features: np.ndarray, prices: np.ndarray,
                 window: int = 30, tx_cost: float = 0.001):
        super().__init__()
        self.features = features
        self.prices = prices
        self.window = window
        self.tx_cost = tx_cost

        n_features = features.shape[1]
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(window, n_features), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

        self.current_step = None
        self.prev_allocation = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window
        self.prev_allocation = 0.0
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        allocation = float(np.clip(action[0], 0.0, 1.0))

        price_current = self.prices[self.current_step - 1]
        price_next = self.prices[self.current_step]
        log_return = np.log(price_next / price_current)

        tx_penalty = self.tx_cost * abs(allocation - self.prev_allocation)
        reward = float(log_return * allocation - tx_penalty)

        self.prev_allocation = allocation
        self.current_step += 1

        terminated = self.current_step >= len(self.prices) - 1
        obs = self._get_obs()

        info = {"portfolio_return": log_return * allocation, "allocation": allocation}
        return obs, reward, terminated, False, info

    def _get_obs(self):
        start = self.current_step - self.window
        end = self.current_step
        return self.features[start:end].astype(np.float32)
```

**Step 4:** Run test — expect PASS.

**Step 5:** Commit: `feat(env): add Gymnasium trading environment`

### Task 2.2: SB3 compatibility wrapper

| | |
|---|---|
| **Файл** | `src/env/trading_env.py` (дополнение) |
| **Что делает** | Flatten observation для MlpPolicy SB3 (30×N → 30*N вектор) |
| **Input** | TradingEnv |
| **Output** | Env совместимый с `sb3.common.env_checker.check_env` |
| **Тест** | `tests/test_trading_env.py` — `check_env(env)` не бросает exception |

Commit: `feat(env): add SB3 compatibility (flat observation)`

---

## Slice 3: Agent-1 Baseline — Train & Evaluate (~4h)

**Цель:** Обучить PPO на ценовых данных, получить первые метрики, визуализации.

### Task 3.1: Конфигурация агентов

| | |
|---|---|
| **Файл** | `src/agents/config.py` |
| **Что делает** | Dataclass/dict с гиперпараметрами PPO, путями, настройками |
| **Input** | — |
| **Output** | Конфиг объект с learning_rate, n_steps, batch_size, net_arch и т.д. |
| **Тест** | `tests/test_config.py` — проверка значений по умолчанию |

Commit: `feat(agents): add PPO config dataclass`

### Task 3.2: Training script

| | |
|---|---|
| **Файл** | `src/agents/train.py` |
| **Что делает** | Загружает данные, создаёт env, тренирует PPO, сохраняет модель, логирует в mlflow |
| **Input** | Config (asset, agent_type, seed) |
| **Output** | Сохранённая модель в `experiments/{agent_name}/{timestamp}/model.zip` |
| **Тест** | `tests/test_train.py` — smoke test: 1000 steps на dummy данных, модель сохраняется |

**Step 1: Write failing test**

```python
# tests/test_train.py
import tempfile
from src.agents.train import train_agent
from src.agents.config import AgentConfig

def test_train_smoke():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = AgentConfig(
            total_timesteps=1000,
            save_dir=tmpdir,
            seed=42,
        )
        model_path = train_agent(config, dummy=True)
        assert model_path.exists()
```

**Step 2-5:** TDD цикл.

Commit: `feat(agents): add PPO training script with mlflow logging`

### Task 3.3: Evaluation metrics

| | |
|---|---|
| **Файл** | `src/eval/metrics.py` |
| **Что делает** | Вычисляет Sharpe, Sortino, Max Drawdown, Calmar, Total Return |
| **Input** | Массив daily returns |
| **Output** | Dict с метриками |
| **Тест** | `tests/test_metrics.py` — проверка на известных данных |

**Step 1: Write failing test**

```python
# tests/test_metrics.py
import numpy as np
from src.eval.metrics import compute_metrics

def test_sharpe_ratio():
    returns = np.array([0.01, 0.02, -0.005, 0.015, 0.01])
    metrics = compute_metrics(returns)
    assert "sharpe_ratio" in metrics
    assert metrics["sharpe_ratio"] > 0  # positive returns -> positive sharpe

def test_max_drawdown():
    returns = np.array([0.1, -0.2, 0.05, -0.1, 0.03])
    metrics = compute_metrics(returns)
    assert 0 <= metrics["max_drawdown"] <= 1

def test_total_return():
    returns = np.array([0.1, 0.1])  # (1.1 * 1.1 - 1) = 0.21
    metrics = compute_metrics(returns)
    assert abs(metrics["total_return"] - 0.21) < 1e-6
```

**Step 2-5:** TDD цикл.

Commit: `feat(eval): add financial metrics (Sharpe, Sortino, MDD, Calmar)`

### Task 3.4: Backtesting & visualization

| | |
|---|---|
| **Файл** | `src/eval/backtest.py` |
| **Что делает** | Прогоняет обученную модель на test данных, строит equity curve |
| **Input** | Обученная модель (path), test features, test prices |
| **Output** | Dict метрик + equity curve plot (сохраняется в results/) |
| **Тест** | `tests/test_backtest.py` — smoke test на dummy модели |

Commit: `feat(eval): add backtesting with equity curve plots`

### Task 3.5: Bootstrap confidence intervals

| | |
|---|---|
| **Файл** | `src/eval/bootstrap.py` |
| **Что делает** | Bootstrap CI (95%) для Sharpe и Total Return по N seeds |
| **Input** | List of return arrays (по seeds) |
| **Output** | Dict: mean, std, ci_lower, ci_upper для каждой метрики |
| **Тест** | `tests/test_bootstrap.py` — CI содержит true mean на синтетических данных |

Commit: `feat(eval): add bootstrap confidence intervals`

---

## Slice 4: News Data Pipeline (~3h)

**Цель:** Загрузить, очистить и подготовить новостные данные для NLP.

### Task 4.1: Загрузка новостных датасетов

| | |
|---|---|
| **Файл** | `src/data/news_collector.py` |
| **Что делает** | Загружает `edaschau/bitcoin_news` с HuggingFace, опционально Kaggle sentiment dataset |
| **Input** | Название датасета, период фильтрации |
| **Output** | `data/raw/bitcoin_news.parquet` |
| **Тест** | `tests/test_news_collector.py` — проверка колонок (title, text, date), фильтрации по дате |

Commit: `feat(data): add news dataset loader from HuggingFace`

### Task 4.2: Препроцессинг новостей

| | |
|---|---|
| **Файл** | `src/data/news_preprocessor.py` |
| **Что делает** | Дедупликация, фильтрация по дате, группировка по торговым дням (UTC) |
| **Input** | Raw news DataFrame |
| **Output** | `data/processed/news_by_day.parquet` — одна строка на день, список текстов |
| **Тест** | `tests/test_news_preprocessor.py` — нет дубликатов, даты в диапазоне, группировка корректна |

Commit: `feat(data): add news preprocessing and daily grouping`

---

## Slice 5: NLP Pipeline + Agent-2 (+Sentiment) (~4h)

**Цель:** FinBERT sentiment scoring, расширение environment, обучение Agent-2.

### Task 5.1: FinBERT sentiment scoring

| | |
|---|---|
| **Файл** | `src/features/sentiment.py` |
| **Что делает** | Прогоняет тексты через ProsusAI/finbert, возвращает score [-1, +1] |
| **Input** | List[str] текстов новостей |
| **Output** | float sentiment score (mean по текстам) |
| **Тест** | `tests/test_sentiment.py` — positive text -> score > 0, negative -> score < 0 |

**Step 1: Write failing test**

```python
# tests/test_sentiment.py
from src.features.sentiment import compute_sentiment

def test_positive_sentiment():
    texts = ["Bitcoin surges to all-time high, investors celebrate massive gains"]
    score = compute_sentiment(texts)
    assert score > 0.0

def test_negative_sentiment():
    texts = ["Crypto market crashes, billions wiped out in massive sell-off"]
    score = compute_sentiment(texts)
    assert score < 0.0

def test_empty_texts_fallback():
    score = compute_sentiment([])
    assert score == 0.0
```

**Step 2-5:** TDD цикл.

Commit: `feat(features): add FinBERT sentiment scoring`

### Task 5.2: Daily sentiment features

| | |
|---|---|
| **Файл** | `src/features/build_sentiment_features.py` |
| **Что делает** | Для каждого торгового дня: compute sentiment, merge с price features |
| **Input** | `data/processed/news_by_day.parquet`, price features |
| **Output** | `data/processed/btc_features_sentiment.parquet` (price features + sentiment column) |
| **Тест** | `tests/test_build_sentiment.py` — sentiment колонка существует, fallback = 0 для дней без новостей |

Commit: `feat(features): add daily sentiment feature pipeline`

### Task 5.3: Agent-2 training & evaluation

| | |
|---|---|
| **Файл** | `src/agents/train.py` (расширение) |
| **Что делает** | Тренирует PPO с sentiment features в observation, оценивает |
| **Input** | Config с agent_type="sentiment" |
| **Output** | Модель + метрики для Agent-2 |
| **Тест** | Smoke test — обучение 1000 steps, метрики вычисляются |

Commit: `feat(agents): add Agent-2 sentiment training config`

---

## Slice 6: NLP Pipeline + Agent-3 (+Embeddings) (~4h)

**Цель:** Sentence embeddings, Linear compression, расширение environment, обучение Agent-3.

### Task 6.1: Sentence embeddings

| | |
|---|---|
| **Файл** | `src/features/embeddings.py` |
| **Что делает** | all-MiniLM-L6-v2 → 384d embedding → mean pooling по новостям дня |
| **Input** | List[str] текстов |
| **Output** | np.ndarray shape (384,) — mean embedding |
| **Тест** | `tests/test_embeddings.py` — output shape, non-zero, empty fallback = zeros |

Commit: `feat(features): add sentence embedding computation`

### Task 6.2: Linear compression (384 -> 32)

| | |
|---|---|
| **Файл** | `src/features/embedding_compressor.py` |
| **Что делает** | Обучает Linear(384, 32) на train embeddings (PCA или autoencoder), сохраняет |
| **Input** | Матрица embeddings (N_days, 384) |
| **Output** | Compressed embeddings (N_days, 32), сохранённый compressor |
| **Тест** | `tests/test_compressor.py` — output shape (N, 32), reconstruction loss уменьшается |

Commit: `feat(features): add embedding compressor (384 -> 32)`

### Task 6.3: Daily embedding features + Agent-3

| | |
|---|---|
| **Файл** | `src/features/build_embedding_features.py` |
| **Что делает** | Для каждого дня: compute embedding → compress → merge с price features |
| **Input** | news_by_day, price features |
| **Output** | `data/processed/btc_features_embeddings.parquet` (price + 32 embedding cols) |

Commit: `feat(features): add daily embedding feature pipeline`

### Task 6.4: Agent-3 training & evaluation

Аналогично Task 5.3 но с embedding features.

Commit: `feat(agents): add Agent-3 embeddings training config`

---

## Slice 7: Ablation Study & Comparison (~3h)

**Цель:** Запустить все конфигурации (6 = 3 агента x 2 актива), собрать метрики, визуализации.

### Task 7.1: Experiment runner

| | |
|---|---|
| **Файл** | `src/agents/run_ablation.py` |
| **Что делает** | Запускает все 6 конфигураций × 3-5 seeds, логирует в mlflow |
| **Input** | Список конфигураций |
| **Output** | Все модели в experiments/, mlflow runs |
| **Тест** | Smoke test на 1 конфигурации |

Commit: `feat(agents): add ablation study runner`

### Task 7.2: Results aggregation & visualization

| | |
|---|---|
| **Файл** | `src/eval/visualize.py` |
| **Что делает** | Equity curves, bar charts (Sharpe, Return, MDD), heatmap агенты × метрики, training curves |
| **Input** | mlflow experiment data, backtest results |
| **Output** | Plots в `results/` (PNG/PDF) |
| **Тест** | `tests/test_visualize.py` — функции не бросают exceptions на dummy данных |

Commit: `feat(eval): add comparison visualizations`

### Task 7.3: Buy & Hold baseline

| | |
|---|---|
| **Файл** | `src/eval/baselines.py` |
| **Что делает** | Вычисляет Buy & Hold метрики для BTC и ETH на test периоде |
| **Input** | Test prices |
| **Output** | Dict метрик для B&H |
| **Тест** | `tests/test_baselines.py` — total return = (price_end / price_start) - 1 |

Commit: `feat(eval): add Buy & Hold baseline`

---

## Slice 8: Algorithm Comparison (PPO vs A2C vs SAC) (~3h)

**Цель:** Сравнить 3 RL алгоритма на лучшем агенте из ablation.

### Task 8.1: Multi-algorithm training

| | |
|---|---|
| **Файл** | `src/agents/train.py` (расширение) |
| **Что делает** | Поддержка A2C и SAC через config parameter |
| **Input** | Config с algorithm="A2C" или "SAC" |
| **Output** | Обученные модели |
| **Тест** | Smoke test для каждого алгоритма |

Commit: `feat(agents): add A2C and SAC support`

### Task 8.2: Algorithm comparison runner & plots

| | |
|---|---|
| **Файл** | `src/agents/run_algo_comparison.py` |
| **Что делает** | Запуск PPO/A2C/SAC × лучший агент × 3-5 seeds |
| **Input** | Best agent config из ablation |
| **Output** | Comparison plots, metrics table |

Commit: `feat(agents): add algorithm comparison experiment`

---

## Slice 9: Final Report Artifacts (~2h)

**Цель:** Собрать все результаты в финальные таблицы и визуализации.

### Task 9.1: Summary tables

| | |
|---|---|
| **Файл** | `src/eval/report.py` |
| **Что делает** | Генерирует LaTeX/markdown таблицы результатов ablation, algo comparison |
| **Input** | mlflow data |
| **Output** | `results/tables/` — markdown и LaTeX таблицы |

Commit: `feat(eval): add result summary tables`

### Task 9.2: Notebook для воспроизводимости

| | |
|---|---|
| **Файл** | `notebooks/full_pipeline.ipynb` |
| **Что делает** | End-to-end notebook: data → train → evaluate → visualize |
| **Input** | Все модули проекта |
| **Output** | Визуализации, таблицы inline |

Commit: `docs: add reproducibility notebook`

---

## (Optional) Slice 10: Fine-tune FinBERT + Agent-4 (~4h)

### Task 10.1: Fine-tune FinBERT на крипто-новостях

| | |
|---|---|
| **Файл** | `src/features/finetune_finbert.py` |
| **Что делает** | Fine-tune ProsusAI/finbert на Kaggle crypto sentiment dataset |
| **Input** | Kaggle sentiment dataset |
| **Output** | Fine-tuned model в `experiments/finbert_finetuned/` |

### Task 10.2: Agent-4 (sentiment + embeddings fusion)

| | |
|---|---|
| **Файл** | Расширение config + features pipeline |
| **Что делает** | Объединяет sentiment + 32d embeddings в observation |
| **Input** | Price features + sentiment + embeddings |
| **Output** | Модель Agent-4, метрики |

---

## Summary: Порядок и зависимости

```
Slice 1 (Price Data) ──→ Slice 2 (Environment) ──→ Slice 3 (Agent-1 Baseline)
                                                           │
Slice 4 (News Data) ──→ Slice 5 (Sentiment + Agent-2) ────┤
                    └──→ Slice 6 (Embeddings + Agent-3) ───┤
                                                           ↓
                                                   Slice 7 (Ablation)
                                                           ↓
                                                   Slice 8 (Algo Comparison)
                                                           ↓
                                                   Slice 9 (Report)
                                                           ↓
                                                   Slice 10 (Optional)
```

**Параллельные потоки:** Slice 4 можно начать параллельно со Slice 2-3.

**Оценка:** ~30 часов основная часть + ~4 часа опциональная.
