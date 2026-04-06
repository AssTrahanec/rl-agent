# Максимальное улучшение Embeddings агента

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Максимизировать Sharpe embeddings агента и снизить разброс между seeds. Только embeddings — baseline/sentiment не трогаем.

**Architecture:** 6 улучшений в коде + ночной прогон обучения на 7 seeds × 2M timesteps + сравнение и документация.

**Tech Stack:** pandas, numpy, stable-baselines3, pytest

---

## Текущая ситуация

| Метрика | Baseline (3 seeds, 500K) | Embeddings v1 (3 seeds, 500K) |
|---------|--------------------------|-------------------------------|
| Sharpe | 1.358 ± 0.436 | 1.310 ± 0.520 |
| Return | 43.7% ± 18.2% | 54.3% ± 35.7% |
| MaxDD | 17.5% ± 5.3% | 24.4% ± 7.9% |
| Sortino | 1.758 ± 0.612 | 1.858 ± 0.911 |

**Проблемы:**
- Высокий разброс между seeds (std=0.520)
- 500K timesteps мало для 83 фич (obs space = 2490)
- LR константный 3e-4 — агент скачет на поздних этапах
- Сеть [256, 256] мала для 2490 входов
- Нет lag features — новости уже в цене к закрытию дня
- Mean pooling размазывает экстремальные новости среди нейтральных
- Нет информации о "фоне" новостной активности (rolling)
- Теряется "направление" вчерашних новостей (lag embeddings)

---

## Все улучшения

| # | Улучшение | Новые фичи | Зачем |
|---|-----------|-----------|-------|
| 1 | Lag news_count (t-1, t-2) | +2 | Вчерашняя/позавчерашняя новостная активность |
| 2 | Rolling news_count (7d) | +1 | Фон новостной активности за неделю |
| 3 | Lag top-3 PCA (t-1, t-2) | +6 | "Направление" вчерашних/позавчерашних новостей |
| 4 | Sentiment extremes (min, max, spread) | +3 | Одна сильная новость среди нейтральных не теряется |
| 5 | Linear LR decay (3e-4 → 0) | — | Стабильная сходимость, меньше разброс |
| 6 | Сеть [512, 256] | — | Больше capacity для расширенного feature space |

**Итого фич:** 83 (было) + 2 lag + 1 rolling + 6 PCA lag + 3 sentiment extremes = **95 фич**

---

## Task 1: Lag features — утилита

**Files:**
- Create: `src/features/lag_features.py`
- Create: `tests/test_lag_features.py`

- [ ] **Step 1: Написать тесты для lag утилиты**

Создать `tests/test_lag_features.py`:

```python
import pandas as pd
import numpy as np
from src.features.lag_features import add_lag_features, add_rolling_features


def test_lag_adds_correct_columns():
    """add_lag_features creates _lag1 and _lag2 columns."""
    idx = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
    df = pd.DataFrame({"sentiment": [0.1, 0.5, -0.3, 0.8, 0.0]}, index=idx)
    result = add_lag_features(df, columns=["sentiment"], lags=[1, 2])
    assert "sentiment_lag1" in result.columns
    assert "sentiment_lag2" in result.columns


def test_lag_values_are_shifted():
    """Lag-1 should equal previous day's value."""
    idx = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
    df = pd.DataFrame({"sentiment": [0.1, 0.5, -0.3, 0.8, 0.0]}, index=idx)
    result = add_lag_features(df, columns=["sentiment"], lags=[1, 2])
    assert result["sentiment_lag1"].iloc[2] == 0.5
    assert result["sentiment_lag2"].iloc[2] == 0.1


def test_lag_fills_nan_with_zero():
    """First rows where lag is unavailable should be filled with 0."""
    idx = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
    df = pd.DataFrame({"sentiment": [0.1, 0.5, -0.3, 0.8, 0.0]}, index=idx)
    result = add_lag_features(df, columns=["sentiment"], lags=[1, 2])
    assert result["sentiment_lag1"].iloc[0] == 0.0
    assert result["sentiment_lag2"].iloc[0] == 0.0
    assert result["sentiment_lag2"].iloc[1] == 0.0


def test_lag_preserves_original_columns():
    """Original columns should remain unchanged."""
    idx = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
    df = pd.DataFrame({
        "price": [100, 101, 102, 103, 104],
        "sentiment": [0.1, 0.5, -0.3, 0.8, 0.0],
    }, index=idx)
    result = add_lag_features(df, columns=["sentiment"], lags=[1, 2])
    pd.testing.assert_series_equal(result["price"], df["price"])
    pd.testing.assert_series_equal(result["sentiment"], df["sentiment"])


def test_lag_multiple_columns():
    """Can lag multiple columns at once."""
    idx = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
    df = pd.DataFrame({
        "sentiment": [0.1, 0.5, -0.3, 0.8, 0.0],
        "news_count": [3, 5, 0, 2, 7],
    }, index=idx)
    result = add_lag_features(df, columns=["sentiment", "news_count"], lags=[1, 2])
    assert "sentiment_lag1" in result.columns
    assert "news_count_lag1" in result.columns
    assert "news_count_lag2" in result.columns
    assert result["news_count_lag1"].iloc[2] == 5


def test_rolling_adds_column():
    """add_rolling_features creates rolling mean column."""
    idx = pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC")
    df = pd.DataFrame({"news_count": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}, index=idx)
    result = add_rolling_features(df, columns=["news_count"], window=3)
    assert "news_count_roll3" in result.columns


def test_rolling_values_correct():
    """Rolling mean should average over window."""
    idx = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
    df = pd.DataFrame({"val": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=idx)
    result = add_rolling_features(df, columns=["val"], window=3)
    # Row 3 (index 2): mean(1, 2, 3) = 2.0
    assert result["val_roll3"].iloc[2] == 2.0
    # Row 4 (index 3): mean(2, 3, 4) = 3.0
    assert result["val_roll3"].iloc[3] == 3.0


def test_rolling_fills_nan_with_zero():
    """First rows where window is incomplete should be 0."""
    idx = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
    df = pd.DataFrame({"val": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=idx)
    result = add_rolling_features(df, columns=["val"], window=3)
    assert result["val_roll3"].iloc[0] == 0.0
    assert result["val_roll3"].iloc[1] == 0.0
```

- [ ] **Step 2: Запустить тесты, убедиться что падают**

```bash
python -m pytest tests/test_lag_features.py -v
```
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Реализовать lag и rolling утилиты**

Создать `src/features/lag_features.py`:

```python
"""Add lagged and rolling versions of features to capture delayed market reactions."""
from typing import List

import pandas as pd


def add_lag_features(
    df: pd.DataFrame,
    columns: List[str],
    lags: List[int] = [1, 2],
) -> pd.DataFrame:
    """Add time-lagged copies of specified columns.

    Args:
        df: DataFrame with DatetimeIndex.
        columns: Column names to create lags for.
        lags: List of lag periods (default: [1, 2] = yesterday and day before).

    Returns:
        DataFrame with added {col}_lag{n} columns. NaN filled with 0.0.
    """
    result = df.copy()
    for col in columns:
        for lag in lags:
            result[f"{col}_lag{lag}"] = result[col].shift(lag).fillna(0.0)
    return result


def add_rolling_features(
    df: pd.DataFrame,
    columns: List[str],
    window: int = 7,
) -> pd.DataFrame:
    """Add rolling mean of specified columns.

    Args:
        df: DataFrame with DatetimeIndex.
        columns: Column names to compute rolling mean for.
        window: Rolling window size in days (default: 7).

    Returns:
        DataFrame with added {col}_roll{window} columns. NaN filled with 0.0.
    """
    result = df.copy()
    for col in columns:
        result[f"{col}_roll{window}"] = (
            result[col].rolling(window=window, min_periods=window).mean().fillna(0.0)
        )
    return result
```

- [ ] **Step 4: Запустить тесты**

```bash
python -m pytest tests/test_lag_features.py -v
```
Expected: 9 PASS

- [ ] **Step 5: Коммит**

```bash
git add src/features/lag_features.py tests/test_lag_features.py
git commit -m "feat: add lag and rolling feature utilities"
```

---

## Task 2: Sentiment extremes + lag/rolling в build_embedding_features

**Files:**
- Modify: `src/features/build_embedding_features.py`
- Modify: `tests/test_build_embedding.py`

- [ ] **Step 1: Добавить тесты**

Добавить в конец `tests/test_build_embedding.py`:

```python
def test_sentiment_extreme_columns_exist():
    """build_embedding_features adds sentiment_max, sentiment_min, sentiment_spread."""
    prices = _make_price_features()
    news = _make_news_by_day()
    comp = _mock_compressor()
    with patch("src.features.build_embedding_features.compute_embeddings") as mock_emb, \
         patch("src.features.build_embedding_features.compute_sentiment") as mock_sent:
        mock_emb.return_value = np.zeros(768, dtype=np.float32)
        mock_sent.side_effect = lambda texts: 0.5 if "surges" in texts[0] else -0.3
        result = build_embedding_features(prices, news, compressor=comp)
    assert "sentiment_max" in result.columns
    assert "sentiment_min" in result.columns
    assert "sentiment_spread" in result.columns


def test_sentiment_extremes_values():
    """sentiment_max/min capture per-article extremes, not mean."""
    prices = _make_price_features()
    # Day with 2 articles: one positive, one negative
    news = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-02"], utc=True),
        "texts": [["Great news for BTC", "Terrible crash coming"]],
    })
    comp = _mock_compressor()
    sentiment_values = iter([0.8, -0.6])
    with patch("src.features.build_embedding_features.compute_embeddings") as mock_emb, \
         patch("src.features.build_embedding_features.compute_sentiment") as mock_sent:
        mock_emb.return_value = np.zeros(768, dtype=np.float32)
        mock_sent.side_effect = lambda texts: next(sentiment_values)
        result = build_embedding_features(prices, news, compressor=comp)
    day = pd.Timestamp("2024-01-02", tz="UTC")
    assert result.loc[day, "sentiment_max"] == 0.8
    assert result.loc[day, "sentiment_min"] == -0.6
    assert abs(result.loc[day, "sentiment_spread"] - 1.4) < 1e-6


def test_pca_lag_columns_exist():
    """build_embedding_features adds emb_0_lag1..emb_2_lag2 (top-3 PCA lags)."""
    prices = _make_price_features()
    news = _make_news_by_day()
    comp = _mock_compressor()
    with patch("src.features.build_embedding_features.compute_embeddings") as mock_emb, \
         patch("src.features.build_embedding_features.compute_sentiment", return_value=0.5):
        mock_emb.return_value = np.zeros(768, dtype=np.float32)
        result = build_embedding_features(prices, news, compressor=comp)
    for i in range(3):
        assert f"emb_{i}_lag1" in result.columns
        assert f"emb_{i}_lag2" in result.columns


def test_news_count_lag_columns_exist():
    """build_embedding_features adds news_count_lag1, news_count_lag2, news_count_roll7."""
    prices = _make_price_features()
    news = _make_news_by_day()
    comp = _mock_compressor()
    with patch("src.features.build_embedding_features.compute_embeddings") as mock_emb, \
         patch("src.features.build_embedding_features.compute_sentiment", return_value=0.5):
        mock_emb.return_value = np.zeros(768, dtype=np.float32)
        result = build_embedding_features(prices, news, compressor=comp)
    assert "news_count_lag1" in result.columns
    assert "news_count_lag2" in result.columns
    assert "news_count_roll7" in result.columns


def test_news_count_lag_values_correct():
    """news_count_lag1 should be previous day's news_count."""
    prices = _make_price_features()
    news = _make_news_by_day()
    comp = _mock_compressor()
    with patch("src.features.build_embedding_features.compute_embeddings") as mock_emb, \
         patch("src.features.build_embedding_features.compute_sentiment", return_value=0.5):
        mock_emb.return_value = np.zeros(768, dtype=np.float32)
        result = build_embedding_features(prices, news, compressor=comp)
    day_after = pd.Timestamp("2024-01-03", tz="UTC")
    assert result.loc[day_after, "news_count_lag1"] == 1
```

- [ ] **Step 2: Запустить, убедиться что падают**

```bash
python -m pytest tests/test_build_embedding.py::test_sentiment_extreme_columns_exist -v
```
Expected: FAIL

- [ ] **Step 3: Переписать build_embedding_features.py**

Полная замена `src/features/build_embedding_features.py`:

```python
"""Build daily embedding features by merging compressed embeddings with price features."""
import logging

import numpy as np
import pandas as pd

from src.features.embeddings import compute_embeddings
from src.features.sentiment import compute_sentiment
from src.features.lag_features import add_lag_features, add_rolling_features

logger = logging.getLogger(__name__)

COMPRESSED_DIM = 64  # increased from 32
TOP_PCA_LAGS = 3     # lag top-3 PCA components


def build_embedding_features(
    price_features: pd.DataFrame,
    news_by_day: pd.DataFrame,
    compressor,
) -> pd.DataFrame:
    """Merge compressed daily embeddings with price features.

    Adds:
    - 64 emb_0..emb_63 columns (PCA-compressed embeddings, weighted by |sentiment|)
    - news_count, news_count_lag1, news_count_lag2, news_count_roll7
    - sentiment_max, sentiment_min, sentiment_spread (per-day extremes)
    - emb_0_lag1..emb_2_lag2 (top-3 PCA lags for "news direction" memory)

    Days without news get zeros for all NLP features.
    """
    result = price_features.copy()

    # Initialize embedding columns with zeros
    emb_cols = [f"emb_{i}" for i in range(COMPRESSED_DIM)]
    for col in emb_cols:
        result[col] = 0.0
    result["news_count"] = 0
    result["sentiment_max"] = 0.0
    result["sentiment_min"] = 0.0
    result["sentiment_spread"] = 0.0

    for _, row in news_by_day.iterrows():
        day = row["date"].normalize()
        texts = row["texts"]
        if day not in result.index:
            continue

        n = len(texts)
        result.loc[day, "news_count"] = n

        # Compute per-article sentiment scores
        sentiment_scores = [compute_sentiment([t]) for t in texts]
        result.loc[day, "sentiment_max"] = max(sentiment_scores)
        result.loc[day, "sentiment_min"] = min(sentiment_scores)
        result.loc[day, "sentiment_spread"] = max(sentiment_scores) - min(sentiment_scores)

        # Weighted pooling: |sentiment| so extreme articles weigh more
        weights = [abs(s) + 0.1 for s in sentiment_scores]

        raw_emb = compute_embeddings(texts, weights=weights)
        compressed = compressor.transform(raw_emb.reshape(1, -1))[0]
        result.loc[day, emb_cols] = compressed

    # Lag and rolling features for news_count
    result = add_lag_features(result, columns=["news_count"], lags=[1, 2])
    result = add_rolling_features(result, columns=["news_count"], window=7)

    # Lag top-3 PCA components (captures "news direction" from yesterday/day before)
    top_pca_cols = [f"emb_{i}" for i in range(TOP_PCA_LAGS)]
    result = add_lag_features(result, columns=top_pca_cols, lags=[1, 2])

    logger.info(
        f"Added {COMPRESSED_DIM} emb + 3 sentiment extremes + lags/rolling "
        f"to {len(result)} rows ({len(news_by_day)} days with news)"
    )
    return result
```

- [ ] **Step 4: Запустить все тесты build_embedding**

```bash
python -m pytest tests/test_build_embedding.py -v
```
Expected: все PASS (old + new tests)

- [ ] **Step 5: Коммит**

```bash
git add src/features/build_embedding_features.py tests/test_build_embedding.py
git commit -m "feat: add sentiment extremes, PCA lags, rolling news_count to embeddings"
```

---

## Task 3: Linear LR decay + сеть [512, 256] + обновить FEATURE_COUNTS

**Files:**
- Modify: `src/agents/config.py`
- Modify: `src/agents/train.py`
- Modify: `tests/test_train.py`

- [ ] **Step 1: Добавить lr_schedule в AgentConfig**

В `src/agents/config.py` добавить поле после `activation_fn`:

```python
    lr_schedule: str = "constant"    # "constant" or "linear" (decay to 0)
```

- [ ] **Step 2: Обновить train.py — FEATURE_COUNTS, EMBEDDINGS_NET_ARCH, linear schedule**

В `src/agents/train.py`:

1. Обновить FEATURE_COUNTS:
```python
FEATURE_COUNTS = {
    "baseline": 18,
    "sentiment": 19,     # baseline + 1 sentiment score
    "embeddings": 95,    # 18 base + 64 emb + news_count(3: raw+lag1+lag2) + roll7 + 3 sent extremes + 6 PCA lags
    "fusion": 98,        # embeddings(95) + 3 sentiment (raw + lag1 + lag2)
}

# Larger network for high-dimensional agents
EMBEDDINGS_NET_ARCH = [512, 256]
```

2. Добавить функцию linear schedule перед `train_agent()`:
```python
def _linear_schedule(initial_lr: float):
    """Linear LR decay from initial_lr to 0."""
    def schedule(progress_remaining: float) -> float:
        return progress_remaining * initial_lr
    return schedule
```

3. В `train_agent()` заменить блок создания модели (строки 108-119):
```python
    algo_cls = ALGO_MAP[config.algorithm]

    # Learning rate: constant or linear decay
    lr = config.learning_rate
    if config.lr_schedule == "linear":
        lr = _linear_schedule(config.learning_rate)

    # Override net_arch for high-dimensional embeddings
    policy_kwargs = config.policy_kwargs()
    if config.agent_type in ("embeddings", "fusion") and config.net_arch == [256, 256]:
        policy_kwargs["net_arch"] = EMBEDDINGS_NET_ARCH

    model = algo_cls(
        "MlpPolicy",
        env,
        learning_rate=lr,
        seed=config.seed,
        verbose=0,
        device="cpu",
        policy_kwargs=policy_kwargs,
        **_algo_specific_kwargs(config),
    )
```

- [ ] **Step 3: Обновить тесты**

В `tests/test_train.py` заменить тесты FEATURE_COUNTS:

```python
def test_feature_counts_embeddings_updated():
    """embeddings = 95 features total."""
    from src.agents.train import FEATURE_COUNTS
    assert FEATURE_COUNTS["embeddings"] == 95


def test_feature_counts_fusion_updated():
    """fusion = 98 features total."""
    from src.agents.train import FEATURE_COUNTS
    assert FEATURE_COUNTS["fusion"] == 98
```

Добавить новые тесты:

```python
def test_train_embeddings_with_linear_lr():
    """Embeddings agent trains with linear LR schedule."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        config = AgentConfig(
            total_timesteps=512,
            save_dir=tmpdir,
            seed=42,
            agent_type="embeddings",
            lr_schedule="linear",
        )
        model_path = train_agent(config, dummy=True)
        assert model_path.exists()


def test_embeddings_uses_larger_network():
    """Embeddings agent should use [512, 256] network."""
    from src.agents.train import EMBEDDINGS_NET_ARCH
    assert EMBEDDINGS_NET_ARCH == [512, 256]
```

- [ ] **Step 4: Запустить тесты**

```bash
python -m pytest tests/test_train.py -v
```
Expected: все PASS

- [ ] **Step 5: Коммит**

```bash
git add src/agents/config.py src/agents/train.py tests/test_train.py
git commit -m "feat: add linear LR decay, larger network, update FEATURE_COUNTS to 95"
```

---

## Task 4: Обновить run_all.py для linear LR

**Files:**
- Modify: `experiments/run_all.py`

- [ ] **Step 1: Обновить run_all.py**

В `experiments/run_all.py` в функции `run_all()`, при создании `AgentConfig` (строка ~94), добавить `lr_schedule="linear"`:

```python
            config = AgentConfig(
                agent_type=agent_type,
                algorithm="PPO",
                total_timesteps=total_timesteps,
                seed=seed,
                save_dir=save_dir,
                lr_schedule="linear",
            )
```

- [ ] **Step 2: Smoke test**

```bash
PYTHONPATH=. python experiments/run_all.py --dummy --total-timesteps 1000 --agent-types embeddings --seeds 42
```

Expected: trains with 95 features, linear LR, [512, 256] network.

- [ ] **Step 3: Коммит**

```bash
git add experiments/run_all.py
git commit -m "feat: enable linear LR schedule in run_all.py"
```

---

## Task 5: Пересобрать embedding features и запустить ночной прогон

**Files:**
- Run: `scripts/rebuild_embedding_features.py`
- Run: `experiments/run_all.py`

- [ ] **Step 1: Пересобрать parquet**

```bash
PYTHONPATH=. python scripts/rebuild_embedding_features.py
```

Время: ~30-40 мин (модель FinLang уже скачана).

- [ ] **Step 2: Проверить результат**

```bash
PYTHONPATH=. python -c "
import pandas as pd
df = pd.read_parquet('data/processed/btc_embedding_features.parquet')
price_cols = {'open','high','low','close','volume','raw_close'}
feat_cols = [c for c in df.columns if c.lower() not in price_cols]
lag_cols = [c for c in df.columns if 'lag' in c]
roll_cols = [c for c in df.columns if 'roll' in c]
sent_cols = [c for c in df.columns if 'sentiment' in c]
print(f'Shape: {df.shape}')
print(f'Feature cols: {len(feat_cols)}')
print(f'Lag cols: {lag_cols}')
print(f'Roll cols: {roll_cols}')
print(f'Sentiment cols: {sent_cols}')
"
```

Expected: 95 features. Lag cols: `news_count_lag1`, `news_count_lag2`, `emb_0_lag1`, `emb_0_lag2`, `emb_1_lag1`, `emb_1_lag2`, `emb_2_lag1`, `emb_2_lag2`. Roll cols: `news_count_roll7`. Sentiment cols: `sentiment_max`, `sentiment_min`, `sentiment_spread`.

- [ ] **Step 3: Запустить обучение embeddings (7 seeds × 2M)**

```bash
PYTHONPATH=. python experiments/run_all.py \
    --total-timesteps 2000000 \
    --asset BTC/USDT \
    --agent-types embeddings \
    --seeds 42,43,44,45,46,47,48
```

Время: ~7 × 25 мин = ~3 часа.

После завершения сохранить:
```bash
cp results/metrics.csv results/embeddings_v2_results.csv
```

- [ ] **Step 4: Запустить baseline для сравнения (7 seeds × 2M)**

```bash
PYTHONPATH=. python experiments/run_all.py \
    --total-timesteps 2000000 \
    --asset BTC/USDT \
    --agent-types baseline \
    --seeds 42,43,44,45,46,47,48
```

Время: ~7 × 12 мин = ~1.5 часа. Baseline тоже с linear LR для честности.

- [ ] **Step 5: Объединить результаты**

```bash
PYTHONPATH=. python -c "
import pandas as pd, numpy as np
emb = pd.read_csv('results/embeddings_v2_results.csv')
base = pd.read_csv('results/metrics.csv')
combined = pd.concat([base, emb], ignore_index=True)
combined.to_csv('results/metrics.csv', index=False)

for agent in ['baseline', 'embeddings']:
    rows = combined[combined.agent_type == agent]
    s, r, d, so = rows.sharpe_ratio, rows.total_return, rows.max_drawdown, rows.sortino_ratio
    print(f'{agent:12} Sharpe={s.mean():.3f}±{s.std():.3f}  Return={r.mean()*100:.1f}%±{r.std()*100:.1f}%  MaxDD={d.mean()*100:.1f}%±{d.std()*100:.1f}%  Sortino={so.mean():.3f}±{so.std():.3f}')
"
```

- [ ] **Step 6: Коммит**

```bash
git add results/metrics.csv results/embeddings_v2_results.csv
git commit -m "feat: embeddings v2 vs baseline — 7 seeds × 2M timesteps"
```

---

## Task 6: Документация результатов и сравнение

**Files:**
- Create: `docs/embeddings_v2_results.md`
- Modify: `docs/experiment_summary.md`

- [ ] **Step 1: Создать docs/embeddings_v2_results.md**

Документ должен содержать:

1. **Что изменили** — таблица всех улучшений (lag, rolling, sentiment extremes, PCA lags, LR decay, bigger network, 2M steps, 7 seeds)

2. **Embeddings v2 результаты** — посеедовая таблица (7 seeds: Sharpe, Return, MaxDD, Sortino, Calmar)

3. **Baseline результаты** — посеедовая таблица (7 seeds)

4. **Сравнение v2 vs baseline** — таблица mean±std (Sharpe, Return, MaxDD, Sortino) с колонкой Δ

5. **Сравнение v2 vs v1 (предыдущий embeddings)** — таблица:

| Метрика | Emb v1 (3s, 500K) | Emb v2 (7s, 2M) | Δ |
|---------|-------------------|-----------------|---|
| Sharpe | 1.310 ± 0.520 | ??? | |
| Return | 54.3% ± 35.7% | ??? | |
| MaxDD | 24.4% ± 7.9% | ??? | |

6. **Анализ** — что сработало, что нет, что дальше

7. **Полный рейтинг стратегий** — обновлённый рейтинг всех стратегий по Sharpe (baseline, embeddings v1, embeddings v2, ensembles, buy&hold)

- [ ] **Step 2: Обновить docs/experiment_summary.md**

Добавить секцию "1.6 Embeddings v2" с кратким summary и ссылкой на полный отчёт.

- [ ] **Step 3: Коммит**

```bash
git add docs/embeddings_v2_results.md docs/experiment_summary.md
git commit -m "docs: add embeddings v2 results and full comparison"
```

---

## Итоговый подсчёт фич (embeddings v2)

```
18  baseline price features (SMA, EMA, MACD, RSI, BB, ATR, OBV, Stoch, returns)
64  PCA-compressed FinLang embeddings (emb_0..emb_63)
 1  news_count
 2  news_count_lag1, news_count_lag2
 1  news_count_roll7
 3  sentiment_max, sentiment_min, sentiment_spread
 6  emb_0_lag1, emb_0_lag2, emb_1_lag1, emb_1_lag2, emb_2_lag1, emb_2_lag2
──
95  TOTAL
```

## Верификация

```bash
# Все unit тесты
python -m pytest tests/test_lag_features.py tests/test_build_embedding.py tests/test_train.py -v -k "not integration"

# Smoke test (end-to-end с dummy данными)
PYTHONPATH=. python experiments/run_all.py --dummy --total-timesteps 1000 --agent-types embeddings --seeds 42

# Проверить parquet
PYTHONPATH=. python -c "
import pandas as pd
df = pd.read_parquet('data/processed/btc_embedding_features.parquet')
price_cols = {'open','high','low','close','volume','raw_close'}
feat_cols = [c for c in df.columns if c.lower() not in price_cols]
print(f'Features: {len(feat_cols)} (expected 95)')
"
```

## Ожидаемое время

| Этап | Время |
|------|-------|
| Tasks 1-4 (код + тесты) | ~20 мин |
| Task 5 step 1 (пересборка parquet) | ~35 мин |
| Task 5 step 3 (embeddings 7×2M) | ~3 часа |
| Task 5 step 4 (baseline 7×2M) | ~1.5 часа |
| Task 6 (документация) | ~10 мин |
| **Итого** | **~5 часов** (ставить на ночь) |
