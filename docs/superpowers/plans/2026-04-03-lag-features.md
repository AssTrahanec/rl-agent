# Lag Features для Sentiment и Embeddings агентов

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Добавить lag-1 и lag-2 фичи для sentiment и news_count, чтобы агент мог ловить запаздывающую реакцию рынка на новости (гипотеза: новость сегодня → движение цены завтра/послезавтра).

**Architecture:** Добавляем `add_lag_features(df, columns, lags)` утилиту в новый файл `src/features/lag_features.py`. Вызываем её в конце `build_sentiment_features()` и `build_embedding_features()`. Обновляем FEATURE_COUNTS. Пересобираем parquet и переобучаем.

**Tech Stack:** pandas, numpy, pytest, stable-baselines3 PPO

---

## Контекст

Текущая проблема: sentiment и embeddings за день N подаются агенту в день N, но к закрытию дня новость уже отыграна. Добавляя lag-1 (вчерашний sentiment) и lag-2, агент может ловить:
- Запаздывающую реакцию (новость вышла, но рынок среагировал через день)
- Продолжение тренда (сильный sentiment вчера → продолжение движения сегодня)

## Файловая структура

| Действие | Файл | Что меняем |
|----------|------|------------|
| Create | `src/features/lag_features.py` | `add_lag_features()` утилита |
| Create | `tests/test_lag_features.py` | Тесты для lag утилиты |
| Modify | `src/features/build_sentiment_features.py` | Вызов add_lag_features для sentiment |
| Modify | `src/features/build_embedding_features.py` | Вызов add_lag_features для news_count |
| Modify | `src/agents/train.py:45-50` | FEATURE_COUNTS +2 для sentiment, +2 для embeddings |
| Modify | `tests/test_build_sentiment.py` | Обновить тесты для новых колонок |
| Modify | `tests/test_build_embedding.py` | Обновить тесты для новых колонок |
| Modify | `tests/test_train.py` | Обновить тесты FEATURE_COUNTS |

### Итоговые фичи после изменений

| Агент | Было | Стало | Новые колонки |
|-------|------|-------|---------------|
| sentiment | 19 | 21 | sentiment_lag1, sentiment_lag2 |
| embeddings | 83 | 85 | news_count_lag1, news_count_lag2 |
| fusion | 84 | 88 | все 4 лага |

---

### Task 1: Утилита add_lag_features

**Files:**
- Create: `src/features/lag_features.py`
- Create: `tests/test_lag_features.py`

- [ ] **Step 1: Написать тесты**

Создать `tests/test_lag_features.py`:

```python
import pandas as pd
import numpy as np
from src.features.lag_features import add_lag_features


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
    # Day 3 lag1 = Day 2 value
    assert result["sentiment_lag1"].iloc[2] == 0.5
    # Day 3 lag2 = Day 1 value
    assert result["sentiment_lag2"].iloc[2] == 0.1


def test_lag_fills_nan_with_zero():
    """First rows where lag is unavailable should be filled with 0."""
    idx = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
    df = pd.DataFrame({"sentiment": [0.1, 0.5, -0.3, 0.8, 0.0]}, index=idx)
    result = add_lag_features(df, columns=["sentiment"], lags=[1, 2])
    assert result["sentiment_lag1"].iloc[0] == 0.0  # no previous day
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
    assert result["news_count_lag1"].iloc[2] == 5  # day 2 value
```

- [ ] **Step 2: Запустить тесты, убедиться что падают**

```bash
python -m pytest tests/test_lag_features.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'src.features.lag_features'`

- [ ] **Step 3: Реализовать add_lag_features**

Создать `src/features/lag_features.py`:

```python
"""Add lagged versions of features to capture delayed market reactions."""
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
```

- [ ] **Step 4: Запустить тесты**

```bash
python -m pytest tests/test_lag_features.py -v
```
Expected: все 5 PASS

- [ ] **Step 5: Коммит**

```bash
git add src/features/lag_features.py tests/test_lag_features.py
git commit -m "feat: add lag_features utility for delayed market reaction"
```

---

### Task 2: Lag features в build_sentiment_features

**Files:**
- Modify: `src/features/build_sentiment_features.py`
- Modify: `tests/test_build_sentiment.py`

- [ ] **Step 1: Добавить тест**

Добавить в конец `tests/test_build_sentiment.py`:

```python
def test_sentiment_lag_columns_exist():
    """build_sentiment_features adds sentiment_lag1 and sentiment_lag2."""
    prices = _make_price_features()
    news = _make_news_by_day()
    with patch("src.features.build_sentiment_features.compute_sentiment") as mock_sent:
        mock_sent.return_value = 0.5
        result = build_sentiment_features(prices, news)
    assert "sentiment_lag1" in result.columns
    assert "sentiment_lag2" in result.columns


def test_sentiment_lag_values_correct():
    """sentiment_lag1 should be previous day's sentiment."""
    prices = _make_price_features()
    news = _make_news_by_day()
    with patch("src.features.build_sentiment_features.compute_sentiment") as mock_sent:
        mock_sent.return_value = 0.7
        result = build_sentiment_features(prices, news)
    # 2024-01-02 has sentiment 0.7; 2024-01-03 should have lag1=0.7
    day_after_news = pd.Timestamp("2024-01-03", tz="UTC")
    assert result.loc[day_after_news, "sentiment_lag1"] == 0.7
```

- [ ] **Step 2: Запустить тесты, убедиться что падают**

```bash
python -m pytest tests/test_build_sentiment.py::test_sentiment_lag_columns_exist -v
```
Expected: FAIL — `sentiment_lag1` не существует

- [ ] **Step 3: Обновить build_sentiment_features.py**

Добавить import и вызов в конце функции `build_sentiment_features`:

```python
"""Build daily sentiment features by merging news sentiment with price features.

Usage:
    python -m src.features.build_sentiment_features
"""
import logging

import pandas as pd

from src.features.sentiment import compute_sentiment
from src.features.lag_features import add_lag_features

logger = logging.getLogger(__name__)


def build_sentiment_features(
    price_features: pd.DataFrame,
    news_by_day: pd.DataFrame,
) -> pd.DataFrame:
    """Merge daily sentiment scores with price features.

    Args:
        price_features: DataFrame indexed by date with normalized price features.
        news_by_day: DataFrame with columns [date, texts] from news_preprocessor.

    Returns:
        price_features with added 'sentiment', 'sentiment_lag1', 'sentiment_lag2'
        columns. Days without news get 0.0.
    """
    result = price_features.copy()
    result["sentiment"] = 0.0

    # Build date -> sentiment mapping
    for _, row in news_by_day.iterrows():
        day = row["date"].normalize()
        texts = row["texts"]
        score = compute_sentiment(texts)
        if day in result.index:
            result.loc[day, "sentiment"] = score

    # Add lagged sentiment features
    result = add_lag_features(result, columns=["sentiment"], lags=[1, 2])

    logger.info(
        f"Added sentiment + 2 lags to {len(result)} rows "
        f"({len(news_by_day)} days with news)"
    )
    return result
```

- [ ] **Step 4: Запустить все тесты build_sentiment**

```bash
python -m pytest tests/test_build_sentiment.py -v
```
Expected: все 6 PASS

- [ ] **Step 5: Коммит**

```bash
git add src/features/build_sentiment_features.py tests/test_build_sentiment.py
git commit -m "feat: add sentiment lag1/lag2 features to sentiment agent"
```

---

### Task 3: Lag features в build_embedding_features

**Files:**
- Modify: `src/features/build_embedding_features.py`
- Modify: `tests/test_build_embedding.py`

- [ ] **Step 1: Добавить тест**

Добавить в конец `tests/test_build_embedding.py`:

```python
def test_news_count_lag_columns_exist():
    """build_embedding_features adds news_count_lag1 and news_count_lag2."""
    prices = _make_price_features()
    news = _make_news_by_day()
    comp = _mock_compressor()
    with patch("src.features.build_embedding_features.compute_embeddings") as mock_emb, \
         patch("src.features.build_embedding_features.compute_sentiment", return_value=0.5):
        mock_emb.return_value = np.zeros(768, dtype=np.float32)
        result = build_embedding_features(prices, news, compressor=comp)
    assert "news_count_lag1" in result.columns
    assert "news_count_lag2" in result.columns


def test_news_count_lag_values_correct():
    """news_count_lag1 should be previous day's news_count."""
    prices = _make_price_features()
    news = _make_news_by_day()
    comp = _mock_compressor()
    with patch("src.features.build_embedding_features.compute_embeddings") as mock_emb, \
         patch("src.features.build_embedding_features.compute_sentiment", return_value=0.5):
        mock_emb.return_value = np.zeros(768, dtype=np.float32)
        result = build_embedding_features(prices, news, compressor=comp)
    # 2024-01-02 has 1 article; 2024-01-03 should have lag1=1
    day_after = pd.Timestamp("2024-01-03", tz="UTC")
    assert result.loc[day_after, "news_count_lag1"] == 1
```

- [ ] **Step 2: Запустить, убедиться что падают**

```bash
python -m pytest tests/test_build_embedding.py::test_news_count_lag_columns_exist -v
```
Expected: FAIL

- [ ] **Step 3: Обновить build_embedding_features.py**

Добавить import `add_lag_features` и вызов перед return:

В начале файла добавить import:
```python
from src.features.lag_features import add_lag_features
```

Перед строкой `logger.info(` добавить:
```python
    # Add lagged news_count features
    result = add_lag_features(result, columns=["news_count"], lags=[1, 2])
```

- [ ] **Step 4: Запустить все тесты build_embedding**

```bash
python -m pytest tests/test_build_embedding.py -v
```
Expected: все 8 PASS

- [ ] **Step 5: Коммит**

```bash
git add src/features/build_embedding_features.py tests/test_build_embedding.py
git commit -m "feat: add news_count lag1/lag2 features to embeddings agent"
```

---

### Task 4: Обновить FEATURE_COUNTS и тесты

**Files:**
- Modify: `src/agents/train.py:45-50`
- Modify: `tests/test_train.py`

- [ ] **Step 1: Обновить FEATURE_COUNTS**

В `src/agents/train.py` строки 45-50, изменить:

```python
FEATURE_COUNTS = {
    "baseline": 18,
    "sentiment": 21,     # baseline(18) + sentiment + sentiment_lag1 + sentiment_lag2
    "embeddings": 85,    # baseline(18) + 64 emb + news_count + news_count_lag1 + news_count_lag2
    "fusion": 88,        # baseline(18) + 3 sentiment + 64 emb + 3 news_count
}
```

- [ ] **Step 2: Обновить тесты**

В `tests/test_train.py` обновить:

```python
def test_feature_counts_embeddings_updated():
    """embeddings agent feature count reflects 18 baseline + 64 emb + 3 news_count."""
    from src.agents.train import FEATURE_COUNTS
    assert FEATURE_COUNTS["embeddings"] == 85  # 18 + 64 + 1 + 2 lags


def test_feature_counts_fusion_updated():
    """fusion agent feature count reflects all features."""
    from src.agents.train import FEATURE_COUNTS
    assert FEATURE_COUNTS["fusion"] == 88  # 18 + 3 sentiment + 64 emb + 3 news_count
```

- [ ] **Step 3: Запустить тесты**

```bash
python -m pytest tests/test_train.py -v
```
Expected: все PASS

- [ ] **Step 4: Коммит**

```bash
git add src/agents/train.py tests/test_train.py
git commit -m "feat: update FEATURE_COUNTS for lag features"
```

---

### Task 5: Пересобрать данные и переобучить

**Files:**
- Modify: `scripts/rebuild_embedding_features.py` (не нужно — lag добавляется автоматически при build)
- Run: training pipeline

- [ ] **Step 1: Пересобрать sentiment features**

```bash
PYTHONPATH=. python -c "
import pandas as pd
from src.data.news_preprocessor import preprocess_news
from src.features.build_sentiment_features import build_sentiment_features

price_features = pd.read_parquet('data/processed/btc_features.parquet')
raw_news = pd.read_parquet('data/raw/bitcoin_news.parquet')
raw_news = raw_news.rename(columns={'article_text': 'text', 'date_time': 'date'})
raw_news['date'] = pd.to_datetime(raw_news['date'], utc=True)
news_by_day = preprocess_news(raw_news)

result = build_sentiment_features(price_features, news_by_day)
result.to_parquet('data/processed/btc_sentiment_features.parquet')
print(f'Shape: {result.shape}')
print(f'Columns: {[c for c in result.columns if \"sentiment\" in c]}')
"
```

Expected: shape has 21 feature columns (18 price + sentiment + sentiment_lag1 + sentiment_lag2), `['sentiment', 'sentiment_lag1', 'sentiment_lag2']` printed.

- [ ] **Step 2: Пересобрать embedding features**

```bash
PYTHONPATH=. python scripts/rebuild_embedding_features.py
```

Expected: parquet обновлён с news_count_lag1, news_count_lag2 (85 features total).

Проверить:
```bash
PYTHONPATH=. python -c "
import pandas as pd
df = pd.read_parquet('data/processed/btc_embedding_features.parquet')
lag_cols = [c for c in df.columns if 'lag' in c]
print(f'Shape: {df.shape}')
print(f'Lag columns: {lag_cols}')
"
```

- [ ] **Step 3: Обучить embeddings + baseline (3 seeds)**

```bash
PYTHONPATH=. python experiments/run_all.py --total-timesteps 500000 --asset BTC/USDT --agent-types baseline,embeddings --seeds 42,43,44
```

- [ ] **Step 4: Сравнить результаты**

```bash
PYTHONPATH=. python -c "
import pandas as pd
import numpy as np
df = pd.read_csv('results/metrics.csv')
for agent in ['baseline', 'embeddings']:
    rows = df[df.agent_type == agent]
    print(f'{agent:12} Sharpe={rows.sharpe_ratio.mean():.3f}±{rows.sharpe_ratio.std():.3f}  Return={rows.total_return.mean()*100:.1f}%  MaxDD={rows.max_drawdown.mean()*100:.1f}%')
"
```

- [ ] **Step 5: Коммит результатов**

```bash
git add results/metrics.csv
git commit -m "feat: retrain with lag features, compare results"
```

---

## Верификация

```bash
# Все тесты
python -m pytest tests/test_lag_features.py tests/test_build_sentiment.py tests/test_build_embedding.py tests/test_train.py -v

# Smoke test
PYTHONPATH=. python experiments/run_all.py --dummy --total-timesteps 1000 --agent-types embeddings,sentiment --seeds 42
```
