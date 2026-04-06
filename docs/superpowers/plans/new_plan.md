# Улучшенный Embeddings Агент: FinLang + Weighted Pooling + PCA 64d

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Улучшить агента с эмбеддингами заменив all-MiniLM-L6-v2 на финансово-специфичную модель FinLang/finance-embeddings-investopedia, добавив взвешенный pooling и увеличив PCA с 32 до 64 измерений — затем сравнить Sharpe нового агента с текущим (1.76).

**Architecture:** Меняем только `src/features/embeddings.py` (модель + pooling) и `src/features/build_embedding_features.py` (PCA dim + news_count фича). Пересобираем `data/processed/btc_embedding_features.parquet`. Переобучаем агент и сравниваем метрики. Все остальные агенты (baseline, sentiment, fusion) не трогаем.

**Tech Stack:** sentence-transformers, scikit-learn PCA, stable-baselines3 PPO, pandas, numpy, pytest

---

## Контекст и мотивация

Текущий embeddings агент показывает Sharpe 1.76 vs baseline 1.80 — хуже несмотря на новостные данные. Три причины:

1. **all-MiniLM-L6-v2** — общая модель (Wikipedia, форумы), не понимает финансовые тексты
2. **Mean pooling** — важные новости "тонут" среди скучных; вес всех одинаковый
3. **PCA 32d** — теряем 91% информации при сжатии 384→32

Исправляем все три проблемы за одну пересборку данных.

**Weighted pooling логика:** взвешиваем каждую статью по |sentiment_score| — чем сильнее эмоция (позитивная или негативная), тем важнее статья. Нейтральные статьи получают малый вес.

---

## Файловая структура

| Действие | Файл | Что меняем |
|----------|------|------------|
| Modify | `src/features/embeddings.py` | Модель → FinLang, добавить weighted pooling |
| Modify | `src/features/build_embedding_features.py` | PCA 32→64, добавить news_count фичу |
| Modify | `src/agents/train.py` | FEATURE_COUNTS["embeddings"]: 50→83 (18+64+1) |
| Modify | `tests/test_embeddings.py` | Тесты для weighted pooling |
| Modify | `tests/test_build_embedding.py` | Тесты для 64d + news_count |

---

## Возможные дальнейшие улучшения (после этого плана)

После выполнения плана можно сравнить и продолжить:

1. **CryptoBERT для sentiment** — заменить FinBERT → `kk08/CryptoBERT` в `src/features/sentiment.py`. Ожидаемый прирост: +3-8% точности классификации на крипто-тексте.

2. **Lag features для sentiment** — добавить sentiment[-1] и sentiment[-2] как отдельные признаки. Гипотеза: новость влияет на цену на следующий день.

3. **Количество новостей как фича** — уже входит в этот план (news_count), но можно расширить: добавить rolling average за 7 дней.

4. **PCA → Autoencoder** — заменить линейное PCA на нейросетевой автоэнкодер (64d latent). Сохраняет нелинейные паттерны. Сложнее, потенциально лучше.

5. **Fusion агент с улучшенными моделями** — объединить CryptoBERT sentiment + FinLang embeddings в один агент (Agent-4 fusion). Код уже есть в `src/features/build_fusion_features.py`.

6. **Twitter/Reddit данные** — текущие данные только из новостей. Добавить социальные сети через другой HuggingFace датасет.

---

### Task 1: Weighted pooling в embeddings.py

**Files:**
- Modify: `src/features/embeddings.py`
- Modify: `tests/test_embeddings.py`

- [ ] **Step 1: Написать тесты для weighted pooling**

Добавить в конец `tests/test_embeddings.py`:

```python
def test_weighted_pooling_differs_from_mean():
    """Weighted pooling with unequal weights differs from mean pooling."""
    from src.features.embeddings import compute_embeddings

    class MockModel:
        def encode(self, texts, show_progress_bar=False):
            # Two very different embeddings
            return np.array([
                [1.0, 0.0, 0.0] + [0.0] * 381,
                [0.0, 1.0, 0.0] + [0.0] * 381,
            ], dtype=np.float32)

    weights = [0.9, 0.1]  # first article is much more important
    result = compute_embeddings(["text1", "text2"], model=MockModel(), weights=weights)
    mean_result = compute_embeddings(["text1", "text2"], model=MockModel())

    # With heavy weight on first article, result[0] should be closer to 1.0
    assert result[0] > mean_result[0], "Weighted pooling should differ from mean"


def test_weighted_pooling_equal_weights_matches_mean():
    """Equal weights produce same result as mean pooling."""
    from src.features.embeddings import compute_embeddings

    class MockModel:
        def encode(self, texts, show_progress_bar=False):
            return np.array([
                [1.0, 0.0] + [0.0] * 382,
                [0.0, 1.0] + [0.0] * 382,
            ], dtype=np.float32)

    weights = [0.5, 0.5]
    result_weighted = compute_embeddings(["t1", "t2"], model=MockModel(), weights=weights)
    result_mean = compute_embeddings(["t1", "t2"], model=MockModel())
    np.testing.assert_allclose(result_weighted, result_mean, atol=1e-6)


def test_empty_weights_falls_back_to_mean():
    """None weights falls back to mean pooling."""
    from src.features.embeddings import compute_embeddings

    class MockModel:
        def encode(self, texts, show_progress_bar=False):
            return np.ones((len(texts), 384), dtype=np.float32)

    result = compute_embeddings(["text"], model=MockModel(), weights=None)
    assert result.shape == (384,)
```

- [ ] **Step 2: Запустить тесты, убедиться что падают**

```bash
python -m pytest tests/test_embeddings.py::test_weighted_pooling_differs_from_mean -v
```
Expected: FAIL — `compute_embeddings` не принимает `weights`

- [ ] **Step 3: Реализовать weighted pooling + сменить модель**

Заменить весь файл `src/features/embeddings.py`:

```python
"""Sentence embeddings via FinLang/finance-embeddings-investopedia for news texts."""
import logging
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 768  # finance-embeddings-investopedia outputs 768d
_model = None


def _get_model():
    """Lazy-load the sentence-transformer model."""
    global _model
    if _model is None:
        import torch
        from sentence_transformers import SentenceTransformer
        logger.info("Loading FinLang/finance-embeddings-investopedia...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = SentenceTransformer("FinLang/finance-embeddings-investopedia", device=device)
    return _model


def compute_embeddings(
    texts: List[str],
    model: Optional[object] = None,
    weights: Optional[List[float]] = None,
) -> np.ndarray:
    """Compute weighted mean-pooled sentence embedding from list of texts.

    Args:
        texts: List of news article texts.
        model: Optional pre-loaded SentenceTransformer (for testing).
        weights: Optional list of weights per article. If None, uses mean pooling.
                 Weights are normalized to sum=1 internally.

    Returns:
        np.ndarray of shape (768,). Returns zeros if texts is empty.
    """
    if not texts:
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)

    st_model = model if model is not None else _get_model()
    embeddings = st_model.encode(texts, show_progress_bar=False)

    if weights is None or len(weights) != len(texts):
        mean_embedding = np.mean(embeddings, axis=0)
    else:
        w = np.array(weights, dtype=np.float32)
        w_sum = w.sum()
        if w_sum < 1e-8:
            mean_embedding = np.mean(embeddings, axis=0)
        else:
            w = w / w_sum
            mean_embedding = np.average(embeddings, axis=0, weights=w)

    return mean_embedding.astype(np.float32)
```

- [ ] **Step 4: Запустить все тесты embeddings**

```bash
python -m pytest tests/test_embeddings.py -v
```
Expected: все PASS

- [ ] **Step 5: Коммит**

```bash
git add src/features/embeddings.py tests/test_embeddings.py
git commit -m "feat: switch to FinLang finance embeddings + weighted pooling"
```

---

### Task 2: PCA 64d + news_count в build_embedding_features.py

**Files:**
- Modify: `src/features/build_embedding_features.py`
- Modify: `tests/test_build_embedding.py`

- [ ] **Step 1: Написать тесты**

Добавить в конец `tests/test_build_embedding.py`:

```python
def test_news_count_column_added():
    """build_embedding_features adds news_count column."""
    import pandas as pd
    import numpy as np
    from unittest.mock import patch
    from src.features.build_embedding_features import build_embedding_features

    idx = pd.date_range("2021-01-01", periods=5, freq="D", tz="UTC")
    price_features = pd.DataFrame({"close": [100.0] * 5}, index=idx)

    news_by_day = pd.DataFrame({
        "date": [pd.Timestamp("2021-01-01", tz="UTC")],
        "texts": [["article1", "article2", "article3"]],
    })

    class MockCompressor:
        def transform(self, X):
            return np.zeros((X.shape[0], 64), dtype=np.float32)

    with patch("src.features.build_embedding_features.compute_embeddings",
               return_value=np.zeros(768, dtype=np.float32)):
        result = build_embedding_features(price_features, news_by_day, MockCompressor())

    assert "news_count" in result.columns
    assert result.loc[pd.Timestamp("2021-01-01", tz="UTC"), "news_count"] == 3
    assert result["news_count"].iloc[1] == 0  # no news day


def test_embedding_columns_are_64d():
    """build_embedding_features produces 64 emb columns when compressor outputs 64d."""
    import pandas as pd
    import numpy as np
    from unittest.mock import patch
    from src.features.build_embedding_features import build_embedding_features

    idx = pd.date_range("2021-01-01", periods=3, freq="D", tz="UTC")
    price_features = pd.DataFrame({"close": [100.0] * 3}, index=idx)
    news_by_day = pd.DataFrame({
        "date": [pd.Timestamp("2021-01-01", tz="UTC")],
        "texts": [["text"]],
    })

    class MockCompressor64:
        def transform(self, X):
            return np.zeros((X.shape[0], 64), dtype=np.float32)

    with patch("src.features.build_embedding_features.compute_embeddings",
               return_value=np.zeros(768, dtype=np.float32)):
        result = build_embedding_features(price_features, news_by_day, MockCompressor64())

    emb_cols = [c for c in result.columns if c.startswith("emb_")]
    assert len(emb_cols) == 64
```

- [ ] **Step 2: Запустить тесты, убедиться что падают**

```bash
python -m pytest tests/test_build_embedding.py::test_news_count_column_added tests/test_build_embedding.py::test_embedding_columns_are_64d -v
```
Expected: FAIL

- [ ] **Step 3: Обновить build_embedding_features.py**

Заменить весь файл `src/features/build_embedding_features.py`:

```python
"""Build daily embedding features by merging compressed embeddings with price features."""
import logging

import numpy as np
import pandas as pd

from src.features.embeddings import compute_embeddings
from src.features.sentiment import compute_sentiment

logger = logging.getLogger(__name__)

COMPRESSED_DIM = 64  # increased from 32


def build_embedding_features(
    price_features: pd.DataFrame,
    news_by_day: pd.DataFrame,
    compressor,
) -> pd.DataFrame:
    """Merge compressed daily embeddings with price features.

    Uses weighted pooling: articles weighted by |sentiment_score| so that
    strongly positive/negative articles contribute more than neutral ones.

    Args:
        price_features: DataFrame indexed by date with normalized price features.
        news_by_day: DataFrame with columns [date, texts] from news_preprocessor.
        compressor: Fitted EmbeddingCompressor with transform() method (outputs 64d).

    Returns:
        price_features with 64 emb_0..emb_63 columns and news_count column added.
        Days without news get zeros for embeddings and 0 for news_count.
    """
    result = price_features.copy()

    emb_cols = [f"emb_{i}" for i in range(COMPRESSED_DIM)]
    for col in emb_cols:
        result[col] = 0.0
    result["news_count"] = 0

    for _, row in news_by_day.iterrows():
        day = row["date"].normalize()
        texts = row["texts"]
        if day not in result.index:
            continue

        n = len(texts)
        result.loc[day, "news_count"] = n

        # Compute sentiment weights: |score| so extreme articles weigh more
        sentiment_scores = [compute_sentiment([t]) for t in texts]
        weights = [abs(s) + 0.1 for s in sentiment_scores]  # +0.1 ensures no zero weights

        raw_emb = compute_embeddings(texts, weights=weights)
        compressed = compressor.transform(raw_emb.reshape(1, -1))[0]
        result.loc[day, emb_cols] = compressed

    logger.info(
        f"Added {COMPRESSED_DIM} embedding features + news_count to {len(result)} rows "
        f"({len(news_by_day)} days with news)"
    )
    return result
```

- [ ] **Step 4: Запустить все тесты build_embedding**

```bash
python -m pytest tests/test_build_embedding.py -v
```
Expected: все PASS (старые тесты могут потребовать обновления mock для 64d)

**Если старые тесты упали** из-за 32d → 64d, обновить в `tests/test_build_embedding.py` все места где написано `range(32)` на `range(64)` и `[f"emb_{i}" for i in range(32)]` на `range(64)`.

- [ ] **Step 5: Коммит**

```bash
git add src/features/build_embedding_features.py tests/test_build_embedding.py
git commit -m "feat: increase PCA to 64d, add news_count feature, weighted pooling by sentiment"
```

---

### Task 3: Обновить FEATURE_COUNTS в train.py

**Files:**
- Modify: `src/agents/train.py:45-50`

- [ ] **Step 1: Написать тест**

Добавить в `tests/test_train.py`:

```python
def test_feature_counts_embeddings_updated():
    """embeddings agent feature count reflects 18 baseline + 64 emb + 1 news_count."""
    from src.agents.train import FEATURE_COUNTS
    assert FEATURE_COUNTS["embeddings"] == 83  # 18 + 64 + 1
```

- [ ] **Step 2: Запустить, убедиться что падает**

```bash
python -m pytest tests/test_train.py::test_feature_counts_embeddings_updated -v
```
Expected: FAIL (текущее значение 50)

- [ ] **Step 3: Обновить FEATURE_COUNTS**

В `src/agents/train.py` строки 45-50, изменить:

```python
FEATURE_COUNTS = {
    "baseline": 18,
    "sentiment": 19,     # baseline + 1 sentiment score
    "embeddings": 83,    # baseline(18) + 64 compressed embeddings + 1 news_count
    "fusion": 84,        # baseline(18) + 1 sentiment + 64 embeddings + 1 news_count
}
```

- [ ] **Step 4: Запустить тесты train**

```bash
python -m pytest tests/test_train.py -v
```
Expected: все PASS

- [ ] **Step 5: Коммит**

```bash
git add src/agents/train.py tests/test_train.py
git commit -m "feat: update FEATURE_COUNTS for 64d embeddings + news_count"
```

---

### Task 4: Обновить EmbeddingCompressor для 768→64

**Files:**
- Modify: `src/features/embedding_compressor.py:15`

- [ ] **Step 1: Написать тест**

Добавить в `tests/test_compressor.py`:

```python
def test_compressor_768_to_64():
    """EmbeddingCompressor works with 768d input and 64d output."""
    from src.features.embedding_compressor import EmbeddingCompressor
    import numpy as np

    compressor = EmbeddingCompressor(input_dim=768, output_dim=64)
    data = np.random.randn(200, 768).astype(np.float32)
    compressor.fit(data)
    result = compressor.transform(data[:5])
    assert result.shape == (5, 64)
    assert compressor.explained_variance_ratio() > 0
```

- [ ] **Step 2: Запустить, убедиться что PASS**

```bash
python -m pytest tests/test_compressor.py::test_compressor_768_to_64 -v
```
Expected: PASS — компрессор уже принимает произвольные размеры (параметрический)

- [ ] **Step 3: Коммит (только если были изменения)**

Если тест PASS без изменений — компрессор уже поддерживает 768→64, коммит не нужен. Если нет — исправить дефолтные значения в `embedding_compressor.py:15`:

```python
def __init__(self, input_dim: int = 768, output_dim: int = 64):
```

```bash
git add src/features/embedding_compressor.py tests/test_compressor.py
git commit -m "feat: update EmbeddingCompressor defaults to 768->64"
```

---

### Task 5: Пересобрать embedding features и переобучить агента

**Files:**
- Run: `src/features/build_embedding_features.py`
- Run: `experiments/run_all.py`

- [ ] **Step 1: Пересобрать btc_embedding_features.parquet**

```bash
PYTHONPATH=. python -m src.features.build_embedding_features
```

Expected: создаётся `data/processed/btc_embedding_features.parquet` с 64 emb колонками + news_count. Займёт ~2-3 часа (новая модель тяжелее).

Проверить:

```bash
python -c "
import pandas as pd
df = pd.read_parquet('data/processed/btc_embedding_features.parquet')
emb_cols = [c for c in df.columns if c.startswith('emb_')]
print(f'Embedding cols: {len(emb_cols)}')  # должно быть 64
print(f'Has news_count: {\"news_count\" in df.columns}')  # True
print(f'Shape: {df.shape}')
"
```

- [ ] **Step 2: Переобучить embeddings агента (3 seeds)**

```bash
PYTHONPATH=. python experiments/run_all.py --total-timesteps 500000 --asset BTC/USDT --agent-types embeddings --seeds 42,43,44
```

Expected: новые результаты в `results/metrics.csv` для embeddings агента.

- [ ] **Step 3: Сравнить результаты**

```bash
python -c "
import pandas as pd
df = pd.read_csv('results/metrics.csv')
emb = df[df.agent_type == 'embeddings']
base = df[df.agent_type == 'baseline']
print('Embeddings:', emb.sharpe_ratio.mean().round(3), '+-', emb.sharpe_ratio.std().round(3))
print('Baseline:  ', base.sharpe_ratio.mean().round(3), '+-', base.sharpe_ratio.std().round(3))
"
```

- [ ] **Step 4: Записать результаты и коммит**

```bash
git add results/metrics.csv results/improved_metrics.csv
git commit -m "feat: add improved embeddings agent results (FinLang + weighted pooling + PCA 64d)"
```

---

## Верификация end-to-end

После выполнения всех задач:

```bash
# 1. Все тесты проходят
python -m pytest tests/test_embeddings.py tests/test_compressor.py tests/test_build_embedding.py tests/test_train.py -v

# 2. Smoke test с dummy данными
PYTHONPATH=. python experiments/run_all.py --dummy --total-timesteps 1000 --agent-types embeddings --seeds 42

# 3. Проверить parquet
python -c "
import pandas as pd
df = pd.read_parquet('data/processed/btc_embedding_features.parquet')
assert len([c for c in df.columns if c.startswith('emb_')]) == 64
assert 'news_count' in df.columns
print('OK: 64 emb cols + news_count')
"
```

Ожидаемый итог: embeddings агент Sharpe > 1.80 (лучше baseline) или как минимум сопоставимый результат с диагностикой почему.
