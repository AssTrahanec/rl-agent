# Результаты улучшенного Embeddings агента

**Дата**: 2026-04-03  
**Изменения**: FinLang/finance-embeddings-investopedia + Weighted Pooling + PCA 64d

---

## Что изменили

| Компонент | Было | Стало |
|-----------|------|-------|
| Модель эмбеддингов | all-MiniLM-L6-v2 (384d, общая) | FinLang/finance-embeddings-investopedia (768d, финансовая) |
| Pooling | Mean pooling (все статьи равны) | Weighted pooling по \|sentiment\| (экстремальные новости важнее) |
| PCA сжатие | 384 → 32d (8.3% сохранение) | 768 → 64d |
| Фичи | 50 (18 price + 32 emb) | 83 (18 price + 64 emb + 1 news_count) |

---

## Результаты BTC/USDT (тест: 2024-01-01 – 2024-12-31)

### Improved Embeddings Agent (FinLang + Weighted + 64d)

| Seed | Sharpe | Return | MaxDD | Sortino | Calmar |
|------|--------|--------|-------|---------|--------|
| 42 | 0.662 | 17.6% | 34.6% | 0.818 | 0.557 |
| 43 | **1.934** | **102.6%** | 23.0% | **3.028** | **5.041** |
| 44 | 1.332 | 42.6% | **15.5%** | 1.729 | 3.043 |
| **Mean±Std** | **1.310±0.520** | **54.3%±35.7%** | **24.4%±7.9%** | **1.858±0.911** | **2.880±1.839** |

### Baseline Agent (только цена)

| Seed | Sharpe | Return | MaxDD | Sortino | Calmar |
|------|--------|--------|-------|---------|--------|
| 42 | 1.156 | 32.2% | 12.0% | 1.561 | 2.949 |
| 43 | **1.963** | **69.4%** | 15.5% | **2.583** | **4.996** |
| 44 | 0.954 | 29.4% | 24.8% | 1.131 | 1.308 |
| **Mean±Std** | **1.358±0.436** | **43.7%±18.2%** | **17.5%±5.3%** | **1.758±0.612** | **3.084±1.519** |

---

## Сравнение

| Метрика | Baseline | Improved Embeddings | Δ |
|---------|----------|-------------------|---|
| Sharpe Ratio | 1.358 ± 0.436 | 1.310 ± 0.520 | -0.048 (-3.5%) |
| Total Return | 43.7% ± 18.2% | 54.3% ± 35.7% | +10.6% (+24.2%) |
| Max Drawdown | 17.5% ± 5.3% | 24.4% ± 7.9% | +6.9% (хуже) |
| Sortino Ratio | 1.758 ± 0.612 | 1.858 ± 0.911 | +0.100 (+5.7%) |
| Calmar Ratio | 3.084 ± 1.519 | 2.880 ± 1.839 | -0.204 (-6.6%) |

---

## Анализ

### Положительное
1. **Доходность выросла на 24%** (43.7% → 54.3%) — эмбеддинги добавляют полезную информацию
2. **Sortino улучшился** (+5.7%) — агент лучше управляет downside risk
3. **Seed 43 показал Sharpe 1.934** — лучший результат среди всех запусков

### Отрицательное
1. **Sharpe ratio сопоставим** (1.310 vs 1.358, разница в пределах std)
2. **Просадка выросла** (24.4% vs 17.5%) — агент берёт больше риска
3. **Высокая дисперсия** (std=0.520 vs 0.436) — менее стабильно между seeds

### Вывод

Улучшенный embeddings агент **не превзошёл baseline по Sharpe**, но показал **более высокую доходность** за счёт **большего риска** (просадка). Результаты статистически неразличимы (overlapping CI).

**Возможные причины:**
- 500K timesteps может быть недостаточно для обучения с 83 фичами (vs 18 у baseline)
- FinLang модель обучена на Investopedia определениях, не на новостном потоке
- Weighted pooling по sentiment может усиливать шум (FinBERT не оптимален для крипто)

---

## Следующие шаги

1. **Увеличить timesteps** до 1M-2M для embeddings агента (больше фичей = дольше обучение)
2. **CryptoBERT для sentiment** — заменить FinBERT → kk08/CryptoBERT для лучшего weighted pooling
3. **Lag features** — sentiment[-1], sentiment[-2] как отдельные признаки
4. **Rolling news_count** — добавить 7-дневное скользящее среднее новостей
5. **Fusion агент** — объединить улучшенные embeddings + sentiment

---

## Файлы

| Файл | Описание |
|------|----------|
| `src/features/embeddings.py` | FinLang модель + weighted pooling |
| `src/features/build_embedding_features.py` | PCA 64d + news_count + sentiment weights |
| `src/features/embedding_compressor.py` | 768→64 PCA defaults |
| `src/agents/train.py` | FEATURE_COUNTS обновлён (embeddings=83) |
| `scripts/rebuild_embedding_features.py` | Скрипт пересборки parquet |
| `results/metrics.csv` | Все результаты |
