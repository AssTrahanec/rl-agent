# Design: ETH/USDT 4h Embeddings Model (аналог BTC)

**Date:** 2026-04-14  
**Status:** Approved

---

## Цель

Обучить RL-агентов (SAC, PPO, A2C) на ETH/USDT 4h данных с embeddings + DSR reward — полный аналог лучших BTC-моделей из планов `2026-04-12-embeddings-3agents` и `2026-04-13-dsr-sentiment-reward`.

---

## Архитектура

### Данные

| Файл | Содержание |
|------|-----------|
| `data/raw/eth_4h_ohlcv.parquet` | ETH/USDT 4h OHLCV с Binance, 2020-01-01–2024-12-31 |
| `data/raw/eth_news.parquet` | Ethereum-новости с HuggingFace `ashraq/crypto-news`, фильтр по ключевым словам ETH |
| `data/processed/eth_4h_embedding_features.parquet` | Итоговый feature файл, 101 колонка (6 OHLCV + 95 features), аналог `btc_4h_embedding_features.parquet` |
| `data/processed/eth_4h_compressor.pkl` | PCA компрессор 384→64, обученный только на train-новостях 2020–2023 |

### Schema (101 колонка — идентично BTC 4h)

**OHLCV (6):** `open, high, low, close, volume, raw_close`

**Price/Tech (17):** `sma_7, sma_25, ema_12, ema_26, macd, macd_signal, macd_hist, rsi_14, bb_upper, bb_lower, bb_mid, bb_width, atr_14, obv, stoch_k, stoch_d, return_1d, return_5d`

**Embeddings (64):** `emb_0 .. emb_63`

**NLP/lag (14):** `news_count, sentiment_max, sentiment_min, sentiment_spread, news_count_lag1, news_count_lag2, news_count_roll7, emb_0_lag1, emb_0_lag2, emb_1_lag1, emb_1_lag2, emb_2_lag1, emb_2_lag2`

**Итого feature cols (non-OHLCV): 95**

### Параметры обучения

| Параметр | Значение |
|----------|----------|
| Asset | ETH/USDT |
| Timeframe | 4h |
| Train | 2020-01-01 – 2023-12-31 |
| OOS | 2024-01-01 – 2024-12-31 |
| Window | 30 |
| tx_cost | 0.001 |
| reward_type | `dsr` |
| sentiment_lambda | 0.1 |
| net_arch | [128, 128] |
| Seeds | [42, 123, 7, 2024, 99] |

**SAC:** lr=7.3e-4, buffer=300k, learning_starts=10k, batch=256, tau=0.02, train_freq=8, gradient_steps=8, use_sde=True, total_timesteps=200k, device=cuda

**PPO:** lr=3e-4, lr_schedule=linear, n_steps=2048, batch=64, n_epochs=10, gae_lambda=0.95, clip=0.2, ent_coef=0.01, use_sde=True, total_timesteps=500k, device=cpu

**A2C:** lr=7e-4, lr_schedule=linear, n_steps=16, gae_lambda=1.0, ent_coef=0.01, use_sde=True, normalize_advantage=True, total_timesteps=500k, device=cpu

---

## Компоненты (новые файлы)

### 1. `scripts/collect_eth_news.py`
Загружает `ashraq/crypto-news` с HuggingFace, фильтрует строки содержащие "ethereum", "ETH" или "ether" в заголовке/тексте, сохраняет в `data/raw/eth_news.parquet`.

Колонки на выходе: `[title, text, date]` — стандарт для `news_preprocessor`.

### 2. `scripts/build_eth_4h_dataset.py`
По образцу `scripts/build_4h_dataset.py`. Шаги:
1. Скачать ETH 4h OHLCV (`fetch_ohlcv("ETH/USDT", ...)`)
2. Технические индикаторы + raw_close + rolling z-score normalization
3. Загрузить `eth_news.parquet`, запустить `preprocess_news_4h()`
4. Вычислить raw embeddings (all-MiniLM-L6-v2) — только на train-периоде (≤ 2023-12-31) для fit PCA
5. Fit нового `EmbeddingCompressor(input_dim=384, output_dim=64)`, сохранить в `eth_4h_compressor.pkl`
6. Transform всех новостей (train + test) через fitted compressor
7. Добавить lag/rolling features
8. Сохранить `eth_4h_embedding_features.parquet`

**Важно:** PCA компрессор обучается только на train-данных (2020-01-01–2023-12-31), применяется ко всему периоду.

### 3. `src/agents/config.py` — три новые factory-функции
```python
sac_eth_embeddings_dsr_config(seed: int) -> AgentConfig
ppo_eth_embeddings_dsr_config(seed: int) -> AgentConfig
a2c_eth_embeddings_dsr_config(seed: int) -> AgentConfig
```
Идентичны BTC DSR конфигам, но ничего не меняется в самом AgentConfig — asset и timeframe передаются в `train_agent()`.

### 4. `scripts/train_eth_embeddings.py`
Оркестратор обучения + OOS. По образцу `scripts/run_dsr_experiment.py`:
- Обучает SAC × 5 seeds, PPO × 5 seeds, A2C × 5 seeds на ETH 4h train (2020–2023)
- OOS backtest 2024 через `run_backtest()`
- Сохраняет `results/oos_2024_eth_embeddings.csv`
- Печатает сравнительную таблицу (Sharpe, Return, MaxDD per algo)

---

## Что не меняем

- `TradingEnv` — уже поддерживает DSR reward + sentiment_signal
- `train_agent()` — уже поддерживает ETH/USDT + timeframe="4h"
- `load_features_for_agent()` — уже маппит ETH/USDT → `eth_4h_embedding_features.parquet`
- `run_backtest()` — без изменений
- Все существующие BTC модели и результаты — не трогаем

---

## Ожидаемые выходы

| Файл | Описание |
|------|---------|
| `data/raw/eth_news.parquet` | ETH-новости с HF |
| `data/processed/eth_4h_embedding_features.parquet` | 101 cols, ~10956 rows, 2020–2024 |
| `data/processed/eth_4h_compressor.pkl` | PCA 384→64, fit на 2020–2023 |
| `results/oos_2024_eth_embeddings.csv` | 15 строк (SAC×5 + PPO×5 + A2C×5), OOS метрики |

---

## Критерии успеха

- `eth_4h_embedding_features.parquet`: shape (~10956, 101), 95 feature cols, 0 NaN, `news_count` ненулевой для большинства строк
- Обучение завершается без ошибок для всех 15 моделей
- `oos_2024_eth_embeddings.csv` содержит 15 строк с корректными метриками (Sharpe, total_return, max_drawdown)
- PCA компрессор обучен только на train-данных (leakage check)
