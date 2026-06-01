# Разбор кода: архитектура, конфигурация и карта модулей

> Обзор кода `dsr_experiment/` для защиты. Актуально для текущего пайплайна:
> **DQN (дискретные действия) + DSR-reward, 41 признак** (8 индикаторов + `sentiment_mean`
> + 32 PCA-эмбеддинга), аннуализация **√2190**.
>
> Детальный построчный путь новости — [`news_path_walkthrough.md`](news_path_walkthrough.md).
> Детальный разбор дашборда — [`dashboard_walkthrough.md`](dashboard_walkthrough.md).

---

## 0. Конфигурация — `config.yaml`

Все параметры эксперимента в одном файле; загружается в типизированные dataclass'ы
через `lib/config_loader.py` (загрузчик **терпим** к устаревшим/лишним ключам — старые
конфиги тоже грузятся).

| Секция | Ключи | Смысл |
|---|---|---|
| `data` | asset=`BTC/USDT`, timeframe=`4h`, пути к parquet/компрессору | Что и откуда брать |
| `periods` | `train` 2020–2023, `oos_2024`, `oos_2025` | Train/test split по датам |
| `news` | `hf_dataset: edaschau/bitcoin_news`, `dedup_threshold` | Источник новостного корпуса |
| `embeddings` | `raw_dim: 768`, **`compressed_dim: 32`** | Размерности FinLang → PCA |
| `sentiment` | `model_name: ProsusAI/finbert` | Модель тональности |
| `features` | `normalize_window: 30` | Окно rolling z-score |
| `env` | `window: 30`, `tx_cost: 0.001`, `sentiment_lambda: 0.3`, `action_space_type: discrete` | Параметры среды |
| `experiment` | `seeds: [42,123,7,11,99]`, **`algos: ["DQN"]`**, `oos_periods` | Что запускать |
| `agent_dqn` / `agent_sac` | гиперпараметры SB3 | 200k шагов, net_arch [128,128] |

> **Почему `compressed_dim: 32`.** У новостей «эффективная размерность» мала — первые
> компоненты PCA несут почти всю дисперсию. 32 хватает, а лишнее — шум. (Раньше было 64.)

> **Почему награда — только DSR.** Раньше в `env` были `reward_type` (basic/risk_adjusted/dsr),
> `allow_short`, `volatility_penalty`, `dsr_eta` — все эти ветки удалены. Осталась одна:
> **DSR + sentiment-бонус**. `dsr_eta` теперь константа в коде (`_DSR_ETA = 0.01`).

> **Почему `algos: ["DQN"]`.** PPO убран из кода целиком (был, но в финальный результат не
> вошёл). Основной агент — DQN; SAC поддерживается, но переобучается отдельно.

---

## 1. Карта кода по файлам

### Сборка данных
| Файл | Ключевые функции | Что делает |
|---|---|---|
| `build_data.py` | `build_train`, `build_oos`, `_build_baseline`, `_attach_nlp_features_minimal` | Оркестратор: OHLCV + новости → 41-фичный parquet |
| `lib/features/price.py` | `fetch_ohlcv`, **`add_technical_indicators_minimal`** (8 инд.), `rolling_zscore_normalize` | Цены + индикаторы + нормализация |
| `lib/features/news.py` | `load_news_from_hf`, `preprocess_news_4h` | Загрузка HF-корпуса + группировка в 4h-окна |
| `lib/features/sentiment.py` | `compute_sentiment_scores`, `_get_pipeline` | FinBERT-тональность ∈ [−1,+1] |
| `lib/features/embeddings.py` | `_get_model`, `compute_embeddings`, `EmbeddingCompressor` | FinLang (768d) + PCA-компрессор (768→32) |

### Обучение и оценка
| Файл | Ключевые функции | Что делает |
|---|---|---|
| `lib/data_loader.py` | `load_train`, `load_oos` | parquet → (матрица 41 фичи, цены `raw_close`, сигнал `sentiment_mean`); флаг `exclude_news` для аблации |
| `lib/env.py` | `TradingEnv`, `_compute_dsr`, `_get_obs` | Среда: DSR-награда + sentiment-бонус; Discrete(3) для DQN |
| `lib/train.py` | `train_agent`, `_algo_kwargs` | SB3 **DQN/SAC** (PPO убран) |
| `lib/backtest.py` | `run_backtest`, `load_model` | Детерминированный rollout → метрики + equity |
| `lib/metrics.py` | `compute_metrics` | Sharpe/Sortino/MaxDD/Calmar/… аннуализация **√2190** |
| `lib/bootstrap.py` | `bootstrap_ci` | 95% CI по сидам |
| `lib/interpretability.py` | `identify_feature_groups`, `permutation_importance` | Важность групп фич (цена/sentiment/эмбеддинги) |
| `run.py` | `train_phase`, `oos_phase` | Оркестратор: train × seeds + OOS |

> **Что удалено относительно старой версии.** `lib/features/lag.py` (лаги/rolling — больше не
> нужны, окно 30 несёт временной контекст); в `interpretability.py` убраны вспомогательные
> `action_feature_correlation` / `action_distribution_by_sentiment_regime`; в `env`/`train`/`run`
> убраны PPO, VecNormalize, ветки reward, `allow_short`.

---

## 2. Поток данных (одним взглядом)

```
OHLCV (ccxt/Binance, 4h)                Новости (HF: edaschau/bitcoin_news)
   │                                        │
   │ add_technical_indicators_minimal       │ preprocess_news_4h (4h-окна)
   │ → 8 индикаторов                        ├─ FinBERT → sentiment ∈ [−1,+1]
   │ rolling z-score (окно 30, трейлинг)    └─ FinLang → 768d
   │                                            │ PCA.fit (только train) → 768→32
   │                                            │ взвеш. по тональности среднее → 32
   ▼                                            ▼
   └────────►  features.parquet: 8 инд + sentiment_mean + emb_0..31 = 41 фича
                     │
                     │ data_loader: матрица 41 фичи + цены + сигнал sentiment_mean
                     ▼
              env: окно 30 строк → наблюдение 30×41+1 = 1231
                     │  reward = DSR(R_t) + λ·sentiment·log_return
                     ▼
              DQN (Discrete: Hold/Buy/Sell) × 5 сидов → OOS-метрики
```

---

## 3. Торговая среда `lib/env.py` (подробно)

Одна ветка награды — **DSR + sentiment-бонус** (старые basic/risk_adjusted удалены).

**Действие → доля** (дискретно, DQN):
```python
if a == 1:   allocation = 1.0          # Buy — всё в BTC
elif a == 2: allocation = 0.0          # Sell — всё в деньги
else:        allocation = prev_alloc   # Hold
```

**Базовая отдача с комиссией:**
```python
log_return = log(price_next / price_curr)
R_t = log_return * allocation − tx_cost * |Δallocation|      # tx_cost = 0.001
```

**DSR — Differential Sharpe Ratio** (Moody & Saffell, `_compute_dsr`):
```python
dA = R_t − A;  dB = R_t² − B;  denom = B − A²
reward = (B·dA − 0.5·A·dB) / denom^1.5          # при denom>0
A += 0.01·dA;  B += 0.01·dB                      # η = 0.01 (константа)
```
> DSR вознаграждает прирост доходности **с поправкой на риск** инкрементально, на каждом
> шаге — не нужно ждать конца эпизода, чтобы посчитать Sharpe.

**Sentiment-бонус:** `reward += sentiment_lambda · sentiment_t · log_return_t` (λ = 0.3).
Так новости влияют дважды: через наблюдение (эмбеддинги + скаляр) и через форму награды.

**Наблюдение** (`_get_obs`): последние 30 строк × 41 фича, сплющенные, + текущая доля = **1231**.

---

## 4. Метрики `lib/metrics.py` — аннуализация √2190

```python
PERIODS_PER_YEAR_4H = 365 * 6   # 2190 четырёхчасовых баров в году (крипта 24/7)
sharpe  = mean(r)/std(r) * sqrt(2190)
sortino = mean(r)/downside_std(r) * sqrt(2190)
```
> **Почему √2190, а не √365.** Доходности считаются **по 4h-барам** (один шаг env = один бар).
> Стандартная аннуализация Sharpe = √N × per-period Sharpe, где N — число периодов в году.
> Для 4h-баров N = 365×6 = 2190. (Раньше было √365 — занижало Sharpe в ~2.45×.)

Все метрики сохранены: total/annualized return, Sharpe, Sortino, MaxDD, Calmar, win_rate,
profit_factor, time_in_market. `bootstrap.py` строит 95% CI по сидам.

---

## 5. Честность результата (нет утечки из будущего)

1. **Нормализация строго в прошлое.** `rolling_zscore_normalize` (окно [t−29, t]) использует
   только прошлое и текущий бар — pandas `.rolling()` каузален.
2. **PCA обучается только на train** (`build_train`) и применяется к OOS без переобучения.

---

## 6. Ответы на возможные вопросы

**«Сколько признаков и почему так мало?»**
41 = 8 индикаторов + 1 `sentiment_mean` + 32 эмбеддинга. По меркам литературы (FinRL, SAPPO)
этого достаточно; больше — переобучение на шум. Наблюдение агента = 30×41+1 = 1231.

**«Почему DQN, а не PPO/SAC?»**
DQN с дискретными действиями (Hold/Buy/Sell) хорошо подходит для одного актива и небольшого
датасета (FinRL-Meta: меньше параметров → быстрее учится, меньше переобучается). PPO убран.
SAC (непрерывная доля) поддерживается для сравнения, но основной результат — на DQN.

**«Почему один скаляр тональности, а не несколько агрегатов?»**
Раньше было 5 (`max/min/mean/std/spread`) + `news_count` — сильно скоррелированы. Осталась
`sentiment_mean`: SAPPO показывает, что одного скаляра достаточно, остальное зашумляет state.

**«Почему PCA до 32 и тот же на OOS?»**
PCA — линейное преобразование, обученное на train. Переобучить его на OOS = утечка будущего.
32 компоненты сохраняют почти всю дисперсию новостных эмбеддингов.
