# Дашборд: разбор кода — что откуда берётся и зачем

Подробный разбор архитектуры дашборда для подготовки к защите. Все ссылки на файлы кликабельны.

> **Актуально для текущего пайплайна:** 41 признак (8 индикаторов + `sentiment_mean` + 32 эмбеддинга),
> PCA 768→32, ансамбль **DQN × 5 сидов** из снапшота `run_2026-06-01_minimal_dqn5`. Наблюдение
> агента = 30 × 41 + 1 = **1231** число. Лаги/rolling и 5 sentiment-агрегатов убраны.

---

## Структура проекта

```
dsr_experiment/
├── dashboard/
│   ├── app.py                      ← главная страница
│   ├── pages/
│   │   ├── 1_Strategy.py           ← вкладка «Стратегия»
│   │   ├── 2_Validation.py         ← вкладка «Валидация»
│   │   └── 3_News_Analyzer.py      ← вкладка «Анализатор новости»
│   ├── utils/                      ← общие модули (12 файлов)
│   └── cache/                      ← локальный кэш новостей
├── data/
│   ├── raw/ohlcv.parquet           ← кэш свечей
│   ├── raw/news.parquet            ← обучающий новостной корпус
│   └── train/compressor.pkl        ← обученный PCA 768→32
├── experiments/
│   └── run_2026-06-01_minimal_dqn5/  ← снимок: DQN × 5 сидов (41 фича)
│       ├── models/
│       │   └── DQN/DQN_seed42_…/model.zip   (5 сидов)
│       └── results/
└── lib/                            ← переиспользуется обучающим кодом
    ├── features/                   ← FinBERT, FinLang, PCA, 8 индикаторов
    └── env.py                      ← торговая среда (используется в backtest)
```

> **Зачем такая структура.** Дашборд намеренно переиспользует те же модули `lib/`, что и обучение. Это гарантирует, что на этапе инференса признаки считаются точно так же, как при обучении. Если бы я скопировал код в `dashboard/utils/` отдельно, рано или поздно версии разъехались бы, и модель получала бы признаки в чуть другом формате — это типичная причина «работает на train, не работает в проде».

---

## 1. Источники данных

### 1.1. Цены биткоина (OHLCV)

**Модуль:** [`dashboard/utils/live_data.py`](../dashboard/utils/live_data.py), функция `fetch_live_ohlcv()`

Свечи берутся **с биржи Binance** через библиотеку `ccxt`. Если интернет недоступен — fallback на кэш `data/raw/ohlcv.parquet`.

```python
@st.cache_data(ttl=900)
def fetch_live_ohlcv(symbol="BTC/USDT", timeframe="4h",
                     lookback_bars=120, use_live=True):
    if use_live:
        from lib.features.price import fetch_ohlcv
        df = fetch_ohlcv(symbol, start, end, timeframe=timeframe)
        if len(df) >= 31:
            return df, "live"
    # Fallback на кэш
    df = pd.read_parquet(RAW_OHLCV).tail(lookback_bars)
    return df, "cached_raw"
```

> **Зачем кэш на 15 минут (`ttl=900`).** Свеча 4-часовая закрывается раз в 4 часа. Делать запрос к Binance чаще, чем раз в 15 минут, бессмысленно: данные не изменятся, а лимит на запросы можно превысить. 15 минут — компромисс: пользователь видит свежую свечу через 15 минут после открытия новой.

> **Зачем fallback на кэш.** Дашборд должен открываться и работать даже без интернета (например, при демонстрации на защите без Wi-Fi). Кэш `data/raw/ohlcv.parquet` готовится один раз при первом запуске обучения и остаётся в репозитории.

### 1.2. Свежие новости

**Модуль:** [`dashboard/utils/news_feed.py`](../dashboard/utils/news_feed.py):
- NewsAPI: константы и URL — строки **26–29**, функция `fetch_newsapi_once()` — строки **124–202**
- RSS: константы `RSS_SOURCES` — строки **31–34**, функция `fetch_rss_once()` — строки **67–102**
- Объединение двух источников: `fetch_news_once()` — строки **205–223**
- Сохранение в parquet: `save_feed()` / `merge_new()` — строки **286–304**
- Регулярка BTC-ключевиков — строки **36–39**
- Hash для дедупликации `_uid()` — строки **57–58**

Два независимых источника:

**A. NewsAPI** (платный, ключ в `.env`):
```python
NEWSAPI_KEY = os.environ.get("NEWSAPI_KEY", "").strip()
NEWSAPI_URL = "https://newsapi.org/v2/everything"
NEWSAPI_LOOKBACK_DAYS = 5    # последние 5 дней

params = {
    "q": "bitcoin OR btc OR cryptocurrency",
    "from": from_iso, "to": to_iso,
    "language": "en", "sortBy": "publishedAt",
    "pageSize": 100, "apiKey": NEWSAPI_KEY,
}
```

> **Зачем именно NewsAPI.** Это самый дешёвый и простой способ получить структурированный поток англоязычных финансовых новостей с историей. Free tier позволяет 100 запросов в день — достаточно для одного дашборда. Альтернатива (Bloomberg, Refinitiv) стоит тысячи долларов в месяц.

> **Зачем глубина 5 дней.** Окно состояния агента — 30 четырёхчасовых баров = 5 дней. Брать больше новостей бессмысленно: они не попадут в state.

**B. RSS-фиды** (бесплатно, без ключа):
```python
RSS_SOURCES = [
    ("CoinDesk", "https://www.coindesk.com/arc/outboundfeeds/rss/?outputType=xml"),
    ("Cointelegraph", "https://cointelegraph.com/rss"),
]
```

> **Зачем два источника параллельно.** NewsAPI может не быть ключа или превышен лимит. RSS — гарантированный fallback без ключей. Если работают оба — результаты дедуплицируются и объединяются.

Каждая статья фильтруется регуляркой по BTC-ключевикам:
```python
BTC_KEYWORDS = re.compile(
    r"\b(bitcoin|btc|crypto|cryptocurrenc(y|ies)|satoshi)\b",
    re.IGNORECASE,
)
```

> **Зачем фильтр.** RSS-фид CoinDesk даёт новости про **всю** криптоиндустрию: альткоины, NFT, регулирование Ethereum и т. д. Модель обучена реагировать только на BTC-новости, поэтому остальное надо отсечь.

Дедупликация — по md5-хэшу `title + link`:
```python
def _uid(title: str, link: str) -> str:
    return hashlib.md5(f"{title}|{link}".encode()).hexdigest()[:16]
```

> **Зачем хэш по `title + link`.** Одна и та же новость может приходить и через NewsAPI, и через RSS. Хэш заголовка + ссылки — простой способ идентифицировать одну и ту же статью.

### 1.3. Обученные модели

**Модуль:** [`dashboard/utils/snapshot.py`](../dashboard/utils/snapshot.py): регулярка имён `_MODEL_DIR_RE` — строка **17**, `discover_models()` — строки **38–74**, `load_aggregate_csv()` — строки **94–99**, `load_seed_npz()` — строки **103–113**.

Снимок эксперимента лежит в `experiments/run_2026-06-01_minimal_dqn5/models/`. Внутри подпапка `DQN/` с **5 моделями** (по одной на случайный сид: 42, 123, 7, 11, 99).

```python
@st.cache_data(ttl=300)
def discover_models(snapshot: str) -> dict[str, list[dict]]:
    """Сканирует experiments/{snapshot}/models/{ALGO}/{ALGO}_seed{S}_{TS}/model.zip"""
    out = {}
    for algo_dir in (SNAPSHOTS_DIR / snapshot / "models").iterdir():
        algo = algo_dir.name        # "DQN" или "SAC"
        entries = []
        for run_dir in algo_dir.iterdir():
            m = _MODEL_DIR_RE.match(run_dir.name)
            # Папки вида "DQN_seed42_20260421_004100"
            zip_path = run_dir / "model.zip"
            if zip_path.exists():
                entries.append({"seed": int(m.group("seed")),
                                "path": str(zip_path)})
        out[algo] = sorted(entries, key=lambda x: x["seed"])
    return out
```

> **Зачем «снимки» (snapshots).** Эксперимент может повторяться много раз. Каждая модель попадает в свою папку с временной меткой. Так можно держать несколько версий моделей одновременно и переключаться между ними без перезаписи.

### 1.4. PCA-компрессор

**Модуль:** [`dashboard/utils/model_loader.py`](../dashboard/utils/model_loader.py), функция `load_compressor()`.

Файл `data/train/compressor.pkl` обучен один раз на тренировочных данных. Дашборд просто его загружает:

```python
@st.cache_resource
def load_compressor():
    from lib.features.embeddings import EmbeddingCompressor
    return EmbeddingCompressor.load(str(TRAIN_COMPRESSOR))
```

> **Зачем тот же компрессор, а не пересчитывать.** PCA — это линейное преобразование с конкретной матрицей весов 768×32. Эта матрица подобрана так, чтобы максимально сохранить дисперсию **тренировочных** эмбеддингов. Если на инференсе применить другой PCA (например, переобученный на свежих данных), модель получит признаки в другой системе координат и выдаст случайные ответы. Это та же причина, по которой нельзя менять архитектуру нейросети между обучением и инференсом.

---

## 2. Обработка одной новости: пайплайн

Конкретный пример — пользователь ввёл новость:
> "The SEC has approved the first spot Bitcoin ETF, opening the door for large institutional investment into cryptocurrency."

**Модуль:** [`dashboard/utils/news_impact.py`](../dashboard/utils/news_impact.py): константы — строки **4–8**, `inject_news_features()` — строки **11–35**, `score_news_batch()` — строки **38–73**, `analyze_news_impact()` — строки **76–138**.

### Шаг 1. Тональность через FinBERT

```python
from lib.features.sentiment import compute_sentiment_scores

scores = [float(s) for s in compute_sentiment_scores(texts)]
# Для примера выше: scores = [0.84]
```

FinBERT (`ProsusAI/finbert` из HuggingFace) выдаёт три вероятности: `positive`, `neutral`, `negative`. Tональность вычисляется как:
```
s = P(positive) − P(negative)  ∈ [−1, +1]
```

> **Зачем именно FinBERT, а не обычный BERT.** Обычный BERT обучен на Википедии и книгах. Финансовые тексты он понимает плохо: «bear market» он воспринимает как «медведь + рынок», а не как «падающий рынок». FinBERT дообучен на финансовом корпусе Reuters и Bloomberg, поэтому правильно интерпретирует терминологию.

> **Зачем формула `P(pos) − P(neg)`, а не argmax.** Argmax даёт только метку класса. Разность вероятностей даёт **силу** тональности: новость с уверенным позитивом (0.95) важнее новости со слабым позитивом (0.55).

### Шаг 2. Смысловой вектор через FinLang

```python
from lib.features.embeddings import compute_embeddings

# Sentiment-weighted embedding: эмоционально сильные новости весят больше
signed_w = [s + np.sign(s) * 0.1 if s != 0 else 0.1 for s in scores]
raw_emb = compute_embeddings(texts, weights=signed_w)   # 768-мерный вектор
```

> **Зачем FinLang отдельно от FinBERT.** FinBERT даёт **направление** (хорошо/плохо). FinLang даёт **содержание** (про что именно новость). Две новости с одинаковым FinBERT-скором +0.9 могут быть совершенно разными: «SEC одобрил ETF» и «Tesla купила биткоинов». Агенту полезно различать их.

> **Зачем взвешенное среднее, а не простое.** В одном 4-часовом окне может быть 50 новостей: 49 о мелочах и 1 про объявление SEC. Простое среднее размывает сильный сигнал слабыми. Веса по силе тональности дают сильным новостям больший вклад в итоговый вектор.

### Шаг 3. Сжатие через PCA

```python
compressor = load_compressor()
emb = compressor.transform(raw_emb.reshape(1, -1))[0]   # 768 → 32
```

> **Зачем сжатие.** 768 чисел на окно × 30 окон = 23 040 признаков только от новостей — на 4 годах данных сеть мгновенно переобучается на шум. Сжатие до **32** + 8 индикаторов + 1 `sentiment_mean` = **41 признак** на бар; наблюдение 30×41 + 1 = **1231** — модель учится стабильно.

> **Зачем именно 32.** У новостей «эффективная размерность» мала — первые компоненты PCA несут почти всю дисперсию, а лишние измерения добавляют шум. 32 достаточно, чтобы сохранить смысл, и при этом компактно. (Раньше было 64 — оказалось избыточно.)

### Шаг 4. Сборка sentiment-агрегатов

```python
stats = {
    "sentiment_mean":   float(arr.mean()),
    "sentiment_max":    float(arr.max()),
    "sentiment_min":    float(arr.min()),
    "sentiment_std":    float(arr.std()) if len(arr) > 1 else 0.0,
    "sentiment_spread": float(arr.max() - arr.min()),
    "news_count":       float(len(texts)),
}
```

> **Что из этого реально видит модель.** В минимальном наборе агент получает **только `sentiment_mean`** — один скаляр тональности окна. Остальные агрегаты (`max/min/std/spread`, `news_count`) по-прежнему считаются, но идут **в интерфейс для наглядности**, а не в наблюдение. Литература (SAPPO, sentiment-augmented PPO) показывает, что одного скаляра тональности достаточно, а лишние агрегаты в state только зашумляют обучение.

### Шаг 5. Подстановка в состояние агента

**Модуль:** [`dashboard/utils/news_impact.py`](../dashboard/utils/news_impact.py) → `inject_news_features()`, строки **11–35**.

```python
def inject_news_features(features, feature_columns, stats, emb_64):
    """Подставляет sentiment_mean + emb_0..31 во ВСЕ строки окна."""
    out = np.array(features, dtype=np.float32, copy=True)
    col_idx = {name: i for i, name in enumerate(feature_columns)}

    for name in _NEWS_STAT_COLUMNS:
        if name in col_idx and name in stats:
            out[:, col_idx[name]] = float(stats[name])
    for i in range(_EMB_DIM):
        name = f"emb_{i}"
        if name in col_idx:
            out[:, col_idx[name]] = float(emb_64[i])
    return out
```

> **Зачем подставлять во ВСЕ 30 строк, а не только в последнюю.** Когда модель обучалась, новости были в каждом 4-часовом окне обучающего корпуса. То есть в state-матрице 30×F все строки имели реальные новостные признаки. Если на инференсе заполнить только последнюю строку, а остальные оставить нулями, state будет **не таким**, какой модель видела при обучении, — и она выдаст случайный ответ. Дублирование текущих новостей во все строки делает state ближе к обучающему распределению.

---

## 3. Подготовка состояния и прогон через ансамбль

### 3.1. Сборка фичей в реальном времени

**Модуль:** [`dashboard/utils/features_live.py`](../dashboard/utils/features_live.py): константы — строки **17–21**, `expected_feature_count()` — строки **24–30**, `expected_feature_columns()` — строки **33–36**, `build_live_features()` — строки **39–97**, `build_live_obs()` — строки **100–106**.

Логика 1-в-1 повторяет `build_data.py` из обучения:

```python
def build_live_features(ohlcv):
    # 1. 8 индикаторов (ema_26, macd, rsi_14, bb_width, atr_14, obv, stoch_k, return_1d)
    df = add_technical_indicators_minimal(ohlcv)

    # 2. Сохраняем "сырую" цену для бэктеста
    df["raw_close"] = df["close"].copy()

    # 3. Rolling z-score нормализация окном 30 баров
    df[cols_norm] = rolling_zscore_normalize(df[cols_norm], window=30)

    # 4. Нулевая инициализация новостных колонок (на барах без новостей)
    for i in range(EMB_DIM):          # EMB_DIM = 32
        df[f"emb_{i}"] = 0.0
    df["sentiment_mean"] = 0.0

    # 5. Берём ровно те 41 колонку, что в обучающем parquet, в том же порядке
    target_cols = expected_feature_columns()
    features = df.reindex(columns=target_cols).to_numpy(dtype=np.float32)

    return features, prices, df
```

> **Зачем сохранять `raw_close` отдельно.** Все остальные фичи нормализуются rolling z-score → центр около 0, разброс около 1. Но для расчёта реального P&L при бэктесте нужна **сырая** цена в долларах. Поэтому колонка `close` дублируется в `raw_close` до нормализации.

> **Зачем rolling z-score, а не глобальная нормализация.** Глобальная нормализация (по всему датасету) — это утечка будущего: на 2020 году state знал бы что-то о статистике 2024 года. Rolling по 30 барам — использует только прошлое, как агент в реальной торговле.

> **Зачем sanity check shape.** Это страховка. Если по ошибке индикатор добавит лишнюю колонку или одну из них переименует, state будет неправильной формы — модель примет любое значение, но ответ будет случайным. Проверка через `expected_feature_count()` сразу падает с понятной ошибкой.

### 3.2. Формирование observation

```python
def build_live_obs(features, prev_allocation=0.0, window=30):
    """Воспроизводит TradingEnv._get_obs:
    плющит последние 30 строк фичей и добавляет prev_allocation."""
    return np.append(features[-window:].flatten(), prev_allocation).astype(np.float32)
```

> **Зачем добавлять `prev_allocation` в observation.** Без этой информации агент не знает, в позиции он сейчас или в кэше. А действие зависит от текущего состояния: если уже в позиции — может надо удержать; если в кэше — может надо войти. Без `prev_allocation` агент не различал бы эти ситуации.

> **Зачем `flatten()`, а не подавать матрицу 30×F.** Используются MLP-сети (а не CNN/RNN). MLP принимает вектор фиксированной длины. `flatten` превращает 30×F в одномерный вектор длины 30·F.

### 3.3. Прогон через ансамбль из 5 моделей

**Модуль:** [`dashboard/utils/feed_decisions.py`](../dashboard/utils/feed_decisions.py): `_build_features()` — строки **~30–75**, `_run_ensemble()` — строки **78–113**, `_majority_winner()` — строки **116–124**, `decide_for_buckets()` — строки **127+**.

```python
def _run_ensemble(obs, seeds, algo, prev_alloc):
    """DQN: majority vote по BUY/HOLD/SELL.
       SAC: медиана непрерывных аллокаций."""
    allocs, labels = [], []
    for _seed, path in seeds:
        model = load_sb3_model(path, algo_hint=algo)
        action, _ = model.predict(obs, deterministic=True)
        if algo == "DQN":
            a = int(np.asarray(action).flatten()[0])
            if a == 1:
                allocs.append(1.0); labels.append("BUY")
            elif a == 2:
                allocs.append(0.0); labels.append("SELL")
            else:
                allocs.append(prev_alloc); labels.append("HOLD")
        else:  # SAC
            allocs.append(float(np.clip(action.flatten()[0], 0.0, 1.0)))

    if algo == "DQN":
        votes = {"BUY": labels.count("BUY"),
                 "HOLD": labels.count("HOLD"),
                 "SELL": labels.count("SELL")}
        winner = _majority_winner(votes)
        allocation = 1.0 if winner == "BUY" else 0.0 if winner == "SELL" else prev_alloc
        return {"allocation": allocation, "votes": votes, "winner": winner}

    return {"allocation": float(np.median(allocs))}  # SAC
```

> **Зачем `deterministic=True`.** Для DQN — argmax по Q-функции вместо ε-greedy. Для SAC — берётся mean политики, а не sample. Это нужно, чтобы при перезапуске дашборда на одних и тех же данных получалось одно и то же решение. Иначе пользователь будет видеть разные голоса от прогона к прогону, и это будет выглядеть как баг.

> **Зачем majority vote для DQN, а медиана для SAC.** У DQN action дискретный (3 варианта). У SAC непрерывный (число от 0 до 1). По дискретным значениям нельзя считать медиану осмысленно. По непрерывным — можно. Медиана устойчивее к выбросам, чем среднее (если 4 модели сказали 0.5, а 1 сказала 1.0 — среднее даст 0.6, а медиана 0.5).

> **Зачем ансамбль из 5 моделей, а не одна.** RL-обучение очень шумное: одна модель может случайно дать как очень хороший, так и провальный результат. Усреднение по 5 сидам показывает **типичное** поведение, а не лучшее или худшее. Кроме того, ансамбль даёт меру **уверенности**: 4–5 из 5 за BUY — надёжный сигнал; 2-2-1 — рынок в неопределённом состоянии.

### 3.4. Загрузка моделей: трюк с replay buffer

**Модуль:** [`dashboard/utils/model_loader.py`](../dashboard/utils/model_loader.py), функция `load_sb3_model()`.

```python
@st.cache_resource
def load_sb3_model(model_path, algo_hint=None):
    """Off-policy модели (SAC, DQN) сериализуют размер replay buffer.
    Наивный .load() пытается перераспределить буфер для обучения
    (300k × obs_dim ≈ 3 ГБ на модель). Для инференса буфер не нужен —
    переопределяем buffer_size=1 через custom_objects.
    """
    from stable_baselines3 import PPO, SAC, DQN
    custom = {"buffer_size": 1}

    hint_map = {"SAC": SAC, "DQN": DQN, "PPO": PPO}
    if algo_hint in hint_map:
        cls = hint_map[algo_hint]
        return cls.load(model_path, device="cpu", custom_objects=custom)
```

> **Зачем `custom_objects={"buffer_size": 1}`.** Без этого 5 моделей DQN съели бы **~15 ГБ оперативки** только на буферах опыта (≈3 ГБ каждая), которые при инференсе вообще не используются. С `buffer_size=1` модель загружается за миллисекунды и весит десятки мегабайт.

> **Зачем `device="cpu"`.** На CPU инференс одной MLP-сети с парой сотен нейронов выполняется за миллисекунды. Запуск модели на GPU для 5 моделей даст overhead на копирование данных GPU↔RAM больший, чем сам инференс.

> **Зачем `@st.cache_resource`.** Streamlit перерендеривает страницу при каждом действии пользователя. Без кэширования модель грузилась бы из файла каждый раз — это сотни миллисекунд на модель × 5 моделей = секунды задержки на каждый клик. `@st.cache_resource` держит модель в памяти процесса.

---

## 4. News Analyzer: сравнение «с новостью» vs «без»

**Модуль:** [`dashboard/utils/news_impact.py`](../dashboard/utils/news_impact.py) → `analyze_news_impact()`, строки **76–138**.

Идея: показать **чистый вес** новости.

```python
def analyze_news_impact(background, tested):
    """
    decision_without — что модель думает СЕЙЧАС (на текущем 5-дневном фоне).
    decision_with    — что она думала бы, если бы единственным сигналом
                       была эта новость.
    """
    ohlcv = fetch_ohlcv()
    features, _, _ = build_live_features(ohlcv)

    # «Без» — текущий новостной фон последних 5 дней.
    if background:
        bg = score_news_batch(background)
        feat_without = inject_news_features(features, columns, bg["stats"], bg["emb_64"])
    else:
        feat_without = features

    obs_without = build_live_obs(feat_without)
    decision_without = _run_ensemble(obs_without, seeds_paths, "DQN", 0.0)

    # «С» — ТОЛЬКО проверяемая новость как единственный сигнал.
    tested_only = score_news_batch(tested)
    feat_with = inject_news_features(features, columns, tested_only["stats"], tested_only["emb_64"])
    obs_with = build_live_obs(feat_with)
    decision_with = _run_ensemble(obs_with, seeds_paths, "DQN", 0.0)

    return {
        "decision_without": decision_without,
        "decision_with":    decision_with,
    }
```

> **Зачем сравнивать не `background+tested` vs `background`, а `background` vs `только tested`.** Если запустить ансамбль на полном фоне из ~100 свежих новостей и просто добавить одну новую, она растворится в усреднении: `sentiment_mean` и средний эмбеддинг почти не сдвинутся, ансамбль выдаст тот же ответ. Чтобы увидеть **чистый вес** новости, надо подать её **в одиночку**.

> **Зачем вообще брать фон из 5 дней.** Без фона модель видит state без новостей вообще (нулевые sentiment-колонки), что нетипично для обучающего распределения и даст неправильный baseline. Фон даёт realistic baseline, относительно которого считается «насколько эта новость двигает решение».

---

## 5. Карта модулей

| Модуль | Файл | Назначение | Зачем |
|---|---|---|---|
| `app.py` | [link](../dashboard/app.py) | Главная страница | Навигация + сводка лучших моделей |
| `paths.py` | [link](../dashboard/utils/paths.py) | Константы путей | Один источник истины для путей, чтобы не было опечаток в строках |
| `live_data.py` | [link](../dashboard/utils/live_data.py) | OHLCV с Binance | Свежие цены для всех вкладок |
| `news_feed.py` | [link](../dashboard/utils/news_feed.py) | Новости NewsAPI + RSS | Свежие новости для Strategy и News Analyzer |
| `news_live.py` | [link](../dashboard/utils/news_live.py) | Грузит FinLang-эмбеддер | Singleton-загрузка ML-модели в память |
| `model_loader.py` | [link](../dashboard/utils/model_loader.py) | Грузит SB3-модели и PCA | Трюк с `buffer_size=1` против OOM |
| `model_catalog.py` | [link](../dashboard/utils/model_catalog.py) | Каталог снимков | Список доступных «версий» моделей |
| `snapshot.py` | [link](../dashboard/utils/snapshot.py) | Папки `experiments/` | Парсинг имён `DQN_seed42_…` |
| `features_live.py` | [link](../dashboard/utils/features_live.py) | Live-сборка фичей | Воспроизведение pipeline обучения |
| `feed_decisions.py` | [link](../dashboard/utils/feed_decisions.py) | Прогон ансамбля | Majority vote / медиана |
| `news_impact.py` | [link](../dashboard/utils/news_impact.py) | News Analyzer | Сравнение «с/без» новости |
| `style.py` | [link](../dashboard/utils/style.py) | CSS Streamlit | Кастомизация UI |

---

## 6. Полная цепочка для одного запроса

```
[Текст новости]
      │
      ▼
score_news_batch()
      │   FinBERT → sentiment ∈ [−1, +1]
      │   FinLang → embedding 768d
      │   PCA compressor.transform() → 32d
      │   sentiment_mean (для модели) + остальные агрегаты (для UI)
      ▼
build_live_features()
      │   OHLCV (с Binance) → 8 indicators → z-score normalize
      ▼
inject_news_features()
      │   подставляет sentiment_mean + emb_0..31 в ВСЕ 30 строк окна
      ▼
build_live_obs()
      │   flatten(30 × 41 признак) + [prev_allocation] = 1231
      ▼
_run_ensemble()
      │   5 раз model.predict(obs)
      │   majority vote по {BUY, HOLD, SELL}
      ▼
decision_with / decision_without
      │
      ▼
Streamlit рисует распределение голосов
```

---

## 7. Как объяснить за 30 секунд на защите

> «Дашборд устроен как pipeline. Свежие свечи биткоина приходят с Binance через библиотеку ccxt, новости — с NewsAPI и RSS-фидов CoinDesk и Cointelegraph. Каждая новость прогоняется через две модели: FinBERT даёт оценку тональности, FinLang — смысловой вектор из 768 чисел. Этот вектор сжимается тем же PCA-компрессором, что использовался при обучении, до 32 признаков. Вместе с 8 индикаторами и тональностью получается 41 признак на бар; состояние агента — матрица из последних 30 четырёхчасовых баров (1231 число). Состояние подаётся параллельно в ансамбль из 5 обученных моделей DQN, каждая независимо голосует BUY, HOLD или SELL. Распределение голосов и показывается пользователю.»

---

## 8. Ответы на возможные вопросы

**«Почему вы используете тот же PCA, что и при обучении?»**
PCA — это линейное преобразование, обученное на тренировочных данных. Если на инференсе применить другой PCA (например, переобученный на свежих данных), модель получит признаки в другом пространстве и выдаст случайные ответы.

**«Зачем кэширование на 15 минут?»**
Свечи 4-часовые — новая появляется не чаще, чем раз в 4 часа. 15 минут — компромисс между свежестью и снижением нагрузки на API Binance.

**«Почему ансамбль, а не одна лучшая модель?»**
Одна модель — шумный ответ, зависит от случайной инициализации. Распределение голосов 5 моделей даёт уверенность сигнала: 4–5 из 5 за BUY — надёжно, 2-2-1 — рынок в неопределённости.

**«Что делает News Analyzer и зачем «без новости» vs «с новостью»?»**
Одна новость в потоке из 100 фоновых растворяется в усреднении. Поэтому показывается чистый вес: один прогон на «текущем фоне 5 дней», второй — на «только эта новость». Разница голосов — вклад именно этой новости.

**«Можно ли торговать по дашборду на реальные деньги?»**
Нет, это демонстрационный прототип. Для реальной торговли нужны: учёт проскальзывания, обработка сбоев API, тестирование на большем интервале, риск-менеджмент.
