# Подробный разбор кода: путь новости и живой дашборд

> **Назначение.** Доклад для защиты ВКР. Можно зачитывать вслух: каждый шаг — это
> «простыми словами» → реальный кусок кода → построчный разбор → «зачем». Код процитирован
> прямо в документе, поэтому смотреть можно даже без открытого редактора.
>
> **Номера строк** указаны по текущему состоянию кода (ветка `news-analyzer`). Если после
> правок номера разойдутся — ищите по имени функции (оно дано в каждом блоке).

**О чём система.** Торговый агент на BTC/USDT принимает решение «купить / держать / продать»
раз в 4 часа, опираясь на цену **и** новостной фон. Новости проходят через две нейросети:
**FinBERT** (тональность) и **FinLang** (смысл), сжимаются методом PCA и подаются агенту
(обучение с подкреплением, RL) вместе с 8 ценовыми индикаторами.

---

## Содержание

- [Общая схема пути новости](#общая-схема)
- [Шаг 1. Загрузка новостей](#шаг-1--загрузка-новостей--libfeaturesnewspy)
- [Шаг 2. Группировка в 4-часовые окна](#шаг-2--группировка-в-4-часовые-окна)
- [Шаг 3. Тональность (FinBERT)](#шаг-3--тональность-новости-finbert)
- [Шаг 4. Смысл (FinLang-эмбеддинги)](#шаг-4--смысл-новости-finlang)
- [Шаг 5. Обучение сжатия (PCA, только на train)](#шаг-5--обучение-сжатия-pca)
- [Шаг 6. Взвешивание по тональности и сжатие](#шаг-6--взвешивание-по-тональности-и-сжатие)
- [Шаг 7. Сборка таблицы признаков (41 признак)](#шаг-7--сборка-таблицы-признаков)
- [Шаг 8. Загрузка признаков в модель](#шаг-8--загрузка-признаков-в-модель--libdata_loaderpy)
- [Шаг 9. Наблюдение и награда агента](#шаг-9--наблюдение-и-награда-агента--libenvpy)
- [Почему нет утечки из будущего](#почему-нет-утечки-из-будущего)
- [Дашборд](#дашборд--живая-демонстрация)
- [Словарь терминов](#словарь-терминов)

---

## Общая схема

```
   Новостной датасет (HuggingFace: edaschau/bitcoin_news)
        │
        ▼
   [1] load_news_from_hf        →  таблица [текст, дата] за период
        │
        ▼
   [2] preprocess_news_4h       →  {4h-окно → список текстов}
        │
        ├──► [3] FinBERT   каждый текст → тональность ∈ [−1, +1]
        │
        └──► [4] FinLang   каждый текст → вектор 768d (смысл)
                 │
                 ├─ [5] PCA.fit(все train-вектора)        →  compressor.pkl (768→32)
                 └─ [6] вектора окна × тональность → среднее → PCA  →  32 числа
        │
        ▼
   [7] строка окна: sentiment_mean + emb_0..emb_31  (+ 8 индикаторов)  =  41 признак
        →  features.parquet
        │
        ▼
   [8] data_loader: parquet → матрица признаков + цены + сигнал тональности
        │
        ▼
   [9] env: окно 30 строк = наблюдение (1231) + sentiment-бонус в награде
```

---

## Шаг 1 — Загрузка новостей · `lib/features/news.py`

**Простыми словами.** Берём открытый датасет биткойн-новостей, приводим все даты к UTC,
выкидываем битые даты и дубликаты заголовков, оставляем только новости нужного периода.

**Код** (`load_news_from_hf`, строки 12–46):

```python
12  def load_news_from_hf(dataset_name, start, end,
13                        text_col="article_text", date_col="date_time", split="train"):
24      from datasets import load_dataset
26      ds = load_dataset(dataset_name)
27      df = ds[split].to_pandas()                      # → pandas-таблица
        # переименование колонок источника к единому виду text/date
37      df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
38      df = df.dropna(subset=["date"])                 # убрать битые даты
40          df = df.drop_duplicates(subset=["title"])   # убрать повторы по заголовку
42      start_ts = pd.Timestamp(start, tz="UTC")
43      end_ts = pd.Timestamp(end, tz="UTC")
44      df = df[(df["date"] >= start_ts) & (df["date"] <= end_ts)]
46      return df[["text", "date", ...]]
```

**Разбор.**
- Стр. 26–27: грузим датасет с HuggingFace и превращаем в таблицу.
- Стр. 37: `to_datetime(..., utc=True)` — единый часовой пояс; `errors="coerce"` помечает
  битые даты как пустые, стр. 38 их удаляет.
- Стр. 40: дедуп по заголовку — грубая чистка точных дублей.
- Стр. 42–44: фильтр по диапазону периода (обучающего или тестового).

**Зачем.** Нужна чистая таблица «текст + дата», синхронная по времени с ценами.

---

## Шаг 2 — Группировка в 4-часовые окна

**Простыми словами.** Цена идёт 4-часовыми свечами. Каждую новость округляем вниз до начала
её 4-часового окна и собираем все новости окна в один список.

**Код** (`preprocess_news_4h`, строки 49–66):

```python
49  def preprocess_news_4h(df):
55      df = df.copy()
56      df["date"] = df["date"].dt.floor("4h")          # округлить вниз: 00:00,04:00,08:00...
57      grouped = (
58          df.groupby("date")["text"]
59            .apply(list)                               # все тексты окна → список
61            .rename(columns={"text": "texts"})
62            .sort_values("date")
63      )
66      return grouped                                   # → [date, texts]
```

**Разбор.**
- Стр. 56: `dt.floor("4h")` ставит всем новостям одного интервала одинаковую метку времени.
- Стр. 58–59: `groupby + apply(list)` собирает тексты окна в список `texts`.

**Зачем.** Агент решает раз в 4 часа — новостной фон должен быть агрегирован по той же сетке.

---

## Шаг 3 — Тональность новости (FinBERT)

**Простыми словами.** Каждую новость прогоняем через FinBERT (нейросеть для финансовых
текстов). Она говорит: позитив / негатив / нейтрал и уверенность. Превращаем в одно число
от −1 до +1.

**Код** (`lib/features/sentiment.py`, `compute_sentiment_scores`, строки 28–35):

```python
28  def compute_sentiment_scores(texts, model=None):
30      if not texts:
31          return []
32      pipe = model if model is not None else _get_pipeline()   # ProsusAI/finbert
33      results = pipe(texts, truncation=True, max_length=512)
34      label_map = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
35      return [float(r["score"] * label_map.get(r["label"].lower(), 0.0)) for r in results]
```

**Разбор.**
- Стр. 32: `_get_pipeline()` лениво грузит `ProsusAI/finbert` (один раз, на GPU если есть).
- Стр. 33: `truncation=True, max_length=512` — длинные статьи обрезаются до лимита модели.
- Стр. 34–35: метка × уверенность → число. Пример: `positive` с уверенностью 0.9 → **+0.9**;
  `negative` 0.8 → **−0.8**; `neutral` → **0**.

**Зачем.** Это «эмоциональный заряд» новости — насколько она хороша/плоха для рынка.

---

## Шаг 4 — Смысл новости (FinLang)

**Простыми словами.** Параллельно прогоняем каждую новость через FinLang — модель, которая
превращает текст в вектор из 768 чисел, кодирующий **о чём** новость (а не только её
тональность).

**Код** (`lib/features/embeddings.py`, `_get_model`, строки 18–27):

```python
18  def _get_model(model_name="FinLang/finance-embeddings-investopedia"):
19      global _model, _model_name_loaded
20      if _model is None or _model_name_loaded != model_name:
22          from sentence_transformers import SentenceTransformer
23          device = "cuda" if torch.cuda.is_available() else "cpu"
25          _model = SentenceTransformer(model_name, device=device)
27      return _model
```

Само кодирование — вызов `st_model.encode(texts)` → матрица `(число новостей, 768)`.

**Разбор.**
- Стр. 20–25: модель грузится один раз (singleton) на GPU, если доступен.
- `encode` отдаёт по 768-мерному вектору на каждую статью.

**Зачем.** Тональность отвечает «насколько хорошо/плохо», эмбеддинг — «о чём речь». Два
разных среза информации, оба полезны агенту.

---

## Шаг 5 — Обучение сжатия (PCA)

**Простыми словами.** 768 чисел на новость — много и шумно. Обучаем PCA сжимать 768 → 32,
причём **только на обучающих новостях**. Этим же сжатием потом обрабатываем тестовые.

**Код** (`build_data.py`, `build_train`, строки 162–182):

```python
162     # Fit compressor on ALL train embeddings
163     logger.info("Computing raw train embeddings for PCA fit...")
164     news_4h = preprocess_news_4h(news_train)
165     st_model = _get_model(cfg.embeddings.model_name)
166     all_emb = []
167     for idx, row in news_4h.iterrows():
168         for text in row["texts"]:
169             emb = st_model.encode([text], show_progress_bar=False)[0]
170             all_emb.append(emb)                       # копим все train-вектора
173     all_emb = np.array(all_emb, dtype=np.float32)
176     compressor = EmbeddingCompressor(
177         input_dim=cfg.embeddings.raw_dim,             # 768
178         output_dim=cfg.embeddings.compressed_dim,     # 32
179     )
180     compressor.fit(all_emb)                           # обучаем PCA
182     compressor.save(cfg.data.paths.compressor)        # → data/train/compressor.pkl
```

А само обучение PCA — в `lib/features/embeddings.py`, `EmbeddingCompressor.fit`, строки 63–72:

```python
63  def fit(self, embeddings):
67      self._pca.fit(embeddings)                         # sklearn PCA, n_components=32
68      logger.info(f"PCA fitted: {self.input_dim} -> {self.output_dim}, "
70                  f"explained var: {self.explained_variance_ratio():.3f}")
72      return self
```

**Разбор.**
- Стр. 167–170: прогоняем **все** train-новости через FinLang и копим 768-вектора.
  Это самый долгий этап сборки (тексты кодируются по одному).
- Стр. 176–180: создаём PCA на 32 компоненты и обучаем его.
- Стр. 182: сохраняем компрессор в файл; в `build_oos` (строка 205) он **загружается**, а не
  переобучается.

**Зачем.** Меньше признаков → агент учится быстрее, меньше переобучается. А обучение PCA
строго на train — гарантия, что в тест не «протекает» будущее.

---

## Шаг 6 — Взвешивание по тональности и сжатие

**Простыми словами.** В одном окне обычно несколько новостей. Объединяем их смысловые
векторы в один, но **не простым средним**: новости с сильной тональностью весят больше,
нейтральные — меньше. Полученный 768-вектор сжимаем PCA в 32 числа.

**Код** (`build_data.py`, `_attach_nlp_features_minimal`, строки 117–146):

```python
117     result = baseline.copy()
118     emb_cols = [f"emb_{i}" for i in range(cfg.embeddings.compressed_dim)]   # emb_0..emb_31
119     for c in emb_cols:
120         result[c] = 0.0                                # по умолчанию — нули (нет новостей)
121     result["sentiment_mean"] = 0.0
123     news_4h = preprocess_news_4h(news_slice)
124     st_model = _get_model(cfg.embeddings.model_name)
126     for _, row in news_4h.iterrows():
127         ws = row["date"]; texts = row["texts"]
129         if ws not in result.index:
130             continue
131         scores = compute_sentiment_scores(texts)       # FinBERT по статьям окна
134         result.loc[ws, "sentiment_mean"] = float(np.mean(scores))   # средняя тональность
136         # Эмбеддинги — взвешенное среднее по тональности, PCA-сжатие
137         raw_embs = st_model.encode(texts, show_progress_bar=False)   # (N, 768)
138         signed_w = np.array(
139             [s + np.sign(s) * 0.1 if s != 0 else 0.1 for s in scores], dtype=np.float32)
142         denom = np.abs(signed_w).sum()
143         signed_w = signed_w / denom if denom > 0 else np.ones_like(signed_w)/len(signed_w)
144         agg_emb = (raw_embs * signed_w[:, None]).sum(axis=0)   # одно взвешенное среднее 768d
145         compressed = compressor.transform(agg_emb.reshape(1, -1))[0]   # 768 → 32
146         result.loc[ws, emb_cols] = compressed
```

**Разбор.**
- Стр. 118–121: заранее заводим 32 колонки эмбеддинга и `sentiment_mean`, заполняем нулями —
  у окон без новостей так и останутся нули.
- Стр. 131, 134: тональности статей окна → их среднее в `sentiment_mean`.
- Стр. 138–139: вес статьи = `тональность + знак·0.1`. Сильно заряженные статьи (|s| велико)
  весят больше; нейтральные получают минимум 0.1.
- Стр. 142–143: нормировка весов на сумму модулей.
- Стр. 144: взвешенное среднее 768-векторов = один вектор «смысла окна».
- Стр. 145–146: PCA сжимает его до 32 чисел `emb_0..emb_31`.

**Зачем.** Важные (эмоционально заряженные) новости должны сильнее влиять на «смысл окна»,
чем проходные нейтральные сообщения.

---

## Шаг 7 — Сборка таблицы признаков

**Простыми словами.** Складываем всё, что агент будет видеть: 8 ценовых индикаторов +
1 тональность + 32 эмбеддинга = **41 признак** на каждое окно.

**Код** (`build_data.py`, `_build_baseline`, строки 77–82):

```python
77  def _build_baseline(ohlcv_slice, normalize_window):
78      df = add_technical_indicators_minimal(ohlcv_slice)   # 8 индикаторов
79      df["raw_close"] = df["close"].copy()                 # сохранить реальную цену
80      cols = [c for c in df.columns if c != "raw_close"]
81      df[cols] = rolling_zscore_normalize(df[cols], window=normalize_window)  # нормализация
82      return df
```

8 индикаторов считаются в `lib/features/price.py`, `add_technical_indicators_minimal`
(строки 49–100). Это **по одному-двум индикаторам на каждую из 5 категорий рынка** —
минимально достаточный набор, чтобы агент «видел» тренд, моментум, волатильность, объём
и свежий импульс, но без избыточности (литература FinRL использует похожие 4–10):

| Индикатор | Что это (простыми словами) | Категория | Зачем агенту |
|---|---|---|---|
| **`ema_26`** | Экспоненциальное скользящее среднее цены за 26 баров (~4.3 дня): сглаженный «центр тяжести» цены, недавние бары весят больше | **Тренд** | Цена выше своей EMA — бычий контекст, ниже — медвежий |
| **`macd`** | Разница быстрого и медленного средних: `EMA-12 − EMA-26` | **Тренд / моментум** | Знак и величина = направление и сила тренда; смена знака = разворот |
| **`rsi_14`** | Индекс относительной силы (0–100): какая доля из последних 14 баров была ростом | **Моментум / разворот** | >70 — перекуплен (риск отката вниз), <30 — перепродан (вероятен отскок вверх) |
| **`bb_width`** | Ширина полос Боллинджера `4·σ₂₀ / SMA₂₀`: насколько широк «коридор» цены | **Волатильность (режим)** | Узкие полосы = затишье перед движением; широкие = высокая турбулентность |
| **`atr_14`** | Average True Range за 14 баров: средний реальный размах бара | **Волатильность (модуль)** | Насколько «крупные» свечи сейчас — прямая мера риска позиции |
| **`obv`** | On-Balance Volume: накопленный объём со знаком движения цены | **Объём** | Растёт вместе с ценой — движение подтверждено объёмом (сильный тренд) |
| **`stoch_k`** | Stochastic %K (0–100): где закрытие внутри диапазона [min, max] за 14 баров | **Моментум / позиция** | Близко к 100 — у верха диапазона, близко к 0 — у низа |
| **`return_1d`** | Доходность за 1 бар: `close.pct_change()` | **Доходность** | Самый свежий импульс — куда и насколько цена дёрнулась только что |

> **Почему этого достаточно.** Каждая категория отвечает на свой вопрос: «куда идём» (тренд),
> «с какой силой» (моментум), «насколько трясёт» (волатильность), «подтверждено ли объёмом»
> (объём), «что прямо сейчас» (доходность). Добавлять по 5 индикаторов в каждую категорию
> (как было в старой версии с 20 индикаторами) — значит давать почти одно и то же разными
> формулами: модель переобучается на шум. Поэтому оставлен минимум.

Нормализация (`rolling_zscore_normalize`, строки 156–161):

```python
156 def rolling_zscore_normalize(df, window=30):
158     rolling_mean = df.rolling(window=window, min_periods=1).mean()
159     rolling_std = df.rolling(window=window, min_periods=1).std().replace(0, 1)
160     normalized = (df - rolling_mean) / rolling_std
161     return normalized.fillna(0)
```

**Итоговый набор признаков:**

| Группа | Сколько | Что это |
|---|---|---|
| Ценовые индикаторы | **8** | EMA-26, MACD, RSI-14, ширина Bollinger, ATR-14, OBV, Stoch %K, дневная доходность |
| Тональность | **1** | `sentiment_mean` |
| Смысл новостей | **32** | `emb_0 … emb_31` |
| **Итого** | **41** | один набор на каждое 4-часовое окно |

**Разбор нормализации.**
- Стр. 158–160: скользящий z-score — из каждого значения вычитается среднее по окну 30 и
  делится на стандартное отклонение. `rolling()` в pandas **трейлинговый**: берёт только
  прошлое и текущий бар `[t−29, t]`.

**Зачем именно 8+1+32.** Сознательно компактно: по литературе (FinRL, SAPPO) большего не
нужно, а меньше признаков = понятнее и устойчивее.

---

## Шаг 8 — Загрузка признаков в модель · `lib/data_loader.py`

**Простыми словами.** Читаем `features.parquet` и делим на: реальные цены (для P&L), матрицу
41 признака («глаза» агента) и отдельно сигнал тональности (для награды).

**Код** (`_load_features_parquet`, ключевые строки 45–76):

```python
45      if "raw_close" in df.columns:
46          prices = df["raw_close"].to_numpy(dtype=np.float64)      # реальные цены для P&L
50      feature_cols = [c for c in df.columns if c.lower() not in _PRICE_COLUMNS_EXT]
51      if exclude_news:                                            # режим --no-news (аблация)
53          feature_cols = [c for c in feature_cols if not _is_news_column(c)]
62      features = df[feature_cols].to_numpy(dtype=np.float32)      # матрица 41 признака
69          sentiment = np.zeros(len(df), dtype=np.float32)        # при --no-news сигнал = 0
71          sentiment = (df["sentiment_mean"].to_numpy(dtype=np.float32)
73                       if "sentiment_mean" in df.columns
74                       else np.zeros(len(df), dtype=np.float32))   # сигнал тональности
```

**Разбор.**
- Стр. 45–46: `raw_close` отделяется как реальная цена — по ней считается доходность.
- Стр. 50: всё, кроме `open/high/low/close/volume/raw_close`, идёт в матрицу признаков (41).
- Стр. 51–53: при `--no-news` новостные колонки выбрасываются (аблация «без новостей»).
- Стр. 71–75: колонка `sentiment_mean` вынимается отдельно — она пойдёт в награду.

**Зачем.** Цены — для честной прибыли; признаки — для решений; тональность — и в наблюдение,
и в награду.

---

## Шаг 9 — Наблюдение и награда агента · `lib/env.py`

**Простыми словами.** На каждом шаге агент видит последние 30 окон (≈5 дней) по 41 признаку
плюс свою текущую долю в BTC. Выбирает действие; получает награду DSR + sentiment-бонус.

**Наблюдение** (`_get_obs`, строки 133–136):

```python
133 def _get_obs(self):
134     start = self.current_step - self.window           # window = 30
135     window_obs = self.features[start:self.current_step].flatten()   # 30 × 41 = 1230 чисел
136     return np.append(window_obs, self.prev_allocation).astype(np.float32)   # +1 = 1231
```

**Действие → доля** (`step`, дискретный режим DQN, строки 78–86):

```python
78      if self.action_space_type == "discrete":
81          if a == 1:   allocation = 1.0                  # Buy — всё в BTC
83          elif a == 2: allocation = 0.0                  # Sell — всё в деньги
85          else:        allocation = self.prev_allocation # Hold — без изменений
```

**Базовая отдача с комиссией** (строки 91–97):

```python
91      price_curr = self.prices[self.current_step - 1]
92      price_next = self.prices[self.current_step]
93      log_return = float(np.log(price_next / price_curr))   # логарифмическая доходность
95      delta = abs(allocation - self.prev_allocation)
96      tx_penalty = self.tx_cost * delta                     # комиссия 0.1% за смену позиции
97      R_t = log_return * allocation - tx_penalty
```

**Награда DSR** (`_compute_dsr`, строки 118–131):

```python
118 def _compute_dsr(self, R_t):
120     dA = R_t - self._dsr_A
121     dB = R_t**2 - self._dsr_B
122     denom = self._dsr_B - self._dsr_A**2
123     if denom < 1e-12:
124         reward = float(R_t)
126     else:
127         reward = float((self._dsr_B*dA - 0.5*self._dsr_A*dB) / (denom**1.5))
129     self._dsr_A += eta * dA                               # экспоненц. обновление моментов
130     self._dsr_B += eta * dB
131     return reward
```

**Sentiment-бонус** (строки 108–110):

```python
108     if self.sentiment_signal is not None:
109         sent = float(self.sentiment_signal[self.current_step])
110         reward += self.sentiment_lambda * sent * log_return   # λ = 0.3
```

**Разбор.**
- `_get_obs`: последние 30 строк (по 41 числу) расплющиваются в 1230 чисел, плюс текущая
  доля → **наблюдение из 1231 числа**. Это всё, что видит агент.
- DQN выбирает Hold/Buy/Sell → доля 0% или 100%. (SAC — непрерывная доля, ветка `else`.)
- `R_t` — доходность позиции минус комиссия за изменение доли.
- DSR — дифференциальный коэффициент Шарпа (Moody & Saffell): рекурсивно обновляет
  экспоненциальные моменты `A`, `B` и выдаёт «прирост качества с учётом риска».
- Sentiment-бонус: если тональность совпала с направлением цены — небольшая добавка.

**Зачем sentiment-бонус.** Новости влияют на агента **дважды**: как вход (смысл + тональность
в наблюдении) и как форма награды.

---

## Почему нет утечки из будущего

Главный вопрос рецензента к торговым системам. Две защиты:

1. **Нормализация строго «в прошлое».** Скользящий z-score на шаге *t* использует только
   окно `[t−29, t]` (`price.py:158-159`, метод `rolling()` каузален) — никаких будущих баров.
2. **PCA обучается только на train** (`build_data.py:180`) и применяется к тесту без
   переобучения (`build_data.py:205`).

Значит, на каждом решении агент опирается лишь на информацию, доступную в тот момент.

---

## Дашборд · живая демонстрация

Streamlit-приложение: подтягивает свежие BTC-новости, прогоняет их через тот же конвейер,
опрашивает ансамбль из 10 сетей и показывает решение BUY / HOLD / SELL прямо сейчас.

### Архитектура

- `dashboard/app.py` — лендинг: сводка по результатам + навигация.
- `pages/1_Strategy.py` — «Решение прямо сейчас».
- `pages/2_Validation.py` — «Валидация на истории» (2024, 2025 vs Buy & Hold).
- `pages/3_News_Analyzer.py` — «Анализатор новости».

### Как принимается живое решение · `pages/1_Strategy.py`

**Автообновление ленты** (строки 52–62):

```python
52  feed = load_cached_feed()
54  stale = feed.empty or (now_utc - feed["ts"].max()) > pd.Timedelta(hours=4)
55  if stale:
56      with st.spinner("Подтягиваю свежие новости и считаю sentiment FinBERT..."):
58          feed, n_new, _ = refresh_feed(force=False)     # тянем новости + FinBERT
```

**Прогон ансамбля по окнам** (строки 70–75):

```python
70  buckets = group_by_bucket(feed, lookback_buckets=30)   # лента → 30 окон по 4ч
71  seeds = [(m["seed"], m["path"]) for m in snapshot.discover_models(...)[entry.algo]]
74  decisions = decide_for_buckets(buckets, seeds, entry.algo)   # каждое окно → решение
75  decisions = compute_portfolio(decisions, initial_capital=1.0)
```

### Построение признаков и голосование · `dashboard/utils/feed_decisions.py`

**Те же признаки, что в обучении** (`_build_features`, строки 47–66): в последнюю строку окна
вписываются реальные новостные фичи — тональность и взвешенный по ней эмбеддинг (точно как в
шаге 6):

```python
60      raw = get_embedder().encode(texts, show_progress_bar=False)
61      w = np.array([s + np.sign(s)*0.1 if s != 0 else 0.1 for s in scores])
62      w = w / np.abs(w).sum()
63      agg = (raw * w[:, None]).sum(axis=0)               # взвешенное среднее
64      compressed = load_compressor().transform(agg.reshape(1, -1))[0]   # тот же PCA
```

**Голосование ансамбля** (`_run_ensemble`, строки 91–113):

```python
91      action, _ = model.predict(obs, deterministic=True)   # каждая из 10 сетей
92      if algo == "DQN":
95          if a == 1: allocs.append(1.0); labels.append("BUY")
97          elif a == 2: allocs.append(0.0); labels.append("SELL")
99          else: allocs.append(prev_alloc); labels.append("HOLD")
106     if algo == "DQN":                                   # большинство голосов
108         winner = _majority_winner(votes)                # → 0% или 100%
113     return {"allocation": float(np.median(allocs)), ...}   # SAC → медиана
```

**Связь с обучением.** Наблюдение собирается точь-в-точь как `env._get_obs`
(`feed_decisions.py:152`: последние 30 строк + `prev_alloc`). Поэтому обученная сеть
применяется к live-данным без дообучения — это и делает демонстрацию убедительной.

---

## Словарь терминов

| Термин | Простое объяснение |
|---|---|
| **RL / агент** | Обучение с подкреплением; агент учится действиям, максимизирующим награду. |
| **DQN** | RL с дискретными действиями (держать / купить / продать). Хорош для одного актива и небольших данных. |
| **SAC** | RL с непрерывными действиями (любая доля 0–100%). |
| **FinBERT** | Нейросеть для финансовых текстов; выдаёт тональность. |
| **FinLang-эмбеддинг** | Вектор из 768 чисел, кодирующий смысл текста. |
| **PCA** | Метод главных компонент; сжимает много признаков в несколько информативных. |
| **DSR** | Дифференциальный коэффициент Шарпа — награда, поощряющая доходность с поправкой на риск. |
| **sentiment-бонус** | Добавка к награде, когда тональность совпадает с движением цены. |
| **аллокация** | Доля капитала в BTC (0% = всё в деньгах, 100% = всё в BTC). |
| **log_return** | Логарифмическая доходность `ln(p_next / p_curr)`. |
| **OOS** | Out-of-sample — тестовый период, не виденный при обучении. |
| **ансамбль** | 10 моделей (сидов), решающих вместе; голосование снижает случайность. |
| **Buy & Hold** | Наивная стратегия «купил и держишь» — базовый ориентир. |
| **z-score нормализация** | Перевод признака в «на сколько сигм отклонился от среднего». |
