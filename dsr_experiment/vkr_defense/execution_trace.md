# Полный путь выполнения: от первой команды до результата

> Трассировка вызовов «сверху вниз»: с какой команды всё начинается и какая функция вызывает
> какую. Отступы = глубина вызова. Имена даны как `файл::функция` (надёжнее номеров строк).
>
> Два входа в систему: **A —** собрать данные (`build_data.py`), **B —** обучить и протестировать
> (`run.py`). Сначала запускают A, потом B.

---

# ВХОД A — сборка данных

```bash
cd dsr_experiment
PYTHONPATH=. ../venv/Scripts/python.exe build_data.py --build all
```

## Дерево вызовов

```
build_data.py  (запуск скрипта)
└─ if __name__ == "__main__":  main()                       build_data.py::main
   ├─ argparse → args.config="config.yaml", args.build="all"
   ├─ cfg = load_config("config.yaml")                       config_loader.py::load_config
   │   ├─ yaml.safe_load(file) → raw: dict
   │   ├─ строит Config(...) из секций, КАЖДУЮ через _only_known(cls, raw[...])
   │   │   └─ _only_known отбрасывает устаревшие/лишние ключи (reward_type, agent_ppo…)
   │   └─ _validate(cfg)                                     config_loader.py::_validate
   │       └─ проверки: есть 'train'; algos ⊆ {SAC,DQN}; DQN ⇒ action_space_type='discrete';
   │                    compressed_dim ≤ raw_dim
   │
   ├─ build == "all" ⇒  build_train(cfg)                     build_data.py::build_train
   │   │
   │   ├─ ensure_raw_ohlcv(cfg)                              build_data.py::ensure_raw_ohlcv
   │   │   └─ если data/raw/ohlcv.parquet есть → read_parquet
   │   │      иначе  fetch_ohlcv(asset, …)  (ccxt → Binance)  +  to_parquet (кэш)
   │   │                                                     lib/features/price.py::fetch_ohlcv
   │   ├─ ensure_raw_news(cfg)                               build_data.py::ensure_raw_news
   │   │   └─ если data/raw/news.parquet есть → read_parquet
   │   │      иначе  load_news_from_hf(dataset, start, end)   +  to_parquet
   │   │                                                     lib/features/news.py::load_news_from_hf
   │   │             └─ datasets.load_dataset(HF) → pandas; rename article_text→text, date_time→date;
   │   │                to_datetime(utc); фильтр по датам
   │   │
   │   ├─ _slice_by_period(ohlcv_full, train.start, train.end)   → ohlcv_train
   │   ├─ _slice_by_period(news_full,  train.start, train.end)   → news_train
   │   │
   │   ├─ baseline = _build_baseline(ohlcv_train, normalize_window=30)   build_data.py::_build_baseline
   │   │   ├─ add_technical_indicators_minimal(ohlcv)        price.py::add_technical_indicators_minimal
   │   │   │   └─ считает 8 индикаторов: ema_26, macd, rsi_14, bb_width, atr_14, obv, stoch_k, return_1d
   │   │   ├─ df["raw_close"] = df["close"]                   (сохранить сырую цену для P&L)
   │   │   └─ rolling_zscore_normalize(df[cols], window=30)  price.py::rolling_zscore_normalize
   │   │       └─ скользящий z-score окном 30 (строго трейлинг, без будущего)
   │   │
   │   ├─ — обучение PCA (только на train) —
   │   │   ├─ news_4h = preprocess_news_4h(news_train)        news.py::preprocess_news_4h
   │   │   │   └─ dt.floor("4h"); groupby(date) → {окно → [тексты]}
   │   │   ├─ st_model = _get_model()                         embeddings.py::_get_model
   │   │   │   └─ лениво грузит FinLang (SentenceTransformer, 768d) — singleton
   │   │   ├─ all_texts = [все тексты всех окон]
   │   │   ├─ all_emb = st_model.encode(all_texts, batch=64)  (768d на текст; батчем — быстро)
   │   │   ├─ compressor = EmbeddingCompressor(768, 32)        embeddings.py::EmbeddingCompressor
   │   │   ├─ compressor.fit(all_emb)                          → обучает PCA 768→32
   │   │   └─ compressor.save(compressor.pkl)                  → 📄 data/train/compressor.pkl
   │   │
   │   ├─ features = _attach_nlp_features_minimal(baseline, news_train, cfg, compressor)
   │   │   │                                                  build_data.py::_attach_nlp_features_minimal
   │   │   └─ для каждого 4h-окна с новостями:
   │   │       ├─ scores = compute_sentiment_scores(texts)    sentiment.py::compute_sentiment_scores
   │   │       │   └─ FinBERT pipeline → тональность каждой статьи ∈ [−1,+1]
   │   │       ├─ sentiment_mean = mean(scores)               (пишется в колонку)
   │   │       ├─ raw_embs = st_model.encode(texts)           (768d на статью)
   │   │       ├─ signed_w = тональность + знак·0.1; нормировка
   │   │       ├─ agg = Σ w·emb  (взвешенное среднее 768d)
   │   │       └─ compressed = compressor.transform(agg) → 32  → emb_0..emb_31
   │   │
   │   └─ features.to_parquet(train_features)                 → 📄 data/train/features.parquet (41 фича)
   │
   └─ для k в oos_periods:  build_oos(cfg, k)                 build_data.py::build_oos
       ├─ compressor = EmbeddingCompressor.load(compressor.pkl)  (ГОТОВЫЙ, не переобучаем!)
       ├─ baseline = _build_baseline(ohlcv_oos, 30)
       ├─ features = _attach_nlp_features_minimal(baseline, news_oos, cfg, compressor)
       └─ features.to_parquet(oos_features_path(k))           → 📄 data/oos/{k}_features.parquet
```

**Ключевая идея A.** PCA обучается **только на train** (`build_train`), а на OOS применяется
**загруженный** компрессор (`build_oos` → `EmbeddingCompressor.load`). Это исключает утечку
будущего: тестовые данные не влияют на преобразование признаков.

---

# ВХОД B — обучение и тест

```bash
PYTHONPATH=. ../venv/Scripts/python.exe run.py --algo DQN --seeds 42 123 7 11 99
```

## Дерево вызовов

```
run.py  (запуск скрипта)
└─ if __name__ == "__main__":  main()                        run.py::main
   ├─ argparse → algo="DQN", seeds=[42,123,7,11,99]
   ├─ cfg = load_config("config.yaml")                       (как в A)
   ├─ algos = ["DQN"];  cfg.experiment.seeds = [42,123,7,11,99]
   │
   ├─ НЕ --skip-train ⇒  train_phase(cfg, algos, …)          run.py::train_phase
   │   ├─ features, prices, sentiment = load_train(cfg)      lib/data_loader.py::load_train
   │   │   └─ _load_features_parquet:                         data_loader.py::_load_features_parquet
   │   │       ├─ read_parquet(features.parquet); срез по датам train
   │   │       ├─ prices  = колонка raw_close          (реальные цены для P&L)
   │   │       ├─ features = все колонки кроме price-набора  → матрица 41 фичи
   │   │       └─ sentiment = колонка sentiment_mean   (отдельный сигнал для награды)
   │   │
   │   └─ для seed в [42,123,7,11,99]:
   │       └─ train_agent(cfg, "DQN", seed, features, prices, sentiment)   lib/train.py::train_agent
   │           ├─ agent_cfg = cfg.agent_dqn
   │           ├─ env = _build_env(cfg, features, prices, sentiment)        train.py::_build_env
   │           │   └─ TradingEnv(features, prices, window=30, tx_cost,       lib/env.py::TradingEnv.__init__
   │           │                 sentiment_signal=sentiment, sentiment_lambda=0.3,
   │           │                 action_space_type='discrete')
   │           │       └─ observation_space = Box(30·41+1 = 1231);  action_space = Discrete(3)
   │           ├─ vec_env = DummyVecEnv([lambda: env])
   │           ├─ model = DQN("MlpPolicy", vec_env, lr, seed, net_arch=[128,128], **_algo_kwargs)
   │           │                                                            stable_baselines3.DQN
   │           ├─ model.learn(total_timesteps=200000, callback=ProgressCallback)
   │           │   └─ ЦИКЛ ОБУЧЕНИЯ (внутри SB3), много раз:
   │           │       env.reset() → obs                                    env.py::reset → _get_obs
   │           │       action = политика(obs)
   │           │       env.step(action):                                    env.py::step
   │           │         ├─ action → allocation (Hold=prev / Buy=1 / Sell=0)
   │           │         ├─ log_return = log(p_next/p_curr)
   │           │         ├─ R_t = log_return·allocation − tx_cost·|Δ|
   │           │         ├─ reward = _compute_dsr(R_t)                      env.py::_compute_dsr
   │           │         │   └─ дифференциальный Sharpe (моменты A,B, η=0.01)
   │           │         ├─ reward += sentiment_lambda·sentiment_t·log_return   (sentiment-бонус)
   │           │         └─ возвращает obs', reward, done, info{log_return, allocation}
   │           └─ model.save(…)                               → 📄 models/DQN/DQN_seed{S}_{ts}/model.zip
   │
   └─ НЕ --skip-oos ⇒  oos_phase(cfg, algos, model_paths, oos_keys, …)     run.py::oos_phase
       └─ для period в ["oos_2024","oos_2025"]:
           ├─ features, prices, _ = load_oos(cfg, period)     data_loader.py::load_oos
           └─ для (path, seed) в моделях:
               ├─ res = run_backtest(features, prices, model_path=path, window=30, tx_cost,
               │                     action_space_type='discrete')          lib/backtest.py::run_backtest
               │   ├─ env = TradingEnv(features, prices, …, 'discrete')
               │   ├─ model = load_model(path)                backtest.py::load_model  (DQN.load)
               │   ├─ ЦИКЛ БЭКТЕСТА (детерминированный):
               │   │   obs = env.reset()
               │   │   while not done:
               │   │     action = model.predict(obs, deterministic=True)
               │   │     obs, _, done, info = env.step(action)
               │   │     step_returns.append( exp(log_ret·alloc) − 1 )
               │   ├─ equity_curve = cumprod(1+step_returns)
               │   └─ metrics = compute_metrics(step_returns, allocations)  lib/metrics.py::compute_metrics
               │       └─ Sharpe = mean/std·√2190; Sortino; MaxDD; Calmar; win_rate; …
               ├─ rows.append({algorithm, seed, period, **metrics})
               └─ np.savez(…seed{S}.npz, daily_returns, allocations, equity_curve)
                                                              → 📄 results/oos_{period}_DQN_seed{S}.npz
       ├─ csv merge → 📄 results/oos_{period}.csv             (строка на (algo, seed))
       └─ _print_summary(rows)                                → печать mean±std в консоль
```

**Ключевая идея B.** Награда (DSR + sentiment-бонус) нужна только при **обучении** — она формирует
поведение агента. При **бэктесте** метрики считаются не из награды, а из реальных доходностей
(`info["log_return"] × allocation`), поэтому изменения в формуле награды не влияют на отчётные
цифры — только на то, чему агент научился.

---

# Что в итоге появляется на диске

| После входа A | После входа B |
|---|---|
| `data/raw/{ohlcv,news}.parquet` (кэш) | `models/DQN/DQN_seed{S}_{ts}/model.zip` |
| `data/train/compressor.pkl` (PCA 768→32) | `results/oos_{period}.csv` (метрики) |
| `data/train/features.parquet` (41 фича) | `results/oos_{period}_DQN_seed{S}.npz` (ряды) |
| `data/oos/oos_{2024,2025}_features.parquet` | |

Детальный разбор каждого шага по смыслу — в [`news_path_walkthrough.md`](news_path_walkthrough.md);
карта файлов — там же в секции «Где что сохраняется».

---

# Самый короткий ответ «как это работает» (для устного)

> «Две команды. Первая — `build_data.py` — качает цены и новости, прогоняет новости через FinBERT
> и FinLang, обучает PCA на train, склеивает 41 признак на каждый 4-часовой бар и пишет parquet.
> Вторая — `run.py` — грузит эти признаки, строит Gymnasium-среду с DSR-наградой, обучает DQN на
> пяти сидах, затем детерминированно прогоняет модели на 2024 и 2025, считает Sharpe/просадку и
> пишет CSV. Дашборд потом читает обученные модели и тот же PCA, и считает решение на свежих
> новостях тем же кодом.»
