# Embeddings 3-Agents — Итоговый отчёт эксперимента

**Дата:** 2026-04-12  
**Данные:** BTC/USDT 4h, Binance  
**Train:** 2020-01-01 – 2023-12-31 (8760 баров)  
**OOS:** 2024-01-01 – 2024-12-31 (2191 бар)  
**Модели:** 15 (PPO × 5 seeds, A2C × 5 seeds, SAC × 5 seeds)  
**Фичи:** 95 (18 тех. индикаторов + 64d PCA embeddings + 13 NLP фичей)

---

## Что было сделано

### Инфраструктура (Tasks 0–10), 188 тестов

- `AgentConfig` расширен: `max_grad_norm`, `use_sde`, `buffer_size`, `tau`, `train_freq`, `gradient_steps`, `learning_starts`, `device`, `normalize_advantage`, `optimize_memory_usage`
- 3 Zoo-grade factory: `ppo_embeddings_config`, `a2c_embeddings_config`, `sac_embeddings_config`
- `FEATURE_COUNTS["embeddings"] = 95`, `EMBEDDINGS_NET_ARCH = [128, 128]`
- `_algo_specific_kwargs` читает всё из конфига (больше нет хардкода)
- SAC получает `device="cuda"`, PPO/A2C — `device="cpu"`
- Оркестратор `run_embeddings_experiments.py` (3 алго × 5 seeds)
- `run_oos_backtest.py` с Bootstrap 95% CI
- `embeddings_report.py` с Mann-Whitney U + Sharpe bar chart

### Обучение

| Алгоритм | Seeds | Steps | Время/seed | Device |
|---|---|---|---|---|
| PPO | 5 | 500k | ~10 мин | CPU |
| A2C | 5 | 500k | ~7 мин | CPU |
| SAC | 5 | 200k | ~48 мин | CUDA |

---

## OOS 2024 — Результаты

### Метрики (mean ± std по 5 seeds)

| Алгоритм | Sharpe ↑ | Total Return ↑ | Max Drawdown ↓ | Calmar ↑ |
|---|---|---|---|---|
| **SAC** | **0.702 ± 0.245** | **0.679 ± 0.365** | 0.231 | **0.501** |
| PPO | 0.474 ± 0.242 | 0.471 ± 0.334 | 0.271 | 0.236 |
| A2C | 0.302 ± 0.236 | 0.261 ± 0.250 | 0.310 | 0.124 |
| **Buy & Hold** | **0.721** | **1.185** | 0.300 | — |

### Детальные результаты по seeds

| Алгоритм | Seed | Sharpe | Total Return | Max DD | Calmar |
|---|---|---|---|---|---|
| PPO | 42 | 0.129 | 0.051 | 0.219 | 0.038 |
| PPO | 123 | 0.467 | 0.416 | 0.281 | 0.216 |
| PPO | 7 | 0.669 | 0.709 | 0.309 | 0.307 |
| PPO | 2024 | 0.371 | 0.289 | 0.308 | 0.142 |
| PPO | 99 | 0.733 | 0.893 | 0.239 | 0.477 |
| A2C | 42 | 0.312 | 0.249 | 0.265 | 0.145 |
| A2C | 123 | 0.458 | 0.443 | 0.357 | 0.179 |
| A2C | 7 | 0.508 | 0.496 | 0.277 | 0.255 |
| A2C | 2024 | 0.324 | 0.254 | 0.327 | 0.119 |
| A2C | 99 | -0.093 | -0.140 | 0.325 | -0.077 |
| SAC | 42 | 0.562 | 0.509 | 0.190 | 0.379 |
| SAC | 123 | 0.576 | 0.509 | 0.341 | 0.211 |
| SAC | 7 | 0.600 | 0.521 | 0.220 | 0.333 |
| SAC | 2024 | 0.634 | 0.526 | 0.288 | 0.257 |
| **SAC** | **99** | **1.138** | **1.332** | **0.116** | **1.323** |

### Bootstrap 95% CI (across seeds per algo)

| Алгоритм | Sharpe CI low | Sharpe CI high | TR CI low | TR CI high |
|---|---|---|---|---|
| PPO | 0.274 | 0.654 | 0.219 | 0.724 |
| A2C | 0.100 | 0.458 | 0.056 | 0.426 |
| SAC | 0.575 | 0.923 | 0.511 | 1.006 |

### Mann-Whitney U (Sharpe, two-sided)

| Пара | p-value | Значимо? |
|---|---|---|
| A2C vs PPO | 0.310 | нет |
| **A2C vs SAC** | **0.008** | **да (p<0.05)** |
| PPO vs SAC | 0.421 | нет |

---

## Интерпретация

**SAC — лучший RL-агент** по Sharpe (0.702), значимо превосходит A2C (p=0.008).  
PPO vs SAC — нет значимой разницы при 5 seeds.

**Buy & Hold** превосходит по Total Return (1.185 vs 0.679) — 2024 год был бычьим (+118%). SAC почти сравнялся по Sharpe-коэффициенту (0.702 vs 0.721), что говорит о хорошем соотношении риск/доход.

**Стабильность:** SAC — все 5 seeds положительны. A2C — один seed ушёл в минус (-0.093).

**Лучший seed:** SAC seed=99: Sharpe=1.138, Total Return=133%, Max DD=11.6%.

---

## Гиперпараметры (Zoo-grade)

| Параметр | PPO | A2C | SAC |
|---|---|---|---|
| learning_rate | 3e-4 (linear) | 7e-4 (linear) | 7.3e-4 (const) |
| n_steps | 2048 | 16 | — |
| batch_size | 64 | — | 256 |
| buffer_size | — | — | 300k |
| learning_starts | — | — | 10k |
| tau | — | — | 0.02 |
| train_freq | — | — | 8 |
| gradient_steps | — | — | 8 |
| ent_coef | 0.01 | 0.01 | auto |
| use_sde | True | True | True |
| net_arch | [128,128] tanh | [128,128] tanh | [128,128] relu |
| total_timesteps | 500k | 500k | 200k |
| device | CPU | CPU | CUDA |

---

## Артефакты

| Файл | Описание |
|---|---|
| `results/embeddings_runs_v4.csv` | Манифест 15 моделей |
| `results/oos_2024_embeddings.csv` | OOS 2024 метрики (15 строк) |
| `results/figures/embeddings_v4/sharpe_bar.png` | Sharpe bar chart |
| `experiments/embeddings/20260412_*/` | Модели + VecNormalize stats |

---

## Следующие шаги

- [ ] OOS 2025-01-01→2025-04-10 (реальные новости, план готов)
- [ ] OOS 2026-01-01→2026-04-12 (NLP=zeros, план готов)
- [ ] Multi-period comparison report

**План:** `docs/superpowers/plans/2026-04-12-oos-2025-2026-backtest.md`
