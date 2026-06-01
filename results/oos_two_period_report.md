# Embeddings 3-Agents — OOS 2024 vs 2025 Report

Periods: **2024** (in-distribution OOS) · **2025** (full NLP, unseen)

Models: 15 (PPO×5, A2C×5, SAC×5), trained on BTC/USDT 4h 2020–2023


## 2024 (OOS baseline)

**Buy & Hold**: Sharpe=0.7212, Total Return=1.1854

| algorithm   |   sharpe_mean |   sharpe_std |   total_return_mean |   total_return_std |   max_dd_mean |   calmar_mean |
|:------------|--------------:|-------------:|--------------------:|-------------------:|--------------:|--------------:|
| A2C         |        0.3017 |       0.2364 |              0.2605 |             0.2495 |        0.3099 |        0.1241 |
| PPO         |        0.4738 |       0.2421 |              0.4714 |             0.3343 |        0.2711 |        0.2359 |
| SAC         |        0.702  |       0.2453 |              0.6791 |             0.3648 |        0.231  |        0.5007 |



## 2025 (NLP)

**Buy & Hold**: Sharpe=-0.2373, Total Return=-0.1285

| algorithm   |   sharpe_mean |   sharpe_std |   total_return_mean |   total_return_std |   max_dd_mean |   calmar_mean |
|:------------|--------------:|-------------:|--------------------:|-------------------:|--------------:|--------------:|
| A2C         |        0.0241 |       0.1504 |             -0.0227 |             0.0456 |        0.2067 |       -0.0706 |
| PPO         |        0.0048 |       0.5165 |             -0.0228 |             0.1477 |        0.1915 |        0.0235 |
| SAC         |       -0.0419 |       0.2736 |             -0.0268 |             0.0647 |        0.1704 |       -0.0727 |



## Figures

- `results\figures\two_period\sharpe_by_period.png` — Sharpe bar chart per period


## Mann-Whitney U (SAC Sharpe: 2024 vs 2025, two-sided)

| pair                              |   U |   p_value |
|:----------------------------------|----:|----------:|
| 2024 (OOS baseline) vs 2025 (NLP) |  25 |   0.00794 |

