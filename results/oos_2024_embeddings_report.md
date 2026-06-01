# Embeddings 3-Agents — OOS 2024 Report

Source: `oos_2024_embeddings.csv`  (n=15 runs)


## Summary (mean ± std over seeds)

| algorithm   |   sharpe_mean |   sharpe_std |   total_return_mean |   total_return_std |   max_dd_mean |   calmar_mean |
|:------------|--------------:|-------------:|--------------------:|-------------------:|--------------:|--------------:|
| A2C         |        0.3017 |       0.2364 |              0.2605 |             0.2495 |        0.3099 |        0.1241 |
| PPO         |        0.4738 |       0.2421 |              0.4714 |             0.3343 |        0.2711 |        0.2359 |
| SAC         |        0.702  |       0.2453 |              0.6791 |             0.3648 |        0.231  |        0.5007 |


## Buy & Hold baseline

- Sharpe: 0.7212
- Total Return: 1.1854


## Mann-Whitney U (Sharpe, two-sided)

| pair       |   U |    p_value |
|:-----------|----:|-----------:|
| A2C vs PPO |   7 | 0.309524   |
| A2C vs SAC |   0 | 0.00793651 |
| PPO vs SAC |   8 | 0.420635   |


## Figures

- `figures/sharpe_bar.png` — Sharpe bar chart with ±std error bars and B&H line
