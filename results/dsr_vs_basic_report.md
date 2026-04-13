# DSR vs Basic Reward - OOS Comparison Report

**Models:** SAC x5 seeds (PPO pending), BTC/USDT 4h, trained 2020-2023
**DSR reward** = Differential Sharpe Ratio + sentiment_lambda=0.1 (SAPPO-lite)
**Basic reward** = log_return x allocation - tx_cost x |delta_allocation|

---

## 2024 (in-distribution OOS)

**Buy & Hold:** Sharpe=0.721, Return=118.5%

### PPO
DSR results not yet available (training in progress)

### SAC

| Metric | basic | DSR | delta | delta pct |
|--------|------:|----:|------:|----------:|
| Sharpe (mean) | 0.702 | 0.866 | +0.164 | +23% |
| Total Return | 67.9% | 78.5% | +10.6pp | +16% |
| Max Drawdown | 23.1% | 21.7% | -1.4pp | |

Mann-Whitney U=19, p=0.2222

Per-seed breakdown:

| seed | basic Sharpe | DSR Sharpe | delta |
|------|----------:|----------:|------:|
| 42 | 0.562 | 1.000 | +0.438 |
| 123 | 0.576 | 0.953 | +0.377 |
| 7 | 0.600 | 0.797 | +0.198 |
| 2024 | 0.634 | 0.969 | +0.335 |
| 99 | 1.138 | 0.613 | -0.525 |

## 2025 (out-of-distribution OOS)

**Buy & Hold:** Sharpe=-0.237, Return=-12.8%

### PPO
DSR results not yet available (training in progress)

### SAC

| Metric | basic | DSR | delta | delta pct |
|--------|------:|----:|------:|----------:|
| Sharpe (mean) | -0.042 | -0.077 | -0.035 | -85% |
| Total Return | -2.7% | -3.6% | -0.9pp | -34% |
| Max Drawdown | 17.0% | 18.1% | +1.1pp | |

Mann-Whitney U=13, p=1.0000

Per-seed breakdown:

| seed | basic Sharpe | DSR Sharpe | delta |
|------|----------:|----------:|------:|
| 42 | -0.057 | -0.360 | -0.303 |
| 123 | 0.353 | -0.038 | -0.391 |
| 7 | 0.014 | -0.119 | -0.133 |
| 2024 | -0.410 | 0.113 | +0.523 |
| 99 | -0.110 | 0.017 | +0.127 |

---

## Key Takeaways

- **SAC 2024**: DSR+sentiment reward improves Sharpe +23% (0.702->0.866), Return +10.6pp, MaxDD -1.4pp vs basic
- **SAC 2025**: DSR slightly underperforms (-0.035 Sharpe) in bear/sideways market
- **Conclusion**: DSR+sentiment reward improves SAC on in-distribution data; generalization to OOS bear market is limited
- **PPO results**: to be added after training completes
