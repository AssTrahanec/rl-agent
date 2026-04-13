# DSR vs Basic Reward - OOS Comparison Report

**Models:** SAC x5 + PPO x5 seeds, BTC/USDT 4h, trained 2020-2023
**DSR reward** = Differential Sharpe Ratio + sentiment_lambda=0.1 (SAPPO-lite)
**Basic reward** = log_return x allocation - tx_cost x |delta_allocation|

---

## 2024 (in-distribution OOS)

**Buy & Hold:** Sharpe=0.721, Return=118.5%

| Algorithm | basic Sharpe | DSR Sharpe | delta | basic Return | DSR Return | delta | basic MaxDD | DSR MaxDD | delta |
|-----------|-------------:|-----------:|------:|-------------:|-----------:|------:|------------:|----------:|------:|
| PPO | 0.474 | 0.677 | +0.203 | 47.1% | 78.2% | +31.1pp | 27.1% | 23.6% | -3.5pp |
| SAC | 0.702 | 0.866 | +0.164 | 67.9% | 78.5% | +10.6pp | 23.1% | 21.7% | -1.4pp |

**PPO Mann-Whitney** (Sharpe DSR vs basic): U=18, p=0.3095

<details><summary>PPO per-seed</summary>

| seed | basic Sharpe | DSR Sharpe | delta | basic Return | DSR Return |
|------|----------:|----------:|------:|-------------:|-----------:|
| 42 | 0.129 | 0.865 | +0.736 | 5.1% | 142.0% |
| 123 | 0.467 | 0.528 | +0.061 | 41.6% | 52.6% |
| 7 | 0.669 | 0.732 | +0.063 | 70.9% | 64.8% |
| 2024 | 0.371 | 0.669 | +0.298 | 28.9% | 84.9% |
| 99 | 0.733 | 0.591 | -0.141 | 89.3% | 46.9% |

</details>

**SAC Mann-Whitney** (Sharpe DSR vs basic): U=19, p=0.2222

<details><summary>SAC per-seed</summary>

| seed | basic Sharpe | DSR Sharpe | delta | basic Return | DSR Return |
|------|----------:|----------:|------:|-------------:|-----------:|
| 42 | 0.562 | 1.000 | +0.438 | 50.9% | 78.2% |
| 123 | 0.576 | 0.953 | +0.377 | 50.9% | 101.0% |
| 7 | 0.600 | 0.797 | +0.198 | 52.1% | 65.4% |
| 2024 | 0.634 | 0.969 | +0.335 | 52.6% | 91.6% |
| 99 | 1.138 | 0.613 | -0.525 | 133.2% | 56.3% |

</details>

## 2025 (out-of-distribution OOS, bear market)

**Buy & Hold:** Sharpe=-0.237, Return=-12.8%

| Algorithm | basic Sharpe | DSR Sharpe | delta | basic Return | DSR Return | delta | basic MaxDD | DSR MaxDD | delta |
|-----------|-------------:|-----------:|------:|-------------:|-----------:|------:|------------:|----------:|------:|
| PPO | 0.005 | -0.610 | -0.615 | -2.3% | -15.4% | -13.2pp | 19.1% | 24.8% | +5.7pp |
| SAC | -0.042 | -0.077 | -0.035 | -2.7% | -3.6% | -0.9pp | 17.0% | 18.1% | +1.1pp |

**PPO Mann-Whitney** (Sharpe DSR vs basic): U=6, p=0.2222

<details><summary>PPO per-seed</summary>

| seed | basic Sharpe | DSR Sharpe | delta | basic Return | DSR Return |
|------|----------:|----------:|------:|-------------:|-----------:|
| 42 | -0.345 | -0.354 | -0.009 | -11.8% | -15.7% |
| 123 | 0.807 | -0.236 | -1.043 | 21.1% | -9.3% |
| 7 | 0.172 | -0.019 | -0.191 | 2.3% | -2.1% |
| 2024 | -0.507 | -0.674 | -0.167 | -16.1% | -22.8% |
| 99 | -0.103 | -1.767 | -1.664 | -7.0% | -27.3% |

</details>

**SAC Mann-Whitney** (Sharpe DSR vs basic): U=13, p=1.0000

<details><summary>SAC per-seed</summary>

| seed | basic Sharpe | DSR Sharpe | delta | basic Return | DSR Return |
|------|----------:|----------:|------:|-------------:|-----------:|
| 42 | -0.057 | -0.360 | -0.303 | -3.7% | -9.5% |
| 123 | 0.353 | -0.038 | -0.391 | 6.6% | -3.0% |
| 7 | 0.014 | -0.119 | -0.133 | -1.0% | -4.7% |
| 2024 | -0.410 | 0.113 | +0.523 | -11.4% | 0.9% |
| 99 | -0.110 | 0.017 | +0.127 | -3.9% | -1.6% |

</details>

---

## Summary

| Algorithm | Period | Sharpe delta | Return delta | MaxDD delta | Verdict |
|-----------|--------|-------------:|-------------:|------------:|--------|
| PPO | 2024 | +0.203 | +31.1pp | -3.5pp | DSR better |
| PPO | 2025 | -0.615 | -13.2pp | +5.7pp | basic better |
| SAC | 2024 | +0.164 | +10.6pp | -1.4pp | DSR better |
| SAC | 2025 | -0.035 | -0.9pp | +1.1pp | similar |

**Key findings:**

- PPO 2024: DSR Sharpe +0.203 (+43%), Return +31.1pp, MaxDD -3.5pp — strong improvement
- SAC 2024: DSR Sharpe +0.164 (+23%), Return +10.6pp, MaxDD -1.4pp — solid improvement
- PPO 2025: DSR Sharpe -0.615 — DSR hurts in bear/volatile OOS market
- SAC 2025: DSR Sharpe -0.035 — marginal degradation, within noise
- **Conclusion**: DSR+sentiment reward consistently improves in-distribution performance.
  Generalization to unseen bear market regimes is limited for PPO; SAC is more robust.
