"""Чарт-доказательство для слайда «Механизм»: парные столбики годовой доходности
DQN на OOS 2025 — с новостями и без, в публикационном стиле с бракетом значимости."""
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("results/ablation_comparison.csv")
row = df[(df["algorithm"] == "DQN") & (df["period"] == "oos_2025")
         & (df["metric"] == "annualized_return")].iloc[0]

with_val = row["with_news_mean"] * 100
no_val = row["no_news_mean"] * 100
with_std = row["with_news_std"] * 100
no_std = row["no_news_std"] * 100
diff_mean = row["diff_mean"] * 100
ci_lo = row["ci_lower"] * 100
ci_hi = row["ci_upper"] * 100

fig, ax = plt.subplots(figsize=(5.5, 3.8))

labels = ["С новостями", "Без новостей"]
values = [with_val, no_val]
errors = [with_std, no_std]
colors = ["#1F77B4", "#FF7F0E"]

bars = ax.bar(labels, values, color=colors, width=0.5,
              yerr=errors, capsize=8,
              error_kw={"ecolor": "#333333", "linewidth": 1.5})

# value labels above each bar (above the error whisker)
for bar, val, err in zip(bars, values, errors):
    ax.text(bar.get_x() + bar.get_width() / 2, val + err + 4,
            f"{val:.1f}%", ha="center", va="bottom",
            fontsize=16, fontweight="bold")

# significance bracket: horizontal line above both bars + downticks + asterisk
y_top = max(v + e for v, e in zip(values, errors)) + 22
ax.plot([0, 1], [y_top, y_top], color="#333333", linewidth=1.2)
ax.plot([0, 0], [y_top - 3, y_top], color="#333333", linewidth=1.2)
ax.plot([1, 1], [y_top - 3, y_top], color="#333333", linewidth=1.2)
ax.text(0.5, y_top + 1, "*", ha="center", va="bottom",
        fontsize=22, fontweight="bold", color="#333333")

ax.set_ylim(0, y_top + 18)
ax.set_ylabel("Годовая доходность, %", fontsize=11)
ax.set_title("DQN на OOS 2025 — эмпирическая проверка", fontsize=13, pad=14)
ax.grid(True, axis="y", alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# effect size annotation below the chart
fig.text(0.5, -0.02,
         f"Δ = +{diff_mean:.1f} п.п.,  95% CI [+{ci_lo:.1f}; +{ci_hi:.1f}]   "
         "    * — CI разности не содержит 0",
         ha="center", va="top", fontsize=10, color="#333333", style="italic")

fig.tight_layout()
fig.savefig("results/figures/mechanism_proof.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved: results/figures/mechanism_proof.png")
