"""Понятные графики результатов: кривые капитала и ключевые метрики."""
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_mean_equity(results_dir, period, algo):
    # Загружает кривые капитала по всем сидам и усредняет их
    curves = []
    for path in glob.glob(f"{results_dir}/oos_{period}_{algo}_seed*.npz"):
        curves.append(np.load(path)["equity_curve"])
    min_len = min(len(c) for c in curves)
    curves = [c[:min_len] for c in curves]
    return np.mean(curves, axis=0)


def buy_and_hold(period, n_points):
    # Кривая пассивного удержания: цена биткоина, нормированная к старту 1.0
    df = pd.read_parquet(f"data/oos/{period}_features.parquet")
    price = df["raw_close"].to_numpy()
    price = price[-n_points:]
    return price / price[0]


def plot_equity_curves(algo):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, period in zip(axes, ["oos_2024", "oos_2025"]):
        with_news = load_mean_equity("results", period, algo)
        no_news = load_mean_equity("results_no_news", period, algo)
        bh = buy_and_hold(period, len(with_news))

        ax.plot(with_news, label=f"{algo} с новостями", color="tab:blue")
        ax.plot(no_news, label=f"{algo} без новостей", color="tab:orange", linestyle="--")
        ax.plot(bh, label="Buy & Hold", color="gray")
        year = period.replace("oos_", "")
        ax.set_title(f"Рост $1 капитала — {year} год")
        ax.set_xlabel("4-часовые бары")
        ax.set_ylabel("Стоимость портфеля, $")
        ax.legend()
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(f"results/figures/equity_curves_{algo}.png", dpi=150)
    plt.close(fig)
    print(f"Сохранён results/figures/equity_curves_{algo}.png")


def plot_key_metrics(algo):
    df = pd.read_csv("results/ablation_comparison.csv")
    df = df[(df["algorithm"] == algo) & (df["period"] == "oos_2025")]

    metrics = {
        "annualized_return": "Годовая доходность",
        "sharpe_ratio": "Коэффициент Шарпа",
        "max_drawdown": "Максимальная просадка",
        "win_rate": "Доля прибыльных шагов",
    }
    percent_metrics = {"annualized_return", "max_drawdown", "win_rate"}

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for ax, (metric, title) in zip(axes.flat, metrics.items()):
        row = df[df["metric"] == metric].iloc[0]
        with_val = row["with_news_mean"]
        no_val = row["no_news_mean"]

        bars = ax.bar(["С новостями", "Без новостей"], [with_val, no_val],
                      color=["tab:blue", "tab:orange"])
        for bar, val in zip(bars, [with_val, no_val]):
            if metric in percent_metrics:
                label = f"{val * 100:.1f}%"
            else:
                label = f"{val:.2f}"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    label, ha="center", va="bottom", fontsize=12)

        mark = "  —  разница значима" if row["significant"] else ""
        ax.set_title(title + mark)
        ax.grid(alpha=0.3, axis="y")

    fig.suptitle(f"{algo} на 2025 году: с новостями против без новостей", fontsize=13)
    fig.tight_layout()
    fig.savefig(f"results/figures/key_metrics_{algo}.png", dpi=150)
    plt.close(fig)
    print(f"Сохранён results/figures/key_metrics_{algo}.png")


def main():
    for algo in ["DQN", "SAC"]:
        plot_equity_curves(algo)
        plot_key_metrics(algo)


if __name__ == "__main__":
    main()
