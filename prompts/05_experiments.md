# Фаза 5: Эксперименты

Прочитай этот файл и выполни всё что написано ниже.

---

## Задачи

1. Прочитай `docs/PRD.md` и `CLAUDE.md`
2. Используй `superpowers:dispatching-parallel-agents` skill — запусти параллельные агенты для обучения трёх агентов одновременно (baseline, sentiment, embedding)
3. Создай `experiments/run_all.py` — обучает все 3 агента с seeds [42,43,44], train 2019-2022 / test 2023-2024, логирует в wandb, сохраняет метрики в `results/metrics.csv`
4. Создай `notebooks/results_analysis.ipynb` — таблица метрик (mean ± std), barplot Sharpe с error bars, equity curves, статистические тесты (t-test + Wilcoxon), графики в `results/figures/`
5. Используй **Exa** для поиска benchmark результатов из похожих papers — сравни свои метрики с литературой

## Завершение

Обнови CLAUDE.md и WORKFLOW.md (✅ Фаза 5).
