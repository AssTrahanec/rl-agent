# Глава 1. Обзор литературы

## 1.1 Введение

Применение методов машинного обучения к задачам алгоритмической торговли является одним из наиболее активно развивающихся направлений вычислительных финансов. В последние годы особый интерес вызывает сочетание двух парадигм: обучения с подкреплением (Reinforcement Learning, RL) для принятия торговых решений и обработки естественного языка (Natural Language Processing, NLP) для извлечения информации из текстовых источников — новостей, социальных сетей и аналитических отчётов. Настоящий обзор систематизирует существующие работы по трём ключевым направлениям: (1) применение RL в торговле криптовалютами, (2) NLP-методы для анализа финансовых текстов и (3) совместное использование RL и NLP, — а также идентифицирует исследовательские пробелы, которые настоящая работа призвана восполнить.

---

## 1.2 Обучение с подкреплением в алгоритмической торговле

### 1.2.1 Теоретические основы

Задача алгоритмической торговли естественным образом формализуется как марковский процесс принятия решений (MDP): агент наблюдает состояние рынка, выбирает действие (покупка, продажа, удержание позиции) и получает вознаграждение, связанное с доходностью портфеля (Sutton & Barto, 2018). В отличие от классического обучения с учителем, RL-подход не требует размеченных данных вида «правильное действие в момент $t$», а оптимизирует кумулятивную награду, что делает его особенно привлекательным для финансовых приложений с отложенной обратной связью.

### 1.2.2 Обзор алгоритмов

Среди методов глубокого RL, применяемых к торговле, выделяются три группы:

**Методы «актёр-критик» (Actor-Critic).** Алгоритмы PPO (Schulman et al., 2017), A2C (Mnih et al., 2016) и SAC (Haarnoja et al., 2018) являются наиболее распространёнными. PPO демонстрирует высокую устойчивость благодаря клиппированию функции потерь, что предотвращает чрезмерные обновления политики. A2C обеспечивает синхронное обучение с advantage-функцией. SAC использует энтропийную регуляризацию, поощряя более стохастичную политику и активную разведку пространства действий (Raffin et al., 2021).

**Методы на основе Q-обучения.** DQN (Mnih et al., 2015) и его модификации (Double DQN, Dueling DQN) успешно применялись к дискретным пространствам действий, однако ограничены в задачах с непрерывным управлением размером позиции.

**Ансамблевые подходы.** Yang et al. (2020) предложили ансамблевую стратегию, объединяющую PPO, A2C и DDPG для торговли акциями Dow Jones 30, продемонстрировав превосходство над отдельными алгоритмами. Данный подход получил дальнейшее развитие в фреймворке FinRL (Liu et al., 2021), ставшем стандартом де-факто для воспроизводимых исследований в области финансового RL.

### 1.2.3 RL на рынке криптовалют

Криптовалютный рынок характеризуется высокой волатильностью, круглосуточной торговлей и выраженной зависимостью от новостного фона, что создаёт как возможности, так и вызовы для RL-агентов. Систематический обзор 167 исследований за 2017–2025 годы (Chen et al., 2025) показал, что доля гибридных подходов (RL + глубокое обучение) возросла с 15% в 2020 году до 42% в 2025 году. Peng et al. (2023) применили ансамбль DRL-алгоритмов к торговле криптовалютами, интегрируя паттерны свечей как дополнительные признаки.

Тем не менее большинство работ ограничиваются техническими индикаторами, не интегрируя текстовую информацию. Работы, включающие NLP-компоненты, как правило, используют только один тип текстовых признаков (sentiment или embeddings), не проводя прямого сравнения их вклада.

---

## 1.3 NLP в анализе финансовых текстов

### 1.3.1 Sentiment-анализ

Анализ тональности финансовых текстов прошёл путь от словарных методов (Loughran & McDonald, 2011) через модели машинного обучения к предобученным языковым моделям. Ключевым прорывом стала модель FinBERT (Araci, 2019; Huang et al., 2020), представляющая собой дообученную версию BERT на финансовом корпусе. FinBERT достигает F1-score 93.27% и accuracy 91.08% на датасете SEntFiN, значительно превосходя универсальные модели тональности.

В контексте криптовалют Kim et al. (2023) провели сравнительную оценку fine-tuned моделей BERT, DeBERTa и RoBERTa на корпусе криптоновостей, показав преимущество RoBERTa. Ortu et al. (2022) применили BERT для классификации эмоций в комментариях на GitHub и Reddit, связанных с криптовалютами.

Интеграция FinBERT с LSTM-сетями для прогнозирования рыночных трендов подтвердила, что sentiment-признаки значимо повышают точность предсказания рыночных колебаний (Gupta et al., 2023).

### 1.3.2 Sentence embeddings

Помимо скалярного sentiment-score, тексты могут быть представлены в виде плотных векторных вложений (embeddings), сохраняющих семантическую информацию. Модели семейства Sentence-BERT (Reimers & Gurevych, 2019), в частности all-MiniLM-L6-v2, позволяют получить 384-мерные представления предложений, пригодные для downstream-задач.

Сравнительная оценка embedding-представлений для анализа тональности финансовых новостей (Morales et al., 2025) показала, что sentence transformer-представления в сочетании с gradient boosting достигают конкурентоспособных результатов даже в условиях ограниченных ресурсов. Исследование рыночного прогнозирования с использованием Angle-optimized Embedding (AoE) sentence transformer (Li et al., 2024) продемонстрировало, что мультирезолюционные данные (часовые и дневные) в сочетании с новостными эмбеддингами повышают точность прогнозирования.

### 1.3.3 Fine-tuning языковых моделей для финансового домена

Адаптация предобученных моделей к финансовому домену является перспективным направлением. Park et al. (2025) продемонстрировали, что оптимизация FinBERT через квантизацию и отбор подмножеств данных (coreset selection) позволяет развернуть domain-adapted модели с ограниченными вычислительными ресурсами без существенной потери качества.

---

## 1.4 Совместное применение RL и NLP в торговле

### 1.4.1 Sentiment-augmented RL

Ряд работ исследует добавление sentiment-score к observation space RL-агента. Wu et al. (2023) показали, что использование sentiment в комбинации с embeddings является преимуществом для DRL-обучения в multi-modal подходе. Sharma et al. (2024) подтвердили улучшение торговых стратегий при интеграции sentiment-фичей в RL-агентов.

### 1.4.2 Embedding-augmented RL

Использование плотных текстовых представлений в качестве дополнительных признаков для RL-агента является менее исследованным направлением. Zhang et al. (2024) демонстрируют подход к интеграции разнородных текстовых данных в RL-based stock trading, однако не проводят прямого сравнения вклада embeddings и sentiment при прочих равных условиях.

### 1.4.3 LLM-augmented подходы

Новейшие исследования интегрируют большие языковые модели (LLM) непосредственно в цикл принятия решений. Wang et al. (2025) предлагают Meta-RL-Crypto — triple-loop learning framework, где LLM выполняет роли актёра, критика и мета-обучающегося, обрабатывая on-chain метрики, новости и sentiment. Данный подход представляет передовой фронт исследований, однако требует значительных вычислительных ресурсов.

---

## 1.5 Анализ исследовательских пробелов (*Gap Analysis*)

На основании проведённого обзора выявлены следующие пробелы:

1. **Отсутствие прямого сравнения NLP-модальностей.** Большинство работ используют либо sentiment-score, либо embeddings, но не проводят контролируемого ablation study, сопоставляющего вклад каждой модальности при прочих равных условиях. Это затрудняет ответ на фундаментальный вопрос: что информативнее для торгового агента — скалярная оценка тональности или семантическое представление текста?

2. **Ограниченное применение sentence-transformers в RL для криптовалют.** В то время как FinBERT широко применяется для sentiment-анализа, использование sentence-transformer embeddings (в частности, MiniLM) в качестве признаков для RL-агентов на крипторынке остаётся малоизученным.

3. **Недостаточная воспроизводимость.** Многие работы используют проприетарные данные или закрытые торговые системы, что затрудняет верификацию результатов. Исследования на открытых данных с воспроизводимым pipeline остаются редкостью.

4. **Ограниченное сравнение RL-алгоритмов с NLP-фичами.** Работы Yang et al. (2020) и Liu et al. (2021) сравнивают PPO, A2C, SAC на ценовых данных, но аналогичное сравнение в контексте NLP-расширенных observation spaces не проводилось.

5. **Статистическая строгость.** Многие исследования представляют результаты одного запуска без bootstrap confidence intervals или статистических тестов, что ставит под сомнение надёжность выводов.

---

## 1.6 Позиционирование настоящей работы

Настоящее исследование адресует выявленные пробелы посредством:

- **3-way ablation study** — насколько нам известно, одного из первых прямых сравнений трёх конфигураций PPO-агента (baseline, +sentiment, +embeddings) на рынке BTC/ETH с идентичной архитектурой и гиперпараметрами;
- **воспроизводимого pipeline** — использования исключительно открытых данных (ccxt/Binance, HuggingFace datasets), открытых моделей (FinBERT, MiniLM-L6-v2) и библиотеки Stable-Baselines3;
- **статистической строгости** — множественных seeds, bootstrap CI (95%), t-тестов и Mann-Whitney U для оценки значимости различий;
- **сравнения RL-алгоритмов** — PPO vs A2C vs SAC на лучшей конфигурации из ablation study;
- **дополнительных экспериментов** — fine-tuning FinBERT на крипто-новостях и feature fusion (Agent-4: sentiment + embeddings).

---

## Список литературы

1. Araci, D. (2019). FinBERT: Financial Sentiment Analysis with Pre-Trained Language Models. arXiv:1908.10063.
2. Chen, L. et al. (2025). Reinforcement Learning in Financial Decision Making: A Systematic Review of Performance, Challenges, and Implementation Strategies. arXiv:2512.10913.
3. Gupta, R. et al. (2023). Financial Sentiment Analysis Using FinBERT with Application in Predicting Stock Movement. arXiv:2306.02136.
4. Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018). Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor. ICML.
5. Huang, A. H., Wang, H., & Yang, Y. (2020). FinBERT: A Large Language Model for Extracting Information from Financial Text. Contemporary Accounting Research.
6. Kim, S. et al. (2023). Fine-tuning Transformer Models on Cryptocurrency-Specific News Corpus. arXiv.
7. Li, J. et al. (2024). Financial Market Prediction Using Contrastive Learning-Based News Representation and Multi-Resolution Market Data. Canadian Conference on AI.
8. Liu, X.-Y., Yang, H., Gao, J., & Wang, C. (2021). FinRL: Deep Reinforcement Learning Framework to Automate Trading in Quantitative Finance. ACM ICAIF.
9. Loughran, T., & McDonald, B. (2011). When is a Liability not a Liability? Textual Analysis, Dictionaries, and 10-Ks. Journal of Finance, 66(1), 35–65.
10. Mnih, V. et al. (2015). Human-level Control through Deep Reinforcement Learning. Nature, 518, 529–533.
11. Mnih, V. et al. (2016). Asynchronous Methods for Deep Reinforcement Learning. ICML.
12. Morales, A. et al. (2025). Comparative Evaluation of Embedding Representations for Financial News Sentiment Analysis. arXiv:2512.13749.
13. Ortu, M. et al. (2022). Cryptocurrency Sentiment Analysis with BERT on GitHub and Reddit. Applied Sciences.
14. Park, S. et al. (2025). Towards Efficient FinBERT via Quantization and Coreset for Financial Sentiment Analysis. ACL FinNLP Workshop.
15. Peng, Y. et al. (2023). Automated Cryptocurrency Trading Using Ensemble Deep Reinforcement Learning. Expert Systems with Applications, 236.
16. Raffin, A. et al. (2021). Stable-Baselines3: Reliable Reinforcement Learning Implementations. JMLR.
17. Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. EMNLP.
18. Schulman, J. et al. (2017). Proximal Policy Optimization Algorithms. arXiv:1707.06347.
19. Sharma, V. et al. (2024). Enhancing Algorithmic Trading Strategies with Sentiment Analysis: A Reinforcement Learning Approach. Expert Systems with Applications.
20. Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction (2nd ed.). MIT Press.
21. Wang, Z. et al. (2025). Meta-Learning Reinforcement Learning for Crypto-Return Prediction. arXiv:2509.09751.
22. Wu, X. et al. (2023). Deep Reinforcement Learning for Financial Trading Using Multi-Modal Features. Expert Systems with Applications.
23. Yang, H., Liu, X.-Y., Zhong, S., & Walid, A. (2020). Deep Reinforcement Learning for Automated Stock Trading: An Ensemble Strategy. ACM ICAIF.
24. Zhang, Y. et al. (2024). Leveraging Heterogeneous Text Data for Reinforcement Learning-Based Stock Trading Strategies. Springer LNCS.
