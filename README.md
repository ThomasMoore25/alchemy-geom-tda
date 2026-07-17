# Alchemy GeomML + TDA

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python Tests](https://img.shields.io/badge/tests-108%20passing-green.svg)](tests/)
[![Version](https://img.shields.io/badge/version-v32.38-blue.svg)](CHANGELOG)

Предсказание квантово-механических свойств молекул датасета [Alchemy](https://arxiv.org/pdf/1906.09427) (202,579 молекул) с использованием геометрического глубокого обучения и топологического анализа данных.

## Задание

**Программа минимум:** ✅ выполнено
- (а) Простейший полный пайплайн для построения решения с геометрическим ML на датасете с известными геометрическими prior-ами
- (b) Обучить геометрическую модель на исходных данных и с использованием топологических фич
- (с) Enjoy!

**Программа максимум:** ✅ выполнено (v32.38+)
- Обратная задача экстракции prior-ов из TDA → `src/tda/priors.py`
- Пайплайн автоматического выбора оптимальной DL архитектуры → `src/automl/select.py`

## Модели

| # | Модель | Эквивариантность | TDA | Описание |
|---|--------|------------------|-----|----------|
| 1 | **FCNN** | нет | нет | Базовый многослойный перцептрон (на глобальных дескрипторах) |
| 2 | **SchNet** | трансляции + перестановки | нет | Continuous filter convolutions |
| 3 | **EGNN** | E(3) (трансляции + вращения + перестановки) | нет | E(3)-эквивариантная сеть (egnn-pytorch) |
| 4 | **EGNN+TDA** | E(3) | да | EGNN + топологические признаки (Vietoris-Rips, Betti curves) |
| 5 | **EGNN Vector** | E(3) | нет | EGNN с векторным выходом для μ (через Σ qᵢ·(rᵢ−COM)) |
| 6 | **EGNN Vector+TDA** | E(3) | да | EGNN Vector + топологические признаки |

## Таргеты

Из Alchemy `final_version.csv`:
- **mu** (Дебай) — норма вектора дипольного момента |μ|
- **alpha** (a₀³) — изотропная поляризуемость tr(α)/3
- **gap** (Хартри) — HOMO-LUMO gap

## Симметрии задачи

Молекулы обладают симметриями:
- **Трансляции:** сдвиг всей молекулы не меняет свойства
- **Вращения SO(3):** поворот молекулы сохраняет химию
- **Перестановки одинаковых атомов:** порядок нумерации произволен

EGNN кодирует эти симметрии в архитектуру, обеспечивая эквивариантность внутренних признаков и инвариантность скалярных выходов.

## Архитектура

```
                   ┌─────────────────────┐
                   │  TDA-модуль         │
   3D координаты ──┤  Vietoris-Rips      │── TDA-фичи (52D) ──┐
   атомов          │  Betti curves       │                      │
                   │  Persistence entropy│                      │
                   └─────────────────────┘                      │
                                                                ▼
   Атомы +         ┌─────────────────────┐                ┌──────────────┐
   координаты  ───▶│  EGNN               │───────────────▶│  Heads       │──▶ mu, alpha, gap
                   │  (E(3)-эквивариант) │                │  (скаляры)   │
                   │  update_coors=False │                └──────────────┘
                   └─────────────────────┘
```

## Установка

```bash
pip install -r requirements.txt
```

Или вручную:
```bash
pip install torch torch-geometric gudhi rdkit egnn-pytorch pandas matplotlib numpy scipy
```

## Использование

### 1. Загрузка датасета Alchemy

```bash
python data/download_alchemy.py
```

Скачивает ~136 МБ, распаковывает в `data/alchemy/Alchemy-v20191129/`.

### 2. Обучение одной модели

```bash
# EGNN (основная модель)
python src/train.py --model egnn --target all --epochs 100

# EGNN + TDA
python src/train.py --model egnn_tda --target all --epochs 100

# EGNN с векторным выходом mu
python src/train.py --model egnn_vector --target all --epochs 100

# Все доступные модели: fcnn, schnet,
#                       egnn, egnn_tda, egnn_vector, egnn_vector_tda

# Для отладки (на 1000 молекулах)
python src/train.py --model egnn --target all --epochs 5 --max_train 1000
```

### 3. Запуск всех моделей

```bash
# Все 6 EGNN-моделей (по умолчанию)
python src/run_all.py --epochs 9999 --batch_size 512 --multi_gpu --num_workers 4

# Только конкретные модели
python src/run_all.py --models egnn,egnn_tda

# Все модели включая baselines
python src/run_all.py --models all
```

### 4. Multi-GPU (PyG DataParallel)

```bash
python src/run_all.py --multi_gpu --batch_size 1024 --num_workers 4
```

`--multi_gpu` включает `torch_geometric.nn.DataParallel` — корректно работает с PyG Data объектами (в отличие от `nn.DataParallel`).

### 5. Графики обучения

```bash
# Все CSV в директории + comparison
python src/plot.py --input_dir results/experiments/batch_size_512 --save_dir results/figures/batch_size_512 --no-show

# Через Python
from plot import plot_main
plot_main()
```

### 6. Запуск на Kaggle

См. **[KAGGLE_RUN.md](KAGGLE_RUN.md)** — подробная инструкция с примерами ячеек.

**Важно:** на Kaggle всегда запускай через **Save Version (Commit)**, не через интерактивную сессию. Иначе output не сохранится.

## Ключевые параметры обучения

Canonical defaults (v32+, едины для `train.py`, `run_all.py`, `configs/default.yaml`):

| Параметр | По умолчанию | Описание |
|----------|--------------|----------|
| `--config` | нет | Путь к YAML-конфигу (CLI > YAML > argparse default) |
| `--lr` | 1e-3 | Learning rate |
| `--batch_size` | 1024 | Размер батча |
| `--epochs` | 9999 | Максимум эпох (EarlyStopping остановит раньше) |
| `--patience` | 15 | Early Stopping patience |
| `--lr_patience` | 5 | ReduceLROnPlateau patience |
| `--hidden_channels` | 128 | Скрытая размерность |
| `--num_layers` | 4 | Количество слоёв EGNN |
| `--cutoff` | 5.0 | Радиус отсечения + нормализация координат |
| `--k_neighbors` | 16 | Число соседей в kNN-графе |
| `--m_dim` | 32 | Размерность m в EGNN_Sparse |
| `--tda_mode` | concat | Способ интеграции TDA: `concat` или `film` |
| `--es_mode` | and | Early stopping: `and`, `or`, `loss_only` |
| `--noise` | 0.0 | Шум в координатах (для robustness test) |
| `--noise_mode` | test_only | `test_only`/`train_val_test`/`train_only`/`eval_only` |
| `--device` | auto | `auto`/`cpu`/`cuda`/`cuda:N` |
| `--multi_gpu` | off | Включить PyG DataParallel |
| `--num_workers` | 4 | DataLoader workers |
| `--output_dir` | auto | `results/experiments/batch_size_{bs}` |

## Robustness evaluation (v32+)

После обучения модели можно оценить её устойчивость к шуму в координатах:

```bash
python src/eval_robustness.py \
    --model egnn --target all \
    --checkpoint checkpoints/egnn_all_best.pt \
    --target_stats results/experiments/batch_size_512/target_stats_egnn_all.json \
    --noise_sigma 0.0,0.05,0.10,0.15
```

Сохраняет CSV с метриками при разных sigma в `results/robustness/<model>_robustness.csv`.

## AutoML: автоматический выбор архитектуры (программа максимум, v32.38+)

Обратная задача: по TDA-фичам датасета извлечь геометрические priors и
автоматически рекомендовать оптимальную архитектуру.

```bash
python src/automl/select.py \
    --data_dir data/alchemy \
    --n_molecules 100 \
    --threshold 0.95 \
    --output_json results/automl/recommendation.json
```

С quick-train сравнением candidate-моделей:

```bash
python src/automl/select.py \
    --data_dir data/alchemy \
    --n_molecules 500 \
    --epochs 3 \
    --quick_train \
    --candidates fcnn,schnet,egnn,egnn_tda \
    --output_json results/automl/recommendation.json
```

**Как это работает:**

1. Загружает N молекул из датасета (только координаты).
2. Для каждой молекулы применяет случайные изометрические преобразования
   (translation, rotation, permutation) и сравнивает TDA-фичи до/после.
3. Если TDA инвариантна к преобразованию (score >= threshold) →
   соответствующий prior присутствует в данных.
4. Рекомендует архитектуру:
   - Все три E(3) симметрии → `egnn` (E(3)-эквивариантная сеть)
   - Только translation + permutation → `schnet` (без rotation)
   - Только permutation → `fcnn` (baseline достаточно)
   - Сильных priors нет → `fcnn` (baseline достаточно)
5. Опционально: quick-train несколько моделей на подвыборке и
   сравнить val_loss.

**На Alchemy датасете** (30 молекул, threshold=0.95):
- translation_invariance: 1.0000
- rotation_invariance: 1.0000
- permutation_invariance: 1.0000
- **Recommended: `egnn`** (все три E(3) симметрии присутствуют)

Отчёт сохраняется в `results/automl/recommendation.json` со всеми
деталями priors и quick-train результатов.

## Docker (v32+)

Для воспроизводимости окружения:

```bash
# Сборка
docker build -t alchemy-geom-tda .

# Запуск (CPU only)
docker run -v $(pwd)/data:/app/data -v $(pwd)/results:/app/results \
    alchemy-geom-tda python src/train.py --model egnn --device cpu --max_train 1000

# Запуск (с GPU — требует nvidia-docker)
docker run --gpus all -v $(pwd)/data:/app/data -v $(pwd)/results:/app/results \
    alchemy-geom-tda python src/train.py --model egnn
```

Data и results монтируются как volumes — данные не дублируются в image.

## Логирование

Каждая эпоха логируется в формате:
```
Epoch  10/9999 | train_loss=0.8623 | val_loss=0.7821 | mu_mae=0.5537 | alpha_mae=1.4746 | gap_mae=0.0158 | ES=[val_loss: 0/15 | val_mu_mae: 0/15 | val_alpha_mae: 0/15 | val_gap_mae: 0/15] | lr=5.00e-04 | 162.1s | RESET:val_loss,val_mu_mae,val_alpha_mae,val_gap_mae | ★ best
```

- **ES** — счётчики Early Stopping для каждой метрики
- **RESET:...** — какие метрики улучшились в эту эпоху
- **★ best** — отметка лучшей эпохи (без сохранения на диск)

После Early Stopping:
- `→ Сохранён best checkpoint → egnn_tda_all_best.pt` — чекпойнт сохранён
- `=== Финальная оценка на test ===` — test метрики
- `История сохранена в results/.../history_egnn_tda_all_<timestamp>.csv` — CSV с историей

В конце `run_all.py` создаётся `summary_<timestamp>.csv` со сводкой всех моделей.

## Структура репозитория

```
alchemy-geom-tda/
├── README.md                       # Этот файл
├── KAGGLE_RUN.md                   # Инструкция для Kaggle (Commit vs Interactive)
├── requirements.txt                # Runtime зависимости (v32+ pinned)
├── requirements-dev.txt            # + pytest, ruff, black, mypy
├── pyproject.toml                  # pytest/ruff/black/mypy config
├── LICENSE                         # MIT (v32+)
├── CITATION.cff                    # Для GitHub "Cite this repository"
├── .gitignore
├── .gitattributes                  # LFS tracking (опционально)
├── data/
│   ├── __init__.py
│   └── download_alchemy.py         # Скачивание Alchemy v20191129 + SHA256 check
├── src/
│   ├── __init__.py                 # __version__, package metadata
│   ├── data.py                     # Парсинг SDF + final_version.csv
│   ├── dataset.py                  # PyG AlchemyDataset + кэш по хешу
│   ├── utils.py                    # Сиды, логирование
│   ├── metrics.py                  # MAE, RMSE, R² для mu/alpha/gap
│   ├── train.py                    # Главный скрипт обучения + --config
│   ├── run_all.py                  # subprocess запуск нескольких моделей
│   ├── eval_robustness.py          # Прогон при разном шуме (v32+)
│   ├── plot.py                     # Графики обучения + parity plot
│   ├── early_stopping.py           # Multi-metric ES (--es_mode and/or/loss_only)
│   ├── models/
│   │   ├── __init__.py             # Экспорты 6 моделей + 6 build_* (v32+)
│   │   ├── fcnn.py                 # FCNN baseline
│   │   ├── schnet.py               # SchNet baseline
│   │   ├── egnn.py                 # EGNN (update_coors=False)
│   │   ├── egnn_tda.py             # EGNN + TDA (concat или film)
│   │   ├── egnn_vector.py          # EGNN с векторным mu
│   │   ├── egnn_vector_tda.py      # EGNN Vector + TDA
│   │   └── knn.py                  # kNN graph O(Σ n_i²) (v32+, был O(N²))
│   └── tda/
│       ├── __init__.py             # Экспорты TDA функций (v32+)
│       ├── features.py             # Vietoris-Rips, Betti curves (векторизовано v32+)
│       └── film.py                 # FiLM conditioning
├── configs/
│   └── default.yaml                # Canonical defaults (v32+ unified)
├── notebooks/
│   └── 01_eda.py                   # EDA датасета (v32+ fixed)
├── tests/                          # 89 pytest тестов (v32+)
│   ├── conftest.py
│   ├── test_knn.py
│   ├── test_early_stopping.py
│   ├── test_metrics.py
│   ├── test_tda_features.py
│   ├── test_data.py
│   ├── test_models.py
│   ├── test_train.py
│   └── test_dataset_cache.py
└── results/
    ├── table.md                    # Сводная таблица результатов
    ├── experiments/
    │   └── batch_size_256/         # Истории обучения (CSV)
    └── figures/
        └── batch_size_256/         # Графики (PNG)
```

## Результаты

### Baselines (v26.1, bs=256, полный датасет 202k молекул)

| Модель | mu_mae | alpha_mae | gap_mae | test_loss | Эпох |
|--------|--------|-----------|---------|-----------|------|
| FCNN   | 0.851  | 2.271     | 0.0261  | 1.235     | 176  |
| SchNet | 0.131  | 0.442     | 0.0033  | 0.186     | 226  |
| EGNN   | 0.179  | 0.393     | 0.0041  | 0.227     | 131  |

### EGNN-семья (v29, bs=512, полный датасет)

| Модель            | mu_mae | alpha_mae | gap_mae | test_loss | Эпох | Время (ч) |
|-------------------|--------|-----------|---------|-----------|------|-----------|
| EGNN              | 0.245  | 0.460     | 0.0046  | 0.285     | 164  | 3.77      |
| EGNN+TDA          | 0.274  | 0.498     | 0.0053  | 0.321     | 179  | 4.29      |
| EGNN Vector       | 0.710  | 0.390     | 0.0041  | 0.575     | 140  | 3.47      |
| EGNN Vector+TDA   | 0.704  | 0.357     | 0.0039  | 0.565     | 145  | 3.60      |

Подробности и интерпретация — в `results/table.md`.

## Ссылки

- **Alchemy dataset:** Chen et al., 2019. [arXiv:1906.09427](https://arxiv.org/pdf/1906.09427)
- **Скачивание данных:** https://alchemy.tencent.com/data/alchemy-v20191129.zip
- **EGNN:** Satorras et al., 2021. [arXiv:2107.02994](https://arxiv.org/abs/2107.02994)
- **egnn-pytorch:** https://github.com/lucidrains/egnn-pytorch
- **Equivariant ML обзор:** Weiler, 2023. [блог](https://maurice-weiler.gitlab.io/blog_post/cnn-book_1_equivariant_networks/)
- **TDA обзор:** Chazal & Michel, 2019. [arXiv:1904.11044](https://arxiv.org/pdf/1904.11044)
- **GUDHI:** https://gudhi.inria.fr/
