# Alchemy GeomML + TDA

Предсказание квантово-механических свойств молекул датасета [Alchemy](https://arxiv.org/pdf/1906.09427) (202,579 молекул) с использованием геометрического глубокого обучения и топологического анализа данных.

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
| `--multi_gpu` | off | Включить PyG DataParallel |
| `--num_workers` | 4 | DataLoader workers |
| `--output_dir` | auto | `results/experiments/batch_size_{bs}` |

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
├── requirements.txt                # Зависимости
├── .gitignore
├── data/
│   └── download_alchemy.py         # Скачивание Alchemy v20191129
├── src/
│   ├── __init__.py
│   ├── data.py                     # Парсинг SDF + final_version.csv
│   ├── dataset.py                  # PyG AlchemyDataset (InMemoryDataset)
│   ├── utils.py                    # Сиды, логирование, AverageMeter
│   ├── metrics.py                  # MAE для mu/alpha/gap
│   ├── train.py                    # Главный скрипт обучения
│   ├── run_all.py                  # Запуск нескольких моделей + summary CSV
│   ├── plot.py                     # Графики обучения из CSV
│   ├── early_stopping.py           # Multi-metric Early Stopping
│   ├── models/
│   │   ├── __init__.py
│   │   ├── fcnn.py                 # FCNN baseline
│   │   ├── schnet.py               # SchNet baseline
│   │   ├── egnn.py                 # EGNN (update_coors=False)
│   │   ├── egnn_tda.py             # EGNN + TDA (concat)
│   │   ├── egnn_vector.py          # EGNN с векторным mu
│   │   ├── egnn_vector_tda.py      # EGNN Vector + TDA
│   │   └── knn.py                  # Свой kNN graph (без pyg-lib)
│   └── tda/
│       ├── __init__.py
│       ├── features.py             # Vietoris-Rips, Betti curves (GUDHI)
│       └── film.py                 # FiLM conditioning
├── configs/
│   └── default.yaml
├── notebooks/
│   └── 01_eda.py                   # EDA датасета
└── results/
    ├── table.md                    # Сводная таблица результатов
    ├── experiments/
    │   └── batch_size_256/         # Истории обучения (CSV)
    └── figures/
        └── batch_size_256/         # Графики (PNG)
```

## Результаты (v26.1, bs=256, полный датасет)

| Модель | mu_mae | alpha_mae | gap_mae | test_loss |
|--------|--------|-----------|---------|-----------|
| FCNN | 0.85 | 2.27 | 0.026 | 1.24 |
| SchNet | 0.13 | 0.44 | 0.003 | 0.19 |
| EGNN | 0.18 | 0.39 | 0.004 | 0.23 |

EGNN+TDA, EGNN Vector, EGNN Vector+TDA — обучаются (см. `KAGGLE_RUN.md`).

## Ссылки

- **Alchemy dataset:** Chen et al., 2019. [arXiv:1906.09427](https://arxiv.org/pdf/1906.09427)
- **Скачивание данных:** https://alchemy.tencent.com/data/alchemy-v20191129.zip
- **EGNN:** Satorras et al., 2021. [arXiv:2107.02994](https://arxiv.org/abs/2107.02994)
- **egnn-pytorch:** https://github.com/lucidrains/egnn-pytorch
- **Equivariant ML обзор:** Weiler, 2023. [блог](https://maurice-weiler.gitlab.io/blog_post/cnn-book_1_equivariant_networks/)
- **TDA обзор:** Chazal & Michel, 2019. [arXiv:1904.11044](https://arxiv.org/pdf/1904.11044)
- **GUDHI:** https://gudhi.inria.fr/
