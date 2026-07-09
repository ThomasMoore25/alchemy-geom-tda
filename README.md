# Alchemy GeomML + TDA

Предсказание свойств молекул датасета [Alchemy](https://arxiv.org/pdf/1906.09427) с использованием:

- **Геометрического ML:** E(3)-эквивариантная нейросеть PaiNN
- **Топологического анализа данных (TDA):** персистентная гомология облака 3D-точек атомов (Vietoris-Rips) + интеграция через FiLM conditioning
- **Baselines:** FCNN, SchNet

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

PaiNN кодирует эти симметрии в архитектуру, обеспечивая эквивариантность внутренних
признаков и инвариантность скалярных выходов.

## Архитектура

```
                   ┌─────────────────────┐
                   │  TDA-модуль         │
   3D координаты ──┤  Vietoris-Rips      │── TDA-фичи (52D) ──┐
   атомов          │  Betti curves       │                      │ FiLM
                   │  Persistence entropy│                      │ conditioning
                   └─────────────────────┘                      │
                                                                ▼
   Атомы +         ┌─────────────────────┐                ┌──────────────┐
   координаты  ───▶│  PaiNN              │───────────────▶│  Heads       │──▶ mu, alpha, gap
                   │  (E(3)-эквивариант) │                │  (скаляры)   │
                   └─────────────────────┘                └──────────────┘
```

## Установка

```bash
pip install -r requirements.txt
```

## Использование

### 1. Загрузка датасета Alchemy

```bash
python data/download_alchemy.py
```

Скачивает ~136 МБ, распаковывает в `data/alchemy/Alchemy-v20191129/`.

### 2. Обучение моделей

```bash
# FCNN baseline
python src/train.py --model fcnn --target all --epochs 50

# SchNet baseline
python src/train.py --model schnet --target all --epochs 50

# PaiNN (основная модель)
python src/train.py --model painn --target all --epochs 100

# PaiNN + TDA (наша финальная модель)
python src/train.py --model painn_tda --target all --epochs 100

# Для отладки (на 1000 молекулах)
python src/train.py --model painn --target all --epochs 5 --max_train 1000
```

### 3. Тестирование робастности к шуму

```bash
python src/train.py --model painn_tda --target all --eval_only \
    --checkpoint checkpoints/painn_tda_all_best.pt --noise 0.10
```

## Структура репозитория

```
alchemy-geom-tda/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── download_alchemy.py        # Скачивание Alchemy v20191129
├── src/
│   ├── data.py                    # Парсинг SDF + final_version.csv
│   ├── dataset.py                 # PyG AlchemyDataset
│   ├── utils.py                   # Сиды, логирование
│   ├── metrics.py                 # MAE для mu/alpha/gap
│   ├── train.py                   # Главный скрипт обучения
│   ├── models/
│   │   ├── fcnn.py                # FCNN baseline
│   │   ├── schnet.py              # SchNet baseline
│   │   ├── painn.py               # PaiNN с скалярными выходами
│   │   └── painn_tda.py           # PaiNN + TDA через FiLM
│   └── tda/
│       ├── features.py            # Vietoris-Rips, Betti curves
│       └── film.py                # FiLM conditioning
├── configs/
│   └── default.yaml
├── notebooks/
│   └── 01_eda.py                  # EDA датасета
└── results/
    └── table.md
```

## Ссылки

- **Alchemy dataset:** Chen et al., 2019. [arXiv:1906.09427](https://arxiv.org/pdf/1906.09427)
- **Скачивание данных:** https://alchemy.tencent.com/data/alchemy-v20191129.zip
- **PaiNN:** Schütt et al., 2021. [arXiv:2102.03150](https://arxiv.org/abs/2102.03150)
- **Equivariant ML обзор:** Weiler, 2023. [блог](https://maurice-weiler.gitlab.io/blog_post/cnn-book_1_equivariant_networks/)
- **TDA обзор:** Chazal & Michel, 2019. [arXiv:1904.11044](https://arxiv.org/pdf/1904.11044)
