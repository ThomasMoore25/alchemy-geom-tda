# Инструкция по запуску

## Что это

Полный пайплайн геометрического ML с топологическими фичами на датасете Alchemy (202,579 молекул):
6 моделей (FCNN, SchNet, EGNN, EGNN+TDA, EGNN Vector, EGNN Vector+TDA), Early Stopping,
multi-GPU через PyG DataParallel, AutoML для выбора архитектуры.

## Установка (1 раз)

```bash
git clone https://github.com/ThomasMoore25/alchemy-geom-tda.git
cd alchemy-geom-tda
pip install -r requirements.txt
python data/download_alchemy.py
```

## Быстрый запуск

### 1. Обучить одну модель (1-2 часа на GPU)

Параметры оптимизированы для быстрой сходимости:
- `--epochs 9999` (EarlyStopping остановит раньше)
- `--patience 10` (быстрее стопаем)
- `--es_mode or` (стоп при первой плохой метрике)
- `--batch_size 1024` (ускоряет обучение)

```bash
python src/train.py --model egnn --target all \
    --epochs 9999 --batch_size 1024 \
    --patience 10 --es_mode or \
    --device cuda --num_workers 4
```

### 2. Обучить все 7 моделей сразу (3-6 часов на GPU)

```bash
python src/run_all.py --models all \
    --epochs 9999 --batch_size 1024 \
    --patience 10 --es_mode or \
    --device cuda --num_workers 4 --multi_gpu
```

### 3. EGNN Tensor — часть B (вектор μ + тензор α, 1-2 часа)

Физически корректная модель: предсказывает полный вектор дипольного момента
μ ∈ R³ и полный тензор поляризуемости α ∈ R^(3×3).

```bash
python src/train.py --model egnn_tensor --target all \
    --predict_tensor_alpha \
    --epochs 9999 --batch_size 1024 \
    --patience 10 --es_mode or \
    --device cuda --num_workers 4
```

### 4. Построить графики обучения

```bash
python src/plot.py --input_dir results/experiments/batch_size_1024 \
    --save_dir results/figures/batch_size_1024 --no-show
```

### 5. Оценка устойчивости к шуму

```bash
python src/eval_robustness.py \
    --model egnn --target all \
    --checkpoint checkpoints/egnn_all_best.pt \
    --target_stats results/experiments/batch_size_1024/target_stats_egnn_all.json \
    --noise_sigma 0.0,0.05,0.10,0.15 --device cuda
```

### 6. AutoML — автоматически выбрать архитектуру (программа максимум)

```bash
python src/automl/run.py --data_dir data/alchemy \
    --n_molecules 100 --threshold 0.95 \
    --output_json results/automl/recommendation.json
```

## Ключевые параметры

| Параметр | По умолчанию | Описание |
|---|---|---|
| `--model` | (обязательный) | `fcnn` / `schnet` / `egnn` / `egnn_tda` / `egnn_vector` / `egnn_vector_tda` / `egnn_tensor` (часть B) |
| `--target` | `all` | `mu` / `alpha` / `gap` / `all` |
| `--epochs` | `9999` | Максимум (EarlyStopping остановит раньше) |
| `--batch_size` | `1024` | Размер батча |
| `--lr` | `1e-3` | Learning rate |
| `--device` | `auto` | `cpu` / `cuda` / `cuda:0` / `cuda:1` |
| `--multi_gpu` | off | PyG DataParallel для multi-GPU |
| `--num_workers` | `4` | DataLoader workers |
| `--tda_mode` | `concat` | `concat` / `film` — способ интеграции TDA |
| `--es_mode` | `and` | `and` / `or` / `loss_only` — EarlyStopping режим |
| `--patience` | `15` | EarlyStopping patience |
| `--noise` | `0.0` | Шум в координатах для robustness |
| `--noise_mode` | `test_only` | `test_only` / `train_val_test` / `train_only` / `eval_only` |
| `--config` | нет | Путь к YAML-конфигу (`configs/default.yaml`) |

## EarlyStopping (многометричный)

Из репозитория (v27+), поддерживает 3 режима (`--es_mode`):
- `and` (по умолчанию) — остановка, когда ВСЕ метрики перестают улучшаться
- `or` — остановка, когда ХОТЯ БЫ ОДНА метрика перестала улучшаться (быстрее)
- `loss_only` — только val_loss (классический ES)

Отслеживает: `val_loss`, `val_mu_mae`, `val_alpha_mae`, `val_gap_mae`.
Best checkpoint сохраняется по `val_loss` в RAM, на диск — один раз в конце.

## Multi-GPU (PyG DataParallel)

Флаг `--multi_gpu` включает `torch_geometric.nn.DataParallel` — корректно работает
с PyG `Data` объектами (в отличие от `nn.DataParallel`). Требует `device_count() > 1`.

```bash
python src/run_all.py --multi_gpu --batch_size 1024 --num_workers 4
# 2 GPU по 512 молекул на каждый
```

`DataListLoader` возвращает `list[Data]` вместо `Batch` — это контракт PyG DataParallel.

## Структура выводов

После обучения в `results/experiments/batch_size_<BS>/`:
- `history_<model>_<target>_<timestamp>.csv` — метрики по эпохам
- `args_<model>_<target>.json` — все гиперпараметры (для воспроизводимости)
- `target_stats_<model>_<target>.json` — mean/std для денормализации
- `summary_<timestamp>.csv` — сводка всех моделей (после `run_all.py`)

В `checkpoints/`:
- `<model>_<target>_best.pt` — лучший чекпойнт (state_dict)

## Smoke-тест (проверка установки, 5 минут на CPU)

```bash
python src/train.py --model egnn --target all \
    --epochs 3 --max_train 100 --max_val 20 --max_test 20 \
    --batch_size 32 --device cpu
```

Если converged (loss падает) — установка корректна.

## TDD: тесты

```bash
pytest                          # все 108 тестов
pytest tests/test_priors.py     # только AutoML тесты
ruff check src/ tests/ data/    # линтер
```

## Ноутбук

См. `notebooks/alchemy_geom_tda_quickstart.ipynb` — интерактивный туториал
с пошаговым запуском всех этапов.
