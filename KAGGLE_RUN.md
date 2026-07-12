# Как запускать на Kaggle (v29)

## ⚠️ ВАЖНО: Commit vs Interactive

Kaggle различает два режима:

| Режим | Что происходит с output |
|-------|-------------------------|
| **Interactive** (запуск ячейки в браузере) | Output виден в браузере, но **НЕ сохраняется** на сервере. Если ядро умирает (12h timeout, OOM, закрыл вкладку) — **всё пропадает** |
| **Save Version / Commit** | Kaggle перезапускает ноутбук в фоне. После завершения output **сохраняется навсегда** |

**ВСЕГДА запускай через Save Version (Commit)**, а не через обычный запуск ячейки.

## Шаги для запуска на Kaggle

### 1. Загрузить v29_1.zip

Скачай `alchemy-geom-tda-v29_1.zip` и загрузи в Kaggle через **Add Data → Upload**.

### 2. Создать ноутбук

Создай новый notebook. В первой ячейке:

```python
%cd /kaggle/working
!rm -rf alchemy-geom-tda
!git clone https://github.com/zzz20/alchemy-geom-tda.git
%cd alchemy-geom-tda

# Распаковываем v29_1
!unzip -o /kaggle/input/<имя-загруженного-dataset>/alchemy-geom-tda-v29_1.zip

# ЗАЩИТА: удалить корневые .py если есть (могут тенть src/)
!rm -f train.py run_all.py plot.py early_stopping.py

# Проверка v29-фич
!echo "=== Проверка v29 ==="
!echo "PyGDataParallel: $(grep -c 'PyGDataParallel' src/train.py)"
!echo "DataListLoader: $(grep -c 'DataListLoader' src/train.py)"
!echo "multi_gpu: $(grep -c 'multi_gpu' src/train.py)"
!echo "num_workers: $(grep -c 'num_workers' src/train.py)"
!echo "★ best: $(grep -c '★ best' src/train.py)"
!echo "Корневых .py: $(ls *.py 2>/dev/null | wc -l)"
```

### 3. Установить зависимости

Вторая ячейка:

```python
!pip install torch-geometric gudhi rdkit egnn-pytorch -q
```

### 4. Скачать датасет Alchemy

Третья ячейка:

```python
!python data/download_alchemy.py
```

### 5. Запустить обучение

Четвёртая ячейка (последняя):

```python
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # раскомментируй если проблемы с multi-GPU

import sys
sys.path.insert(0, '/kaggle/working/alchemy-geom-tda/src')

sys.argv = ['run_all.py',
    '--epochs', '9999',
    '--batch_size', '1024',         # 2 GPU по 512 — попробуй, если OOM → 512
    '--hidden_channels', '128',
    '--num_layers', '4',
    '--lr', '1e-3',                  # 2× больше дефолта — быстрее сходится
    '--lr_patience', '3',            # быстрее снижать lr при plateau
    '--device', 'cuda',
    '--patience', '15',
    '--models', 'egnn,egnn_tda,egnn_vector,egnn_vector_tda',
    '--num_workers', '4',
    '--multi_gpu',                   # ВКЛЮЧАЕМ PyG DataParallel для 2 GPU
]

from run_all import main; main()
```

### 6. ⚠️ СОХРАНИТЬ ЧЕРЕЗ COMMIT

**НЕ запускай последнюю ячейку в interactive режиме.** Вместо этого:

1. Нажми **Save Version** в правом верхнем углу
2. Выбери **Save & Run All (Commit)**
3. Дай название версии (например: `v29 egnn+tda bs512`)
4. Нажми **Save**

Kaggle запустит ноутбук **в фоновом режиме** с нуля. После завершения:
- Output сохранится навсегда
- Появится в **Versions** tab
- Можно скачать через `kaggle kernels output`

### 7. Следить за прогрессом

После Commit:
- Зайди в **Versions** tab
- Открой последнюю версию
- Там будет live log выполнения
- Можно смотреть на процесс без блокировки

### 8. Скачать результаты

После завершения Commit:

```bash
# Через kaggle CLI
pip install kaggle
kaggle kernels output zahartedeev/<notebook-slug> -p ~/alchemy_output

# Или через браузер
# Kaggle → notebook → Output tab → Download
```

## Что сохранится

После успешного Commit в Output будут:
- `checkpoints/<model>_all_best.pt` — чекпойнт каждой модели
- `results/experiments/batch_size_512/history_<model>_all_<ts>.csv` — история
- `results/experiments/batch_size_512/summary_<ts>.csv` — сводка
- `data/alchemy/processed/*.pt` — кэш TDA (для повторных запусков)

## Если 12h лимит истечёт

При Commit Kaggle даёт **12h на выполнение**. Если не успел:
- Все файлы созданные до таймаута **сохранятся**
- Но если `torch.save()` не успел вызваться (модель не дошла до Early Stopping) — чекпойнта не будет
- В v29 `torch.save()` вызывается **один раз в конце** — если нужно periodic save, попроси v30

## Частые проблемы

### OOM (CUDA out of memory)
Уменьши `--batch_size` до 256:
```python
'--batch_size', '256',
```

### DataParallel device mismatch
Раскомментируй `os.environ['CUDA_VISIBLE_DEVICES'] = '0'` в начале ноутбука.

### Не нужны все 4 модели
Сократи список:
```python
'--models', 'egnn,egnn_tda',  # только 2 модели
```

## Поддержание актуальной документации

При каждом изменении кода:

1. **README.md** — обновить:
   - Список моделей и их описания
   - Параметры обучения (если поменялись дефолты)
   - Структуру репозитория (новые/удалённые файлы)
   - Результаты в таблице (после завершения обучения)

2. **KAGGLE_RUN.md** — обновить:
   - Параметры запуска в ячейках (если поменялись)
   - Частые проблемы (если нашли новые)

3. **results/table.md** — обновить:
   - Метрики после каждого завершённого обучения
   - Статус моделей (✅/⏳/❌)

4. **Docstrings в .py файлах** — обновить:
   - Заголовок файла (если архитектура изменилась)
   - Примеры запуска (если CLI параметры поменялись)
   - Убрать упоминания старых версий (v17, v26, etc.)

5. **Чек-лист** (если в нём появились новые пункты) — обновить:
   - Все пункты должны быть актуальны
   - Удалить устаревшие правила

**Принцип:** документация должна соответствовать коду. Если код изменился — документация тоже.
