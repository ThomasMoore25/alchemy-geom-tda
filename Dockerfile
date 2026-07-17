# Dockerfile for alchemy-geom-tda (v32+)
#
# Базовый образ с CUDA + Python. Для CPU-only замените на python:3.11-slim.
#
# Сборка:
#   docker build -t alchemy-geom-tda .
#
# Запуск (с GPU):
#   docker run --gpus all -v $(pwd)/data:/app/data -v $(pwd)/results:/app/results alchemy-geom-tda
#
# Запуск (CPU only):
#   docker run -v $(pwd)/data:/app/data -v $(pwd)/results:/app/results alchemy-geom-tda python src/train.py --model egnn --device cpu
#
# Версии: см. requirements.txt. Для воспроизводимости pinned в pyproject.toml.

FROM python:3.11-slim

# Установка системных зависимостей для rdkit и gudhi
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libboost-all-dev \
    libeigen3-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Сначала копируем только зависимости — для кэширования слоя
COPY requirements.txt requirements-dev.txt pyproject.toml ./

# Установка Python-зависимостей
# Используем --system-packages, чтобы избежать venv (в Docker не нужно)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Копируем исходный код
COPY src/ ./src/
COPY data/ ./data/
COPY configs/ ./configs/
COPY notebooks/ ./notebooks/
COPY tests/ ./tests/
COPY README.md LICENSE ./

# Создаём директории для данных и результатов (точки монтирования)
RUN mkdir -p /app/data/alchemy /app/results /app/checkpoints /app/logs

# Python path
ENV PYTHONPATH=/app:/app/src

# Дефолтная команда — помощь
CMD ["python", "src/train.py", "--help"]
