"""Pytest configuration: добавляет src/ в sys.path для импортов."""
import sys
from pathlib import Path

# Добавляем src/ в sys.path для импортов вида `from models.egnn import ...`
SRC = Path(__file__).parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Добавляем корень проекта для импортов вида `from src.data import ...`
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
