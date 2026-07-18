"""src.automl — автоматический выбор DL архитектуры (программа максимум).

Удобный импорт:
    from src.automl import recommend_architecture_for_dataset
"""
from .select import load_molecules_for_priors, quick_train_candidates

__all__ = [
    "load_molecules_for_priors",
    "quick_train_candidates",
]
