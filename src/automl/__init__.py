"""src.automl — автоматический выбор DL архитектуры (программа максимум).

Удобный импорт:
    from src.automl import load_molecules_for_priors, quick_train_candidates
    from src.automl.run import main as automl_main
"""
from .run import load_molecules_for_priors, quick_train_candidates

__all__ = [
    "load_molecules_for_priors",
    "quick_train_candidates",
]
