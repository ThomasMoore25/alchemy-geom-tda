"""src.tda — экспорты TDA-модуля (v32+).

Удобный импорт:
    from src.tda import extract_tda_features, tda_feature_dim
    from src.tda import FiLMModulation, FiLMNodeModulation
"""

from .features import (
    betti_curve,
    compute_persistence,
    extract_tda_features,
    extract_tda_features_batch,
    persistence_entropy,
    tda_feature_dim,
)
from .film import FiLMModulation, FiLMNodeModulation

__all__ = [
    "extract_tda_features",
    "extract_tda_features_batch",
    "tda_feature_dim",
    "compute_persistence",
    "betti_curve",
    "persistence_entropy",
    "FiLMModulation",
    "FiLMNodeModulation",
]
