"""Тесты для src.tda.features — функции TDA без GUDHI."""

from tda.features import tda_feature_dim


def test_tda_feature_dim_default():
    """Дефолтная размерность: 16*3 + 3 + 1 = 52."""
    assert tda_feature_dim() == 52


def test_tda_feature_dim_custom_n_bins():
    """Кастомный n_bins: n_bins*3 + 3 + 1."""
    assert tda_feature_dim(n_bins=8) == 8 * 3 + 3 + 1  # = 28
    assert tda_feature_dim(n_bins=32) == 32 * 3 + 3 + 1  # = 100
    assert tda_feature_dim(n_bins=64) == 64 * 3 + 3 + 1  # = 196


def test_tda_feature_dim_custom_max_dim():
    """Кастомный max_dim: n_bins*(max_dim+1) + (max_dim+1) + 1."""
    assert tda_feature_dim(n_bins=16, max_dim=1) == 16 * 2 + 2 + 1  # = 35
    assert tda_feature_dim(n_bins=16, max_dim=3) == 16 * 4 + 4 + 1  # = 69


def test_tda_feature_dim_formula():
    """Формула: n_bins * (max_dim + 1) + (max_dim + 1) + 1."""
    for n_bins in [4, 8, 16, 32, 64]:
        for max_dim in [0, 1, 2, 3]:
            expected = n_bins * (max_dim + 1) + (max_dim + 1) + 1
            assert tda_feature_dim(n_bins=n_bins, max_dim=max_dim) == expected


def test_tda_feature_dim_consistent_with_model_default():
    """Размерность по умолчанию совпадает с захардкоженной в EGNNTDA."""
    # Из src/models/egnn_tda.py: tda_dim=52 по умолчанию
    assert tda_feature_dim(n_bins=16, max_dim=2) == 52
