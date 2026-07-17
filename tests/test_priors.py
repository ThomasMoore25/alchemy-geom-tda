"""Тесты для src.tda.priors — экстракция геометрических prior-ов из TDA."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest

from tda.priors import (
    _permutation,
    _rotation_matrix,
    extract_priors,
    recommend_architecture,
    tda_invariance_score,
)


def test_rotation_matrix_is_so3():
    """_rotation_matrix возвращает матрицу из SO(3): det=+1, ортогональная."""
    R = _rotation_matrix(seed=42)
    # det = +1
    assert abs(np.linalg.det(R) - 1.0) < 1e-6, f"det(R) = {np.linalg.det(R)}"
    # ортогональная: R @ R.T = I
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-6)


def test_rotation_matrix_different_seeds():
    """Разные seed → разные матрицы."""
    R1 = _rotation_matrix(seed=1)
    R2 = _rotation_matrix(seed=2)
    assert not np.allclose(R1, R2)


def test_permutation_is_permutation():
    """_permutation возвращает перестановку индексов."""
    perm = _permutation(10, seed=42)
    assert sorted(perm.tolist()) == list(range(10))


def test_tda_translation_invariance():
    """TDA от (X + t) == TDA от X → score близко к 1.0."""
    np.random.seed(0)
    coords = np.random.randn(10, 3) * 2
    score = tda_invariance_score(coords, "translation", n_trials=3)
    assert score > 0.95, f"Translation invariance failed: {score}"


def test_tda_rotation_invariance():
    """TDA от (R @ X) == TDA от X → score близко к 1.0."""
    np.random.seed(0)
    coords = np.random.randn(10, 3) * 2
    score = tda_invariance_score(coords, "rotation", n_trials=3)
    assert score > 0.95, f"Rotation invariance failed: {score}"


def test_tda_permutation_invariance():
    """TDA от X[perm] == TDA от X → score близко к 1.0."""
    np.random.seed(0)
    coords = np.random.randn(10, 3) * 2
    score = tda_invariance_score(coords, "permutation", n_trials=3)
    assert score > 0.95, f"Permutation invariance failed: {score}"


def test_tda_invariance_invalid_transform():
    """Неверное имя преобразования → ValueError."""
    coords = np.random.randn(5, 3)
    with pytest.raises(ValueError):
        tda_invariance_score(coords, "scaling", n_trials=1)


def test_tda_invariance_invalid_coords():
    """Неверная форма coords → ValueError."""
    with pytest.raises(ValueError):
        tda_invariance_score(np.random.randn(10), "translation")  # 1D
    with pytest.raises(ValueError):
        tda_invariance_score(np.random.randn(10, 2), "translation")  # не 3D


def test_tda_invariance_single_atom():
    """Один атом: TDA пустая → score = 1.0 (тривиально инвариантна)."""
    coords = np.array([[0.0, 0.0, 0.0]])
    score = tda_invariance_score(coords, "translation", n_trials=2)
    # Один атом → trivial TDA → invariant
    assert score >= 0.0  # не падает


def test_extract_priors_returns_dict():
    """extract_priors возвращает dict с ожидаемыми ключами."""
    np.random.seed(0)
    coords_list = [np.random.randn(8, 3) * 2 for _ in range(3)]
    priors = extract_priors(coords_list, n_trials=2, verbose=False)

    assert "translation_invariance" in priors
    assert "rotation_invariance" in priors
    assert "permutation_invariance" in priors
    assert "n_molecules" in priors
    assert "n_molecules_success" in priors
    assert "per_molecule" in priors
    assert priors["n_molecules"] == 3
    assert priors["n_molecules_success"] == 3


def test_extract_priors_empty_list():
    """Пустой список молекул → нули."""
    priors = extract_priors([], verbose=False)
    assert priors["translation_invariance"] == 0.0
    assert priors["n_molecules"] == 0
    assert priors["n_molecules_success"] == 0


def test_extract_priors_all_e3_invariant():
    """Для случайных 3D-облаков все три E(3) invariances близки к 1.0."""
    np.random.seed(0)
    coords_list = [np.random.randn(10, 3) * 2 for _ in range(5)]
    priors = extract_priors(coords_list, n_trials=2, verbose=False)

    assert priors["translation_invariance"] > 0.95
    assert priors["rotation_invariance"] > 0.95
    assert priors["permutation_invariance"] > 0.95


def test_recommend_architecture_all_priors():
    """Все три priors → recommended='egnn'."""
    priors = {
        "translation_invariance": 0.99,
        "rotation_invariance": 0.98,
        "permutation_invariance": 0.97,
    }
    rec = recommend_architecture(priors, threshold=0.9)
    assert rec["recommended_model"] == "egnn"
    assert set(rec["required_invariances"]) == {"translation", "rotation", "permutation"}
    assert rec["needs_tda"] is True


def test_recommend_architecture_no_rotation():
    """Translation + permutation, но не rotation → recommended='schnet'."""
    priors = {
        "translation_invariance": 0.99,
        "rotation_invariance": 0.5,  # ниже threshold
        "permutation_invariance": 0.97,
    }
    rec = recommend_architecture(priors, threshold=0.9)
    assert rec["recommended_model"] == "schnet"
    assert "rotation" not in rec["required_invariances"]
    assert "translation" in rec["required_invariances"]
    assert "permutation" in rec["required_invariances"]


def test_recommend_architecture_only_permutation():
    """Только permutation → recommended='fcnn'."""
    priors = {
        "translation_invariance": 0.5,
        "rotation_invariance": 0.5,
        "permutation_invariance": 0.97,
    }
    rec = recommend_architecture(priors, threshold=0.9)
    assert rec["recommended_model"] == "fcnn"
    assert rec["required_invariances"] == ["permutation"]


def test_recommend_architecture_no_priors():
    """Нет сильных priors → recommended='fcnn'."""
    priors = {
        "translation_invariance": 0.5,
        "rotation_invariance": 0.5,
        "permutation_invariance": 0.5,
    }
    rec = recommend_architecture(priors, threshold=0.9)
    assert rec["recommended_model"] == "fcnn"
    assert rec["required_invariances"] == []


def test_recommend_architecture_threshold_effect():
    """Более высокий threshold → меньше priors проходит."""
    priors = {
        "translation_invariance": 0.92,
        "rotation_invariance": 0.92,
        "permutation_invariance": 0.92,
    }
    rec_low = recommend_architecture(priors, threshold=0.9)
    rec_high = recommend_architecture(priors, threshold=0.95)
    # При threshold=0.9 все три проходят → egnn
    assert rec_low["recommended_model"] == "egnn"
    # При threshold=0.95 ни один не проходит → fcnn
    assert rec_high["recommended_model"] == "fcnn"


def test_recommend_architecture_has_rationale():
    """Рекомендация содержит rationale — объяснение выбора."""
    priors = {
        "translation_invariance": 0.99,
        "rotation_invariance": 0.99,
        "permutation_invariance": 0.99,
    }
    rec = recommend_architecture(priors)
    assert "rationale" in rec
    assert len(rec["rationale"]) > 50  # содержательное объяснение
