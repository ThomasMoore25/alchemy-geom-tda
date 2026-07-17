"""Экстракция геометрических prior-ов из топологического анализа данных.

Программа максимум (обратная задача): по TDA-фичам датасета определить,
какие геометрические симметрии (priors) присутствуют в данных, и какие
архитектурные свойства модели нужны для их учёта.

Идея:
  1. Считаем TDA-фичи (Betti curves H_0/H_1/H_2, persistence entropy)
     для каждого объекта датасета.
  2. Применяем изометрические преобразования (поворот, сдвиг, перестановку)
     и смотрим, как меняются TDA-фичи.
  3. Если TDA инвариантна к преобразованию → соответствующий prior
     присутствует в данных.
  4. Рекомендуем архитектуру с подходящими эквивариантностями.

Поддерживаемые priors:
  - translation_invariance: TDA от (X + t) == TDA от X
  - rotation_invariance: TDA от (R @ X) == TDA от X
  - permutation_invariance: TDA от X[perm] == TDA от X
  - topology_invariance: разные молекулы с похожей топологией имеют похожие TDA
"""
from __future__ import annotations

from typing import Literal

import numpy as np

from .features import (
    extract_tda_features,
)


def _rotation_matrix(seed: int = 42) -> np.ndarray:
    """Случайная матрица поворота SO(3) через QR-разложение."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((3, 3))
    Q, R = np.linalg.qr(A)
    # Корректируем знак, чтобы det(Q) = +1 (собственный поворот)
    Q = Q @ np.diag(np.sign(np.diag(R)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def _permutation(n: int, seed: int = 42) -> np.ndarray:
    """Случайная перестановка индексов длины n."""
    rng = np.random.default_rng(seed)
    perm = np.arange(n)
    rng.shuffle(perm)
    return perm


def tda_invariance_score(
    coords: np.ndarray,
    transform: Literal["translation", "rotation", "permutation"],
    n_trials: int = 5,
    n_bins: int = 16,
    max_radius: float = 5.0,
    seed: int = 42,
) -> float:
    """Вычислить степень инвариантности TDA к заданному преобразованию.

    Args:
        coords: (N, 3) координаты атомов одной молекулы
        transform: тип преобразования
        n_trials: число случайных испытаний
        n_bins: число бинов Betti curve
        max_radius: радиус фильтрации
        seed: сид для генератора случайных чисел

    Returns:
        score в [0, 1]:
          1.0 = полная инвариантность (TDA не меняется)
          0.0 = TDA полностью меняется
    """
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"coords must be (N, 3), got {coords.shape}")

    # Базовые TDA-фичи
    base_features = extract_tda_features(
        coords, n_bins=n_bins, max_radius=max_radius
    )
    base_norm = np.linalg.norm(base_features)

    if base_norm < 1e-10:
        return 1.0  # пустая топология — тривиально инвариантна

    deltas = []
    for trial in range(n_trials):
        trial_seed = seed + trial
        if transform == "translation":
            rng = np.random.default_rng(trial_seed)
            t = rng.standard_normal(3) * 2.0
            transformed = coords + t[None, :]
        elif transform == "rotation":
            R = _rotation_matrix(seed=trial_seed)
            transformed = coords @ R.T
        elif transform == "permutation":
            perm = _permutation(len(coords), seed=trial_seed)
            transformed = coords[perm]
        else:
            raise ValueError(f"Unknown transform: {transform}")

        # TDA от преобразованных координат
        transformed_features = extract_tda_features(
            transformed, n_bins=n_bins, max_radius=max_radius
        )

        # Относительная разница
        delta = np.linalg.norm(transformed_features - base_features) / base_norm
        deltas.append(delta)

    # Score = 1 - среднее относительное отклонение
    mean_delta = float(np.mean(deltas))
    score = max(0.0, 1.0 - mean_delta)
    return score


def extract_priors(
    coords_list: list[np.ndarray],
    n_bins: int = 16,
    max_radius: float = 5.0,
    n_trials: int = 3,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """Извлечь геометрические priors из датасета через TDA.

    Args:
        coords_list: список массивов (N_i, 3) — координаты молекул
        n_bins: число бинов Betti curve
        max_radius: радиус фильтрации TDA
        n_trials: число случайных преобразований на молекулу
        seed: сид
        verbose: печатать прогресс

    Returns:
        dict с ключами:
          - translation_invariance: средний score [0, 1]
          - rotation_invariance: средний score [0, 1]
          - permutation_invariance: средний score [0, 1]
          - n_molecules: число молекул
          - n_molecules_success: число успешно обработанных
          - per_molecule: list of per-molecule dicts
    """
    per_molecule = []
    success = 0

    for i, coords in enumerate(coords_list):
        if verbose and i % max(1, len(coords_list) // 10) == 0:
            print(f"  TDA prior extraction: {i}/{len(coords_list)}")

        coords = np.asarray(coords, dtype=np.float64)
        if coords.ndim != 2 or coords.shape[0] < 2:
            continue

        try:
            scores = {
                "translation": tda_invariance_score(
                    coords, "translation", n_trials=n_trials,
                    n_bins=n_bins, max_radius=max_radius, seed=seed,
                ),
                "rotation": tda_invariance_score(
                    coords, "rotation", n_trials=n_trials,
                    n_bins=n_bins, max_radius=max_radius, seed=seed,
                ),
                "permutation": tda_invariance_score(
                    coords, "permutation", n_trials=n_trials,
                    n_bins=n_bins, max_radius=max_radius, seed=seed,
                ),
            }
            per_molecule.append(scores)
            success += 1
        except Exception as e:
            if verbose:
                print(f"    [WARN] molecule {i}: {e}")
            continue

    if not per_molecule:
        return {
            "translation_invariance": 0.0,
            "rotation_invariance": 0.0,
            "permutation_invariance": 0.0,
            "n_molecules": len(coords_list),
            "n_molecules_success": 0,
            "per_molecule": [],
        }

    # Усредняем по молекулам
    result = {
        "translation_invariance": float(np.mean([m["translation"] for m in per_molecule])),
        "rotation_invariance": float(np.mean([m["rotation"] for m in per_molecule])),
        "permutation_invariance": float(np.mean([m["permutation"] for m in per_molecule])),
        "n_molecules": len(coords_list),
        "n_molecules_success": success,
        "per_molecule": per_molecule,
    }

    if verbose:
        print(f"\nExtracted priors (n={success} molecules):")
        print(f"  translation_invariance:  {result['translation_invariance']:.4f}")
        print(f"  rotation_invariance:     {result['rotation_invariance']:.4f}")
        print(f"  permutation_invariance:  {result['permutation_invariance']:.4f}")

    return result


def recommend_architecture(priors: dict, threshold: float = 0.9) -> dict:
    """На основе извлечённых prior-ов рекомендовать архитектуру.

    Args:
        priors: dict из extract_priors()
        threshold: score >= threshold считается "сильным" prior-ом

    Returns:
        dict с ключами:
          - required_invariances: list of str
          - recommended_model: имя модели из репозитория
          - needs_tda: bool (всегда True — TDA даёт дополнительные фичи)
          - rationale: str с объяснением
    """
    required = []
    if priors["translation_invariance"] >= threshold:
        required.append("translation")
    if priors["rotation_invariance"] >= threshold:
        required.append("rotation")
    if priors["permutation_invariance"] >= threshold:
        required.append("permutation")

    # Маппинг priors → архитектура
    # E(3) = translation + rotation + permutation
    if {"translation", "rotation", "permutation"}.issubset(required):
        recommended = "egnn"  # E(3)-эквивариантная
        rationale = (
            "Все три E(3) симметрии присутствуют как сильные priors. "
            "Рекомендуется EGNN (E(3)-эквивариантная сеть). "
            "TDA-фичи могут дополнительно улучшить метрики через --tda_mode film."
        )
    elif {"translation", "permutation"}.issubset(required):
        recommended = "schnet"
        rationale = (
            "Трансляции и перестановки — сильные priors, но вращение — слабое. "
            "Рекомендуется SchNet (инвариантен к translation + permutation). "
            "EGNN может переобучиться, если rotation не настоящий prior."
        )
    elif "permutation" in required:
        recommended = "fcnn"
        rationale = (
            "Только перестановки — сильный prior. "
            "Базового FCNN с permutation-invariant pooling достаточно. "
            "Геометрические модели избыточны."
        )
    else:
        recommended = "fcnn"
        rationale = (
            "Сильных геометрических priors не обнаружено. "
            "Начните с FCNN baseline, дальше проверьте, помогает ли TDA."
        )

    return {
        "required_invariances": required,
        "recommended_model": recommended,
        "needs_tda": True,  # TDA всегда может дать дополнительный сигнал
        "rationale": rationale,
    }
