"""Тесты для src.data — атомные типы, массы, базовые функции без датасета."""
import pytest
import numpy as np

from src.data import (
    ATOM_TYPES,
    ATOMIC_MASSES,
    ATOM_TO_IDX,
    atoms_to_features,
    bonds_to_edge_index,
    mol_to_arrays,
    stratified_split_by_gap,
)


def test_atom_types_complete():
    """ATOM_TYPES содержит все 7 элементов Alchemy: H, C, N, O, F, S, Cl."""
    assert ATOM_TYPES == ["H", "C", "N", "O", "F", "S", "Cl"]


def test_atomic_masses_keys_match_atom_types():
    """ATOMIC_MASSES содержит массу для каждого ATOM_TYPES."""
    for atom in ATOM_TYPES:
        assert atom in ATOMIC_MASSES, f"Missing mass for {atom}"


def test_atom_to_idx_mapping():
    """ATOM_TO_IDX — корректный mapping."""
    assert ATOM_TO_IDX["H"] == 0
    assert ATOM_TO_IDX["Cl"] == 6
    assert len(ATOM_TO_IDX) == 7


def test_atoms_to_features_shape():
    """atoms_to_features возвращает (8,): 7 one-hot + 1 mass."""
    feat = atoms_to_features("C")
    assert feat.shape == (8,)
    # one-hot для C = [0, 1, 0, 0, 0, 0, 0]
    assert feat[1] == 1.0
    # mass для C = 12.011
    assert abs(feat[7] - 12.011) < 1e-3


def test_atoms_to_features_all_atoms():
    """Все 7 атомов дают корректные one-hot + mass."""
    for i, atom in enumerate(ATOM_TYPES):
        feat = atoms_to_features(atom)
        assert feat[i] == 1.0, f"one-hot failed for {atom}"
        assert feat[7] > 0, f"mass missing for {atom}"


def test_atoms_to_features_unknown_atom():
    """Неизвестный атом: one-hot пустой, mass=0."""
    feat = atoms_to_features("Xe")
    assert feat.sum() == 0.0  # все нули


def test_bonds_to_edge_index_basic():
    """1 связь → 2 ребра (оба направления)."""
    bonds = [(0, 1, 1.0)]  # single bond
    edge_index, edge_attr = bonds_to_edge_index(bonds, n_atoms=2)
    assert edge_index.shape == (2, 2)
    assert edge_attr.shape == (2, 4)
    # one-hot single bond = [1, 0, 0, 0]
    assert edge_attr[0, 0] == 1.0
    assert edge_attr[1, 0] == 1.0


def test_bonds_to_edge_index_bond_types():
    """Все 4 типа связей: 1.0, 2.0, 3.0, 1.5."""
    bonds = [
        (0, 1, 1.0),  # single -> [1,0,0,0]
        (1, 2, 2.0),  # double -> [0,1,0,0]
        (2, 3, 3.0),  # triple -> [0,0,1,0]
        (3, 0, 1.5),  # aromatic -> [0,0,0,1]
    ]
    edge_index, edge_attr = bonds_to_edge_index(bonds, n_atoms=4)
    assert edge_index.shape == (2, 8)  # 4 bonds × 2 directions
    assert edge_attr.shape == (8, 4)
    # Проверяем все 4 one-hot
    assert edge_attr[0, 0] == 1.0  # single
    assert edge_attr[2, 1] == 1.0  # double
    assert edge_attr[4, 2] == 1.0  # triple
    assert edge_attr[6, 3] == 1.0  # aromatic


def test_bonds_to_edge_index_empty():
    """Нет связей → пустые edge_index и edge_attr."""
    edge_index, edge_attr = bonds_to_edge_index([], n_atoms=5)
    assert edge_index.shape == (2, 0)
    assert edge_attr.shape == (0, 4)


def test_mol_to_arrays_centers_at_com():
    """mol_to_arrays центрирует координаты в центр масс."""
    mol = {
        "atoms": [
            ("H", 0.0, 0.0, 0.0),
            ("H", 2.0, 0.0, 0.0),
        ],
        "bonds": [],
    }
    arr = mol_to_arrays(mol)
    # COM для двух атомов H массой 1.008: (0+2)/2 = 1.0
    # После центрирования: (-1, 0, 0) и (1, 0, 0)
    assert arr["pos"].shape == (2, 3)
    assert abs(arr["pos"][0, 0] - (-1.0)) < 1e-6
    assert abs(arr["pos"][1, 0] - 1.0) < 1e-6


def test_mol_to_arrays_features_shape():
    """mol_to_arrays возвращает x (N,8), pos (N,3), edge_index (2,E), edge_attr (E,4)."""
    mol = {
        "atoms": [
            ("C", 0.0, 0.0, 0.0),
            ("O", 1.0, 0.0, 0.0),
            ("H", 2.0, 0.0, 0.0),
        ],
        "bonds": [(0, 1, 1.0), (1, 2, 1.0)],
    }
    arr = mol_to_arrays(mol)
    assert arr["x"].shape == (3, 8)
    assert arr["pos"].shape == (3, 3)
    assert arr["edge_index"].shape == (2, 4)  # 2 bonds × 2 directions
    assert arr["edge_attr"].shape == (4, 4)


def test_stratified_split_no_leak():
    """stratified_split_by_gap: train/val/test не пересекаются."""
    import pandas as pd

    gdb_indices = list(range(1000))
    props = pd.DataFrame({
        "gdb_idx": gdb_indices,
        "mu": np.random.rand(1000),
        "alpha": np.random.rand(1000) * 10,
        "gap": np.random.rand(1000) * 0.5,
        "HOMO": np.random.rand(1000) * -0.2,
        "LUMO": np.random.rand(1000) * 0.1,
    })

    train, val, test = stratified_split_by_gap(gdb_indices, props, seed=42)
    train_set, val_set, test_set = set(train), set(val), set(test)
    # Нет пересечений
    assert not (train_set & val_set), "train ∩ val not empty"
    assert not (train_set & test_set), "train ∩ test not empty"
    assert not (val_set & test_set), "val ∩ test not empty"
    # Все индексы покрыты
    assert len(train_set | val_set | test_set) == 1000


def test_stratified_split_sizes():
    """Стратифицированный split даёт ~80/10/10."""
    import pandas as pd

    gdb_indices = list(range(1000))
    props = pd.DataFrame({
        "gdb_idx": gdb_indices,
        "mu": np.random.rand(1000),
        "alpha": np.random.rand(1000) * 10,
        "gap": np.random.rand(1000) * 0.5,
        "HOMO": np.random.rand(1000) * -0.2,
        "LUMO": np.random.rand(1000) * 0.1,
    })

    train, val, test = stratified_split_by_gap(gdb_indices, props, seed=42)
    # ~80/10/10
    assert 700 < len(train) < 900
    assert 50 < len(val) < 200
    assert 50 < len(test) < 200
    assert len(train) + len(val) + len(test) == 1000


def test_stratified_split_reproducible():
    """Тот же seed → тот же split."""
    import pandas as pd

    gdb_indices = list(range(1000))
    props = pd.DataFrame({
        "gdb_idx": gdb_indices,
        "mu": np.random.rand(1000),
        "alpha": np.random.rand(1000) * 10,
        "gap": np.random.rand(1000) * 0.5,
        "HOMO": np.random.rand(1000) * -0.2,
        "LUMO": np.random.rand(1000) * 0.1,
    })

    train1, val1, test1 = stratified_split_by_gap(gdb_indices, props, seed=42)
    train2, val2, test2 = stratified_split_by_gap(gdb_indices, props, seed=42)
    assert train1 == train2
    assert val1 == val2
    assert test1 == test2
