"""
Парсинг датасета Alchemy.

Alchemy v20191129 содержит:
  - SDF файлы в data/alchemy/Alchemy-v20191129/atom_{9,10,11,12}/
  - final_version.csv со свойствами:
      mu, alpha, gap, HOMO, LUMO, U0, U, H, G, Cv, zpve, R2

Каждый SDF файл назван по gdb_idx (например, 1000170.sdf).
В CSV есть колонка gdb_idx для маппинга.

Свойства в Alchemy — СКАЛЯРЫ:
  - mu    (Дебай)  — норма вектора диполя |μ|
  - alpha (a₀³)    — изотропная поляризуемость tr(α)/3
  - gap   (Хартри) — HOMO-LUMO gap

Для части B (программа максимум) вектор μ и тензор α вычисляются через
PySCF отдельно — см. src/dipole_pyscf.py
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Химические элементы в Alchemy: H, C, N, O, F, S, Cl
ATOM_TYPES = ["H", "C", "N", "O", "F", "S", "Cl"]
ATOM_TO_IDX = {a: i for i, a in enumerate(ATOM_TYPES)}
ATOMIC_MASSES = {
    "H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999,
    "F": 18.998, "S": 32.06, "Cl": 35.45,
}

# Имена колонок в final_version.csv (с переносами строк в названиях)
COL_GDB_IDX = "gdb_idx"
# Считаем колонки по позиции, т.к. в названиях есть \n
# 0: gdb_idx
# 1: atom number
# 2: zpve
# 3: Cv
# 4: gap
# 5: G
# 6: HOMO
# 7: U
# 8: alpha
# 9: U0
# 10: H
# 11: LUMO
# 12: mu
# 13: R2
COL_IDX = {
    "gdb_idx": 0, "atom_number": 1, "zpve": 2, "Cv": 3, "gap": 4,
    "G": 5, "HOMO": 6, "U": 7, "alpha": 8, "U0": 9, "H": 10,
    "LUMO": 11, "mu": 12, "R2": 13,
}


def load_properties_csv(csv_path: str | Path) -> pd.DataFrame:
    """Загрузить final_version.csv, переименовать колонки в короткие имена."""
    df = pd.read_csv(csv_path)
    # Переименуем колонки по позиции (в названиях есть \n)
    short_names = list(COL_IDX.keys())
    df.columns = short_names
    # Оставляем только нужные колонки
    return df[["gdb_idx", "mu", "alpha", "gap", "HOMO", "LUMO"]].copy()


def find_sdf_files(data_root: str | Path) -> dict[int, Path]:
    """Найти все SDF файлы и вернуть словарь {gdb_idx: path}."""
    data_root = Path(data_root)
    sdf_files = {}
    for atom_dir in data_root.iterdir():
        if not atom_dir.is_dir() or not atom_dir.name.startswith("atom_"):
            continue
        for sdf in atom_dir.glob("*.sdf"):
            try:
                gdb_idx = int(sdf.stem)
                sdf_files[gdb_idx] = sdf
            except ValueError:
                continue
    return sdf_files


def parse_sdf(sdf_path: str | Path) -> dict | None:
    """Парсит один SDF файл.

    Возвращает словарь:
      {
        'atoms': list of (symbol, x, y, z),
        'bonds': list of (i, j, bond_type),
      }
    """
    from rdkit import Chem

    suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=False)
    mol = next(iter(suppl), None)
    if mol is None:
        return None

    conf = mol.GetConformer()
    n_atoms = mol.GetNumAtoms()

    atoms = []
    for i in range(n_atoms):
        atom = mol.GetAtomWithIdx(i)
        pos = conf.GetAtomPosition(i)
        atoms.append((atom.GetSymbol(), float(pos.x), float(pos.y), float(pos.z)))

    bonds = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        btype = bond.GetBondTypeAsDouble()
        bonds.append((i, j, btype))

    return {"atoms": atoms, "bonds": bonds}


def atoms_to_features(symbol: str) -> np.ndarray:
    """Узловой признак: one-hot по типу атома + mass. Размерность 8."""
    feat = np.zeros(len(ATOM_TYPES) + 1, dtype=np.float32)
    if symbol in ATOM_TO_IDX:
        feat[ATOM_TO_IDX[symbol]] = 1.0
        feat[-1] = ATOMIC_MASSES.get(symbol, 0.0)
    return feat


def bonds_to_edge_index(bonds: list, n_atoms: int) -> tuple[np.ndarray, np.ndarray]:
    """Связи → edge_index (2, 2E) и edge_attr (2E, 4) — one-hot по типу."""
    if not bonds:
        return (np.zeros((2, 0), dtype=np.int64),
                np.zeros((0, 4), dtype=np.float32))

    src, dst, types = [], [], []
    for i, j, btype in bonds:
        src.extend([i, j])
        dst.extend([j, i])
        types.extend([btype, btype])

    edge_index = np.array([src, dst], dtype=np.int64)
    edge_attr = np.zeros((len(src), 4), dtype=np.float32)
    for k, b in enumerate(types):
        if b == 1.0:
            edge_attr[k, 0] = 1.0
        elif b == 1.5:
            edge_attr[k, 3] = 1.0
        elif b == 2.0:
            edge_attr[k, 1] = 1.0
        elif b == 3.0:
            edge_attr[k, 2] = 1.0
    return edge_index, edge_attr


def mol_to_arrays(mol_dict: dict) -> dict:
    """Молекула → словарь numpy массивов для PyG Data."""
    atoms = mol_dict["atoms"]
    n = len(atoms)

    x = np.stack([atoms_to_features(a[0]) for a in atoms])  # (N, 8)
    pos = np.array([[a[1], a[2], a[3]] for a in atoms], dtype=np.float32)

    # Центрируем в центр масс (трансляционная инвариантность)
    masses = np.array([ATOMIC_MASSES.get(a[0], 1.0) for a in atoms], dtype=np.float32)
    com = (pos * masses[:, None]).sum(0) / masses.sum()
    pos = pos - com

    edge_index, edge_attr = bonds_to_edge_index(mol_dict["bonds"], n)

    return {
        "x": x,
        "pos": pos,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
    }


def stratified_split_by_gap(
    gdb_indices: list[int],
    properties: pd.DataFrame,
    test_size: float = 0.1,
    val_size: float = 0.1,
    seed: int = 42,
) -> tuple[list[int], list[int], list[int]]:
    """Stratified split по HOMO-LUMO gap (как в статье Alchemy)."""
    rng = np.random.default_rng(seed)
    # Сортируем по gap
    df = properties[properties["gdb_idx"].isin(gdb_indices)].copy()
    df = df.sort_values("gap").reset_index(drop=True)

    train, val, test = [], [], []
    chunk = 10
    for i in range(0, len(df), chunk):
        block = df["gdb_idx"].iloc[i:i + chunk].tolist()
        rng.shuffle(block)
        n_tr = int(len(block) * (1 - test_size - val_size))
        n_val = int(len(block) * val_size)
        train.extend(block[:n_tr])
        val.extend(block[n_tr:n_tr + n_val])
        test.extend(block[n_tr + n_val:])

    return train, val, test


if __name__ == "__main__":
    # Quick test
    data_root = "data/alchemy/Alchemy-v20191129"
    if not Path(data_root).exists():
        print("Сначала запустите data/download_alchemy.py")
    else:
        print("Загружаю CSV ...")
        df = load_properties_csv(f"{data_root}/final_version.csv")
        print(f"Свойств: {len(df)} молекул")
        print(df.head())

        print("\nИщу SDF файлы ...")
        sdf_files = find_sdf_files(data_root)
        print(f"Найдено SDF: {len(sdf_files)}")

        # Парсим первый SDF
        if sdf_files:
            gdb_idx, path = next(iter(sdf_files.items()))
            print(f"\nПарсю SDF для gdb_idx={gdb_idx}:")
            mol = parse_sdf(path)
            if mol:
                print(f"  Атомов: {len(mol['atoms'])}")
                print(f"  Связей: {len(mol['bonds'])}")
                arr = mol_to_arrays(mol)
                print(f"  x: {arr['x'].shape}")
                print(f"  pos: {arr['pos'].shape}")
