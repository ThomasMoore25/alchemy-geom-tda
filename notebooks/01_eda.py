# EDA: исследование датасета Alchemy
"""Ноутбук для исследования датасета Alchemy v20191129.

Использует актуальный API из src.data: find_sdf_files, parse_sdf,
mol_to_arrays, load_properties_csv, stratified_split_by_gap.

Запуск:
  python notebooks/01_eda.py
  (или через jupyter: переименовать в .ipynb при необходимости)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt

import matplotlib.font_manager as fm
try:
    fm.fontManager.addfont('/usr/share/fonts/truetype/chinese/NotoSansSC-Regular.ttf')
except Exception:
    pass
try:
    fm.fontManager.addfont('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
except Exception:
    pass
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Noto Sans SC']
plt.rcParams['axes.unicode_minus'] = False

from src.data import (
    ATOM_TYPES,
    ATOMIC_MASSES,
    find_sdf_files,
    parse_sdf,
    mol_to_arrays,
    load_properties_csv,
)

plt.rcParams.update({
    "figure.figsize": (10, 6),
    "font.size": 12,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def load_sample(data_root: str = "data/alchemy/Alchemy-v20191129", n_max: int = 2000):
    """Загрузить n_max молекул с свойствами из final_version.csv.

    Возвращает:
        molecules: list of dict с ключами 'atoms', 'bonds', 'arr', 'props'
    """
    data_root = Path(data_root)
    csv_path = data_root / "final_version.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV не найден: {csv_path}. "
            "Запустите: python data/download_alchemy.py"
        )

    print(f"Загружаю свойства из {csv_path} ...")
    props = load_properties_csv(csv_path)
    props_dict = props.set_index("gdb_idx").to_dict("index")
    print(f"  Свойств: {len(props)} молекул")

    print(f"Ищу SDF в {data_root} ...")
    sdf_files = find_sdf_files(data_root)
    print(f"  Найдено SDF: {len(sdf_files)}")

    valid_gdb = sorted(set(sdf_files.keys()) & set(props_dict.keys()))
    print(f"  Валидных (SDF + CSV): {len(valid_gdb)}")

    if n_max is not None and n_max < len(valid_gdb):
        valid_gdb = valid_gdb[:n_max]
        print(f"  Ограничился первыми {n_max}")

    molecules = []
    for i, gdb_idx in enumerate(valid_gdb):
        if i % 500 == 0:
            print(f"  Обработано {i}/{len(valid_gdb)}")
        mol = parse_sdf(sdf_files[gdb_idx])
        if mol is None:
            continue
        arr = mol_to_arrays(mol)
        molecules.append({
            "gdb_idx": gdb_idx,
            "atoms": mol["atoms"],
            "bonds": mol["bonds"],
            "arr": arr,
            "props": props_dict[gdb_idx],
        })
    print(f"  Загружено молекул: {len(molecules)}")
    return molecules


def analyze_atom_distribution(molecules):
    """Распределение типов атомов и размеров молекул."""
    print("\n=== Распределение типов атомов ===")
    atom_counts = {a: 0 for a in ATOM_TYPES}
    sizes = []

    for mol in molecules:
        sizes.append(len(mol["atoms"]))
        for symbol, *_ in mol["atoms"]:
            if symbol in atom_counts:
                atom_counts[symbol] += 1

    print("Атомы:")
    for a, c in sorted(atom_counts.items(), key=lambda x: -x[1]):
        print(f"  {a}: {c}")

    print(f"\nРазмер молекул: mean={np.mean(sizes):.1f}, "
          f"min={min(sizes)}, max={max(sizes)}, median={np.median(sizes):.1f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    axes[0].bar(atom_counts.keys(), atom_counts.values())
    axes[0].set_title("Распределение типов атомов")
    axes[0].set_ylabel("Количество")

    axes[1].hist(sizes, bins=30, edgecolor="black")
    axes[1].set_title("Распределение размеров молекул")
    axes[1].set_xlabel("Число атомов")
    axes[1].set_ylabel("Частота")

    out = Path("results/figures/eda_atoms.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=100)
    plt.close()
    print(f"Сохранено: {out}")


def analyze_targets(molecules):
    """Распределение mu, alpha, gap из final_version.csv."""
    print("\n=== Целевые свойства ===")

    mus = [m["props"]["mu"] for m in molecules]
    alphas = [m["props"]["alpha"] for m in molecules]
    gaps = [m["props"]["gap"] for m in molecules]

    print(f"mu    (Дебай): n={len(mus)},  mean={np.mean(mus):.3f}, "
          f"std={np.std(mus):.3f}, min={np.min(mus):.3f}, max={np.max(mus):.3f}")
    print(f"alpha (a0^3):  n={len(alphas)}, mean={np.mean(alphas):.3f}, "
          f"std={np.std(alphas):.3f}, min={np.min(alphas):.3f}, max={np.max(alphas):.3f}")
    print(f"gap   (Hartree): n={len(gaps)},  mean={np.mean(gaps):.4f}, "
          f"std={np.std(gaps):.4f}, min={np.min(gaps):.4f}, max={np.max(gaps):.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

    axes[0].hist(mus, bins=40, edgecolor="black", color="steelblue")
    axes[0].set_title("Дипольный момент |μ|")
    axes[0].set_xlabel("Дебай")

    axes[1].hist(alphas, bins=40, edgecolor="black", color="darkorange")
    axes[1].set_title("Изотропная поляризуемость tr(α)/3")
    axes[1].set_xlabel("a₀³")

    axes[2].hist(gaps, bins=40, edgecolor="black", color="green")
    axes[2].set_title("HOMO-LUMO gap")
    axes[2].set_xlabel("Хартри")

    out = Path("results/figures/eda_targets.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=100)
    plt.close()
    print(f"Сохранено: {out}")


def demonstrate_symmetry(molecules, n_examples: int = 3):
    """Демонстрация E(3) симметрий: поворот молекулы не меняет скалярные свойства.

    Берём вектор из первых двух атомов (как proxy для диполя, поскольку
    истинный вектор μ в final_version.csv не дан — дана только его норма),
    и показываем, что при повороте координат норма сохраняется.
    """
    print("\n=== Демонстрация симметрий ===")
    try:
        from scipy.spatial.transform import Rotation
    except ImportError:
        print("  scipy не установлен — пропускаю")
        return

    for i in range(min(n_examples, len(molecules))):
        mol = molecules[i]
        coords = mol["arr"]["pos"]  # уже центрировано в COM
        # proxy-вектор: разность позиций первых двух атомов
        if coords.shape[0] < 2:
            continue
        v = coords[0] - coords[1]
        v_norm = float(np.linalg.norm(v))

        R = Rotation.random().as_matrix()
        v_rotated = R @ v
        v_rotated_norm = float(np.linalg.norm(v_rotated))

        # Глобальные дескрипторы (гистограмма атомов) — инвариантны
        atom_onehot = mol["arr"]["x"][:, :len(ATOM_TYPES)]
        hist_before = atom_onehot.sum(axis=0)

        print(f"\nМолекула {i} (gdb_idx={mol['gdb_idx']}): "
              f"{len(mol['atoms'])} атомов")
        print(f"  |v| исходный:      {v_norm:.6f}")
        print(f"  |v| повёрнутый:    {v_rotated_norm:.6f}")
        print(f"  Гистограмма атомов: {dict(zip(ATOM_TYPES, hist_before.astype(int).tolist()))}")
        print(f"  (скалярные свойства и нормы векторов сохраняются при повороте)")


if __name__ == "__main__":
    molecules = load_sample(n_max=2000)
    if not molecules:
        print("Нет молекул для анализа. Проверьте путь к данным.")
        sys.exit(1)
    analyze_atom_distribution(molecules)
    analyze_targets(molecules)
    demonstrate_symmetry(molecules)
    print("\n=== EDA завершён ===")
    print("Графики сохранены в results/figures/")
