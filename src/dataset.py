"""
PyG датасет для Alchemy v20191129.

Особенности:
  - InMemoryDataset с кэшированием обработанных данных в data/alchemy/processed/
  - Stratified split по HOMO-LUMO gap (как в статье Alchemy)
  - Train/Val/Test: 162063/20257/20259 молекул (80/10/10)
  - TDA-фичи вычисляются один раз и кэшируются (52D на молекулу)
  - Поддержка max_samples для отладки (лимит на каждый сплит отдельно)
  - Имя .pt-кэша зависит от хеша всех параметров, влияющих на содержимое
    (split, max_samples, tda, n_bins, max_radius, seed, atom_types, cache_version).
    Любое изменение логики process() или этих параметров автоматически
    инвалидирует кэш (v32+).
"""
import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, Data

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data import (
    ATOM_TYPES,
    ATOMIC_MASSES,
    load_properties_csv, find_sdf_files, parse_sdf, mol_to_arrays,
    stratified_split_by_gap,
)

# Bump this when changing logic of process(), mol_to_arrays, parse_sdf,
# stratified_split_by_gap, or Data field schema. Forces cache invalidation.
CACHE_VERSION = "v32"


def _tda_worker(coords: np.ndarray, n_bins: int = 16, max_radius: float = 5.0) -> np.ndarray:
    """Worker-функция для multiprocessing.Pool — вычисляет TDA-фичи одной молекулы.

    Должна быть на уровне модуля (не лямбда), чтобы быть picklable.
    """
    from src.tda.features import extract_tda_features
    return extract_tda_features(coords, n_bins=n_bins, max_radius=max_radius)


class AlchemyDataset(InMemoryDataset):
    """PyG датасет для Alchemy v20191129.

    Args:
        root: путь к data/alchemy (где лежит папка Alchemy-v20191129)
        split: 'train' | 'val' | 'test' | 'all'
        max_samples: лимит молекул В ЭТОМ СПЛИТЕ (а не в общем пуле)
        tda_features: вычислить и добавить TDA-фичи
        n_bins: число бинов для Betti curves
        max_radius: радиус фильтрации TDA
        seed: сид для split
        n_jobs: число процессов для TDA-расчёта (v32+). 1 = последовательно.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        max_samples: int | None = None,
        tda_features: bool = False,
        n_bins: int = 16,
        max_radius: float = 5.0,
        seed: int = 42,
        n_jobs: int = 1,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.split = split
        self.max_samples = max_samples
        self.tda_features = tda_features
        self.n_bins = n_bins
        self.max_radius = max_radius
        self.seed = seed
        self.n_jobs = max(1, n_jobs)
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["Alchemy-v20191129/final_version.csv"]

    @property
    def processed_file_names(self):
        # Стабильный хеш от всех параметров, влияющих на содержимое .pt.
        # При любом изменении → новый хеш → PyG пересчитает кэш.
        # n_jobs НЕ входит в хеш — от него не зависит содержимое, только скорость.
        params = {
            "cache_version": CACHE_VERSION,
            "split": self.split,
            "max_samples": self.max_samples,
            "tda_features": self.tda_features,
            "n_bins": self.n_bins,
            "max_radius": self.max_radius,
            "seed": self.seed,
            "atom_types": ATOM_TYPES,
            "atomic_masses": ATOMIC_MASSES,
            # Сигнатура Data-объекта (какие поля хранятся)
            "data_fields": ["x", "pos", "edge_index", "edge_attr",
                            "mu", "alpha", "gap", "gdb_idx", "tda"],
        }
        param_str = repr(sorted(params.items()))
        h = hashlib.sha1(param_str.encode("utf-8")).hexdigest()[:12]
        # Сохраняем split/max_samples в имени для читаемости (без влияния на логику):
        # кэш инвалидируется по хешу, не по этой строке.
        readable = f"alchemy_{CACHE_VERSION}_{self.split}"
        if self.max_samples is not None:
            readable += f"_max{self.max_samples}"
        if self.tda_features:
            readable += f"_tda{self.n_bins}"
        return [f"{readable}_{h}.pt"]

    def download(self):
        """Скачать Alchemy, если данных нет.

        v32: вместо FileNotFoundError вызываем data.download_alchemy.download_alchemy(),
        чтобы PyG InMemoryDataset сам управлял жизненным циклом данных.
        Это соответствует контракту базового класса.
        """
        if (Path(self.root) / "Alchemy-v20191129" / "final_version.csv").exists():
            return  # уже скачано
        # Импортируем лениво, чтобы избежать циклических зависимостей
        import sys
        repo_root = Path(__file__).parent.parent.parent
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from data.download_alchemy import download_alchemy
        # download_alchemy по умолчанию использует DATA_DIR = data/alchemy
        # Если self.root отличается, нужно явно указать
        default_root = repo_root / "data" / "alchemy"
        if Path(self.root) != default_root:
            # Пользователь указал кастомный путь — скачиваем туда
            # (download_alchemy не принимает путь, поэтому просто вызываем,
            # а потом сообщим пользователю, если путь нестандартный)
            print(f"[WARN] download_alchemy() использует дефолтный путь {default_root}. "
                  f"Если вы указали root={self.root}, убедитесь, что данные там.")
        download_alchemy()

    def process(self):
        """Парсинг SDF и создание Data объектов.

        ИСПРАВЛЕНО: split делается ОДИН РАЗ для всего датасета,
        потом max_samples ограничивает каждый сплит отдельно.
        Никакой утечки данных.
        """
        data_root = Path(self.root) / "Alchemy-v20191129"
        csv_path = data_root / "final_version.csv"

        print(f"[{self.split}] Загружаю свойства из {csv_path} ...")
        props = load_properties_csv(csv_path)
        print(f"  Свойств: {len(props)} молекул")

        print(f"[{self.split}] Ищу SDF файлы ...")
        sdf_files = find_sdf_files(data_root)
        print(f"  Найдено SDF: {len(sdf_files)}")

        # Берём только те, у кого есть и SDF, и свойства
        valid_gdb = sorted(set(sdf_files.keys()) & set(props["gdb_idx"].tolist()))
        print(f"  Валидных молекул (SDF + CSV): {len(valid_gdb)}")

        # === ИСПРАВЛЕНИЕ: делаем split ОДИН РАЗ для ВСЕХ молекул ===
        print(f"[{self.split}] Делаю stratified split по gap ...")
        train_idx, val_idx, test_idx = stratified_split_by_gap(
            valid_gdb, props, seed=self.seed
        )
        print(f"  Всего: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

        # Выбираем нужный сплит
        if self.split == "all":
            indices = valid_gdb
        elif self.split == "train":
            indices = train_idx
        elif self.split == "val":
            indices = val_idx
        elif self.split == "test":
            indices = test_idx
        else:
            raise ValueError(f"Unknown split: {self.split}")

        # === ИСПРАВЛЕНИЕ: max_samples ограничивает УЖЕ выбранный сплит ===
        if self.max_samples is not None:
            indices = indices[:self.max_samples]
        print(f"  {self.split}: {len(indices)} молекул (после max_samples)")

        # Строим Data объекты
        data_list = []
        props_dict = props.set_index("gdb_idx").to_dict("index")

        # v32: предпарсим все SDF и сохраним arr["pos"] для TDA
        # Это позволяет распараллелить TDA-расчёт через multiprocessing
        parsed_data = []  # list of (gdb_idx, arr, props_row)
        for i, gdb_idx in enumerate(indices):
            if i % 5000 == 0:
                print(f"  Парсинг SDF: {i}/{len(indices)}")
            mol = parse_sdf(sdf_files[gdb_idx])
            if mol is None:
                continue
            arr = mol_to_arrays(mol)
            props_row = props_dict[gdb_idx]
            parsed_data.append((gdb_idx, arr, props_row))

        # v32: TDA-расчёт — параллельный через multiprocessing.Pool если n_jobs > 1
        tda_features_list = None
        if self.tda_features:
            from src.tda.features import extract_tda_features

            coords_list = [arr["pos"] for _, arr, _ in parsed_data]
            n = len(coords_list)
            print(f"  TDA-расчёт для {n} молекул (n_jobs={self.n_jobs}) ...")

            if self.n_jobs > 1 and n > 100:
                # Параллельный режим
                from multiprocessing import Pool
                import functools

                worker = functools.partial(
                    _tda_worker,
                    n_bins=self.n_bins,
                    max_radius=self.max_radius,
                )
                with Pool(self.n_jobs) as pool:
                    # chunksize для уменьшения overhead
                    chunksize = max(1, n // (self.n_jobs * 10))
                    tda_features_list = pool.map(
                        worker, coords_list, chunksize=chunksize
                    )
            else:
                # Последовательный режим
                tda_features_list = []
                for i, c in enumerate(coords_list):
                    if i % 5000 == 0:
                        print(f"  TDA: {i}/{n}")
                    tda_features_list.append(
                        extract_tda_features(c, n_bins=self.n_bins,
                                              max_radius=self.max_radius)
                    )

        # Собираем Data объекты
        for i, (gdb_idx, arr, props_row) in enumerate(parsed_data):
            data = Data(
                x=torch.from_numpy(arr["x"]),
                pos=torch.from_numpy(arr["pos"]),
                edge_index=torch.from_numpy(arr["edge_index"]),
                edge_attr=torch.from_numpy(arr["edge_attr"]),
                mu=torch.tensor([props_row["mu"]], dtype=torch.float32),
                alpha=torch.tensor([props_row["alpha"]], dtype=torch.float32),
                gap=torch.tensor([props_row["gap"]], dtype=torch.float32),
                gdb_idx=torch.tensor([gdb_idx], dtype=torch.long),
            )

            if tda_features_list is not None:
                tda = tda_features_list[i]
                data.tda = torch.from_numpy(tda).unsqueeze(0)  # (1, 52) для PyG

            data_list.append(data)

        print(f"[{self.split}] Сохраняю {len(data_list)} молекул в {self.processed_paths[0]}")
        self.save(data_list, self.processed_paths[0])
