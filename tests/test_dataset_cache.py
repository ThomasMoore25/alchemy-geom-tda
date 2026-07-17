"""Тесты для src.dataset::AlchemyDataset — кэш-хеширование и параметры."""
import sys
import pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dataset import AlchemyDataset, CACHE_VERSION


class _NS:
    """Простой namespace для вызова property.fget без создания экземпляра."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def _get_filename(**kwargs) -> str:
    """Получить имя processed_file для заданных параметров."""
    defaults = dict(
        split="train", max_samples=None, tda_features=False,
        n_bins=16, max_radius=5.0, seed=42,
    )
    defaults.update(kwargs)
    ns = _NS(**defaults)
    return AlchemyDataset.processed_file_names.fget(ns)[0]


def test_cache_filename_has_version():
    """Имя файла содержит CACHE_VERSION."""
    name = _get_filename()
    assert CACHE_VERSION in name, f"{CACHE_VERSION} not in {name}"


def test_cache_filename_has_split():
    """Имя файла содержит split."""
    assert "train" in _get_filename(split="train")
    assert "val" in _get_filename(split="val")
    assert "test" in _get_filename(split="test")


def test_cache_filename_different_splits_different_hash():
    """Разные split -> разные хеши."""
    n_train = _get_filename(split="train")
    n_val = _get_filename(split="val")
    n_test = _get_filename(split="test")
    assert n_train != n_val != n_test != n_train


def test_cache_filename_tda_changes_hash():
    """Включение TDA -> другой хеш."""
    n_no_tda = _get_filename(tda_features=False)
    n_tda16 = _get_filename(tda_features=True, n_bins=16)
    assert n_no_tda != n_tda16
    assert "tda16" in n_tda16


def test_cache_filename_n_bins_changes_hash():
    """Разные n_bins -> разные хеши."""
    n16 = _get_filename(tda_features=True, n_bins=16)
    n32 = _get_filename(tda_features=True, n_bins=32)
    assert n16 != n32


def test_cache_filename_seed_changes_hash():
    """Разные seed -> разные хеши."""
    n42 = _get_filename(seed=42)
    n123 = _get_filename(seed=123)
    assert n42 != n123


def test_cache_filename_max_samples_changes_hash():
    """max_samples -> другой хеш."""
    n_full = _get_filename(max_samples=None)
    n_1000 = _get_filename(max_samples=1000)
    assert n_full != n_1000
    assert "max1000" in n_1000


def test_cache_filename_stable():
    """Те же параметры -> то же имя."""
    n1 = _get_filename()
    n2 = _get_filename()
    assert n1 == n2


def test_cache_version_is_v32():
    """CACHE_VERSION = 'v32' (после рефакторинга кэша)."""
    assert CACHE_VERSION == "v32"


def test_n_jobs_not_in_cache_hash():
    """n_jobs не влияет на хеш кэша (от него зависит только скорость).

    Проверяем через конкретный вызов: один и тот же набор параметров
    с разными n_jobs должен дать тот же хеш.
    """
    # _get_filename не принимает n_jobs, но если бы создал два инстанса
    # AlchemyDataset с разными n_jobs, их processed_file_names должны совпасть.
    # Проверяем, что params dict, который формирует хеш, не включает n_jobs.
    import inspect
    src = inspect.getsource(AlchemyDataset.processed_file_names.fget)
    # Находим блок params = { ... }
    assert '"n_jobs"' not in src, "n_jobs should NOT be in cache hash params dict"


def test_tda_worker_callable():
    """_tda_worker можно вызвать и возвращает ndarray правильной размерности."""
    import numpy as np
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    # _tda_worker импортирует gudhi лениво, должен работать
    # если gudhi установлен
    try:
        import gudhi  # noqa
    except ImportError:
        pytest.skip("gudhi not installed")
    from dataset import _tda_worker
    coords = np.random.randn(10, 3).astype(np.float32)
    result = _tda_worker(coords, n_bins=16, max_radius=5.0)
    assert result.shape == (52,)
    assert result.dtype == np.float32


def test_tda_worker_is_picklable():
    """_tda_worker должна быть picklable для multiprocessing.Pool."""
    import pickle
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from dataset import _tda_worker
    # Модульная функция picklable по определению
    data = pickle.dumps(_tda_worker)
    restored = pickle.loads(data)
    assert restored is _tda_worker
