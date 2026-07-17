"""Тесты для src.models.knn::knn_graph_pytorch."""
import pytest
import torch

from models.knn import knn_graph_pytorch


def _edges_set(edge_index: torch.Tensor) -> set[tuple[int, int]]:
    """Множество рёбер (src, dst) для сравнения без учёта порядка."""
    return set(zip(edge_index[0].tolist(), edge_index[1].tolist()))


def test_knn_single_molecule():
    """Один граф (batch=None): k=4 даёт 20 рёбер для 5 узлов."""
    torch.manual_seed(0)
    x = torch.randn(5, 3)
    ei = knn_graph_pytorch(x, k=4, batch=None, loop=False)
    assert ei.shape[0] == 2
    # 5 узлов × 4 соседей = 20 рёбер (без self-loops, n-1=4 доступно)
    assert ei.shape[1] == 20


def test_knn_multi_molecule():
    """Батч из 2 молекул по 10 атомов: k=4 даёт 80 рёбер."""
    torch.manual_seed(0)
    x = torch.randn(20, 3)
    batch = torch.tensor([0] * 10 + [1] * 10)
    ei = knn_graph_pytorch(x, k=4, batch=batch, loop=False)
    assert ei.shape[0] == 2
    assert ei.shape[1] == 80
    # Все рёбра внутри одной молекулы
    same_mol = batch[ei[0]] == batch[ei[1]]
    assert same_mol.all(), "kNN should not cross molecule boundaries"


def test_knn_loop_true_includes_self():
    """loop=True: каждый узел — свой собственный сосед."""
    torch.manual_seed(0)
    x = torch.randn(5, 3)
    ei = knn_graph_pytorch(x, k=3, batch=None, loop=True)
    # С self-loops: 5 узлов × 3 = 15 рёбер
    assert ei.shape[1] == 15
    # Должны быть self-loops
    has_self = (ei[0] == ei[1]).any()
    assert has_self, "loop=True should include self-loops"


def test_knn_loop_false_no_self():
    """loop=False: self-loops отсутствуют."""
    torch.manual_seed(0)
    x = torch.randn(5, 3)
    ei = knn_graph_pytorch(x, k=4, batch=None, loop=False)
    assert (ei[0] != ei[1]).all(), "loop=False should not include self-loops"


def test_knn_k_larger_than_atoms():
    """k > n_atoms: k_actual урезается до n_atoms-1 (без loop)."""
    torch.manual_seed(0)
    x = torch.randn(3, 3)
    ei = knn_graph_pytorch(x, k=10, batch=None, loop=False)
    # 3 узла × (3-1)=2 соседей = 6 рёбер
    assert ei.shape[1] == 6


def test_knn_empty_input():
    """Пустой вход: пустой edge_index."""
    x = torch.zeros(0, 3)
    ei = knn_graph_pytorch(x, k=4, batch=torch.zeros(0, dtype=torch.long))
    assert ei.shape == (2, 0)


def test_knn_unsorted_batch():
    """Неотсортированный batch: рёбра корректно строится через fallback path."""
    torch.manual_seed(0)
    x = torch.randn(20, 3)
    # Перемешанный batch: 0,1,0,1,...
    batch = torch.tensor([1, 0] * 10)
    ei = knn_graph_pytorch(x, k=3, batch=batch, loop=False)
    # Все рёбра внутри одной молекулы
    same_mol = batch[ei[0]] == batch[ei[1]]
    assert same_mol.all()


def test_knn_variable_molecule_sizes():
    """Молекулы разного размера: каждой достаточно своих соседей."""
    torch.manual_seed(0)
    x = torch.randn(30, 3)
    # 4 молекулы: 5, 8, 7, 10 атомов
    batch = torch.tensor([0] * 5 + [1] * 8 + [2] * 7 + [3] * 10)
    ei = knn_graph_pytorch(x, k=4, batch=batch, loop=False)
    # Все рёбра внутри своей молекулы
    same_mol = batch[ei[0]] == batch[ei[1]]
    assert same_mol.all()
    # Самая маленькая молекула (5 атомов) даёт 5*4=20 рёбер
    # Молекулы 8,7,10 атомов дают 8*4+7*4+10*4=100 рёбер
    # Итого 120 рёбер
    assert ei.shape[1] == 120


def test_knn_stable_under_permutation():
    """Те же данные → то же множество рёбер (стабильность)."""
    torch.manual_seed(0)
    x = torch.randn(15, 3)
    batch = torch.tensor([0] * 7 + [1] * 8)
    ei1 = knn_graph_pytorch(x, k=3, batch=batch, loop=False)
    ei2 = knn_graph_pytorch(x, k=3, batch=batch, loop=False)
    assert _edges_set(ei1) == _edges_set(ei2)


def test_knn_memory_block_approach():
    """Проверка, что блочный подход не использует O(N²) память.

    Для N=15360 атомов (bs=1024 × 15 атомов), старый подход требовал
    ~3.7 ГБ только на kNN. Новый блочный — O(15² × 1024) = O(230K).
    """
    torch.manual_seed(0)
    n_mols = 64
    atoms_per_mol = 15
    N = n_mols * atoms_per_mol
    x = torch.randn(N, 3)
    batch = torch.repeat_interleave(torch.arange(n_mols), atoms_per_mol)

    ei = knn_graph_pytorch(x, k=16, batch=batch, loop=False)
    assert ei.shape[0] == 2
    # Все рёбра внутри своей молекулы
    same_mol = batch[ei[0]] == batch[ei[1]]
    assert same_mol.all()
