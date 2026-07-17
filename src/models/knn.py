"""Свой KNN graph — не требует pyg-lib / torch_cluster.

v32: переработан на блочный per-molecule подход.
Раньше: O(N²) по памяти (N — суммарное число атомов в батче),
        для bs=1024 × ~15 атомов = N=15360 → ~3.7 ГБ только на kNN.
Теперь: O(Σ_i n_i²) — для каждой молекулы своя маленькая матрица
        n_i × n_i (≈15×15), суммарно пренебрежимо мало.

Работает на CPU и GPU, без внешних зависимостей кроме torch.
"""
import torch
from torch import Tensor


def _knn_single_molecule(
    x_mol: Tensor, k: int, loop: bool
) -> tuple[Tensor, Tensor]:
    """KNN для одной молекулы.

    Args:
        x_mol: (n, D) координаты атомов одной молекулы
        k: число соседей
        loop: включать ли self-loops

    Returns:
        src, dst: (E,) тензоры индексов рёбер (в локальной нумерации молекулы)
    """
    n = x_mol.shape[0]
    device = x_mol.device

    if n == 0:
        empty = torch.zeros(0, dtype=torch.long, device=device)
        return empty, empty

    # dist: (n, n) — маленькая для одной молекулы
    diff = x_mol.unsqueeze(0) - x_mol.unsqueeze(1)  # (n, n, D)
    dist = (diff ** 2).sum(-1)  # (n, n)

    # Маскируем self-loops если нужно
    if not loop:
        diag_mask = torch.eye(n, dtype=torch.bool, device=device)
        dist = dist.masked_fill(diag_mask, float('inf'))

    # k_actual: не больше, чем есть соседей
    max_neighbours = n if loop else n - 1
    k_actual = min(k, max_neighbours)
    if k_actual < 1:
        empty = torch.zeros(0, dtype=torch.long, device=device)
        return empty, empty

    # topk: для каждой строки — k ближайших
    _, indices = dist.topk(k_actual, dim=-1, largest=False)  # (n, k)

    src = torch.arange(n, device=device).unsqueeze(1).expand(-1, k_actual)  # (n, k)
    src = src.reshape(-1)
    dst = indices.reshape(-1)

    # Удаляем inf-рёбра (когда соседей меньше k)
    gathered = dist.gather(1, indices).reshape(-1)
    valid = torch.isfinite(gathered)
    return src[valid], dst[valid]


def knn_graph_pytorch(
    x: Tensor,
    k: int,
    batch: Tensor | None = None,
    loop: bool = False,
) -> Tensor:
    """Построить kNN граф без pyg-lib.

    Args:
        x: (N, D) — координаты узлов
        k: число ближайших соседей
        batch: (N,) — индекс молекулы для каждого узла (если None — один граф)
        loop: включать ли self-loops

    Returns:
        edge_index: (2, E) — рёбра в формате PyG
    """
    N = x.shape[0]
    device = x.device

    if N == 0:
        return torch.zeros(2, 0, dtype=torch.long, device=device)

    # Если batch не задан — одна молекула
    if batch is None:
        src, dst = _knn_single_molecule(x, k=k, loop=loop)
        return torch.stack([src, dst], dim=0)

    # Blочный per-molecule подход
    src_list = []
    dst_list = []
    offset = 0
    unique_batches, counts = torch.unique_consecutive(batch, return_counts=True)

    # Если batch неотсортирован (молекулы перемешаны), нужно работать через mask
    # torch.unique_consecutive требует отсортированности. Если batch пришёл из PyG —
    # он отсортирован. Но на всякий случай проверим.
    sorted_check = (unique_batches.shape[0] == 1) or \
                   all(unique_batches[i] < unique_batches[i + 1]
                       for i in range(len(unique_batches) - 1))
    if not sorted_check:
        # Batch неотсортирован — группируем вручную
        unique_all = torch.unique(batch)
        for b in unique_all:
            mask = (batch == b)
            idx_local = torch.nonzero(mask, as_tuple=False).squeeze(-1)
            x_mol = x[idx_local]
            src_local, dst_local = _knn_single_molecule(x_mol, k=k, loop=loop)
            # Переводим локальные индексы в глобальные
            src_list.append(idx_local[src_local])
            dst_list.append(idx_local[dst_local])
        if not src_list:
            return torch.zeros(2, 0, dtype=torch.long, device=device)
        src = torch.cat(src_list)
        dst = torch.cat(dst_list)
        return torch.stack([src, dst], dim=0)

    # Быстрый путь: batch отсортирован (стандартный случай PyG)
    for count in counts.tolist():
        count = int(count)
        x_mol = x[offset:offset + count]
        src_local, dst_local = _knn_single_molecule(x_mol, k=k, loop=loop)
        if src_local.numel() > 0:
            src_list.append(src_local + offset)
            dst_list.append(dst_local + offset)
        offset += count

    if not src_list:
        return torch.zeros(2, 0, dtype=torch.long, device=device)

    src = torch.cat(src_list)
    dst = torch.cat(dst_list)
    return torch.stack([src, dst], dim=0)
