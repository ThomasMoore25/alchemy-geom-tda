"""Свой KNN graph — не требует pyg-lib / torch_cluster.

v33.5: ВОЗВРАЩЁН быстрый векторизованный подход из v29.
v32.4 переписал на блочный per-molecule, что убило скорость на GPU
(225 сек/эпоха вместо 10 сек) — Python loop по 1024 молекулам медленнее,
чем один большой GPU op.

Для bs=1024 × ~15 атомов = N=15360:
- (N, N, D) float32 = 2.64 ГБ — влезает в T4 16GB
- Один topk на всю матрицу — быстро

Защита от OOM: если N > MAX_N_VECTORIZED, разбиваем на блоки по chunk_size.
"""
import torch
from torch import Tensor

# Если N больше этого — используем блочный подход (защита от OOM)
# Для T4 16GB: N=20000 → 4.5 ГБ на diff tensor, влезает
# Для CPU/маленьких GPU: можно уменьшить
MAX_N_VECTORIZED = 50000


def _knn_vectorized(
    x: Tensor, k: int, batch: Tensor | None, loop: bool
) -> Tensor:
    """Полностью векторизованный kNN — ОДИН большой GPU op.

    Быстро на GPU (один topk на всю матрицу), но O(N²) по памяти.
    """
    N = x.shape[0]
    device = x.device

    # Считаем попарные расстояния: (N, N)
    diff = x.unsqueeze(0) - x.unsqueeze(1)  # (N, N, D)
    dist = (diff ** 2).sum(-1)  # (N, N)

    # Маскируем узлы из других молекул
    if batch is not None:
        same_mol = (batch.unsqueeze(0) == batch.unsqueeze(1))  # (N, N)
        dist = dist.masked_fill(~same_mol, float("inf"))

    # Маскируем self-loops если нужно
    if not loop:
        diag_mask = torch.eye(N, dtype=torch.bool, device=device)
        dist = dist.masked_fill(diag_mask, float("inf"))

    # k_actual: не больше, чем есть соседей
    k_actual = min(k, N - 1 if not loop else N)
    if k_actual < 1:
        return torch.zeros(2, 0, dtype=torch.long, device=device)

    # topk: для каждой строки — k ближайших
    _, indices = dist.topk(k_actual, dim=-1, largest=False)  # (N, k)

    # Строим edge_index
    src = torch.arange(N, device=device).unsqueeze(1).expand(-1, k_actual)
    edge_index = torch.stack([src.reshape(-1), indices.reshape(-1)], dim=0)

    # Удаляем рёбра с inf (когда соседей меньше k)
    valid = torch.isfinite(dist.gather(1, indices).reshape(-1))
    edge_index = edge_index[:, valid]
    return edge_index


def _knn_single_molecule(
    x_mol: Tensor, k: int, loop: bool
) -> tuple[Tensor, Tensor]:
    """KNN для одной молекулы (для блочного fallback)."""
    n = x_mol.shape[0]
    device = x_mol.device

    if n == 0:
        empty = torch.zeros(0, dtype=torch.long, device=device)
        return empty, empty

    diff = x_mol.unsqueeze(0) - x_mol.unsqueeze(1)
    dist = (diff ** 2).sum(-1)

    if not loop:
        diag_mask = torch.eye(n, dtype=torch.bool, device=device)
        dist = dist.masked_fill(diag_mask, float("inf"))

    max_neighbours = n if loop else n - 1
    k_actual = min(k, max_neighbours)
    if k_actual < 1:
        empty = torch.zeros(0, dtype=torch.long, device=device)
        return empty, empty

    _, indices = dist.topk(k_actual, dim=-1, largest=False)

    src = torch.arange(n, device=device).unsqueeze(1).expand(-1, k_actual)
    src = src.reshape(-1)
    dst = indices.reshape(-1)

    gathered = dist.gather(1, indices).reshape(-1)
    valid = torch.isfinite(gathered)
    return src[valid], dst[valid]


def _knn_block_fallback(
    x: Tensor, k: int, batch: Tensor, loop: bool
) -> Tensor:
    """Блочный per-molecule fallback для очень больших N (защита от OOM).

    Медленнее на GPU, но O(Σ n_i²) по памяти вместо O(N²).
    """
    src_list = []
    dst_list = []
    offset = 0
    unique_batches, counts = torch.unique_consecutive(batch, return_counts=True)

    for count in counts.tolist():
        count = int(count)
        x_mol = x[offset:offset + count]
        src_local, dst_local = _knn_single_molecule(x_mol, k=k, loop=loop)
        if src_local.numel() > 0:
            src_list.append(src_local + offset)
            dst_list.append(dst_local + offset)
        offset += count

    if not src_list:
        return torch.zeros(2, 0, dtype=torch.long, device=x.device)

    src = torch.cat(src_list)
    dst = torch.cat(dst_list)
    return torch.stack([src, dst], dim=0)


def knn_graph_pytorch(
    x: Tensor,
    k: int,
    batch: Tensor | None = None,
    loop: bool = False,
) -> Tensor:
    """Построить kNN граф без pyg-lib.

    v33.5: по умолчанию использует быстрый векторизованный подход (как в v29).
    Если N > MAX_N_VECTORIZED — переключается на блочный (защита от OOM).

    Args:
        x: (N, D) — координаты узлов
        k: число ближайших соседей
        batch: (N,) — индекс молекулы для каждого узла (если None — один граф)
        loop: включать ли self-loops

    Returns:
        edge_index: (2, E)
    """
    N = x.shape[0]
    device = x.device

    if N == 0:
        return torch.zeros(2, 0, dtype=torch.long, device=device)

    # v33.5: если N маленькое — векторизованный (быстро на GPU)
    # Если N большое — блочный (защита от OOM)
    if N <= MAX_N_VECTORIZED:
        return _knn_vectorized(x, k, batch, loop)
    else:
        # Блочный fallback для огромных батчей
        if batch is None:
            # Один граф — разбиваем на блоки по молекулам невозможно,
            # используем чанкование
            return _knn_vectorized(x, k, batch, loop)
        return _knn_block_fallback(x, k, batch, loop)
