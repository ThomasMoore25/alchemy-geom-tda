"""EGNN с правильным API и radius_graph вместо химических связей.

Главные отличия от v7:
  1. Используем radius_graph (радиус 5 Å) вместо химических связей из SDF
     → EGNN видит все пары атомов в радиусе, не только связанных
  2. Вызываем EGNN_Sparse правильно: feats и pos отдельно (не склеенные)
  3. Добавляем edge_attr (расстояния) для лучшей работы
"""
import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import global_add_pool, radius_graph

try:
    from egnn_pytorch import EGNN_Sparse
    EGNN_AVAILABLE = True
except ImportError:
    EGNN_AVAILABLE = False


class EGNNModel(nn.Module):
    """EGNN для скалярных выходов (mu, alpha, gap).

    Args:
        hidden_channels: размер скрытых признаков
        num_layers: число слоёв EGNN
        cutoff: радиус для radius_graph (Å)
        predict_mu, predict_alpha, predict_gap: какие таргеты предсказывать
    """

    def __init__(
        self,
        hidden_channels: int = 128,
        num_layers: int = 4,
        cutoff: float = 5.0,
        predict_mu: bool = True,
        predict_alpha: bool = True,
        predict_gap: bool = True,
        **kwargs,
    ):
        super().__init__()
        if not EGNN_AVAILABLE:
            raise ImportError("egnn-pytorch не установлен: pip install egnn-pytorch")

        self.hidden_channels = hidden_channels
        self.cutoff = cutoff
        self.predict_mu = predict_mu
        self.predict_alpha = predict_alpha
        self.predict_gap = predict_gap

        # Embedding атомов: 8 признаков → hidden
        self.atom_embed = nn.Linear(8, hidden_channels)

        # EGNN слои — с правильным API
        self.egnn_layers = nn.ModuleList([
            EGNN_Sparse(
                feats_dim=hidden_channels,
                pos_dim=3,
                edge_dim=1,          # передаём расстояние как edge_attr
                num_nearest_neighbors=None,  # не ограничиваем
            )
            for _ in range(num_layers)
        ])

        # Heads
        if predict_mu:
            self.mu_head = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels // 2),
                nn.SiLU(),
                nn.Linear(hidden_channels // 2, 1),
            )
        if predict_alpha:
            self.alpha_head = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels // 2),
                nn.SiLU(),
                nn.Linear(hidden_channels // 2, 1),
            )
        if predict_gap:
            self.gap_head = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels // 2),
                nn.SiLU(),
                nn.Linear(hidden_channels // 2, 1),
            )

    def forward(self, batch) -> dict[str, Tensor]:
        h = self.atom_embed(batch.x.float())  # (N, hidden)
        pos = batch.pos  # (N, 3)

        # Строим граф по радиусу — ВСЕ пары атомов в радиусе cutoff
        edge_index = radius_graph(
            pos, r=self.cutoff, batch=batch.batch,
            loop=False, max_num_neighbors=64,
        )
        # edge_attr: расстояния для каждого ребра
        row, col = edge_index
        edge_vec = pos[row] - pos[col]
        edge_dist = edge_vec.norm(dim=-1, keepdim=True)  # (E, 1)

        # Проходим через EGNN слои — ПРАВИЛЬНЫЙ API
        for layer in self.egnn_layers:
            h, pos = layer(h, pos, edge_index, edge_attr=edge_dist)

        # Pooling: суммарный вектор молекулы
        mol_emb = global_add_pool(h, batch.batch)  # (B, hidden)

        out = {}
        if self.predict_mu:
            out["mu"] = self.mu_head(mol_emb)
        if self.predict_alpha:
            out["alpha"] = self.alpha_head(mol_emb)
        if self.predict_gap:
            out["gap"] = self.gap_head(mol_emb)
        return out


def build_egnn(
    predict_mu: bool = True,
    predict_alpha: bool = True,
    predict_gap: bool = True,
    **kwargs,
) -> EGNNModel:
    return EGNNModel(
        predict_mu=predict_mu,
        predict_alpha=predict_alpha,
        predict_gap=predict_gap,
        **kwargs,
    )
