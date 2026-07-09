"""EGNN baseline (и основная модель) — использует готовую реализацию egnn-pytorch.

EGNN: E(n)-EquInvariant Graph Neural Network (Satorras et al., 2021)
https://arxiv.org/abs/2102.09844

E(3)-эквивариантна (сдвиги + повороты + отражения + перестановки).
Использует готовую реализацию из egnn-pytorch (как в проектах прошлого семестра).

Установка: pip install egnn-pytorch
"""
import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.utils import scatter
from torch_geometric.nn import global_add_pool

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
        predict_mu, predict_alpha, predict_gap: какие таргеты предсказывать
    """

    def __init__(
        self,
        hidden_channels: int = 128,
        num_layers: int = 4,
        predict_mu: bool = True,
        predict_alpha: bool = True,
        predict_gap: bool = True,
        **kwargs,
    ):
        super().__init__()
        if not EGNN_AVAILABLE:
            raise ImportError(
                "egnn-pytorch не установлен. Установите: pip install egnn-pytorch"
            )

        self.hidden_channels = hidden_channels
        self.predict_mu = predict_mu
        self.predict_alpha = predict_alpha
        self.predict_gap = predict_gap

        # Embedding атомов: 8 признаков → hidden
        self.atom_embed = nn.Linear(8, hidden_channels)

        # Стек EGNN слоёв
        self.egnn_layers = nn.ModuleList([
            EGNN_Sparse(feats_dim=hidden_channels, pos_dim=3)
            for _ in range(num_layers)
        ])

        # Heads для скалярных выходов
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
        """
        Args:
            batch: PyG Batch с x (N, 8), pos (N, 3), batch (N,), edge_index (2, E)
        """
        # Embedding
        h = self.atom_embed(batch.x.float())  # (N, hidden)
        pos = batch.pos  # (N, 3)

        # EGNN_Sparse принимает edge_index
        edge_index = batch.edge_index if hasattr(batch, 'edge_index') and batch.edge_index.numel() > 0 else None

        # Если edge_index нет — строим полный граф по молекулам
        if edge_index is None or edge_index.numel() == 0:
            from torch_geometric.nn import radius_graph
            edge_index = radius_graph(pos, r=5.0, batch=batch.batch,
                                       loop=False, max_num_neighbors=32)

        # Проходим через слои EGNN
        for layer in self.egnn_layers:
            # EGNN_Sparse: принимает конкатенированный тензор [pos, h]
            combined = torch.cat([pos, h], dim=-1)  # (N, 3 + hidden)
            combined = layer(combined, edge_index, batch=batch.batch)
            pos = combined[:, :3]
            h = combined[:, 3:]

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
