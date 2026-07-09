"""EGNN + TDA: основная модель проекта.

Архитектура (по образцу arutamonofu/dls — проект прошлого семестра):
  1. TDA-фичи извлекаются из 3D координат атомов (Vietoris-Rips + Betti curves)
  2. EGNN кодирует геометрию → графовый эмбеддинг
  3. TDA-фичи конкатенируются с графовым эмбеддингом
  4. Финальный MLP предсказывает mu, alpha, gap
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


class EGNNTDA(nn.Module):
    """EGNN с конкатенацией TDA-фичей перед финальным MLP.

    Args:
        hidden_channels: размер скрытых признаков
        num_layers: число слоёв EGNN
        tda_dim: размерность TDA-фичей
        predict_mu, predict_alpha, predict_gap: какие таргеты предсказывать
    """

    def __init__(
        self,
        hidden_channels: int = 128,
        num_layers: int = 4,
        tda_dim: int = 52,
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
        self.tda_dim = tda_dim
        self.predict_mu = predict_mu
        self.predict_alpha = predict_alpha
        self.predict_gap = predict_gap

        # Embedding атомов
        self.atom_embed = nn.Linear(8, hidden_channels)

        # EGNN слои
        self.egnn_layers = nn.ModuleList([
            EGNN_Sparse(feats_dim=hidden_channels, pos_dim=3)
            for _ in range(num_layers)
        ])

        # Heads с учётом TDA-фичей
        head_in = hidden_channels + tda_dim
        if predict_mu:
            self.mu_head = nn.Sequential(
                nn.Linear(head_in, hidden_channels),
                nn.SiLU(),
                nn.Linear(hidden_channels, 1),
            )
        if predict_alpha:
            self.alpha_head = nn.Sequential(
                nn.Linear(head_in, hidden_channels),
                nn.SiLU(),
                nn.Linear(hidden_channels, 1),
            )
        if predict_gap:
            self.gap_head = nn.Sequential(
                nn.Linear(head_in, hidden_channels),
                nn.SiLU(),
                nn.Linear(hidden_channels, 1),
            )

    def forward(self, batch) -> dict[str, Tensor]:
        h = self.atom_embed(batch.x.float())
        pos = batch.pos

        edge_index = batch.edge_index if hasattr(batch, 'edge_index') and batch.edge_index.numel() > 0 else None
        if edge_index is None or edge_index.numel() == 0:
            edge_index = radius_graph(pos, r=5.0, batch=batch.batch,
                                       loop=False, max_num_neighbors=32)

        for layer in self.egnn_layers:
            combined = torch.cat([pos, h], dim=-1)
            combined = layer(combined, edge_index, batch=batch.batch)
            pos = combined[:, :3]
            h = combined[:, 3:]

        # Pooling
        mol_emb = global_add_pool(h, batch.batch)  # (B, hidden)

        # Конкатенируем с TDA-фичами
        if hasattr(batch, 'tda'):
            tda = batch.tda  # (B, tda_dim)
            mol_emb = torch.cat([mol_emb, tda], dim=-1)  # (B, hidden + tda_dim)

        out = {}
        if self.predict_mu:
            out["mu"] = self.mu_head(mol_emb)
        if self.predict_alpha:
            out["alpha"] = self.alpha_head(mol_emb)
        if self.predict_gap:
            out["gap"] = self.gap_head(mol_emb)
        return out


def build_egnn_tda(
    tda_dim: int = 52,
    predict_mu: bool = True,
    predict_alpha: bool = True,
    predict_gap: bool = True,
    **kwargs,
) -> EGNNTDA:
    return EGNNTDA(
        tda_dim=tda_dim,
        predict_mu=predict_mu,
        predict_alpha=predict_alpha,
        predict_gap=predict_gap,
        **kwargs,
    )
