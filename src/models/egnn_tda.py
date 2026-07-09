"""EGNN + TDA: основная модель проекта.

Архитектура:
  1. TDA-фичи извлекаются из 3D координат атомов (Vietoris-Rips + Betti curves)
  2. EGNN кодирует геометрию через radius_graph (все атомы в радиусе 5 Å)
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
    """EGNN с конкатенацией TDA-фичей перед финальным MLP."""

    def __init__(
        self,
        hidden_channels: int = 128,
        num_layers: int = 4,
        cutoff: float = 5.0,
        tda_dim: int = 52,
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
        self.tda_dim = tda_dim
        self.predict_mu = predict_mu
        self.predict_alpha = predict_alpha
        self.predict_gap = predict_gap

        self.atom_embed = nn.Linear(8, hidden_channels)

        self.egnn_layers = nn.ModuleList([
            EGNN_Sparse(
                feats_dim=hidden_channels,
                pos_dim=3,
                edge_dim=1,
            )
            for _ in range(num_layers)
        ])

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

        edge_index = radius_graph(
            pos, r=self.cutoff, batch=batch.batch,
            loop=False, max_num_neighbors=64,
        )
        row, col = edge_index
        edge_dist = (pos[row] - pos[col]).norm(dim=-1, keepdim=True)

        for layer in self.egnn_layers:
            h, pos = layer(h, pos, edge_index, edge_attr=edge_dist)

        mol_emb = global_add_pool(h, batch.batch)

        if hasattr(batch, 'tda'):
            tda = batch.tda
            mol_emb = torch.cat([mol_emb, tda], dim=-1)

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
