"""EGNN v15: ТОЧНО как в проекте arutamonofu/dls (прошлый семестр).

Ключевые отличия от v14:
  1. Используем ХИМИЧЕСКИЕ СВЯЗИ из SDF (edge_index), а не knn_graph
  2. update_coors=True (дефолт) — как в оригинальной статье EGNN
  3. НЕ нормализуем координаты (в Alchemy они ~1-5 Å, это нормально)
  4. nn.Embedding для типов атомов
  5. global_add_pool для pooling
"""
import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import global_add_pool

try:
    from egnn_pytorch import EGNN_Sparse
    EGNN_AVAILABLE = True
except ImportError:
    EGNN_AVAILABLE = False

NUM_ATOM_TYPES = 7


class EGNNModel(nn.Module):
    """EGNN для скалярных выходов — как в проекте прошлого семестра."""

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
        self.predict_mu = predict_mu
        self.predict_alpha = predict_alpha
        self.predict_gap = predict_gap

        # Embedding атомов (как в проекте прошлого семестра)
        self.atom_embed = nn.Embedding(NUM_ATOM_TYPES, hidden_channels)
        self.node_in_proj = nn.Linear(hidden_channels, hidden_channels)

        # EGNN слои — ДЕФОЛТНЫЕ ПАРАМЕТРЫ (update_coors=True, norm_feats=False)
        self.egnn_layers = nn.ModuleList([
            EGNN_Sparse(
                feats_dim=hidden_channels,
                pos_dim=3,
            )
            for _ in range(num_layers)
        ])

        # Глобальные дескрипторы
        global_dim = NUM_ATOM_TYPES + 2
        head_in = hidden_channels + global_dim

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

    def _global_descriptors(self, batch) -> Tensor:
        atom_onehot = batch.x[:, :NUM_ATOM_TYPES]
        mass = batch.x[:, -1:]
        hist = global_add_pool(atom_onehot, batch.batch)
        ones = torch.ones(mass.shape[0], 1, device=mass.device)
        n_atoms = global_add_pool(ones, batch.batch)
        total_mass = global_add_pool(mass, batch.batch)
        return torch.cat([hist, n_atoms, total_mass], dim=-1)

    def forward(self, batch) -> dict[str, Tensor]:
        # Тип атома из one-hot
        atom_types = batch.x[:, :NUM_ATOM_TYPES].argmax(dim=-1).long()

        # Embedding
        h = self.atom_embed(atom_types)
        h = self.node_in_proj(h)
        pos = batch.pos  # БЕЗ нормализации!

        # Используем ХИМИЧЕСКИЕ СВЯЗИ из SDF (не knn_graph)
        edge_index = batch.edge_index

        # EGNN слои — как в проекте прошлого семестра
        for layer in self.egnn_layers:
            combined = torch.cat([pos, h], dim=-1)
            combined = layer(combined, edge_index, batch=batch.batch)
            pos = combined[:, :3]
            h = combined[:, 3:]

        # Pooling
        mol_emb = global_add_pool(h, batch.batch)

        # Глобальные дескрипторы
        global_desc = self._global_descriptors(batch)
        mol_emb = torch.cat([mol_emb, global_desc], dim=-1)

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
