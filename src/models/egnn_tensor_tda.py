"""EGNN Tensor + TDA: вектор μ + тензор α + топологические фичи.

Объединение:
  - EGNN Tensor (часть B): update_coors=True, вектор μ и тензор α через заряды
  - TDA-фичи (concat или film) для скалярного head (gap)
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

from .knn import knn_graph_pytorch as knn_graph

try:
    from physics import (
        DipolePolarizabilityHead,
        compute_dipole_vector,
        compute_polarizability_tensor,
        polarizability_iso,
    )
except ImportError:
    from ..physics import (
        DipolePolarizabilityHead,
        compute_dipole_vector,
        compute_polarizability_tensor,
        polarizability_iso,
    )

try:
    from tda.film import FiLMModulation
except ImportError:
    from ..tda.film import FiLMModulation

NUM_ATOM_TYPES = 7


class EGNNTensorTDA(nn.Module):
    """EGNN Tensor + TDA: вектор μ + тензор α + TDA-фичи для gap."""

    def __init__(
        self,
        hidden_channels: int = 128,
        num_layers: int = 4,
        cutoff: float = 5.0,
        k_neighbors: int = 16,
        m_dim: int = 32,
        tda_dim: int = 52,
        tda_mode: str = "concat",
        predict_alpha_tensor: bool = True,
        predict_gap: bool = True,
        **kwargs,
    ):
        super().__init__()
        if not EGNN_AVAILABLE:
            raise ImportError("egnn-pytorch не установлен")
        assert tda_mode in ("concat", "film"), f"Unknown tda_mode: {tda_mode}"

        self.hidden_channels = hidden_channels
        self.cutoff = cutoff
        self.k_neighbors = k_neighbors
        self.m_dim = m_dim
        self.tda_dim = tda_dim
        self.tda_mode = tda_mode
        self.predict_alpha_tensor = predict_alpha_tensor
        self.predict_gap = predict_gap

        self.atom_embed = nn.Embedding(NUM_ATOM_TYPES, hidden_channels)

        self.egnn_layers = nn.ModuleList([
            EGNN_Sparse(
                feats_dim=hidden_channels,
                pos_dim=3,
                edge_attr_dim=1,
                update_coors=True,
                update_feats=True,
                norm_feats=False,
                norm_coors=True,
                m_dim=m_dim,
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(hidden_channels)
        self.global_norm = nn.LayerNorm(NUM_ATOM_TYPES + 2)
        self.tda_norm = nn.BatchNorm1d(tda_dim)

        self.dipole_pol_head = DipolePolarizabilityHead(hidden_channels)

        global_dim = NUM_ATOM_TYPES + 2
        if tda_mode == "concat":
            head_in = hidden_channels + global_dim + tda_dim
            self.film = None
        else:
            self.film = FiLMModulation(tda_dim=tda_dim, feat_dim=hidden_channels)
            head_in = hidden_channels + global_dim

        if predict_gap:
            self.gap_head = nn.Sequential(
                nn.Linear(head_in, hidden_channels), nn.SiLU(),
                nn.Linear(hidden_channels, 1)
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
        atom_types = batch.x[:, :NUM_ATOM_TYPES].argmax(dim=-1).long()
        feats = self.atom_embed(atom_types)
        coors = batch.pos / self.cutoff

        edge_index = knn_graph(coors, k=self.k_neighbors, batch=batch.batch, loop=False)
        row, col = edge_index
        edge_dist = (coors[row] - coors[col]).norm(dim=-1, keepdim=True)

        x = torch.cat([coors, feats], dim=-1)
        for layer in self.egnn_layers:
            x = layer(x, edge_index, edge_attr=edge_dist, batch=batch.batch)
        h = x[:, 3:]

        atom_charges = self.dipole_pol_head(h)
        mass = batch.x[:, -1:]
        physical_coors = batch.pos

        mu = compute_dipole_vector(
            atom_charges=atom_charges,
            atom_positions=physical_coors,
            batch_idx=batch.batch,
            atom_masses=mass,
        )
        result = {"mu": mu}

        if self.predict_alpha_tensor:
            alpha_tensor = compute_polarizability_tensor(
                atom_charges=atom_charges,
                atom_positions=physical_coors,
                batch_idx=batch.batch,
                atom_masses=mass,
            )
            result["alpha_tensor"] = alpha_tensor
            result["alpha"] = polarizability_iso(alpha_tensor)

        if self.predict_gap:
            mol_emb = global_add_pool(h, batch.batch)
            mol_emb = self.final_norm(mol_emb)
            global_desc = self._global_descriptors(batch)
            global_desc = self.global_norm(global_desc)

            if not hasattr(batch, "tda"):
                raise ValueError(
                    "Модель expects TDA-фичи, но batch.tda отсутствует. "
                    "Создайте датасет с tda_features=True."
                )
            tda_feat = self.tda_norm(batch.tda)

            if self.tda_mode == "concat":
                mol_emb = torch.cat([mol_emb, global_desc, tda_feat], dim=-1)
            else:
                mol_emb = self.film(mol_emb, tda_feat)
                mol_emb = torch.cat([mol_emb, global_desc], dim=-1)

            result["gap"] = self.gap_head(mol_emb)

        return result


def build_egnn_tensor_tda(tda_dim=52, predict_alpha_tensor=True, predict_gap=True, **kwargs):
    return EGNNTensorTDA(
        tda_dim=tda_dim,
        predict_alpha_tensor=predict_alpha_tensor,
        predict_gap=predict_gap,
        **kwargs
    )
