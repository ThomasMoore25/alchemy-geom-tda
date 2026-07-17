"""EGNN: E(3)-эквивариантная нейросеть для предсказания mu/alpha/gap.

Архитектура:
  - egnn-pytorch EGNN_Sparse, update_coors=False (только обновление признаков)
  - m_dim=32, norm_feats=False, norm_coors=False
  - Собственный knn_graph_pytorch (без pyg-lib)
  - Нормализация координат pos / 5.0
  - nn.Embedding для типов атомов (7 типов: H, C, N, O, F, S, Cl)
  - Глобальные дескрипторы (гистограмма атомов + масса + число атомов)
  - LayerNorm перед heads
  - Отдельные heads для mu, alpha, gap
  - Skip connection для alpha: alpha = Linear(global_desc) + head(mol_emb)

Рекомендуемый lr: 1e-3 (при 5e-4 обучение медленное, при 2e-3 нестабильное)
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

NUM_ATOM_TYPES = 7


class EGNNModel(nn.Module):
    def __init__(
        self,
        hidden_channels: int = 128,
        num_layers: int = 4,
        cutoff: float = 5.0,
        k_neighbors: int = 16,
        m_dim: int = 32,
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
        self.k_neighbors = k_neighbors
        self.m_dim = m_dim
        self.predict_mu = predict_mu
        self.predict_alpha = predict_alpha
        self.predict_gap = predict_gap

        self.atom_embed = nn.Embedding(NUM_ATOM_TYPES, hidden_channels)

        self.egnn_layers = nn.ModuleList([
            EGNN_Sparse(
                feats_dim=hidden_channels,
                pos_dim=3,
                edge_attr_dim=1,
                update_coors=False,
                update_feats=True,
                norm_feats=False,
                norm_coors=False,
                m_dim=m_dim,
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(hidden_channels)

        global_dim = NUM_ATOM_TYPES + 2
        head_in = hidden_channels + global_dim

        # Нормализация глобальных дескрипторов (критично!)
        # Без этого mass~300 доминирует над mol_emb~1 после LayerNorm
        self.global_norm = nn.LayerNorm(global_dim)

        # ОТДЕЛЬНЫЕ heads для каждого таргета
        if predict_mu:
            self.mu_head = nn.Sequential(
                nn.Linear(head_in, hidden_channels), nn.SiLU(),
                nn.Linear(hidden_channels, 1))
        if predict_alpha:
            # Skip connection: alpha напрямую из глобальных дескрипторов
            # alpha ~ молекулярный объём ~ функция от числа и типов атомов
            self.alpha_skip = nn.Linear(global_dim, 1)  # прямой путь
            self.alpha_head = nn.Sequential(
                nn.Linear(head_in, hidden_channels), nn.SiLU(),
                nn.Linear(hidden_channels, 1))  # уточняющий путь
        if predict_gap:
            self.gap_head = nn.Sequential(
                nn.Linear(head_in, hidden_channels), nn.SiLU(),
                nn.Linear(hidden_channels, 1))

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
        coors = batch.pos / self.cutoff  # нормализация координат

        edge_index = knn_graph(coors, k=self.k_neighbors, batch=batch.batch, loop=False)
        row, col = edge_index
        edge_dist = (coors[row] - coors[col]).norm(dim=-1, keepdim=True)

        x = torch.cat([coors, feats], dim=-1)
        for layer in self.egnn_layers:
            x = layer(x, edge_index, edge_attr=edge_dist, batch=batch.batch)
        h = x[:, 3:]

        mol_emb = global_add_pool(h, batch.batch)
        mol_emb = self.final_norm(mol_emb)
        global_desc = self._global_descriptors(batch)
        global_desc = self.global_norm(global_desc)  # Нормализация!
        mol_emb = torch.cat([mol_emb, global_desc], dim=-1)

        result = {}
        if self.predict_mu:
            result["mu"] = self.mu_head(mol_emb)
        if self.predict_alpha:
            # Skip connection: alpha = skip(global_desc) + head(mol_emb)
            result["alpha"] = self.alpha_skip(global_desc) + self.alpha_head(mol_emb)
        if self.predict_gap:
            result["gap"] = self.gap_head(mol_emb)
        return result


def build_egnn(predict_mu=True, predict_alpha=True, predict_gap=True, **kwargs):
    return EGNNModel(predict_mu=predict_mu, predict_alpha=predict_alpha,
                     predict_gap=predict_gap, **kwargs)
