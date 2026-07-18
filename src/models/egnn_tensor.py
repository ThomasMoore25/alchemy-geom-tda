"""EGNN Tensor: EGNN с векторным μ ∈ R³ и тензорным α ∈ R^(3×3).

Часть B (программа максимума): вместо скалярных |μ| и tr(α)/3,
предсказываем полный вектор диполя и полный тензор поляризуемости.

Архитектура:
  1. EGNN слои обновляют координаты (update_coors=True)
  2. charge_head(h_atom) → q_i ∈ R (частичные заряды)
  3. μ = Σᵢ qᵢ · (rᵢ − COM)                       # (B, 3) — вектор
  4. α = Σᵢ qᵢ · (rᵢ − COM) ⊗ (rᵢ − COM)         # (B, 3, 3) — тензор
  5. α_iso = tr(α) / 3                             # (B, 1) — совместимость с Alchemy

E(3)-эквивариантность:
  - При трансляции: COM сдвигается, (rᵢ−COM) не меняется → μ, α не меняются ✓
  - При вращении R: (rᵢ−COM) → R·(rᵢ−COM) → μ → R·μ, α → R·α·Rᵀ ✓
  - При перестановке: сумма не меняется → μ, α не меняются ✓
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

# v33: dual import для совместимости с обоими UX
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

NUM_ATOM_TYPES = 7  # Alchemy: H, C, N, O, F, S, Cl


class EGNNTensorModel(nn.Module):
    """EGNN с эквивариантными векторным μ и тензорным α выходами.

    Args:
        hidden_channels: размер скрытых признаков
        num_layers: число слоёв EGNN
        cutoff: радиус отсечения + нормализация координат
        k_neighbors: число соседей в kNN
        m_dim: размерность m в EGNN_Sparse
        predict_alpha_tensor: если True, предсказывает полный тензор α
                              (иначе только вектор μ)
        predict_gap: предсказывать скалярный gap
    """

    def __init__(
        self,
        hidden_channels: int = 128,
        num_layers: int = 4,
        cutoff: float = 5.0,
        k_neighbors: int = 16,
        m_dim: int = 32,
        predict_alpha_tensor: bool = True,
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
        self.predict_alpha_tensor = predict_alpha_tensor
        self.predict_gap = predict_gap

        self.atom_embed = nn.Embedding(NUM_ATOM_TYPES, hidden_channels)

        # update_coors=True — нужны обновлённые координаты для физически корректного μ и α
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

        # Head для частичных зарядов
        self.dipole_pol_head = DipolePolarizabilityHead(hidden_channels)

        # Скалярные heads для gap (если нужно)
        global_dim = NUM_ATOM_TYPES + 2
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
        x[:, :3]  # (N, 3) — в масштабе pos/cutoff
        h = x[:, 3:]  # (N, hidden)

        # Частичные заряды
        atom_charges = self.dipole_pol_head(h)  # (N, 1)

        # Массы для COM
        mass = batch.x[:, -1:]  # (N, 1)

        # v33.8: используем ОРИГИНАЛЬНЫЕ координаты (batch.pos) для физических
        # расчётов μ и α, а не updated_coors (которые в масштабе pos/cutoff).
        # Раньше: updated_coors = pos/cutoff → μ занижен в cutoff раз,
        #         α занижена в cutoff^2 раз (25x при cutoff=5).
        # Теперь: batch.pos → μ и α в правильных физических единицах.
        physical_coors = batch.pos  # оригинальные координаты в Å

        # Вектор дипольного момента μ ∈ R³ (эквивариантный)
        mu = compute_dipole_vector(
            atom_charges=atom_charges,
            atom_positions=physical_coors,
            batch_idx=batch.batch,
            atom_masses=mass,
        )  # (B, 3)

        result = {"mu": mu}  # вектор!

        # Тензор поляризуемости α ∈ R^(3×3) (эквивариантный)
        if self.predict_alpha_tensor:
            alpha_tensor = compute_polarizability_tensor(
                atom_charges=atom_charges,
                atom_positions=physical_coors,
                batch_idx=batch.batch,
                atom_masses=mass,
            )  # (B, 3, 3)
            result["alpha_tensor"] = alpha_tensor

            # Также возвращаем изотропную поляризуемость для совместимости с Alchemy
            result["alpha"] = polarizability_iso(alpha_tensor)  # (B, 1)

        # Скалярный gap
        if self.predict_gap:
            mol_emb = global_add_pool(h, batch.batch)
            mol_emb = self.final_norm(mol_emb)
            global_desc = self._global_descriptors(batch)
            global_desc = self.global_norm(global_desc)
            mol_emb = torch.cat([mol_emb, global_desc], dim=-1)
            result["gap"] = self.gap_head(mol_emb)

        return result


def build_egnn_tensor(predict_alpha_tensor=True, predict_gap=True, **kwargs):
    return EGNNTensorModel(
        predict_alpha_tensor=predict_alpha_tensor,
        predict_gap=predict_gap,
        **kwargs
    )
