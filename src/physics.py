"""Вектор дипольного момента μ и тензор поляризуемости α.

Часть B (программа максимума): вместо скалярных |μ| и tr(α)/3,
предсказываем полный вектор μ ∈ R³ и полный тензор α ∈ R^(3×3).

Физика:
  Дипольный момент:
      μ = Σᵢ qᵢ · (rᵢ − COM)
  где qᵢ — частичный заряд атома i, rᵢ — его позиция, COM — центр масс.
  Это E(3)-эквивариантный вектор: при повороте R, μ → R·μ.

  Поляризуемость:
      α_ij = ∂μ_i / ∂E_j = Σᵢ qᵢ(E) · (rᵢ − COM)_i · (rᵢ − COM)_j / |rᵢ − COM|
  В первом приближении (Linearized Coupled Perturbed HF):
      α_ij ≈ Σᵢ qᵢ · (rᵢ − COM)_i · (rᵢ − COM)_j
  Это E(3)-эквивариантный тензор 2-го ранга: при повороте R, α → R·α·Rᵀ.

Реализация:
  - partial_charges = MLP(h_atom) — выучиваемые частичные заряды атомов
  - mu = Σᵢ qᵢ · (rᵢ − COM)                          # (B, 3)
  - alpha_tensor = Σᵢ qᵢ · (rᵢ − COM) ⊗ (rᵢ − COM)  # (B, 3, 3)
  - alpha_iso = trace(alpha_tensor) / 3              # (B, 1) — совместимость с Alchemy

Для обучения:
  - target |μ| известен из Alchemy → loss = (|μ_pred| − |μ_target|)²
  - target tr(α)/3 известен из Alchemy → loss = (trace(α_pred)/3 − alpha_target)²
  - дополнительно: soft constraint на симметрию α (должно быть симметричным)
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import global_add_pool


def compute_dipole_vector(
    atom_charges: Tensor,    # (N, 1) — частичные заряды q_i
    atom_positions: Tensor,  # (N, 3) — координаты r_i
    batch_idx: Tensor,       # (N,) — индекс молекулы
    atom_masses: Tensor,     # (N, 1) — массы для COM
) -> Tensor:
    """Вычислить вектор дипольного момента μ ∈ R³ для каждой молекулы.

    μ = Σᵢ qᵢ · (rᵢ − COM)
    COM = Σ(mᵢ · rᵢ) / Σ(mᵢ)

    Args:
        atom_charges: (N, 1) — выучиваемые заряды атомов
        atom_positions: (N, 3) — позиции атомов (центрированные или нет)
        batch_idx: (N,) — индекс молекулы
        atom_masses: (N, 1) — массы атомов для расчёта COM

    Returns:
        mu: (B, 3) — вектор дипольного момента для каждой молекулы

    E(3)-эквивариантность:
        - Трансляция t: rᵢ → rᵢ+t, COM → COM+t, (rᵢ−COM) не меняется → μ не меняется ✓
        - Вращение R: rᵢ → R·rᵢ, COM → R·COM, (rᵢ−COM) → R·(rᵢ−COM) → μ → R·μ ✓
        - Перестановка: сумма не меняется → μ не меняется ✓
    """
    # COM (центр масс) для каждой молекулы
    weighted_pos = atom_positions * atom_masses  # (N, 3)
    sum_weighted = global_add_pool(weighted_pos, batch_idx)  # (B, 3)
    sum_mass = global_add_pool(atom_masses, batch_idx)  # (B, 1)
    com = sum_weighted / (sum_mass + 1e-8)  # (B, 3)

    # Сдвинутые координаты: rᵢ − COM (для каждого атома своей молекулы)
    shifted = atom_positions - com[batch_idx]  # (N, 3)

    # Диполь: μ = Σᵢ qᵢ · (rᵢ − COM)
    per_atom = atom_charges * shifted  # (N, 3) — broadcast (N,1) × (N,3)
    mu = global_add_pool(per_atom, batch_idx)  # (B, 3)
    return mu


def compute_polarizability_tensor(
    atom_charges: Tensor,    # (N, 1) — частичные заряды q_i
    atom_positions: Tensor,  # (N, 3) — координаты r_i
    batch_idx: Tensor,       # (N,) — индекс молекулы
    atom_masses: Tensor,     # (N, 1) — массы для COM
) -> Tensor:
    """Вычислить тензор поляризуемости α ∈ R^(3×3) для каждой молекулы.

    α_ij ≈ Σᵢ qᵢ · (rᵢ − COM)_i · (rᵢ − COM)_j

    В Linearized CPHF приближении: α пропорциональна
    Σᵢ qᵢ · (rᵢ − COM) ⊗ (rᵢ − COM).

    Args:
        atom_charges: (N, 1) — выучиваемые заряды атомов
        atom_positions: (N, 3) — позиции атомов
        batch_idx: (N,) — индекс молекулы
        atom_masses: (N, 1) — массы атомов

    Returns:
        alpha: (B, 3, 3) — тензор поляризуемости для каждой молекулы

    E(3)-эквивариантность:
        - Трансляция: не меняется (как и μ)
        - Вращение R: rᵢ → R·rᵢ → (rᵢ−COM) → R·(rᵢ−COM) → α → R·α·Rᵀ ✓
        - Перестановка: сумма не меняется → α не меняется ✓

    Физический смысл:
        - α симметрична (α_ij = α_ji) — гарантировано построением
        - tr(α)/3 = изотропная поляризуемость (как в Alchemy)
        - собственные значения α = главные поляризуемости
        - анизотропия = (α_max − α_min) / α_iso
    """
    # COM (центр масс)
    weighted_pos = atom_positions * atom_masses  # (N, 3)
    sum_weighted = global_add_pool(weighted_pos, batch_idx)  # (B, 3)
    sum_mass = global_add_pool(atom_masses, batch_idx)  # (B, 1)
    com = sum_weighted / (sum_mass + 1e-8)  # (B, 3)

    # Сдвинутые координаты
    shifted = atom_positions - com[batch_idx]  # (N, 3)

    # Внешнее произведение: (rᵢ−COM) ⊗ (rᵢ−COM) для каждого атома
    # outer_i = shifted_i[:, None] * shifted_i[None, :]  → (3, 3)
    # Векторизованно: (N, 3, 1) × (N, 1, 3) = (N, 3, 3)
    outer = shifted.unsqueeze(-1) * shifted.unsqueeze(-2)  # (N, 3, 3)

    # Взвешиваем по зарядам
    weighted_outer = outer * atom_charges.unsqueeze(-1)  # (N, 3, 3)

    # Суммируем по атомам каждой молекулы
    # global_add_pool ожидает (N, D), поэтому reshape
    N = outer.shape[0]
    weighted_outer_flat = weighted_outer.reshape(N, 9)  # (N, 9)
    alpha_flat = global_add_pool(weighted_outer_flat, batch_idx)  # (B, 9)
    alpha = alpha_flat.reshape(-1, 3, 3)  # (B, 3, 3)

    # Симметризация (на всякий случай, хотя по построению уже симметричная)
    alpha = 0.5 * (alpha + alpha.transpose(-1, -2))
    return alpha


def polarizability_iso(alpha_tensor: Tensor) -> Tensor:
    """Изотропная поляризуемость: tr(α) / 3.

    Args:
        alpha_tensor: (B, 3, 3)

    Returns:
        alpha_iso: (B, 1)
    """
    return torch.diagonal(alpha_tensor, dim1=-2, dim2=-1).sum(dim=-1, keepdim=True) / 3.0


def polarizability_anisotropy(alpha_tensor: Tensor) -> Tensor:
    """Анизотропия поляризуемости: (α_max − α_min) / α_iso.

    Полезная метрика: 0 для сферически симметричных молекул (CH4),
    большая для вытянутых (CO2).

    Args:
        alpha_tensor: (B, 3, 3)

    Returns:
        anisotropy: (B, 1)
    """
    # Собственные значения — главные поляризуемости
    eigvals = torch.linalg.eigvalsh(alpha_tensor)  # (B, 3) — вещественные, т.к. симметричная
    alpha_iso = polarizability_iso(alpha_tensor)  # (B, 1)
    anisotropy = (eigvals.max(dim=-1).values - eigvals.min(dim=-1).values) / (alpha_iso.squeeze(-1) + 1e-8)
    return anisotropy.unsqueeze(-1)


class DipolePolarizabilityHead(nn.Module):
    """Head для предсказания вектора μ и тензора α через частичные заряды.

    Использование:
        head = DipolePolarizabilityHead(hidden_channels=128)
        atom_charges = head(atom_features)  # (N, 1)
        mu = compute_dipole_vector(atom_charges, positions, batch, masses)
        alpha = compute_polarizability_tensor(atom_charges, positions, batch, masses)
    """

    def __init__(self, hidden_channels: int, hidden_dim: int = 64):
        super().__init__()
        # MLP для предсказания частичных зарядов из признаков атомов
        # Инициализация: начинаем с электронейтральной молекулы (q ≈ 0)
        self.charge_mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Инициализация последнего слоя нулями → начальные заряды ≈ 0
        nn.init.zeros_(self.charge_mlp[-1].weight)
        nn.init.zeros_(self.charge_mlp[-1].bias)

    def forward(self, atom_features: Tensor) -> Tensor:
        """Предсказать частичные заряды атомов.

        Args:
            atom_features: (N, hidden_channels) — признаки атомов из EGNN

        Returns:
            charges: (N, 1) — частичные заряды q_i
        """
        return self.charge_mlp(atom_features)


__all__ = [
    "compute_dipole_vector",
    "compute_polarizability_tensor",
    "polarizability_iso",
    "polarizability_anisotropy",
    "DipolePolarizabilityHead",
]
