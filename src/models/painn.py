"""PaiNN для предсказания скалярных свойств молекул Alchemy.

Таргеты (из Alchemy final_version.csv):
  - mu    (1,) — норма вектора диполя (скаляр, l=0)
  - alpha (1,) — изотропная поляризуемость (скаляр, l=0)
  - gap   (1,) — HOMO-LUMO gap (скаляр, l=0)

PaiNN остаётся E(3)-эквивариантным во внутренних признаках (векторные
признаки обновляются корректно), но финальный выход — скаляр после pooling.

Часть B (программа максимум): для векторного выхода μ нужно использовать
отдельную модель — см. painn_vector.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import PaiNN
from torch_geometric.utils import scatter


class PaiNNModel(nn.Module):
    """PaiNN для скалярных выходов (mu, alpha, gap).

    Args:
        hidden_channels: размер скрытых признаков
        num_layers: число слоёв PaiNN
        num_rbf: число радиальных базисных функций
        cutoff: радиус обрезания (Å)
        predict_mu: предсказывать mu (норма диполя)
        predict_alpha: предсказывать alpha (изотропная поляризуемость)
        predict_gap: предсказывать gap (HOMO-LUMO)
    """

    def __init__(
        self,
        hidden_channels: int = 128,
        num_layers: int = 6,
        num_rbf: int = 16,
        cutoff: float = 5.0,
        predict_mu: bool = True,
        predict_alpha: bool = True,
        predict_gap: bool = True,
    ):
        super().__init__()
        self.predict_mu = predict_mu
        self.predict_alpha = predict_alpha
        self.predict_gap = predict_gap

        # Embedding узлов: 8 признаков (7 типов атомов +1) → hidden
        # Alchemy: H, C, N, O, F, S, Cl = 7 типов
        self.atom_embed = nn.Linear(8, hidden_channels)

        # Основная PaiNN сеть
        self.painn = PaiNN(
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            num_rbf=num_rbf,
            cutoff=cutoff,
        )

        # Heads — каждый предсказывает свой скаляр из пулированных признаков
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
            batch: PyG Batch с x (N, 8), pos (N, 3), batch (N,)

        Returns:
            dict с ключами 'mu', 'alpha', 'gap' (каждый (B, 1)) — в зависимости от флагов
        """
        # Embedding атомов
        h = self.atom_embed(batch.x.float())  # (N, hidden)

        # PaiNN: возвращает (scalar, vec)
        h_s, h_v = self.painn(h, batch.pos, batch.batch)
        # h_s: (N, hidden) — скалярные признаки
        # h_v: (N, hidden, 3) — векторные признаки (не используем для скалярного выхода)

        # Pooling: суммарный вектор молекулы из скалярных признаков
        mol_emb = scatter(h_s, batch.batch, dim=0, reduce="sum")  # (B, hidden)

        out = {}
        if self.predict_mu:
            out["mu"] = self.mu_head(mol_emb)  # (B, 1)
        if self.predict_alpha:
            out["alpha"] = self.alpha_head(mol_emb)  # (B, 1)
        if self.predict_gap:
            out["gap"] = self.gap_head(mol_emb)  # (B, 1)
        return out


def build_painn(
    predict_mu: bool = True,
    predict_alpha: bool = True,
    predict_gap: bool = True,
    **kwargs,
) -> PaiNNModel:
    return PaiNNModel(
        predict_mu=predict_mu,
        predict_alpha=predict_alpha,
        predict_gap=predict_gap,
        **kwargs,
    )
