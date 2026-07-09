"""PaiNN + TDA: основная модель проекта.

Архитектура:
  1. TDA-фичи извлекаются из 3D координат атомов (Vietoris-Rips + Betti curves)
  2. TDA-фичи подаются в FiLM conditioning
  3. FiLM модулирует узловые скалярные признаки после половины слоёв PaiNN
  4. Дальше обычный PaiNN + heads для mu/alpha/gap

Эквивариантность сохраняется: TDA-фичи E(3)-инвариантны (топология не меняется
при изометриях), FiLM модуляция γ*h + β сохраняет тип поля (скаляр остаётся скаляром).
"""
import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import PaiNN as PaiNNLayer
from torch_geometric.utils import scatter

from .painn import PaiNNModel
from ..tda.film import FiLMNodeModulation


class PaiNNTDA(PaiNNModel):
    """PaiNN с интеграцией TDA-фичей через FiLM conditioning.

    Args:
        tda_dim: размерность TDA-фичей (по умолчанию 52)
        tda_film_position: после какого слоя вставлять FiLM (по умолчанию num_layers // 2)
        Остальные параметры как у PaiNNModel
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
        tda_dim: int = 52,
        tda_film_position: int | None = None,
    ):
        super().__init__(
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            num_rbf=num_rbf,
            cutoff=cutoff,
            predict_mu=predict_mu,
            predict_alpha=predict_alpha,
            predict_gap=predict_gap,
        )
        # Заменяем единый PaiNN на два блока с FiLM между ними
        if tda_film_position is None:
            tda_film_position = num_layers // 2
        self.tda_film_position = tda_film_position

        self.painn_pre = PaiNNLayer(
            hidden_channels=hidden_channels,
            num_layers=tda_film_position,
            num_rbf=num_rbf,
            cutoff=cutoff,
        )
        self.film = FiLMNodeModulation(tda_dim, hidden_channels)
        self.painn_post = PaiNNLayer(
            hidden_channels=hidden_channels,
            num_layers=num_layers - tda_film_position,
            num_rbf=num_rbf,
            cutoff=cutoff,
        )

    def forward(self, batch) -> dict[str, Tensor]:
        """Переопределённый forward: вставляет FiLM между двумя блоками PaiNN."""
        h = self.atom_embed(batch.x.float())  # (N, hidden)

        # Первый блок PaiNN
        h_s, h_v = self.painn_pre(h, batch.pos, batch.batch)

        # FiLM модуляция скалярных признаков TDA-фичами
        tda = batch.tda  # (B, tda_dim)
        h_s = self.film(h_s, tda, batch.batch)

        # Второй блок PaiNN
        h_s, h_v = self.painn_post(h_s, h_v, batch.pos, batch.batch)

        # Pooling
        mol_emb = scatter(h_s, batch.batch, dim=0, reduce="sum")

        out = {}
        if self.predict_mu:
            out["mu"] = self.mu_head(mol_emb)
        if self.predict_alpha:
            out["alpha"] = self.alpha_head(mol_emb)
        if self.predict_gap:
            out["gap"] = self.gap_head(mol_emb)
        return out


def build_painn_tda(
    tda_dim: int = 52,
    predict_mu: bool = True,
    predict_alpha: bool = True,
    predict_gap: bool = True,
    **kwargs,
) -> PaiNNTDA:
    return PaiNNTDA(
        tda_dim=tda_dim,
        predict_mu=predict_mu,
        predict_alpha=predict_alpha,
        predict_gap=predict_gap,
        **kwargs,
    )
