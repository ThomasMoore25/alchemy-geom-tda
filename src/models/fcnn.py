"""FCNN baseline: полносвязная сеть на агрегированных признаках молекулы.

Никаких индуктивных смещений: ни сдвигов, ни поворотов, ни перестановок.
Фичи молекулы: mean, max, sum по узловым признакам.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter


class FCNNBaseline(nn.Module):
    """Простая MLP регрессия на агрегированных признаках молекулы."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 4,
        out_dim: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        # BatchNorm на входе — критично! Без него mass~300 доминирует
        self.input_norm = nn.BatchNorm1d(in_dim)

        layers = []
        d = in_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(d, hidden_dim))
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(dropout))
            d = hidden_dim
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, batch) -> torch.Tensor:
        x = batch.x  # (N, F)
        batch_idx = batch.batch  # (N,)

        mean = scatter(x, batch_idx, dim=0, reduce="mean")
        mx = scatter(x, batch_idx, dim=0, reduce="max")
        s = scatter(x, batch_idx, dim=0, reduce="sum")
        feat = torch.cat([mean, mx, s], dim=-1)  # (B, 3F)

        # Нормализация входа
        feat = self.input_norm(feat)

        return self.net(feat)


def build_fcnn(in_dim: int, out_dim: int, **kwargs) -> FCNNBaseline:
    return FCNNBaseline(in_dim=in_dim, out_dim=out_dim, **kwargs)
