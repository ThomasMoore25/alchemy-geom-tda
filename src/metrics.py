"""Метрики для скалярных свойств Alchemy: mu, alpha, gap."""
import torch
import torch.nn.functional as F
from torch import Tensor


def mae(pred: Tensor, target: Tensor) -> Tensor:
    """Mean Absolute Error."""
    return (pred - target).abs().mean()


def rmse(pred: Tensor, target: Tensor) -> Tensor:
    return ((pred - target) ** 2).mean().sqrt()


def mu_mae(pred_mu: Tensor, target_mu: Tensor) -> Tensor:
    """MAE для дипольного момента (норма вектора)."""
    return (pred_mu - target_mu).abs().mean()


def alpha_mae(pred_alpha: Tensor, target_alpha: Tensor) -> Tensor:
    """MAE для изотропной поляризуемости."""
    return (pred_alpha - target_alpha).abs().mean()


def gap_mae(pred_gap: Tensor, target_gap: Tensor) -> Tensor:
    """MAE для HOMO-LUMO gap."""
    return (pred_gap - target_gap).abs().mean()


METRIC_REGISTRY = {
    "mae": mae,
    "rmse": rmse,
    "mu_mae": mu_mae,
    "alpha_mae": alpha_mae,
    "gap_mae": gap_mae,
}
