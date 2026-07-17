"""Метрики для скалярных свойств Alchemy: mu, alpha, gap.

v32+: добавлены r2_score и rmse используется в train.py (раньше был объявлен, но не вызывался).
"""
import torch
import torch.nn.functional as F
from torch import Tensor


def mae(pred: Tensor, target: Tensor) -> Tensor:
    """Mean Absolute Error."""
    return (pred - target).abs().mean()


def rmse(pred: Tensor, target: Tensor) -> Tensor:
    """Root Mean Squared Error."""
    return ((pred - target) ** 2).mean().sqrt()


def r2_score(pred: Tensor, target: Tensor) -> Tensor:
    """R² (coefficient of determination).

    R² = 1 - SS_res / SS_tot
    SS_res = sum((pred - target)^2)
    SS_tot = sum((target - mean(target))^2)

    R² = 1: ideal prediction
    R² = 0: predicting mean
    R² < 0: worse than predicting mean
    """
    target_mean = target.mean()
    ss_res = ((pred - target) ** 2).sum()
    ss_tot = ((target - target_mean) ** 2).sum()
    if ss_tot.item() == 0:
        return torch.tensor(float('nan'), device=pred.device)
    return 1 - ss_res / ss_tot


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
    "r2": r2_score,
    "mu_mae": mu_mae,
    "alpha_mae": alpha_mae,
    "gap_mae": gap_mae,
}
