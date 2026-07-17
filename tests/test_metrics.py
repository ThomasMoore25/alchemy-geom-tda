"""Тесты для src.metrics."""
import torch

from metrics import METRIC_REGISTRY, alpha_mae, gap_mae, mae, mu_mae, r2_score, rmse


def test_mae_basic():
    pred = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([1.0, 2.0, 4.0])
    assert abs(mae(pred, target).item() - 1.0 / 3.0) < 1e-6


def test_mae_zero():
    pred = torch.tensor([1.0, 2.0, 3.0])
    assert mae(pred, pred).item() == 0.0


def test_rmse():
    pred = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([1.0, 2.0, 5.0])
    # errors: 0, 0, -2 -> MSE = (0+0+4)/3 = 4/3, RMSE = 2/sqrt(3)
    expected = (4.0 / 3.0) ** 0.5
    assert abs(rmse(pred, target).item() - expected) < 1e-6


def test_r2_perfect_prediction():
    """R² = 1 для идеального предсказания."""
    pred = torch.tensor([1.0, 2.0, 3.0, 4.0])
    target = torch.tensor([1.0, 2.0, 3.0, 4.0])
    r2 = r2_score(pred, target)
    assert abs(r2.item() - 1.0) < 1e-6


def test_r2_predicting_mean():
    """R² = 0 если предсказываем константно среднее таргета."""
    target = torch.tensor([1.0, 2.0, 3.0, 4.0])
    pred = torch.full_like(target, target.mean().item())
    r2 = r2_score(pred, target)
    assert abs(r2.item() - 0.0) < 1e-6


def test_r2_worse_than_mean_is_negative():
    """R² < 0 для предсказания хуже, чем предсказание среднего."""
    target = torch.tensor([1.0, 2.0, 3.0, 4.0])
    pred = torch.tensor([10.0, 20.0, 30.0, 40.0])  # way off
    r2 = r2_score(pred, target)
    assert r2.item() < 0


def test_r2_zero_variance_target_returns_nan():
    """Если target константный (нулевая дисперсия), R² = NaN."""
    pred = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([5.0, 5.0, 5.0])  # constant
    r2 = r2_score(pred, target)
    assert torch.isnan(r2).item()


def test_mu_alpha_gap_mae_are_mae():
    """mu_mae, alpha_mae, gap_mae — просто алиасы для mae."""
    pred = torch.tensor([1.0, 2.0])
    target = torch.tensor([1.5, 2.5])
    assert mu_mae(pred, target).item() == mae(pred, target).item()
    assert alpha_mae(pred, target).item() == mae(pred, target).item()
    assert gap_mae(pred, target).item() == mae(pred, target).item()


def test_metric_registry():
    """METRIC_REGISTRY содержит все 6 метрик."""
    assert 'mae' in METRIC_REGISTRY
    assert 'rmse' in METRIC_REGISTRY
    assert 'r2' in METRIC_REGISTRY
    assert 'mu_mae' in METRIC_REGISTRY
    assert 'alpha_mae' in METRIC_REGISTRY
    assert 'gap_mae' in METRIC_REGISTRY
    assert METRIC_REGISTRY['mae'] is mae
    assert METRIC_REGISTRY['r2'] is r2_score
