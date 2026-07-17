"""Тесты для src.metrics."""
import pytest
import torch

from metrics import mae, rmse, mu_mae, alpha_mae, gap_mae, METRIC_REGISTRY


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


def test_mu_alpha_gap_mae_are_mae():
    """mu_mae, alpha_mae, gap_mae — просто алиасы для mae."""
    pred = torch.tensor([1.0, 2.0])
    target = torch.tensor([1.5, 2.5])
    assert mu_mae(pred, target).item() == mae(pred, target).item()
    assert alpha_mae(pred, target).item() == mae(pred, target).item()
    assert gap_mae(pred, target).item() == mae(pred, target).item()


def test_metric_registry():
    """METRIC_REGISTRY содержит все 5 метрик."""
    assert 'mae' in METRIC_REGISTRY
    assert 'rmse' in METRIC_REGISTRY
    assert 'mu_mae' in METRIC_REGISTRY
    assert 'alpha_mae' in METRIC_REGISTRY
    assert 'gap_mae' in METRIC_REGISTRY
    assert METRIC_REGISTRY['mae'] is mae
