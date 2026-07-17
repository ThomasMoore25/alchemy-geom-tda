"""Тесты для src.train — parse_args, build_model, compute_loss, compute_metrics."""
import sys

import pytest
import torch
from torch_geometric.data import Batch, Data

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent / "src"))

from train import (
    _load_yaml_config,
    _unpack_preds,
    compute_loss,
    compute_metrics,
    parse_args,
)


def make_batch(n_mols: int = 4) -> Batch:
    torch.manual_seed(42)
    data_list = []
    for i in range(n_mols):
        n = 8
        x = torch.zeros(n, 8)
        atom_types = torch.randint(0, 7, (n,))
        for j, t in enumerate(atom_types):
            x[j, t] = 1.0
        x[:, -1] = torch.tensor([1.0, 12.0, 14.0, 16.0, 19.0, 32.0, 35.0])[atom_types]
        pos = torch.randn(n, 3) * 2
        d = Data(x=x, pos=pos,
                 mu=torch.tensor([0.5 + i * 0.1]),
                 alpha=torch.tensor([10.0 + i]),
                 gap=torch.tensor([0.2]))
        data_list.append(d)
    return Batch.from_data_list(data_list)


# ============ parse_args ============

def test_parse_args_defaults():
    sys.argv = ['train.py', '--model', 'egnn']
    args = parse_args()
    assert args.model == 'egnn'
    assert args.lr == 1e-3  # canonical default
    assert args.epochs == 9999
    assert args.num_layers == 4
    assert args.batch_size == 1024
    assert args.patience == 15
    assert args.lr_patience == 5
    assert args.k_neighbors == 16
    assert args.m_dim == 32
    assert args.cutoff == 5.0
    assert args.tda_mode == 'concat'
    assert args.noise_mode == 'test_only'
    assert args.es_mode == 'and'


def test_parse_args_config_yaml_overrides_defaults():
    """--config задаёт дефолты из YAML, CLI может переопределить."""
    sys.argv = ['train.py', '--config', 'configs/default.yaml']
    args = parse_args()
    assert args.model == 'egnn'  # from default.yaml
    assert args.lr == 0.001  # from default.yaml
    assert args.epochs == 9999  # from default.yaml


def test_parse_args_cli_overrides_yaml():
    sys.argv = ['train.py', '--config', 'configs/default.yaml', '--lr', '5e-4']
    args = parse_args()
    assert abs(args.lr - 5e-4) < 1e-10  # CLI wins over YAML


def test_parse_args_missing_model_errors():
    sys.argv = ['train.py', '--lr', '1e-3']
    with pytest.raises(SystemExit):
        parse_args()


def test_parse_args_invalid_choice():
    sys.argv = ['train.py', '--model', 'invalid_model']
    with pytest.raises(SystemExit):
        parse_args()


def test_load_yaml_config():
    cfg = _load_yaml_config('configs/default.yaml')
    assert cfg['model'] == 'egnn'
    assert cfg['lr'] == 0.001
    assert cfg['epochs'] == 9999
    assert cfg['num_layers'] == 4
    assert cfg['batch_size'] == 1024


# ============ compute_loss ============

def test_compute_loss_scalar_pred():
    """Скалярный pred (B,1): обычный MAE loss."""
    batch = make_batch(n_mols=4)
    preds = {
        'mu':    torch.zeros(4, 1),
        'alpha': torch.zeros(4, 1),
        'gap':   torch.zeros(4, 1),
    }
    target_stats = {'mu': (1.0, 0.5), 'alpha': (10.0, 5.0), 'gap': (0.2, 0.05)}
    loss = compute_loss(preds, batch, target='all', target_stats=target_stats)
    assert torch.isfinite(loss)
    assert loss > 0


def test_compute_loss_vector_pred_no_nan_at_zero():
    """Векторный pred = 0: loss конечный (не NaN) благодаря clamp."""
    batch = make_batch(n_mols=4)
    preds = {
        'mu':    torch.zeros(4, 3, requires_grad=True),  # vector, exactly zero
        'alpha': torch.zeros(4, 1, requires_grad=True),
        'gap':   torch.zeros(4, 1, requires_grad=True),
    }
    target_stats = {'mu': (1.0, 0.5), 'alpha': (10.0, 5.0), 'gap': (0.2, 0.05)}
    loss = compute_loss(preds, batch, target='all', target_stats=target_stats)
    assert torch.isfinite(loss), "Loss should be finite at mu_pred=0"
    loss.backward()
    assert torch.isfinite(preds['mu'].grad).all(), "Gradient should be finite"


def test_compute_loss_target_single():
    """target='mu' — только mu loss."""
    batch = make_batch(n_mols=4)
    preds = {
        'mu':    torch.zeros(4, 1),
        'alpha': torch.zeros(4, 1),
        'gap':   torch.zeros(4, 1),
    }
    loss_mu = compute_loss(preds, batch, target='mu')
    loss_all = compute_loss(preds, batch, target='all')
    # loss_mu должен быть меньше loss_all (только mu компонент)
    assert loss_mu < loss_all


# ============ compute_metrics ============

def test_compute_metrics_scalar_pred():
    """Скалярный pred: метрики в физических единицах."""
    batch = make_batch(n_mols=4)
    preds = {
        'mu':    torch.zeros(4, 1),
        'alpha': torch.zeros(4, 1),
        'gap':   torch.zeros(4, 1),
    }
    target_stats = {'mu': (1.0, 0.5), 'alpha': (10.0, 5.0), 'gap': (0.2, 0.05)}
    metrics = compute_metrics(preds, batch, target='all', target_stats=target_stats)
    assert 'mu_mae' in metrics
    assert 'alpha_mae' in metrics
    assert 'gap_mae' in metrics


def test_compute_metrics_vector_pred():
    """Векторный pred: метрика берётся от нормы."""
    batch = make_batch(n_mols=4)
    preds = {'mu': torch.zeros(4, 3)}  # vector zero
    target_stats = {'mu': (1.0, 0.5)}
    metrics = compute_metrics(preds, batch, target='mu', target_stats=target_stats)
    assert 'mu_mae' in metrics
    # pred=0, denorm: 0*0.5 + 1.0 = 1.0; target=0.5+i*0.1; mae = mean(|1.0 - targets|)
    # targets = [0.5, 0.6, 0.7, 0.8], mae = mean([0.5, 0.4, 0.3, 0.2]) = 0.35
    assert abs(metrics['mu_mae'] - 0.35) < 1e-6


def test_compute_metrics_as_item_false_returns_tensor():
    """as_item=False: возвращает tensor, не float."""
    batch = make_batch(n_mols=4)
    preds = {'mu': torch.zeros(4, 1)}
    metrics = compute_metrics(preds, batch, target='mu', as_item=False)
    assert isinstance(metrics['mu_mae'], torch.Tensor)


def test_unpack_preds_dict_passthrough():
    """Dict предсказания проходят как есть."""
    preds = {'mu': torch.zeros(4, 1)}
    assert _unpack_preds(preds, 'all') is preds


def test_unpack_preds_tensor_all_target():
    """Tensor pred (B,3) при target='all' -> dict из 3 колонок."""
    preds = torch.zeros(4, 3)
    out = _unpack_preds(preds, 'all')
    assert 'mu' in out and 'alpha' in out and 'gap' in out
    assert out['mu'].shape == (4, 1)
