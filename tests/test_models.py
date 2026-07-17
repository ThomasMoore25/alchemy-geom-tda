"""Тесты для моделей: forward pass, размерности выходов."""
import pytest
import torch
from torch_geometric.data import Batch, Data

from models.egnn import EGNNModel
from models.egnn_tda import EGNNTDA
from models.egnn_vector import EGNNVectorModel
from models.egnn_vector_tda import EGNNVectorTDA
from models.fcnn import FCNNBaseline
from models.schnet import SchNetWrapper


def make_batch(n_mols: int = 4, with_tda: bool = False) -> Batch:
    """Создать синтетический batch молекул для тестов."""
    torch.manual_seed(42)
    data_list = []
    for i in range(n_mols):
        n = 8  # 8 атомов
        x = torch.zeros(n, 8)
        atom_types = torch.randint(0, 7, (n,))
        for j, t in enumerate(atom_types):
            x[j, t] = 1.0
        x[:, -1] = torch.tensor([1.0, 12.0, 14.0, 16.0, 19.0, 32.0, 35.0])[atom_types]
        pos = torch.randn(n, 3) * 2
        d = Data(
            x=x, pos=pos,
            edge_index=torch.zeros(2, 0, dtype=torch.long),
            edge_attr=torch.zeros(0, 4),
            mu=torch.tensor([0.5 + i * 0.1]),
            alpha=torch.tensor([10.0 + i]),
            gap=torch.tensor([0.2]),
        )
        data_list.append(d)
    batch = Batch.from_data_list(data_list)
    if with_tda:
        batch.tda = torch.randn(n_mols, 52) * 10  # realistic scale
    return batch


# ============ FCNN ============

def test_fcnn_forward_shape():
    """FCNN возвращает тензор (B, out_dim)."""
    batch = make_batch(n_mols=4)
    model = FCNNBaseline(in_dim=8 * 3, hidden_dim=32, n_layers=2, out_dim=3)
    out = model(batch)
    assert out.shape == (4, 3)


def test_fcnn_backward():
    """FCNN backprop работает."""
    batch = make_batch(n_mols=4)
    model = FCNNBaseline(in_dim=8 * 3, hidden_dim=32, n_layers=2, out_dim=3)
    out = model(batch)
    loss = out.sum()
    loss.backward()
    for p in model.parameters():
        assert p.grad is not None


# ============ SchNet ============

def test_schnet_forward_shape():
    """SchNet возвращает тензор (B, out_dim)."""
    batch = make_batch(n_mols=4)
    model = SchNetWrapper(
        hidden_channels=32, num_filters=32, num_interactions=2,
        num_gaussians=16, cutoff=5.0, out_dim=3,
    )
    out = model(batch)
    assert out.shape == (4, 3)


# ============ EGNN ============

def test_egnn_forward_shape():
    """EGNN возвращает dict с mu/alpha/gap shape (B, 1)."""
    batch = make_batch(n_mols=4)
    model = EGNNModel(hidden_channels=32, num_layers=2)
    out = model(batch)
    assert isinstance(out, dict)
    assert out["mu"].shape == (4, 1)
    assert out["alpha"].shape == (4, 1)
    assert out["gap"].shape == (4, 1)


def test_egnn_backward():
    """EGNN backprop работает."""
    batch = make_batch(n_mols=4)
    model = EGNNModel(hidden_channels=32, num_layers=2)
    out = model(batch)
    loss = sum(v.sum() for v in out.values())
    loss.backward()
    for p in model.parameters():
        assert p.grad is not None


def test_egnn_custom_params():
    """EGNN с кастомными cutoff, k_neighbors, m_dim работает."""
    batch = make_batch(n_mols=4)
    model = EGNNModel(
        hidden_channels=32, num_layers=2,
        cutoff=3.0, k_neighbors=8, m_dim=16,
    )
    assert model.cutoff == 3.0
    assert model.k_neighbors == 8
    assert model.m_dim == 16
    out = model(batch)
    assert out["mu"].shape == (4, 1)


# ============ EGNN+TDA ============

def test_egnn_tda_concat_forward():
    """EGNN+TDA concat режим: mu/alpha/gap shape (B, 1)."""
    batch = make_batch(n_mols=4, with_tda=True)
    model = EGNNTDA(hidden_channels=32, num_layers=2, tda_dim=52, tda_mode="concat")
    assert model.tda_mode == "concat"
    assert hasattr(model, "tda_norm")
    out = model(batch)
    assert out["mu"].shape == (4, 1)


def test_egnn_tda_film_forward():
    """EGNN+TDA film режим: mu/alpha/gap shape (B, 1)."""
    batch = make_batch(n_mols=4, with_tda=True)
    model = EGNNTDA(hidden_channels=32, num_layers=2, tda_dim=52, tda_mode="film")
    assert model.tda_mode == "film"
    assert model.film is not None
    out = model(batch)
    assert out["mu"].shape == (4, 1)


def test_egnn_tda_batchnorm_normalizes():
    """BatchNorm1d(tda_dim) действительно нормирует TDA-фичи."""
    batch = make_batch(n_mols=4, with_tda=True)
    model = EGNNTDA(hidden_channels=32, num_layers=2, tda_dim=52)
    model.train()
    for _ in range(10):
        _ = model(batch)
    model.eval()
    normalized = model.tda_norm(batch.tda)
    # После BatchNorm: mean ~ 0, std ~ 1
    assert abs(normalized.mean().item()) < 1.0  # rough
    assert normalized.std().item() < 3.0  # rough


def test_egnn_tda_no_tda_raises():
    """EGNN+TDA без batch.tda: ValueError с понятным сообщением."""
    batch = make_batch(n_mols=4, with_tda=False)
    model = EGNNTDA(hidden_channels=32, num_layers=2, tda_dim=52)
    with pytest.raises(ValueError, match="TDA"):
        model(batch)


def test_egnn_tda_invalid_mode():
    """Неверный tda_mode: AssertionError при construction."""
    with pytest.raises(AssertionError):
        EGNNTDA(hidden_channels=32, num_layers=2, tda_dim=52, tda_mode="invalid")


# ============ EGNN Vector ============

def test_egnn_vector_forward_shape():
    """EGNN Vector возвращает mu shape (B, 3) — вектор!"""
    batch = make_batch(n_mols=4)
    model = EGNNVectorModel(hidden_channels=32, num_layers=2)
    out = model(batch)
    assert out["mu"].shape == (4, 3)  # vector
    assert out["alpha"].shape == (4, 1)  # scalar
    assert out["gap"].shape == (4, 1)  # scalar


def test_egnn_vector_backward():
    """EGNN Vector backprop работает без NaN."""
    batch = make_batch(n_mols=4)
    model = EGNNVectorModel(hidden_channels=32, num_layers=2)
    out = model(batch)
    # loss через mu.norm() — как в compute_loss
    mu_norm = out["mu"].norm(dim=-1, keepdim=True).clamp(min=1e-4)
    loss = mu_norm.sum() + out["alpha"].sum() + out["gap"].sum()
    loss.backward()
    for p in model.parameters():
        assert p.grad is not None
        assert torch.isfinite(p.grad).all(), "NaN in gradient"


# ============ EGNN Vector+TDA ============

def test_egnn_vector_tda_concat_forward():
    """EGNN Vector+TDA concat: mu (B,3), alpha/gap (B,1)."""
    batch = make_batch(n_mols=4, with_tda=True)
    model = EGNNVectorTDA(hidden_channels=32, num_layers=2, tda_dim=52, tda_mode="concat")
    out = model(batch)
    assert out["mu"].shape == (4, 3)
    assert out["alpha"].shape == (4, 1)
    assert out["gap"].shape == (4, 1)


def test_egnn_vector_tda_film_forward():
    """EGNN Vector+TDA film: mu (B,3), alpha/gap (B,1)."""
    batch = make_batch(n_mols=4, with_tda=True)
    model = EGNNVectorTDA(hidden_channels=32, num_layers=2, tda_dim=52, tda_mode="film")
    out = model(batch)
    assert out["mu"].shape == (4, 3)
