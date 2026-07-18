"""Тесты для src.physics — вектор μ и тензор α (часть B)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from torch_geometric.data import Batch, Data

from physics import (
    DipolePolarizabilityHead,
    compute_dipole_vector,
    compute_polarizability_tensor,
    polarizability_anisotropy,
    polarizability_iso,
)


def _make_batch(n_mols=3, n_atoms=8, seed=42):
    """Создать синтетический батч молекул для тестов."""
    torch.manual_seed(seed)
    ds = []
    for _ in range(n_mols):
        x = torch.zeros(n_atoms, 8)
        atom_types = torch.randint(0, 7, (n_atoms,))
        for j, t in enumerate(atom_types):
            x[j, t] = 1.0
        x[:, -1] = torch.tensor([1.0, 12.0, 14.0, 16.0, 19.0, 32.0, 35.0])[atom_types]
        pos = torch.randn(n_atoms, 3) * 2
        d = Data(x=x, pos=pos,
                 edge_index=torch.zeros(2, 0, dtype=torch.long),
                 edge_attr=torch.zeros(0, 4),
                 mu=torch.tensor([0.5]), alpha=torch.tensor([10.0]),
                 gap=torch.tensor([0.2]))
        ds.append(d)
    return Batch.from_data_list(ds)


# ============ compute_dipole_vector ============

def test_dipole_vector_shape():
    """μ имеет форму (B, 3)."""
    batch = _make_batch(n_mols=4, n_atoms=8)
    N = batch.x.shape[0]
    charges = torch.randn(N, 1)
    mass = batch.x[:, -1:]
    mu = compute_dipole_vector(charges, batch.pos, batch.batch, mass)
    assert mu.shape == (4, 3)


def test_dipole_vector_zero_charges():
    """Если все заряды = 0, то μ = 0."""
    batch = _make_batch(n_mols=3, n_atoms=8)
    N = batch.x.shape[0]
    charges = torch.zeros(N, 1)
    mass = batch.x[:, -1:]
    mu = compute_dipole_vector(charges, batch.pos, batch.batch, mass)
    assert torch.allclose(mu, torch.zeros(3, 3), atol=1e-6)


def test_dipole_translation_invariance():
    """μ не меняется при трансляции всех координат на t."""
    batch = _make_batch(n_mols=2, n_atoms=6)
    N = batch.x.shape[0]
    charges = torch.randn(N, 1)
    mass = batch.x[:, -1:]
    mu_orig = compute_dipole_vector(charges, batch.pos, batch.batch, mass)

    t = torch.tensor([5.0, -3.0, 2.0])
    pos_translated = batch.pos + t
    mu_translated = compute_dipole_vector(charges, pos_translated, batch.batch, mass)

    err = (mu_orig - mu_translated).abs().max().item()
    assert err < 1e-5, f"Translation invariance failed: {err}"


def test_dipole_rotation_equivariance():
    """μ поворачивается вместе с координатами: μ(R·X) = R·μ(X)."""
    from scipy.spatial.transform import Rotation

    batch = _make_batch(n_mols=3, n_atoms=8)
    N = batch.x.shape[0]
    charges = torch.randn(N, 1)
    mass = batch.x[:, -1:]
    mu_orig = compute_dipole_vector(charges, batch.pos, batch.batch, mass)

    R = torch.tensor(Rotation.random().as_matrix(), dtype=torch.float32)
    pos_rotated = batch.pos @ R.T
    mu_rotated = compute_dipole_vector(charges, pos_rotated, batch.batch, mass)

    # Ожидаем: mu_rotated = mu_orig @ R^T
    mu_expected = mu_orig @ R.T
    err = (mu_rotated - mu_expected).abs().max().item()
    assert err < 1e-4, f"Rotation equivariance failed: {err}"


def test_dipole_permutation_invariance():
    """μ не меняется при перестановке атомов внутри молекулы."""
    torch.manual_seed(0)
    n_atoms = 8
    x = torch.zeros(n_atoms, 8)
    atom_types = torch.randint(0, 7, (n_atoms,))
    for j, t in enumerate(atom_types):
        x[j, t] = 1.0
    x[:, -1] = torch.tensor([1.0, 12.0, 14.0, 16.0, 19.0, 32.0, 35.0])[atom_types]
    pos = torch.randn(n_atoms, 3) * 2
    charges = torch.randn(n_atoms, 1)
    mass = x[:, -1:]
    batch_idx = torch.zeros(n_atoms, dtype=torch.long)

    mu_orig = compute_dipole_vector(charges, pos, batch_idx, mass)

    perm = torch.randperm(n_atoms)
    mu_perm = compute_dipole_vector(charges[perm], pos[perm], batch_idx, mass[perm])

    err = (mu_orig - mu_perm).abs().max().item()
    assert err < 1e-5, f"Permutation invariance failed: {err}"


# ============ compute_polarizability_tensor ============

def test_polarizability_tensor_shape():
    """α имеет форму (B, 3, 3)."""
    batch = _make_batch(n_mols=4, n_atoms=8)
    N = batch.x.shape[0]
    charges = torch.randn(N, 1)
    mass = batch.x[:, -1:]
    alpha = compute_polarizability_tensor(charges, batch.pos, batch.batch, mass)
    assert alpha.shape == (4, 3, 3)


def test_polarizability_tensor_symmetric():
    """α симметрична: α_ij = α_ji."""
    batch = _make_batch(n_mols=3, n_atoms=8)
    N = batch.x.shape[0]
    charges = torch.randn(N, 1)
    mass = batch.x[:, -1:]
    alpha = compute_polarizability_tensor(charges, batch.pos, batch.batch, mass)

    asym = (alpha - alpha.transpose(-1, -2)).abs().max().item()
    assert asym < 1e-5, f"α not symmetric: {asym}"


def test_polarizability_zero_charges():
    """Если все заряды = 0, то α = 0."""
    batch = _make_batch(n_mols=2, n_atoms=6)
    N = batch.x.shape[0]
    charges = torch.zeros(N, 1)
    mass = batch.x[:, -1:]
    alpha = compute_polarizability_tensor(charges, batch.pos, batch.batch, mass)
    assert torch.allclose(alpha, torch.zeros(2, 3, 3), atol=1e-6)


def test_polarizability_translation_invariance():
    """α не меняется при трансляции координат."""
    batch = _make_batch(n_mols=2, n_atoms=6)
    N = batch.x.shape[0]
    charges = torch.randn(N, 1)
    mass = batch.x[:, -1:]
    alpha_orig = compute_polarizability_tensor(charges, batch.pos, batch.batch, mass)

    t = torch.tensor([5.0, -3.0, 2.0])
    pos_translated = batch.pos + t
    alpha_translated = compute_polarizability_tensor(charges, pos_translated, batch.batch, mass)

    err = (alpha_orig - alpha_translated).abs().max().item()
    assert err < 1e-5


def test_polarizability_rotation_equivariance():
    """α преобразуется как R·α·R^T при повороте координат."""
    from scipy.spatial.transform import Rotation

    batch = _make_batch(n_mols=3, n_atoms=8)
    N = batch.x.shape[0]
    charges = torch.randn(N, 1)
    mass = batch.x[:, -1:]
    alpha_orig = compute_polarizability_tensor(charges, batch.pos, batch.batch, mass)

    R = torch.tensor(Rotation.random().as_matrix(), dtype=torch.float32)
    pos_rotated = batch.pos @ R.T
    alpha_rotated = compute_polarizability_tensor(charges, pos_rotated, batch.batch, mass)

    # Ожидаем: alpha_rotated = R @ alpha_orig @ R^T
    alpha_expected = R @ alpha_orig @ R.T
    err = (alpha_rotated - alpha_expected).abs().max().item()
    assert err < 1e-4


# ============ polarizability_iso / anisotropy ============

def test_polarizability_iso():
    """α_iso = tr(α) / 3."""
    batch = _make_batch(n_mols=2, n_atoms=6)
    N = batch.x.shape[0]
    charges = torch.randn(N, 1)
    mass = batch.x[:, -1:]
    alpha = compute_polarizability_tensor(charges, batch.pos, batch.batch, mass)
    alpha_iso = polarizability_iso(alpha)

    assert alpha_iso.shape == (2, 1)
    expected = torch.diagonal(alpha, dim1=-2, dim2=-1).sum(dim=-1, keepdim=True) / 3.0
    assert torch.allclose(alpha_iso, expected, atol=1e-6)


def test_polarizability_anisotropy():
    """Анизотропия вычисляется без NaN/Inf."""
    batch = _make_batch(n_mols=2, n_atoms=6)
    N = batch.x.shape[0]
    # Положительные заряды → α_iso > 0 → анизотропия >= 0
    charges = torch.rand(N, 1) + 0.1
    mass = batch.x[:, -1:]
    alpha = compute_polarizability_tensor(charges, batch.pos, batch.batch, mass)
    aniso = polarizability_anisotropy(alpha)

    assert aniso.shape == (2, 1)
    assert torch.isfinite(aniso).all()
    # При положительных зарядах α_iso > 0 → анизотропия >= 0
    assert (aniso >= 0).all()


def test_polarizability_anisotropy_spherical():
    """Для сферически симметричной α (α = c·I) анизотропия = 0."""
    # Создаём α = 5 * I (сферически симметричная)
    alpha = 5.0 * torch.eye(3).unsqueeze(0).expand(3, -1, -1)  # (3, 3, 3)
    aniso = polarizability_anisotropy(alpha)
    assert torch.allclose(aniso, torch.zeros(3, 1), atol=1e-5)


# ============ DipolePolarizabilityHead ============

def test_dipole_head_shape():
    """Head возвращает заряды (N, 1)."""
    head = DipolePolarizabilityHead(hidden_channels=32)
    h = torch.randn(20, 32)
    q = head(h)
    assert q.shape == (20, 1)


def test_dipole_head_zero_init():
    """С zero init начальные заряды ≈ 0 (электронейтральная молекула)."""
    head = DipolePolarizabilityHead(hidden_channels=32)
    h = torch.randn(20, 32)
    q = head(h)
    assert q.abs().max().item() < 1e-5, f"Initial charges not zero: {q.abs().max()}"


def test_dipole_head_backward():
    """Backward через head работает."""
    head = DipolePolarizabilityHead(hidden_channels=32)
    h = torch.randn(20, 32, requires_grad=True)
    q = head(h)
    loss = q.sum()
    loss.backward()
    assert h.grad is not None
    assert torch.isfinite(h.grad).all()


# ============ EGNNTensorModel ============

def test_egnn_tensor_forward_shape():
    """EGNNTensorModel возвращает μ (B,3), α_tensor (B,3,3), α (B,1), gap (B,1)."""
    from models.egnn_tensor import EGNNTensorModel

    batch = _make_batch(n_mols=4, n_atoms=8)
    model = EGNNTensorModel(
        hidden_channels=32, num_layers=2,
        predict_alpha_tensor=True, predict_gap=True,
    )
    out = model(batch)
    assert "mu" in out
    assert "alpha_tensor" in out
    assert "alpha" in out
    assert "gap" in out
    assert out["mu"].shape == (4, 3)
    assert out["alpha_tensor"].shape == (4, 3, 3)
    assert out["alpha"].shape == (4, 1)
    assert out["gap"].shape == (4, 1)


def test_egnn_tensor_alpha_symmetric():
    """α_tensor симметричен."""
    from models.egnn_tensor import EGNNTensorModel

    batch = _make_batch(n_mols=3, n_atoms=6)
    model = EGNNTensorModel(hidden_channels=32, num_layers=2, predict_alpha_tensor=True)
    out = model(batch)
    asym = (out["alpha_tensor"] - out["alpha_tensor"].transpose(-1, -2)).abs().max().item()
    assert asym < 1e-5


def test_egnn_tensor_backward():
    """Backward через всю модель работает."""
    from models.egnn_tensor import EGNNTensorModel

    batch = _make_batch(n_mols=3, n_atoms=6)
    model = EGNNTensorModel(hidden_channels=32, num_layers=2)
    out = model(batch)

    # loss = |μ| + α_iso + gap + regularization
    loss = out["mu"].norm(dim=-1).sum() + out["alpha"].sum() + out["gap"].sum()
    sym_reg = (out["alpha_tensor"] - out["alpha_tensor"].transpose(-1, -2)).pow(2).mean()
    loss = loss + 0.01 * sym_reg
    loss.backward()

    for name, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"NaN gradient in {name}"


def test_egnn_tensor_without_alpha():
    """Модель с predict_alpha_tensor=False не возвращает α."""
    from models.egnn_tensor import EGNNTensorModel

    batch = _make_batch(n_mols=2, n_atoms=6)
    model = EGNNTensorModel(
        hidden_channels=32, num_layers=2,
        predict_alpha_tensor=False, predict_gap=True,
    )
    out = model(batch)
    assert "mu" in out
    assert "alpha_tensor" not in out
    assert "alpha" not in out
    assert "gap" in out


def test_egnn_tensor_alpha_iso_matches_trace():
    """α (скаляр) = tr(α_tensor) / 3."""
    from models.egnn_tensor import EGNNTensorModel

    batch = _make_batch(n_mols=3, n_atoms=6)
    model = EGNNTensorModel(hidden_channels=32, num_layers=2)
    out = model(batch)

    expected = torch.diagonal(out["alpha_tensor"], dim1=-2, dim2=-1).sum(dim=-1, keepdim=True) / 3.0
    err = (out["alpha"] - expected).abs().max().item()
    assert err < 1e-5
