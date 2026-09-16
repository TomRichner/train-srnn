"""RMT connectivity: spectral radius, sparsity, Dale's sign structure, export round-trip."""
import numpy as np
import torch
import torch.nn.functional as F

from train_srnn.models.rmt_matrix import RMTMatrix


def _build():
    rmt = RMTMatrix(n=300, f=0.5, indegree=100, seed=42, level_of_chaos=1.0)
    return rmt, rmt.build()


def test_level_of_chaos_scales_weights():
    _, W1 = _build()
    W2 = RMTMatrix(n=300, f=0.5, indegree=100, seed=42, level_of_chaos=2.0).build()
    assert np.allclose(W2, 2.0 * W1)
    rmt, _ = _build()
    assert rmt.spectral_radius > 0


def test_sparsity_and_dales_signs():
    rmt, W = _build()
    assert abs(rmt.S.mean() - rmt.alpha) < 0.02
    n_E = rmt.n_E
    assert W[:, :n_E].mean() > 0 and W[:, n_E:].mean() < 0


def test_export_round_trip_with_dales():
    """The export stores |W| in inverse-softplus space and the column sign separately,
    so the reconstruction is Dale-compliant: sign * |W| on the sparsity support."""
    rmt, W = _build()
    ex = rmt.export_for_srnn(dales=True)
    W_rec = (F.softplus(ex["W_init"]) * ex["dales_sign"][None, :] * ex["sparsity_mask"]).numpy()
    sign = np.where(np.arange(W.shape[1]) < rmt.n_E, 1.0, -1.0)[None, :]
    assert np.abs(W_rec - sign * np.abs(W) * rmt.S).max() < 1e-5


def test_export_without_dales_keeps_mask():
    rmt, _ = _build()
    ex = rmt.export_for_srnn(dales=False, dales_init=False)
    assert np.allclose(ex["sparsity_mask"].numpy(), rmt.S.astype(np.float32))
    assert torch.equal(ex["W_init"], torch.tensor(rmt.W, dtype=torch.float32))


def test_seed_determinism():
    _, W1 = _build()
    _, W2 = _build()
    assert np.allclose(W1, W2)
