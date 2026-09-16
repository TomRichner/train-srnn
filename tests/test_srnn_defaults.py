"""Open-loop dynamics at the default parameters stay bounded and adaptation does its job."""
import numpy as np
import torch

from train_srnn.models.rmt_matrix import RMTMatrix
from train_srnn.models.srnn_cell import SRNNCell, SRNNConfig
from train_srnn.utils.stimulus import step_stimulus

N, FS, T = 64, 200.0, 6.0


def _simulate(solver: str):
    torch.manual_seed(0)
    cfgs = [SRNNConfig(num_units=N, solver=solver, h=1.0 / FS, n_a_E=3, n_a_I=3, n_b_E=2, n_b_I=2),
            SRNNConfig(num_units=N, solver=solver, h=1.0 / FS, n_a_E=0, n_a_I=0, n_b_E=0, n_b_I=0)]
    rmt = RMTMatrix(n=N, f=0.5, indegree=20, seed=7)
    rmt.build()
    cell = SRNNCell(cfgs, N, [rmt.export_for_srnn(dales=True)] * 2)
    with torch.no_grad():
        cell.W_in.copy_(torch.eye(N).expand(2, N, N))
    u = step_stimulus(N, N // 2, int(T * FS), density_E=0.3, amp=1.0, seed=8)
    state = cell.init_state(1)
    r, b, a = [], [], []
    with torch.no_grad():
        for t in range(u.shape[0]):
            _, state = cell(torch.tensor(u[t]).unsqueeze(0), state)
            d = cell.get_diagnostics(state)
            r.append(d["r"][:, 0].numpy()); b.append(d["b_full"][:, 0].numpy()); a.append(d["a_E"][:, 0].numpy())
    return np.stack(r, 1), np.stack(b, 1), np.stack(a, 1), u


def test_bounded_and_adapting():
    r, b, a, u = _simulate("sra1")
    assert np.isfinite(r).all() and np.isfinite(b).all()
    assert r.min() >= 0 and r.max() <= 1 and b.min() >= 0 and b.max() <= 1
    # Variant 1 has no adaptation: its a_E stays at zero and b_full stays at one.
    assert np.all(a[1] == 0) and np.all(b[1] == 1)
    # The adapting variant depresses its stimulated synapses and builds up a_E during the stimulus.
    block = u.shape[0] // 3
    stimulated = u[block] != 0                       # neurons driven during the middle block
    assert b[0, 2 * block - 1, stimulated].mean() < 0.95
    assert a[0, 2 * block - 1].sum() > a[0, block].sum()


def test_solvers_agree():
    r_si, *_ = _simulate("sra1")
    r_rk, *_ = _simulate("rk4")
    assert np.corrcoef(r_si.ravel(), r_rk.ravel())[0, 1] > 0.95
    assert np.abs(r_si - r_rk).mean() < 0.02
