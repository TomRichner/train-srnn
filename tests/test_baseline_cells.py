"""Baseline cells honour the RNNCell contract and their solver options."""
import pytest
import torch

from train_srnn.models.ctrnn_cell import CTGRUCell, CTGRUConfig, CTRNNCell, CTRNNConfig, NODECell
from train_srnn.models.lstm_cell import LSTMCell
from train_srnn.models.ltc_cell import LTCCell, LTCConfig
from train_srnn.models.ode import euler_step, rk4_step

B, I, N = 3, 5, 8
MASK = (torch.arange(N) % 2).float()


def _cells():
    yield "lstm", LSTMCell(I, N)
    for solver in ("semi_implicit", "explicit", "rk4"):
        yield f"ltc-{solver}", LTCCell(I, LTCConfig(num_units=N, solver=solver), W_in_mask=MASK)
    for solver in ("euler", "rk4"):
        yield f"ctrnn-{solver}", CTRNNCell(I, CTRNNConfig(num_units=N, solver=solver), W_in_mask=MASK)
    yield "ctrnn-no-feedback", CTRNNCell(I, CTRNNConfig(num_units=N, global_feedback=False), W_in_mask=MASK)
    yield "node", NODECell(I, CTRNNConfig(num_units=N), W_in_mask=MASK)
    yield "ctgru", CTGRUCell(I, CTGRUConfig(num_units=N, M=3), W_in_mask=MASK)


@pytest.mark.parametrize("name, cell", list(_cells()), ids=lambda c: c if isinstance(c, str) else "")
def test_contract(name, cell):
    torch.manual_seed(0)
    assert cell.K is None and cell.input_size == I and cell.num_units == N
    state = cell.init_state(B)
    assert state.shape == (B, cell.state_size)
    out, new_state = cell(torch.randn(B, I), state)
    assert out.shape == (B, N) and new_state.shape == state.shape
    assert cell.hoist() is None and cell.skip_mask() is None
    cell.constrain_parameters()


def test_node_is_leak_free_rk4():
    cell = NODECell(I, CTRNNConfig(num_units=N, solver="euler", global_feedback=False))
    assert cell.config.solver == "rk4" and cell.config.global_feedback


def test_bad_solver_names_raise():
    with pytest.raises(ValueError):
        LTCCell(I, LTCConfig(num_units=N, solver="heun"))
    with pytest.raises(ValueError):
        CTRNNCell(I, CTRNNConfig(num_units=N, solver="heun"))


def test_ltc_constraints_clamp():
    cell = LTCCell(I, LTCConfig(num_units=N))
    with torch.no_grad():
        cell.W.fill_(-1.0)
    cell.constrain_parameters()
    assert (cell.W >= 1e-5).all()


def test_integrators_on_tuples():
    f = lambda y: (y[1], -y[0])            # harmonic oscillator  # noqa: E731
    y = (torch.tensor([1.0]), torch.tensor([0.0]))
    for _ in range(100):
        y = rk4_step(f, y, 0.01)
    assert torch.allclose(y[0], torch.cos(torch.tensor(1.0)), atol=1e-6)
    ye = euler_step(f, (torch.tensor([1.0]), torch.tensor([0.0])), 0.1)
    assert torch.allclose(ye[0], torch.tensor([1.0])) and torch.allclose(ye[1], torch.tensor([-0.1]))
