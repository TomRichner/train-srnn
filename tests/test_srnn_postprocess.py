"""Version-2 parameter reporting and replay must respect timescale padding."""
import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import postprocess as post
import plot_srnn_timeseries as replay
import simulate_srnn as sim
from train_srnn.config import compose_config
from train_srnn.models.factory import build_model


def _model(per_neuron=False):
    suffix = '-per-neuron' if per_neuron else ''
    names = [name + suffix for name in sim.DEFAULT_VARIANTS]
    cfg = compose_config(['model=srnn', 'task=synthetic', 'model.num_units=7',
                          f'model.variants=[{",".join(names)}]'])
    return build_model(cfg), names


@pytest.mark.parametrize('per_neuron', [False, True])
def test_v2_effective_parameters_and_all_report_panels(tmp_path, per_neuron):
    model, names = _model(per_neuron)
    initial = copy.deepcopy(model.state_dict())
    with torch.no_grad():
        model.cell.log_c_E_gain.add_(.2)
        model.cell.log_tau_b_rec_E_gain.add_(.1)
    final = copy.deepcopy(model.state_dict())
    snaps = [('init', 0, initial), ('last', 1, final)]
    for k, count in enumerate((0, 1, 2)):
        assert post.active_js(final, k, 'E', 'std') == list(range(count))
        params = model.cell.effective_params()
        for key, value in post.effective_taus(final, k).items():
            np.testing.assert_allclose(value, params[key][k].numpy(), rtol=1e-6)
        np.testing.assert_allclose(post.effective_c(final, k, 'E'), params['c_E'][k].numpy(), rtol=1e-6)
        np.testing.assert_allclose(post.effective_W(final, k)[0], params['W'][k].numpy(), rtol=1e-6)
    post.run_per_variant(tmp_path, snaps, names, None)
    for k, name in enumerate(names):
        out = tmp_path / name
        assert all((out / filename).stat().st_size > 0 for filename in
                   ['tau_evolution.png', 'W_EI_evolution.png', 'W_io_evolution.png',
                    'offsets_evolution.png', 'param_table.txt'])
        table = (out / 'param_table.txt').read_text()
        assert 'tau_global' not in table and 'c_0_' not in table
        assert ('c_E (total SFA budget)' in table) == (k > 0)
        assert ('tau_b_rec_E[m=1]' in table) == (k == 2)


def test_simulation_and_replay_support_three_conditions(tmp_path):
    cell = sim.build_cell(sim.DEFAULT_VARIANTS, 7, 'sra1', .0025, 9, .5)
    inputs = np.zeros((8, 7), dtype=np.float32)
    hist = sim.simulate(cell, inputs)
    assert hist['b_full'].shape == (3, 8, 7)
    np.testing.assert_allclose(hist['b_full'][0], 1)
    sim.plot_variants(cell, hist, np.arange(8) * cell.h, 'smoke', tmp_path / 'simulation.png')
    # A model without trained IC must start with fully available synapses.
    model = SimpleNamespace(cell=cell)
    with pytest.warns(UserWarning, match='cell initial state'):
        bufs, indices = replay.replay(model, torch.from_numpy(inputs), 1)
    np.testing.assert_allclose(bufs['b_full'].transpose(1, 0, 2), hist['b_full'])
    replay.render_variant(bufs, (indices + 1) * cell.h, 2, 'mts', cell.configs[2],
                          cell.n_E, cell.n_I, tmp_path / 'replay.png', 'no_input', 0)
    assert (tmp_path / 'replay.png').stat().st_size > 0


def test_lyapunov_excludes_padded_states():
    cell = sim.build_cell(sim.DEFAULT_VARIANTS, 7, 'sra1', .01, 9, .5)
    with torch.no_grad():
        cell.sparsity_masks.zero_()
    # For the no-adaptation variant, dx/dt=-x/tau_d exactly. Padding must not
    # dilute contraction toward a spurious zero Lyapunov exponent.
    inputs = torch.zeros(20, 7)
    result = replay.benettin_replay(SimpleNamespace(cell=cell), inputs, 0, lya_M=2)
    h = cell.h
    expected = np.log(1 - h / .1 + (h / .1) ** 2 / 2) / h
    np.testing.assert_allclose(result['LLE'][0], expected, atol=2e-3)
