"""Equation-level checks independent of learned trajectories and golden snapshots."""
from dataclasses import replace

import pytest
import torch

from train_srnn.models.srnn_cell import SRNNCell, SRNNConfig


def _cell(configs=None, **overrides):
    if configs is None:
        configs = [SRNNConfig(num_units=5, **overrides)]
    n = configs[0].num_units
    exports = []
    for cfg in configs:
        signs = torch.ones(n)
        signs[n // 2:] = -1
        exports.append(dict(W_init=torch.zeros(n, n), sparsity_mask=torch.zeros(n, n),
                            dales_sign=signs, n_E=n // 2, dales=cfg.dales))
    return SRNNCell(configs, 2, exports).double()


def _rhs(cell, state, drive=None):
    a_e, a_i, b_e, b_i, x = cell.unpack_state(state)
    if drive is None:
        drive = torch.zeros_like(x)
    return cell._rhs(x, a_e, a_i, b_e, b_i, drive, cell._effective_W())[0]


def _packed_rhs(cell, state, drive):
    dx, da_e, da_i, db_e, db_i = _rhs(cell, state, drive)
    return cell.pack_state(da_e, da_i, db_e, db_i, dx)


@pytest.mark.parametrize('counts', [(0, 0), (1, 1), (3, 2)])
def test_state_roundtrip_and_dtype_with_odd_population(counts):
    a, b = counts
    cell = _cell(n_a_E=a, n_a_I=a, n_b_E=b, n_b_I=b)
    state = cell.init_state(3)
    assert state.dtype == torch.float64
    pieces = cell.unpack_state(state)
    assert all(p.dtype == torch.float64 for p in pieces)
    assert pieces[2].shape == (1, 3, 2, max(b, 1))
    assert pieces[3].shape == (1, 3, 3, max(b, 1))
    torch.testing.assert_close(cell.pack_state(*pieces), state, rtol=0, atol=0)
    random_state = torch.randn_like(state)
    torch.testing.assert_close(cell.pack_state(*cell.unpack_state(random_state)), random_state)


def test_total_sfa_budget_is_independent_of_timescale_count():
    base = SRNNConfig(num_units=5, n_b_E=0, n_b_I=0)
    cell = _cell([replace(base, n_a_E=a, n_a_I=a) for a in (0, 1, 3)])
    a_e, a_i, b_e, b_i, x = cell.unpack_state(cell.init_state(2))
    a_e = torch.full_like(a_e, .4)
    a_i = torch.full_like(a_i, .4)
    x = torch.full_like(x, .3)
    effective, _, _, _ = cell._drive(x, a_e, a_i, b_e, b_i, None)
    torch.testing.assert_close(effective[0], x[0])
    torch.testing.assert_close(effective[1], effective[2])
    # c is one total budget per neuron, not one budget for each timescale.
    assert cell._c('E').shape == (3, 2)
    expected = x[1] - torch.cat((cell._c('E')[1], cell._c('I')[1])) * .4
    torch.testing.assert_close(effective[1], expected)


def test_std_equilibrium_is_multiplicative_and_driven_by_raw_rate():
    cell = _cell()
    a_e, a_i, b_e, b_i, x = cell.unpack_state(cell.init_state(1))
    x = cell._a_0().unsqueeze(1)  # activation at its midpoint: r = 0.5
    equilibrium = {}
    for side in ('E', 'I'):
        equilibrium[side] = (1 / (1 + .5 * cell._tau_b(side, 'rec') /
                                  cell._tau_b(side, 'rel'))).unsqueeze(1)
    state = cell.pack_state(a_e, a_i, equilibrium['E'], equilibrium['I'], x)
    _, da_e, da_i, db_e, db_i = _rhs(cell, state)
    torch.testing.assert_close(db_e, torch.zeros_like(db_e), atol=1e-15, rtol=0)
    torch.testing.assert_close(db_i, torch.zeros_like(db_i), atol=1e-15, rtol=0)
    torch.testing.assert_close(da_e, .5 / cell._tau_a('E').unsqueeze(1))
    torch.testing.assert_close(da_i, .5 / cell._tau_a('I').unsqueeze(1))
    diag = cell.get_diagnostics(state)
    expected = torch.cat([equilibrium[s].prod(-1) for s in ('E', 'I')], -1)
    torch.testing.assert_close(diag['b_full'], expected)
    torch.testing.assert_close(diag['br'], .5 * expected)


def test_inactive_timescales_are_neutral_and_receive_no_gradient():
    base = SRNNConfig(num_units=5, per_neuron=True)
    cell = _cell([replace(base, n_a_E=a, n_a_I=a, n_b_E=b, n_b_I=b)
                  for a, b in ((0, 0), (1, 1), (3, 2))])
    state = cell.init_state(2)
    a_e, a_i, b_e, b_i, x = cell.unpack_state(state)
    a_e, a_i = torch.full_like(a_e, .1), torch.full_like(a_i, .1)
    b_e, b_i = torch.full_like(b_e, .6), torch.full_like(b_i, .6)
    state = cell.pack_state(a_e, a_i, b_e, b_i, x)
    diag = cell.get_diagnostics(state)
    torch.testing.assert_close(diag['b_full'][0], torch.ones_like(x[0]))
    torch.testing.assert_close(diag['b_full'][1], torch.full_like(x[1], .6))
    torch.testing.assert_close(diag['b_full'][2], torch.full_like(x[2], .36))
    out, new_state = cell(torch.ones(2, 2, dtype=torch.float64), state)
    (out.square().sum() + new_state.square().sum()).backward()
    for side in ('E', 'I'):
        for kind, active in (('tau_a', 1), ('tau_b_rec', 1), ('tau_b_rel', 1)):
            grad = getattr(cell, f'isp_{kind}_{side}_vec').grad
            assert torch.count_nonzero(grad[0]) == 0
            assert torch.count_nonzero(grad[1, :, active:]) == 0
            assert torch.count_nonzero(grad[2]) > 0


@pytest.mark.parametrize('unfolds', [1, 4])
def test_sra1_exact_formula_and_completed_state_readout(unfolds):
    cell = _cell(solver='sra1', ode_unfolds=unfolds)
    inputs = torch.full((2, 2), .7, dtype=torch.float64)
    state = cell.init_state(2)
    drive = cell._input_drive(inputs)
    expected = state
    dt = cell.h / unfolds
    for _ in range(unfolds):
        k1 = _packed_rhs(cell, expected, drive)
        k2 = _packed_rhs(cell, expected + .75 * dt * k1, drive)
        expected = expected + dt / 3 * (k1 + 2 * k2)
    output, actual = cell(inputs, state)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-14)
    torch.testing.assert_close(output, cell.get_diagnostics(actual)['br'])


def test_seed_pairing_independent_of_variant_order_and_batch_width():
    cfg = SRNNConfig(num_units=5, init_seed=13)
    other = replace(cfg, init_seed=27)
    cell = _cell([cfg, other, replace(cfg, n_a_E=1, n_a_I=1, n_b_E=1, n_b_I=1)])
    reordered = _cell([other, cfg])
    single = _cell([cfg])
    for name in ('W_in', 'a_0_vec', 'isp_tau_a_E_vec', 'isp_tau_a_I_vec'):
        torch.testing.assert_close(getattr(cell, name)[0], getattr(reordered, name)[1], rtol=0, atol=0)
        torch.testing.assert_close(getattr(cell, name)[0], getattr(single, name)[0], rtol=0, atol=0)
    torch.testing.assert_close(cell.W_in[0], cell.W_in[2], rtol=0, atol=0)
    torch.testing.assert_close(cell.a_0_vec[0], cell.a_0_vec[2], rtol=0, atol=0)
    x = cell.unpack_state(cell.init_state(2))[-1]
    torch.testing.assert_close(x[0], x[2], rtol=0, atol=0)
    torch.testing.assert_close(x[0], single.unpack_state(single.init_state(2))[-1][0], rtol=0, atol=0)


def test_mixed_batched_cells_match_independent_outputs_and_parameter_gradients():
    base = SRNNConfig(num_units=5, per_neuron=True)
    configs = [replace(base, n_a_E=a, n_a_I=a, n_b_E=b, n_b_I=b)
               for a, b in ((0, 0), (1, 1), (3, 2))]
    batched = _cell(configs)
    inputs = torch.tensor([[.2, -.1], [.5, .7]], dtype=torch.float64)
    output, state = batched(inputs, batched.init_state(2))
    output.square().sum().backward()
    batch_parts = batched.unpack_state(state)
    for k, cfg in enumerate(configs):
        single = _cell([cfg])
        own_output, own_state = single(inputs, single.init_state(2))
        torch.testing.assert_close(output[k], own_output[0], rtol=1e-10, atol=1e-12)
        for index, (bpart, spart) in enumerate(zip(batch_parts, single.unpack_state(own_state))):
            if index < 4:
                count = (cfg.n_a_E, cfg.n_a_I, cfg.n_b_E, cfg.n_b_I)[index]
                if count == 0:
                    continue
                bpart = bpart[..., :count]
            torch.testing.assert_close(bpart[k], spart[0], rtol=1e-10, atol=1e-12)
        own_output.square().sum().backward()
        batch_parameters = dict(batched.named_parameters())
        for name, param in single.named_parameters():
            if param.grad is None:
                continue
            grad = batch_parameters[name].grad[k]
            wanted = param.grad[0]
            # Population/timescale arrays have padding only in the batched cell.
            if grad.shape != wanted.shape:
                grad = grad[..., :wanted.shape[-1]]
            torch.testing.assert_close(grad, wanted, rtol=1e-8, atol=1e-11, msg=name)


def test_dale_gain_cannot_flip_presynaptic_signs():
    cell = _cell()
    with torch.no_grad():
        cell.sparsity_masks.fill_(1)
        cell.W_raw.copy_(torch.linspace(-3, 3, cell.N ** 2).reshape(1, cell.N, cell.N))
        cell.log_W_raw_gain.fill_(-5)
    weights = cell._effective_W()
    assert (weights[..., :cell.n_E] > 0).all()
    assert (weights[..., cell.n_E:] < 0).all()


def test_nominal_timescales_live_in_physical_log_space():
    base = SRNNConfig(num_units=5, tau_a_spread=0)
    cell = _cell([replace(base, n_a_E=1, n_a_I=1, n_b_E=1, n_b_I=1), base])
    for side in ('E', 'I'):
        taus = cell._tau_a(side)
        torch.testing.assert_close(taus[0, :, 0], torch.full_like(taus[0, :, 0], .25), rtol=1e-6, atol=1e-8)
        expected = torch.tensor([.25, 2.5 ** .5, 10.], dtype=taus.dtype).expand_as(taus[1])
        torch.testing.assert_close(taus[1], expected, rtol=1e-6, atol=1e-8)
        torch.testing.assert_close(cell._tau_b(side, 'rec')[1],
                                   torch.tensor([2., 4.], dtype=taus.dtype).expand_as(cell._tau_b(side, 'rec')[1]),
                                   rtol=1e-6, atol=1e-8)


def test_dale_initialization_is_independent_of_training_enforcement():
    from train_srnn.models.rmt_matrix import RMTMatrix

    rmt = RMTMatrix(n=6, f=.5, indegree=6, seed=8)
    rmt.build()
    configs = [SRNNConfig(num_units=6, dales=flag) for flag in (True, False)]
    exports = [rmt.export_for_srnn(dales=c.dales, dales_init=True) for c in configs]
    cell = SRNNCell(configs, 2, exports)
    initial = cell._effective_W()
    torch.testing.assert_close(initial[0], initial[1], rtol=1e-6, atol=1e-7)
    assert (initial[:, :, :3] >= 0).all() and (initial[:, :, 3:] <= 0).all()
    with torch.no_grad():
        cell.W_raw[1].neg_()
    changed = cell._effective_W()
    torch.testing.assert_close(changed[0], initial[0])
    torch.testing.assert_close(changed[1], -initial[1])
