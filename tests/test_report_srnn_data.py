"""Effective v2 report statistics, masking, provenance and epoch semantics."""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from report_srnn_data import collect_run, split_variant
from train_srnn.config import compose_config
from train_srnn.models.factory import build_model


@pytest.fixture
def snapshot():
    cfg = compose_config(['model=srnn', 'task=synthetic', 'model.num_units=8',
        'model.variants=[srnn-no-adapt-seed1,srnn-sfa1-std1-seed1,srnn-sfa3-std2-seed1]'])
    model = build_model(cfg)
    return model, dict(epoch=0, model_version=2, config=OmegaConf.to_container(cfg, resolve=True),
                      variant_names=model.variant_names, model_state_dict=model.state_dict())


def test_active_effective_statistics_and_epochs(tmp_path, snapshot):
    model, ckpt = snapshot
    torch.save(ckpt, tmp_path / 'init.pt')
    ckpt['epoch'] = 4
    torch.save(ckpt, tmp_path / 'epoch_004.pt')
    torch.save(ckpt, tmp_path / 'last.pt')
    report = collect_run(tmp_path)
    assert [(c['tag'], c['completed_epoch']) for c in report['checkpoints']] == [('init', 0), ('last', 5)]
    assert report['provenance'][0]['file'] == 'epoch_004.pt'
    json.dumps(report, allow_nan=False)
    assert report['variant_settings'] == [dict(name=name, skip=False, dales=True, per_neuron=False,
        n_a_E=a, n_a_I=a, n_b_E=b, n_b_I=b) for name, (a, b) in
        zip(model.variant_names, ((0, 0), (1, 1), (3, 2)))]
    native = model.cell.effective_params()
    for k, counts in enumerate(((0, 0), (1, 1), (3, 2))):
        rows = {r['parameter']: r for r in report['rows'] if r['variant'] == model.variant_names[k] and r['epoch'] == 0}
        assert rows['W_relative_change']['mean'] == 0
        assert rows['Dale_violations']['mean'] == 0
        assert rows['tau_d']['n'] == 8
        assert rows['a_0']['n'] == 8
        np.testing.assert_allclose(rows['a_0']['mean'], native['a_0'][k].numpy().mean(), rtol=1e-6)
        for side in ('E', 'I'):
            assert len([p for p in rows if p.startswith(f'tau_a_{side}_')]) == counts[0]
            assert len([p for p in rows if p.startswith(f'tau_b_rec_{side}_')]) == counts[1]
            assert (f'c_{side}' in rows) == bool(counts[0])
            for j in range(counts[1]):
                values = native[f'tau_b_rel_{side}'][k, :, j].numpy()
                row = rows[f'tau_b_rel_{side}_{j+1}']
                np.testing.assert_allclose(row['mean'], values.mean(), rtol=1e-6)
                assert row['n'] == 4
    assert split_variant('srnn-sfa3-std2-no-dales-seed15') == ('srnn-sfa3-std2-no-dales', 15)
    assert split_variant('srnn-no-adapt') == ('srnn-no-adapt', None)


@pytest.mark.parametrize('change', ['config', 'names', 'version', 'mask', 'epoch'])
def test_incompatible_checkpoints_fail(tmp_path, snapshot, change):
    _, ckpt = snapshot
    torch.save(ckpt, tmp_path / 'init.pt')
    other = copy.deepcopy(ckpt)
    if change == 'config':
        other['config']['model']['num_units'] = 9
    elif change == 'names':
        other['variant_names'].reverse()
    elif change == 'version':
        other['model_version'] = 1
    elif change == 'mask':
        other['model_state_dict']['cell.sfa_E_mask'][1, 0, 0] = 0
    elif change == 'epoch':
        other['epoch'] = 7
    torch.save(other, tmp_path / 'epoch_000.pt')
    with pytest.raises(ValueError):
        collect_run(tmp_path)


def test_partial_uploads_are_not_checkpoints(tmp_path):
    (tmp_path / 'last.pt_.gstmp').write_bytes(b'partial')
    with pytest.raises(ValueError, match='No completed checkpoint'):
        collect_run(tmp_path)


def test_trained_gains_changes_and_dale_statistics(tmp_path, snapshot):
    model, ckpt = snapshot
    torch.save(ckpt, tmp_path / 'init.pt')
    with torch.no_grad():
        model.cell.log_W_raw_gain.add_(np.log(2))
        model.cell.log_tau_b_rel_E_gain.add_(np.log(3))
        model.cell.log_c_E_gain.add_(np.log(4))
    ckpt['epoch'] = 0
    torch.save(ckpt, tmp_path / 'last.pt')
    result = collect_run(tmp_path)
    rows = {(r['variant'], r['epoch'], r['parameter']): r for r in result['rows']}
    name = model.variant_names[-1]
    for parameter, factor in [('tau_b_rel_E_2', 3), ('c_E', 4), ('W_abs_E', 2)]:
        np.testing.assert_allclose(rows[name, 1, parameter]['mean'], factor * rows[name, 0, parameter]['mean'], rtol=1e-6)
    assert rows[name, 1, 'W_relative_change']['mean'] == pytest.approx(1, rel=1e-6)


def test_variant_settings_reflect_overrides(tmp_path):
    cfg = compose_config(['model=srnn', 'task=synthetic', 'model.num_units=8',
        'model.variants=[srnn-sfa1-std1-skip-no-dales-per-neuron-seed2]'])
    model = build_model(cfg)
    torch.save(dict(epoch=0, model_version=2, config=OmegaConf.to_container(cfg, resolve=True),
                    variant_names=model.variant_names, model_state_dict=model.state_dict()), tmp_path / 'init.pt')
    setting = collect_run(tmp_path)['variant_settings'][0]
    assert setting == dict(name=model.variant_names[0], skip=True, dales=False, per_neuron=True,
                           n_a_E=1, n_a_I=1, n_b_E=1, n_b_I=1)
