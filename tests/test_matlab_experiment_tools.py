"""Protocol/report checks independent of historical numerical snapshots."""
import csv

import numpy as np
import pytest

from scripts.run_matlab_aligned import active_parameter_counts
from scripts.summarize_matlab_aligned import CONDITIONS, paired_statistics, read_curves
from train_srnn.config import compose_config
from train_srnn.models.factory import build_model


def write_history(tmp_path, omit=None):
    with (tmp_path / 'training_history.csv').open('w') as stream:
        writer = csv.writer(stream)
        writer.writerow(['optimizer_step', 'variant', 'valid_loss'])
        for condition in CONDITIONS:
            for seed in [1, 2]:
                for step in range(0, 2001, 100):
                    if (condition, seed, step) != omit:
                        writer.writerow([step, f'srnn-{condition}-seed{seed}', np.exp(-step / 2000)])


def test_known_log_linear_curve_and_endpoint(tmp_path):
    write_history(tmp_path)
    steps, seeds, curves = read_curves(tmp_path, 2000)
    assert seeds == [1, 2]
    integrate = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
    np.testing.assert_allclose(integrate(np.log(curves['no-adapt']), x=steps, axis=1) / 2000, -.5)
    write_history(tmp_path, omit=('no-adapt', 1, 0))
    with pytest.raises(ValueError, match='Incomplete'):
        read_curves(tmp_path, 2000)


def test_missing_paired_seed_rejected(tmp_path):
    write_history(tmp_path)
    path = tmp_path / 'training_history.csv'
    path.write_text('\n'.join(line for line in path.read_text().splitlines() if 'sfa1-std1-seed2' not in line))
    with pytest.raises(ValueError, match='Incomplete'):
        read_curves(tmp_path, 2000)


def test_signflip_known_three_pairs():
    stats = paired_statistics(np.ones(3), np.random.default_rng(1))
    assert stats['two_sided_p_unadjusted'] == .25
    assert stats['bootstrap_95_percent_interval'] == [1, 1]
    assert stats['paired_dz'] is None


@pytest.mark.parametrize('per_neuron', [False, True])
def test_active_counts_exclude_padding(per_neuron):
    suffix = '-per-neuron' if per_neuron else ''
    names = [f'srnn-{c}{suffix}' for c in CONDITIONS]
    cfg = compose_config(['task=synthetic', 'model=srnn', 'model.num_units=8',
                          'model.variants=[' + ','.join(names) + ']', 'model.variant_seeds=[1]'])
    model = build_model(cfg)
    reports = list(active_parameter_counts(model).values())
    common = reports[0]['total']
    assert reports[0]['by_group']['SFA_E'] == 0
    assert reports[0]['by_group']['STD_E'] == 0
    # Two SFA gains and two STD gains per population. Per-neuron:
    # 1TS adds N*(1 tau_a + 1 c + 2 STD); MTS adds N*(3+1+4).
    assert reports[1]['total'] - common == 8 + (32 if per_neuron else 0)
    assert reports[2]['total'] - common == 8 + (64 if per_neuron else 0)
    assert reports[0]['by_group']['recurrent_nonzero_weights'] == int(model.cell.sparsity_masks[0].count_nonzero())


def test_capacity_conservative_memory_rule():
    from scripts.size_matlab_batch import memory_acceptance
    result = memory_acceptance({'cuda_peak_reserved_bytes': 15000 * 2**20},
                               {'sampled_peak_total_used_mib': 15500, 'device_total_mib': 20000})
    assert result['conservative_used_mib'] == 16024
    assert not result['pass']
    assert memory_acceptance({'cuda_peak_reserved_bytes': 10000 * 2**20},
                             {'sampled_peak_total_used_mib': 14000, 'device_total_mib': 20000})['pass']


def test_capacity_does_not_misclassify_nonfinite_failure(tmp_path):
    import json
    from scripts.size_matlab_batch import explicit_cuda_oom, inspect_profile
    assert explicit_cuda_oom('torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1 GiB')
    assert explicit_cuda_oom('CUDA error: out of memory')
    assert not explicit_cuda_oom('FloatingPointError: Non-finite training loss')
    (tmp_path / 'experiment_metadata.json').write_text(json.dumps({
        'status': 'failed', 'error': 'FloatingPointError: Non-finite training loss'}))
    with pytest.raises(RuntimeError, match='non-capacity'):
        inspect_profile(tmp_path, 1)
    (tmp_path / 'experiment_metadata.json').write_text(json.dumps({
        'status': 'failed', 'error': 'torch.OutOfMemoryError: CUDA out of memory'}))
    assert inspect_profile(tmp_path, 1)['status'] == 'cuda_oom'


def test_profile_source_rejects_old_checkpointing_code(tmp_path):
    from scripts.size_matlab_batch import runtime_source_hashes, validate_profile_source
    (tmp_path / 'scripts').mkdir()
    (tmp_path / 'train_srnn/training').mkdir(parents=True)
    (tmp_path / 'train.py').write_text('# entry')
    (tmp_path / 'scripts/run_matlab_aligned.py').write_text('# runner')
    trainer = tmp_path / 'train_srnn/training/continuous.py'
    trainer.write_text('# checkpoint flag previously ignored')
    old = {'source_sha256': runtime_source_hashes(tmp_path)}
    validate_profile_source(old, runtime_source_hashes(tmp_path))
    trainer.write_text('# checkpoint blocks now implemented')
    with pytest.raises(ValueError, match='continuous.py'):
        validate_profile_source(old, runtime_source_hashes(tmp_path))
    with pytest.raises(ValueError, match='no source checksums'):
        validate_profile_source({}, runtime_source_hashes(tmp_path))
    old['source_sha256']['train_srnn/removed.py'] = 'obsolete'
    with pytest.raises(ValueError, match='file set'):
        validate_profile_source(old, runtime_source_hashes(tmp_path))


def test_profile_config_requires_checkpoint_protocol():
    from types import SimpleNamespace
    from scripts.run_matlab_aligned import overrides
    from scripts.size_matlab_batch import validate_profile_config
    cfg = compose_config(overrides(SimpleNamespace(seeds=3, skip=False, no_dales=False,
        device='cuda', run_name='test', no_compile=False)))
    assert validate_profile_config(cfg) == 3
    cfg.grad_checkpoint = False
    with pytest.raises(ValueError, match='grad_checkpoint'):
        validate_profile_config(cfg)
    cfg.grad_checkpoint = True
    cfg.seed = 99
    with pytest.raises(ValueError, match='seed'):
        validate_profile_config(cfg)


def test_capacity_rejects_nonfinite_memory():
    from scripts.size_matlab_batch import memory_acceptance
    with pytest.raises(ValueError, match='Capacity requires'):
        memory_acceptance({'cuda_peak_reserved_bytes': float('nan')},
                          {'sampled_peak_total_used_mib': 1, 'device_total_mib': 20000})
