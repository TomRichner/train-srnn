"""Measure whole-seed capacity after the pilot without changing its protocol.

Uses isolated profile subprocesses, then repeats the largest passing seed count.
No final training run is launched. A 1-GiB non-allocator allowance is added to
PyTorch's peak reserved memory when applying the 80% whole-device limit.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))



def runtime_source_hashes(root=ROOT):
    """Hash code that can change training memory/numerics, excluding analysis/tests."""
    root = Path(root)
    paths = [root / 'train.py', root / 'scripts/run_matlab_aligned.py']
    paths += sorted((root / 'train_srnn').rglob('*.py'))
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def validate_profile_source(metadata, expected):
    saved = metadata.get('source_sha256')
    if not isinstance(saved, dict):
        raise ValueError('Profile has no source checksums; rerun it with the deployed source')
    runtime_keys = {key for key in saved if key.startswith('train_srnn/') or key in ('train.py', 'scripts/run_matlab_aligned.py')}
    if runtime_keys != set(expected):
        raise ValueError('Profile runtime source file set differs from current deployment')
    mismatched = sorted(key for key, digest in expected.items() if saved[key] != digest)
    if mismatched:
        raise ValueError('Profile runtime source differs from current deployment: ' + ', '.join(mismatched))


def validate_profile_config(cfg):
    from types import SimpleNamespace
    from omegaconf import OmegaConf
    from train_srnn.config import compose_config
    from scripts.run_matlab_aligned import overrides
    seeds = len(cfg.model.variant_seeds)
    if seeds < 1 or list(cfg.model.variant_seeds) != list(range(1, seeds + 1)):
        raise ValueError('Known profile must use consecutive seeds starting at one')
    expected_cfg = compose_config(overrides(SimpleNamespace(seeds=seeds, skip=False, no_dales=False,
        device='cuda', run_name='capacity-check', no_compile=False)))
    for field in ('model', 'task', 'compile', 'closed_loop', 'seed', 'device', 'lr', 'epochs', 'warmup_epochs',
                  'cosine_decay', 'grad_clip', 'grad_checkpoint', 'grad_checkpoint_segment_len',
                  'burn_in', 'burn_in_every', 'freeze_ic_after_burnin', 'amp', 'freeze_params', 'init_ckpt',
                  'checkpoint_interval', 'log_interval', 'early_exit_after_init', 'profile'):
        saved = OmegaConf.to_container(cfg[field], resolve=True) if OmegaConf.is_config(cfg[field]) else cfg[field]
        expected_value = OmegaConf.to_container(expected_cfg[field], resolve=True) if OmegaConf.is_config(expected_cfg[field]) else expected_cfg[field]
        if saved != expected_value:
            raise ValueError(f'Known profile differs from fixed protocol at {field}')
    return seeds


def validate_profile_dataset(metadata, directory):
    expected = {}
    for name in ('train.npz', 'valid.npz', 'test.npz', 'manifest.json'):
        digest = hashlib.sha256()
        with open(Path(directory) / name, 'rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        expected[name] = digest.hexdigest()
    if metadata.get('dataset_sha256') != expected:
        raise ValueError('Profile dataset hashes differ from current dataset')


def memory_acceptance(metadata, gpu):
    """Use the greater of sampled whole-device use and reserved peak + 1 GiB."""
    reserved = metadata.get('cuda_peak_reserved_bytes')
    sampled = gpu.get('sampled_peak_total_used_mib')
    total = gpu.get('device_total_mib')
    if (reserved is None or sampled is None or total is None
            or not all(math.isfinite(float(v)) for v in (reserved, sampled, total))
            or reserved < 0 or sampled < 0 or total <= 0):
        raise ValueError('Capacity requires CUDA allocator peaks and whole-device samples')
    conservative = max(float(sampled), float(reserved) / 2**20 + 1024)
    return {'pass': conservative <= .8 * total,
            'conservative_used_mib': conservative, 'device_total_mib': total,
            'fraction': conservative / total, 'non_allocator_allowance_mib': 1024}


def explicit_cuda_oom(error):
    """Only an explicit CUDA allocation failure can bound capacity."""
    return bool(re.search(r'CUDA(?:\s+error:)?\s+out of memory', error, re.IGNORECASE))


def inspect_profile(path, returncode=0):
    path = Path(path)
    meta_path = path / 'experiment_metadata.json'
    metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    gpu_path = path / 'gpu_memory.json'
    gpu = json.loads(gpu_path.read_text()) if gpu_path.exists() else {}
    if returncode != 0 or metadata.get('status') != 'complete' or gpu.get('exit_code', 0) != 0:
        error = str(metadata.get('error', ''))
        log = path / 'training.log'
        if log.exists():
            error += '\n' + log.read_text(errors='replace')[-50000:]
        if explicit_cuda_oom(error):
            return {'pass': False, 'status': 'cuda_oom', 'run_dir': str(path)}
        raise RuntimeError(f'Profile failed for a non-capacity reason: {path}\n{error[-3000:]}')
    if not metadata.get('profile'):
        raise ValueError(f'Expected a capacity profile, not a full run: {path}')
    result = memory_acceptance(metadata, gpu)
    return {**result, 'status': 'pass' if result['pass'] else 'memory_limit', 'run_dir': str(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--run-prefix', default='matlab-capacity')
    parser.add_argument('--known-profile', type=Path, help='Completed three-condition profile directory or its metadata JSON')
    parser.add_argument('--max-seeds', type=int, help='Optional explicit resource cap; a passing cap is not a measured hardware maximum')
    args = parser.parse_args()
    if args.max_seeds is not None and args.max_seeds < 1:
        parser.error('--max-seeds must be positive')
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source_hashes = runtime_source_hashes()
    summary = {'runtime_source_sha256': source_hashes, 'status': 'running', 'attempts': [], 'max_seeds_cap': args.max_seeds,
               'memory_rule': 'max(sampled whole-device peak, allocator reserved peak + 1024 MiB) <= 80% device capacity',
               'note': 'One-second device sampling can miss transients. The 1-GiB allowance is conservative bookkeeping, not an empirically measured overhead bound.'}
    def save():
        (output / 'capacity_summary.json').write_text(json.dumps(summary, indent=2))
    def attempt(seeds):
        from train_srnn.paths import results_dir
        name = args.run_prefix + f'-s{seeds}-' + time.strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:6]
        run = results_dir() / 'cheetah100' / name
        command = [sys.executable, str(ROOT / 'scripts/run_matlab_aligned.py'), '--profile', '--seeds', str(seeds), '--run-name', name]
        if runtime_source_hashes() != source_hashes:
            raise RuntimeError('Runtime source changed during capacity search; start a new search')
        record = {'seeds': seeds, 'run_dir': str(run), 'command': command, 'status': 'running'}
        summary['attempts'].append(record)
        save()
        with open(output / (name + '.launcher.log'), 'w') as stream:
            code = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT).returncode
        try:
            record.update(inspect_profile(run, code))
            if record['status'] != 'cuda_oom':
                validate_profile_source(json.loads((run / 'experiment_metadata.json').read_text()), source_hashes)
            if runtime_source_hashes() != source_hashes:
                raise RuntimeError('Runtime source changed during capacity search; start a new search')
        except Exception as exc:
            record.update(status='non_capacity_failure', error=str(exc), returncode=code)
            save()
            raise
        save()
        return record['pass']
    try:
        low, high = 0, None
        if args.known_profile:
            known = args.known_profile.resolve()
            known = known.parent if known.is_file() else known
            # Require the exact same fixed protocol, not merely the same model size.
            from omegaconf import OmegaConf
            cfg = OmegaConf.load(known / 'resolved_config.yaml')
            seeds = validate_profile_config(cfg)
            metadata = json.loads((known / 'experiment_metadata.json').read_text())
            validate_profile_source(metadata, source_hashes)
            validate_profile_dataset(metadata, cfg.task.data_dir)
            if args.max_seeds is not None and seeds > args.max_seeds:
                raise ValueError('Known profile exceeds requested seed cap')
            record = {'seeds': seeds, **inspect_profile(known), 'reused': True}
            summary['attempts'].append(record)
            save()
            if record['pass']:
                low = seeds
            else:
                high = seeds
        next_seeds = min(3, args.max_seeds) if args.max_seeds else 3
        if low:
            next_seeds = low * 2
        while high is None:
            if args.max_seeds is not None:
                if low == args.max_seeds:
                    break
                next_seeds = min(next_seeds, args.max_seeds)
            if attempt(next_seeds):
                low = next_seeds
                next_seeds *= 2
            else:
                high = next_seeds
        while high is not None and high - low > 1:
            mid = (low + high) // 2
            if attempt(mid):
                low = mid
            else:
                high = mid
        if low == 0:
            raise RuntimeError('No positive whole-seed count passed the capacity rule')
        if not attempt(low):
            raise RuntimeError('Selected capacity failed confirmation; memory usage is not reproducible, so no capacity is approved')
        summary.update(status='complete', selected_seeds=low,
                       hardware_maximum_bracketed=high is not None,
                       first_failing_seed_bound=high,
                       final_training_command=[sys.executable, str(ROOT / 'scripts/run_matlab_aligned.py'), '--seeds', str(low)])
        print(json.dumps({key: summary[key] for key in ('selected_seeds', 'hardware_maximum_bracketed', 'final_training_command')}, indent=2))
    except BaseException as exc:
        summary.update(status='failed', error=repr(exc))
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
