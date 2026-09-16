"""Durable sequential pilot -> sizing -> final experiment -> reporting.

Launch under nohup or a service on the existing GPU VM to survive a terminal
or UI disconnect. Children inherit the environment and use the current Python
interpreter; no package installation, VM lifecycle operation, or source edit
is performed. A failed stage stops the pipeline and preserves all artifacts.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def training_command(seeds, run_name):
    if seeds < 1:
        raise ValueError('Seed count must be positive')
    return [sys.executable, str(ROOT / 'scripts/run_matlab_aligned.py'),
            '--seeds', str(seeds), '--run-name', run_name]


class Pipeline:
    def __init__(self, directory, initial):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=False)
        self.state = {'status': 'running', 'started_at': timestamp(), 'stages': [], **initial}
        self.save()

    def save(self):
        temporary = self.directory / '.pipeline_status.json.tmp'
        temporary.write_text(json.dumps(self.state, indent=2, allow_nan=False))
        os.replace(temporary, self.directory / 'pipeline_status.json')

    def stage(self, name, command):
        log = self.directory / (name + '.log')
        record = {'name': name, 'command': command, 'log': str(log),
                  'status': 'running', 'started_at': timestamp()}
        self.state['stages'].append(record)
        self.state['current_stage'] = name
        self.save()
        try:
            with open(log, 'w') as stream:
                code = subprocess.run(command, cwd=ROOT, stdout=stream,
                                      stderr=subprocess.STDOUT).returncode
            record['exit_code'] = code
            if code:
                raise RuntimeError(f'Stage {name} exited with {code}; see {log}')
            record['status'] = 'complete'
        except BaseException as exc:
            record.update(status='failed', error=repr(exc))
            self.state.update(status='failed', error=repr(exc))
            raise
        finally:
            record['finished_at'] = timestamp()
            self.save()


def validate_training_result(run, seeds, expected_source):
    from scripts.size_matlab_batch import validate_profile_source
    metadata = json.loads((run / 'experiment_metadata.json').read_text())
    if metadata.get('status') != 'complete' or metadata.get('profile') or metadata.get('optimizer_steps') != 2000:
        raise RuntimeError(f'Incomplete 2,000-step training result: {run}')
    if len(metadata.get('variant_names', [])) != 3 * seeds:
        raise RuntimeError(f'Unexpected variant count: {run}')
    validate_profile_source(metadata, expected_source)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--run-prefix', default='matlab-aligned')
    args = parser.parse_args()
    if not args.run_prefix or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.run_prefix):
        parser.error('--run-prefix must contain only letters, digits, underscores, and hyphens')
    profile = args.profile_dir.resolve()
    pipeline = Pipeline(args.output_dir, {'profile_dir': str(profile)})
    try:
        from omegaconf import OmegaConf
        from train_srnn.paths import results_dir
        from scripts.size_matlab_batch import (runtime_source_hashes, validate_profile_source,
            validate_profile_config, validate_profile_dataset, inspect_profile)
        source = runtime_source_hashes()
        metadata = json.loads((profile / 'experiment_metadata.json').read_text())
        cfg = OmegaConf.load(profile / 'resolved_config.yaml')
        profile_seeds = validate_profile_config(cfg)
        if profile_seeds != 3:
            raise ValueError('Pipeline prerequisite is a successful three-seed checkpointed profile')
        validate_profile_source(metadata, source)
        validate_profile_dataset(metadata, cfg.task.data_dir)
        capacity = inspect_profile(profile)
        if not capacity['pass']:
            raise ValueError('Prerequisite profile does not pass the conservative 80% memory rule')
        losses = metadata.get('profile_losses', {})
        for split in ('initial_valid', 'train', 'valid', 'test'):
            values = losses.get(split, [])
            if len(values) != 9 or not all(math.isfinite(float(v)) for v in values):
                raise ValueError(f'Prerequisite profile lacks nine finite {split} losses')
        if metadata.get('optimizer_steps') != 20 or not (profile / 'last.pt').is_file():
            raise ValueError('Prerequisite profile must include 20 steps and a saved checkpoint')
        token = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:6]
        prefix = args.run_prefix + '-' + token
        pilot_name = prefix + '-pilot-s3'
        pilot = results_dir() / 'cheetah100' / pilot_name
        sizing = pipeline.directory / 'capacity'
        pipeline.state.update(runtime_source_sha256=source, prerequisite_memory=capacity,
                              pilot_run_dir=str(pilot), capacity_dir=str(sizing))
        pipeline.save()

        def unchanged():
            if runtime_source_hashes() != source:
                raise RuntimeError('Runtime source changed during pipeline; refusing to mix implementations')
            validate_profile_dataset(metadata, cfg.task.data_dir)

        unchanged()
        pipeline.stage('pilot_training', training_command(3, pilot_name))
        unchanged()
        validate_training_result(pilot, 3, source)
        pipeline.stage('pilot_summary', [sys.executable, str(ROOT / 'scripts/summarize_matlab_aligned.py'), str(pilot)])
        unchanged()
        pipeline.stage('capacity_search', [sys.executable, str(ROOT / 'scripts/size_matlab_batch.py'),
            '--output-dir', str(sizing), '--known-profile', str(profile), '--run-prefix', prefix + '-capacity'])
        unchanged()
        selection = json.loads((sizing / 'capacity_summary.json').read_text())
        selected = selection.get('selected_seeds')
        if selection.get('status') != 'complete' or not isinstance(selected, int) or selected < 1:
            raise RuntimeError('Capacity search did not produce a confirmed positive seed count')
        final_name = prefix + f'-final-s{selected}'
        final = results_dir() / 'cheetah100' / final_name
        pipeline.state.update(selected_seeds=selected, final_run_dir=str(final))
        pipeline.save()
        pipeline.stage('final_training', training_command(selected, final_name))
        unchanged()
        validate_training_result(final, selected, source)
        pipeline.stage('final_summary', [sys.executable, str(ROOT / 'scripts/summarize_matlab_aligned.py'), str(final)])
        pipeline.state.update(status='complete', finished_at=timestamp(), current_stage=None)
        pipeline.save()
        print(pipeline.directory / 'pipeline_status.json', flush=True)
    except BaseException as exc:
        pipeline.state.update(status='failed', error=repr(exc), finished_at=timestamp())
        pipeline.save()
        raise


if __name__ == '__main__':
    main()
