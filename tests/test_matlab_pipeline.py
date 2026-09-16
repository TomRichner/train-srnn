"""Durable pipeline plumbing tests; never launch training."""
import json
from types import SimpleNamespace

import pytest

from scripts.run_matlab_pipeline import Pipeline, training_command


def test_fresh_training_command_keeps_protocol():
    command = training_command(7, 'test-final-s7')
    assert command[-4:] == ['--seeds', '7', '--run-name', 'test-final-s7']
    assert '--profile' not in command
    assert '--skip' not in command
    assert '--no-dales' not in command
    with pytest.raises(ValueError):
        training_command(0, 'bad')


def test_stage_writes_durable_completed_status(tmp_path, monkeypatch):
    seen = []
    def fake_run(command, **kwargs):
        seen.append(command)
        kwargs['stdout'].write('stage log\n')
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr('scripts.run_matlab_pipeline.subprocess.run', fake_run)
    pipeline = Pipeline(tmp_path / 'pipeline', {'profile_dir': '/fake'})
    pipeline.stage('pilot', ['python', 'fake.py'])
    saved = json.loads((pipeline.directory / 'pipeline_status.json').read_text())
    assert saved['stages'][0]['status'] == 'complete'
    assert saved['stages'][0]['exit_code'] == 0
    assert (pipeline.directory / 'pilot.log').read_text() == 'stage log\n'
    assert seen == [['python', 'fake.py']]
    assert not (pipeline.directory / '.pipeline_status.json.tmp').exists()


def test_failed_stage_halts_and_preserves_status(tmp_path, monkeypatch):
    monkeypatch.setattr('scripts.run_matlab_pipeline.subprocess.run',
                        lambda *args, **kwargs: SimpleNamespace(returncode=7))
    pipeline = Pipeline(tmp_path / 'pipeline', {})
    with pytest.raises(RuntimeError, match='exited with 7'):
        pipeline.stage('capacity', ['fake'])
    saved = json.loads((pipeline.directory / 'pipeline_status.json').read_text())
    assert saved['status'] == 'failed'
    assert saved['stages'][0]['status'] == 'failed'
    assert saved['stages'][0]['exit_code'] == 7
    with pytest.raises(FileExistsError):
        Pipeline(tmp_path / 'pipeline', {})
