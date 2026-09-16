"""The human report is reproducible without PDF tooling and covers every seed."""
import csv
import json
from pathlib import Path
import subprocess
import sys

from omegaconf import OmegaConf
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import report_srnn as report
from train_srnn.config import compose_config
from train_srnn.models.factory import build_model


@pytest.fixture
def saved_run(tmp_path):
    run = tmp_path / 'tiny-run'
    run.mkdir()
    cfg = compose_config(['model=srnn', 'task=synthetic', 'model.num_units=8',
                          'model.variants=[srnn-no-adapt,srnn-sfa1-std1,srnn-sfa3-std2]',
                          'model.variant_seeds=[2,1]', 'epochs=100'])
    model = build_model(cfg)
    names = model.cell.variant_names
    for filename, epoch, change in [('init.pt', -1, 0), ('epoch_004.pt', 4, .1),
                                     ('last.pt', 99, .2)]:
        with torch.no_grad():
            model.cell.log_tau_b_rec_E_gain.add_(change)
            model.cell.log_W_raw_gain.add_(change)
        torch.save({'model_version': 2, 'config': OmegaConf.to_container(cfg, resolve=True),
                    'variant_names': names, 'epoch': epoch,
                    'model_state_dict': model.state_dict()}, run / filename)
    history = []
    for name in names:
        for epoch, step in [(-1, 0), (4, 100), (99, 2000)]:
            history.append(dict(variant=name, epoch=epoch, optimizer_step=step,
                                valid_loss=1 / (1 + step), lr=.0005))
    report.write_csv(run / 'training_history.csv', history)
    report.write_csv(run / 'test_history.csv', [dict(variant=name, tag='last',
                     test_loss=.1, test_metric=.2) for name in names])
    return run, names


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def test_build_report_without_pdf_tools_covers_all_networks(saved_run, monkeypatch):
    run, names = saved_run
    monkeypatch.setattr(report.shutil, 'which', lambda _: None)
    monkeypatch.setattr(report, 'render_pdf', lambda *a, **kw: pytest.fail('PDF was not requested'))
    target = report.build_report(run)
    markdown = target.read_text()
    summary = json.loads((run / 'report_summary.json').read_text())
    selected = summary['selected_variants']
    assert len(selected) == 3
    assert all(name.endswith('-seed1') for name in selected)
    assert [item['completed_epoch'] for item in summary['checkpoints']] == [0, 5, 100]
    rows = read_csv(run / 'report_parameter_history.csv')
    endpoints = read_csv(run / 'report_parameter_initial_final.csv')
    assert {r['variant'] for r in rows} == set(names)
    assert {r['variant'] for r in endpoints} == set(names)
    assert {int(r['epoch']) for r in rows} == {0, 5, 100}
    assert all(r['initial_epoch'] == '0' and r['final_epoch'] == '100' for r in endpoints)
    for name in names:
        parameters = {r['parameter'] for r in rows if r['variant'] == name}
        assert 'a_0' in parameters
        assert ('tau_b_rec_E_2' in parameters) == ('sfa3-std2' in name)
        assert ('tau_b_rec_E_1' in parameters) == ('no-adapt' not in name)
        assert ('c_E' in parameters) == ('no-adapt' not in name)
        assert not any('tau_global' in p or 'c_0' in p for p in parameters)
    assert '`tau_b_rec_E_2`' in markdown
    assert 'tau_global' not in markdown and 'c_0' not in markdown
    assert report.MARKER in markdown
    assert len(read_csv(run / 'report_performance.csv')) == 3
    figures = list((run / 'report_figures').glob('*.png'))
    assert len(figures) >= 10 and all(p.stat().st_size > 0 for p in figures)
    assert not (run / 'report.pdf').exists()


def test_unknown_variant_rejected(saved_run):
    run, _ = saved_run
    with pytest.raises(ValueError, match='Unknown variants'):
        report.build_report(run, variants='missing-seed1')
    assert not (run / 'report.md').exists()


def test_existing_unrelated_report_is_preserved(saved_run):
    run, _ = saved_run
    target = run / 'report.md'
    target.write_text('Original historical report\n')
    with pytest.raises(ValueError, match='Existing report.md'):
        report.build_report(run)
    assert target.read_text() == 'Original historical report\n'


def test_pdf_wrapper_failure_propagates(tmp_path):
    markdown = tmp_path / 'report.md'
    markdown.write_text('A saved report\n')
    wrapper = tmp_path / 'failed-wrapper.sh'
    wrapper.write_text('#!/bin/sh\nexit 23\n')
    with pytest.raises(subprocess.CalledProcessError) as error:
        report.render_pdf(markdown, str(wrapper))
    assert error.value.returncode == 23
    assert markdown.read_text() == 'A saved report\n'


def test_select_all_and_explicit_variants():
    names = ['srnn-sfa1-std1-seed2', 'srnn-no-adapt-seed1', 'srnn-sfa1-std1-seed1']
    assert report.select_variants(names, 'all') == names
    assert report.select_variants(names, names[0]) == names[:1]
    assert report.select_variants(names, None) == names[1:]
