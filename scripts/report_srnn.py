"""Build a version-2 SRNN report from local checkpoints, histories and metadata.

    python scripts/report_srnn.py /path/to/run --pdf
    python scripts/report_srnn.py /path/to/run --variants all --pdf

Markdown, plots and CSV tables are always generated. PDF uses the installed
md2pdf skill wrapper, configurable with --pdf-wrapper or MD2PDF_WRAPPER.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, StrMethodFormatter, NullLocator
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_srnn_data import collect_run

COLORS = {'srnn-no-adapt': '#555555', 'srnn-sfa1-std1': '#0072B2', 'srnn-sfa3-std2': '#DAA520'}
LABELS = {'srnn-no-adapt': 'No adaptation', 'srnn-sfa1-std1': 'SFA1 / STD1', 'srnn-sfa3-std2': 'SFA3 / STD2'}
MARKER = '<!-- srnn-report:v2 -->'


def label(condition):
    return LABELS.get(condition, condition)


def write_csv(path, rows):
    if rows:
        with path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def select_variants(names, spec):
    if spec == 'all':
        return names
    if spec:
        chosen = spec.split(',')
        missing = set(chosen) - set(names)
        if missing:
            raise ValueError(f'Unknown variants: {sorted(missing)}')
        return chosen
    chosen, seen = [], set()
    for name in sorted(names, key=lambda n: int(re.search(r'-seed(\d+)', n).group(1)) if re.search(r'-seed(\d+)', n) else 0):
        condition = re.sub(r'-seed\d+(?=-|$)', '', name)
        if condition not in seen:
            chosen.append(name)
            seen.add(condition)
    return chosen


def plot_parameters(rows, parameters, output, *, condition=None, variant=None, statistic='mean', title=''):
    selected = [r for r in rows if (condition is None or r['condition'] == condition)
                and (variant is None or r['variant'] == variant)]
    parameters = [p for p in parameters if any(r['parameter'] == p for r in selected)]
    if not parameters:
        return False
    columns = min(3, len(parameters))
    height = (len(parameters) + columns - 1) // columns
    fig, axes = plt.subplots(height, columns, figsize=(7.2, 2.15 * height), squeeze=False)
    conditions = list(dict.fromkeys(r['condition'] for r in selected))
    for ax, parameter in zip(axes.flat, parameters):
        for i, group in enumerate(conditions):
            values = [r for r in selected if r['condition'] == group and r['parameter'] == parameter]
            epochs = sorted({r['epoch'] for r in values})
            if not epochs:
                continue
            mu, sd = [], []
            for epoch in epochs:
                at = [r for r in values if r['epoch'] == epoch]
                mu.append(np.mean([r[statistic] for r in at]))
                sd.append(at[0]['std'] if variant and statistic == 'mean'
                          else np.std([r[statistic] for r in at], ddof=1) if len(at) > 1 else 0)
            color = COLORS.get(group, f'C{i % 10}')
            ax.plot(epochs, mu, color=color, lw=1.3, label=label(group))
            ax.fill_between(epochs, np.asarray(mu)-sd, np.asarray(mu)+sd, color=color, alpha=.16)
        ax.set_title(parameter, fontsize=8)
        ax.set_xlabel('Completed epochs', fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=.2)
        if parameter.startswith('tau_'):
            ax.set_ylabel('seconds', fontsize=7)
            ax.set_yscale('log')
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3, prune='both'))
            ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.3g}'))
            ax.yaxis.set_minor_locator(NullLocator())
    for ax in list(axes.flat)[len(parameters):]:
        ax.axis('off')
    if len(conditions) > 1:
        handles = {}
        for ax in axes.flat:
            for handle, name in zip(*ax.get_legend_handles_labels()):
                handles[name] = handle
        fig.legend(handles.values(), handles.keys(), fontsize=6, frameon=False, loc='lower center', ncol=3)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, .06 if len(conditions) > 1 else 0, 1, 1))
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return True


def history_plots(run, output, names):
    history = list(csv.DictReader((run / 'training_history.csv').open()))
    final = list(csv.DictReader((run / 'test_history.csv').open()))
    conditions = list(dict.fromkeys(re.sub(r'-seed\d+(?=-|$)', '', n) for n in names))
    fig, ax = plt.subplots(figsize=(4, 3))
    result = []
    for i, condition in enumerate(conditions):
        members = [n for n in names if re.sub(r'-seed\d+(?=-|$)', '', n) == condition]
        curves = []
        color = COLORS.get(condition, f'C{i % 10}')
        expected_x = None
        for name in members:
            values = [r for r in history if r['variant'] == name and not np.isnan(float(r['valid_loss']))]
            values.sort(key=lambda r: int(r.get('optimizer_step', int(r['epoch'])+1)))
            x = [int(r.get('optimizer_step', int(r['epoch'])+1)) for r in values]
            y = np.array([float(r['valid_loss']) for r in values])
            if not len(y) or not np.isfinite(y).all() or (y <= 0).any():
                raise ValueError(f'Missing/nonpositive/nonfinite validation curve: {name}')
            if expected_x is not None and x != expected_x:
                raise ValueError(f'Inconsistent validation grid: {name}')
            expected_x = x
            ax.plot(x, y, color=color, lw=.6, alpha=.22)
            curves.append(y)
        ax.plot(expected_x, np.mean(curves, axis=0), color=color, lw=1.8, label=label(condition))
        losses, maes = [], []
        for name in members:
            candidates = [r for r in final if r['variant'] == name and r['tag'] == 'last']
            if len(candidates) != 1:
                raise ValueError(f'Expected one final test row for {name}')
            losses.append(float(candidates[0]['test_loss']))
            maes.append(float(candidates[0]['test_metric']))
        if not np.isfinite(losses + maes).all():
            raise ValueError('Nonfinite final test statistics')
        result.append({'condition': condition, 'seeds': len(members), 'test_MSE_mean': float(np.mean(losses)),
                       'test_MSE_SD': float(np.std(losses, ddof=1)) if len(losses)>1 else 0,
                       'test_MAE_mean': float(np.mean(maes))})
    ax.set(xlabel='Optimizer step' if 'optimizer_step' in history[0] else 'Completed epochs',
           ylabel='Validation MSE', yscale='log')
    ax.legend(fontsize=7, frameon=False, loc='lower left')
    fig.tight_layout(); fig.savefig(output / 'learning.png', dpi=200); plt.close(fig)
    one = [r for r in history if r['variant'] == names[0]]
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.plot([int(r.get('optimizer_step', int(r['epoch'])+1)) for r in one], [float(r['lr']) for r in one], 'o-', ms=2)
    ax.set(xlabel='Optimizer step' if 'optimizer_step' in history[0] else 'Completed epochs', ylabel='Learning rate')
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0)); fig.tight_layout(rect=(0, .06 if len(conditions) > 1 else 0, 1, 1))
    fig.savefig(output / 'learning_rate.png', dpi=160); plt.close(fig)
    return result


def render_pdf(markdown, wrapper=None):
    candidates = [wrapper, os.environ.get('MD2PDF_WRAPPER'), shutil.which('md2pdf.sh')]
    candidates += [str(Path.home()/base/'md2pdf/scripts/md2pdf.sh') for base in ('.agents/skills', '.codex/skills', '.claude/skills')]
    script = next((Path(p).expanduser().resolve() for p in candidates if p and Path(p).expanduser().is_file()), None)
    if script is None:
        raise RuntimeError('Markdown saved; PDF needs the md2pdf skill wrapper (--pdf-wrapper PATH).')
    subprocess.run(['bash', str(script), str(markdown), '--', '-V', 'fontsize=10pt',
                    '-V', 'colorlinks=true'], check=True, cwd=markdown.parent)


def build_report(run, output=None, variants=None, pdf=False, pdf_wrapper=None):
    run = Path(run).resolve(); output = Path(output).resolve() if output else run
    output.mkdir(parents=True, exist_ok=True)
    target = output / 'report.md'
    if target.exists() and MARKER not in target.read_text():
        raise ValueError('Existing report.md was not made by this feature; use --output-dir to preserve it.')
    if not (run/'init.pt').is_file() or not (run/'last.pt').is_file():
        raise ValueError('A completed report requires init.pt and last.pt; download the checkpoints first.')
    data = collect_run(run)
    rows, names = data['rows'], data['variant_names']
    selected = select_variants(names, variants)
    figures = output / 'report_figures'; figures.mkdir(exist_ok=True)
    performance = history_plots(run, figures, names)
    write_csv(output/'report_parameter_history.csv', rows)
    endpoints = []
    for name in names:
        by_parameter = {}
        for row in rows:
            if row['variant'] == name:
                by_parameter.setdefault(row['parameter'], []).append(row)
        for parameter, values in by_parameter.items():
            values.sort(key=lambda r:r['epoch']); first,last = values[0],values[-1]
            endpoints.append({'variant':name,'condition':first['condition'],'seed':first['seed'],
                'parameter':parameter,'initial_epoch':first['epoch'],'final_epoch':last['epoch'],
                'initial_mean':first['mean'],'initial_std':first['std'],'final_mean':last['mean'],
                'final_std':last['std'],'n':last['n']})
    write_csv(output/'report_parameter_initial_final.csv', endpoints)
    write_csv(output/'report_performance.csv', performance)
    cfg=data['config']; model=cfg['model']
    md=[MARKER, f'# SRNN training report: {run.name}', '',
        '## Run and interpretation', '',
        f"{len(names)} networks; {cfg['epochs']} epochs; {model['num_units']} neurons; "
        f"solver `{model['solver']}`, {model['ode_unfolds']} internal substeps; "
        f"gradient checkpointing `{cfg.get('grad_checkpoint')}`.", '',
        'Condition names and the saved configuration determine skip and Dale settings. '
        'All checkpoint parameters below are effective (after transforms and masks). '
        'SFA coupling is the total budget c; inactive adaptation slots are omitted. '
        'Each STD recovery and release timescale is reported separately.', '',
        f"Detailed panels use: {', '.join('`'+n+'`' for n in selected)}. "
        'By default these are the lowest-numbered seed in each condition, selected without using outcomes. '
        'All networks enter the aggregate figures and downloadable CSV tables.', '',
        '## Final prediction performance', '',
        '| Condition | Seeds | Test MSE, mean ± SD | Test MAE, mean |',
        '|---|---:|---:|---:|']
    for p in performance:
        md.append(f"| {label(p['condition'])} | {p['seeds']} | {p['test_MSE_mean']:.5g} ± {p['test_MSE_SD']:.3g} | {p['test_MAE_mean']:.5g} |")
    md += ['', 'Test performance was evaluated periodically during training. These are not untouched holdout-test estimates.', '',
           '![Validation curves: thin lines are individual seeds; thick lines are condition means.](report_figures/learning.png){width=80%}',
           '', '\\clearpage', '', '## Learning schedule and provenance', '',
           '![Recorded learning rates at saved history points; the plotted samples do not resolve every warmup step.](report_figures/learning_rate.png){width=95%}', '']
    for filename in ('run_metadata.json','runtime_provenance.json','gpu_memory_summary.json'):
        path=run/filename
        if path.exists():
            obj=json.loads(path.read_text()); data.setdefault('execution',{})[filename]=obj
            if filename=='run_metadata.json':
                md += [f"Source commit: `{obj.get('commit','unknown')}`. Exit code: {obj.get('exit_code','unknown')}. "
                       f"Runtime: {obj.get('duration_seconds','unknown')} seconds.", '']
            elif filename=='runtime_provenance.json':
                md += [f"Python {obj.get('python','unknown')}; PyTorch {obj.get('torch','unknown')}; CUDA {obj.get('cuda','unknown')}.", '']
    md += ['| Condition | Skip | Dale enforced | Per-neuron learning | SFA E/I | STD E/I |',
           '|---|---|---|---|---:|---:|']
    mode_seen=set()
    for setting in data.get('variant_settings',[]):
        condition=re.sub(r'-seed\d+(?=-|$)', '', setting['name'])
        key=(condition,setting['skip'],setting['dales'],setting['per_neuron'],
             setting['n_a_E'],setting['n_a_I'],setting['n_b_E'],setting['n_b_I'])
        if key in mode_seen:
            continue
        mode_seen.add(key)
        md.append(f"| {label(condition)} | {setting['skip']} | {setting['dales']} | {setting['per_neuron']} | "
                  f"{setting['n_a_E']}/{setting['n_a_I']} | {setting['n_b_E']}/{setting['n_b_I']} |")
    md += ['']
    md += [f"Checkpoint epochs: {', '.join(str(c['completed_epoch']) for c in data['checkpoints'])}. "
           'Epoch 0 is initialization; saved epoch indices are converted to completed epochs. '
           'The final checkpoint replaces an intermediate checkpoint at the same epoch.', '',
           'For aggregate parameter panels, each line is the average of the per-network means; '
           'bands are ±1 sample SD across seeds. Input/output weight-spread panels instead summarize '
           'each network’s within-weight SD. Per-seed detail bands denote within-neuron or within-weight '
           'SD, not uncertainty across seeds. Structural zero connections are excluded from recurrent-weight statistics.', '']
    parameters=list(dict.fromkeys(r['parameter'] for r in rows))
    groups=[('Recurrent weights', ['W_E','W_I','W_abs_E','W_abs_I','W_relative_change','Dale_violations'],'mean'),
            ('Input and output weight spread',['W_in','W_out','readout_bias'],'std'),
            ('Adaptation budgets and setpoint',['c_E','c_I','a_0'],'mean')]
    for idx,(title,params,stat) in enumerate(groups):
        file=f'aggregate_{idx}.png'
        if plot_parameters(rows,params,figures/file,statistic=stat,title=title):
            md += ['\\clearpage','',f'## {title}', '',f'![{title} across all seeds.](report_figures/{file}){{width=100%}}','']
    conditions=list(dict.fromkeys(r['condition'] for r in rows))
    for index,condition in enumerate(conditions):
        active=[p for p in parameters if p.startswith('tau_') and any(
            r['parameter']==p and r['condition']==condition for r in rows)]
        for chunk in range(0,len(active),9):
            file=f'taus_{index}_{chunk}.png'
            plot_parameters(rows,active[chunk:chunk+9],figures/file,condition=condition,title=label(condition)+' — time constants')
            md += ['\\clearpage','',f'## {label(condition)}: time constants','',f'![Active dendritic, SFA and STD time constants; inactive components omitted.](report_figures/{file}){{width=100%}}','']
    for index,name in enumerate(selected):
        active_taus=[p for p in parameters if p.startswith('tau_') and any(
            r['parameter']==p and r['variant']==name for r in rows)]
        panels=[(f'taus_{chunk}',active_taus[chunk:chunk+9]) for chunk in range(0,len(active_taus),9)]
        panels.append(('weights',['W_E','W_I','W_in','W_out','readout_bias','c_E','c_I','a_0']))
        for suffix,params in panels:
            file=f'detail_{index}_{suffix}.png'
            plot_parameters(rows,params,figures/file,variant=name,title=name)
            panel_title = 'weights and adaptation' if suffix == 'weights' else f'time constants (part {int(suffix.split("_")[1]) // 9 + 1})'
            md += ['\\clearpage','',f'## {name}: {panel_title}', '',f'![Per-network means with within-network SD bands.](report_figures/{file}){{width=100%}}','']
        md += ['\\clearpage','',f'## {name}: effective parameters','',
               '| Parameter | Initial mean ± SD | Final mean ± SD | n |','|---|---:|---:|---:|']
        for e in endpoints:
            if e['variant']==name:
                md.append(f"| `{e['parameter']}` | {e['initial_mean']:.4g} ± {e['initial_std']:.3g} | {e['final_mean']:.4g} ± {e['final_std']:.3g} | {e['n']} |")
        md += ['', 'Tau values are in seconds. c is a total SFA budget. n counts active entries within this network; it is not the seed count.', '']
    md += ['\\clearpage','','## Complete tables and reproducibility','',
           '[All parameter trajectories](report_parameter_history.csv) · '
           '[All initial/final parameters](report_parameter_initial_final.csv) · '
           '[Final performance](report_performance.csv) · [Machine-readable report](report_summary.json)', '',
           'The CSV files include every network, even when only selected seeds appear in the detail pages. '
           'Weights and time constants describe the fitted model; they are not independent estimates of biological parameters.', '']
    data['performance']=performance; data['selected_variants']=selected
    (output/'report_summary.json').write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    target.write_text('\n'.join(md).rstrip()+'\n')
    if pdf:
        render_pdf(target,pdf_wrapper)
    print(target)
    return target


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--variants',help='Comma-separated exact names or all; default lowest seed per condition')
    parser.add_argument('--pdf',action='store_true')
    parser.add_argument('--pdf-wrapper',help='Path to installed md2pdf.sh wrapper')
    args=parser.parse_args()
    build_report(args.run,args.output_dir,args.variants,args.pdf,args.pdf_wrapper)


if __name__=='__main__':
    main()
