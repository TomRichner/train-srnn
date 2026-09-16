"""Paired three-condition validation-learning summary; require a complete step grid."""
from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
import re

import numpy as np

CONDITIONS = ('no-adapt', 'sfa1-std1', 'sfa3-std2')
LABELS = {'no-adapt': 'No adaptation', 'sfa1-std1': 'One SFA + one STD timescale',
          'sfa3-std2': 'Three SFA + two STD timescales'}


def paired_statistics(difference, rng):
    d = np.asarray(difference, dtype=float)
    n = len(d)
    boot = d[rng.integers(n, size=(2000, n))].mean(axis=1)
    observed = abs(d.mean())
    if n <= 20:
        extreme = total = 0
        for signs in itertools.product((-1, 1), repeat=n):
            extreme += abs(np.dot(signs, d) / n) >= observed - 1e-15
            total += 1
        p = extreme / total
        method = 'exact paired sign-flip'
    else:
        extreme = 0
        for _ in range(100):
            means = (rng.choice((-1, 1), size=(1000, n)) * d).mean(axis=1)
            extreme += int(np.count_nonzero(np.abs(means) >= observed - 1e-15))
        p = (extreme + 1) / 100001
        method = '100000 Monte Carlo paired sign-flips, plus-one correction'
    sd = d.std(ddof=1) if n > 1 else 0
    return {'mean_difference': float(d.mean()), 'bootstrap_95_percent_interval': np.quantile(boot, [.025, .975]).tolist(),
            'paired_dz': float(d.mean() / sd) if sd > 0 else None,
            'two_sided_p_unadjusted': p, 'method': method, 'paired_seed_differences': d.tolist()}


def read_curves(run, end_step):
    curves = {}
    with open(run / 'training_history.csv', newline='') as stream:
        for row in csv.DictReader(stream):
            if 'optimizer_step' not in row:
                raise ValueError('History requires explicit optimizer_step and initial validation at step 0')
            loss = float(row['valid_loss'])
            if np.isnan(loss):
                continue  # non-evaluation training rows
            if not np.isfinite(loss) or loss <= 0:
                raise ValueError('Validation loss must be finite and positive')
            variant = row['variant']
            condition = next((c for c in CONDITIONS if c in variant), None)
            seed = re.search(r'(?:^|-)seed(\d+)(?:-|$)', variant)
            if condition is None or seed is None:
                raise ValueError(f'Cannot identify condition/seed: {variant}')
            key = (condition, int(seed.group(1)))
            step = int(row['optimizer_step'])
            values = curves.setdefault(key, {})
            if step in values and values[step] != loss:
                raise ValueError(f'Conflicting history at {key}, step {step}')
            values[step] = loss
    expected = set(range(0, end_step + 1, 100))
    seeds = sorted({seed for _, seed in curves})
    for condition in CONDITIONS:
        for seed in seeds:
            values = curves.get((condition, seed), {})
            if set(values) != expected:
                raise ValueError(f'Incomplete/unexpected validation grid for {(condition, seed)}: {sorted(values)}')
    if not seeds:
        raise ValueError('No validation curves')
    steps = np.array(sorted(expected))
    arrays = {c: np.array([[curves[c, seed][s] for s in steps] for seed in seeds]) for c in CONDITIONS}
    return steps, seeds, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    steps, seeds, arrays = read_curves(run, 2000)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 3))
    scores = {}
    for condition, color in zip(CONDITIONS, ('#555555', '#0072B2', '#DAA520')):
        values = arrays[condition]
        scores[condition] = np.trapezoid(np.log(values), x=steps, axis=1) / 2000 if hasattr(np, 'trapezoid') else np.trapz(np.log(values), x=steps, axis=1) / 2000
        for curve in values:
            ax.plot(steps, curve, color=color, alpha=.22, lw=.8)
        ax.plot(steps, values.mean(axis=0), color=color, lw=2, label=LABELS[condition])
    ax.set(xlabel='Optimizer step', ylabel='Validation mean squared error', yscale='log')
    ax.legend(frameon=False, fontsize=7.5, loc='lower left')
    fig.tight_layout()
    fig.savefig(run / 'matlab_aligned_learning.svg')
    fig.savefig(run / 'matlab_aligned_learning.png', dpi=300)
    plt.close(fig)
    rng = np.random.default_rng(20260915)
    report = {'seeds': seeds, 'preliminary': len(seeds) <= 3,
              'estimand': 'Trapezoidal integral of natural-log validation loss over optimizer steps 0..2000, divided by 2000; lower is better.',
              'uncertainty_note': 'Paired bootstrap across seeds; intervals are descriptive with three seeds. P values are unadjusted for three comparisons.',
              'scores_by_condition_in_seed_order': {c: s.tolist() for c, s in scores.items()}, 'comparisons': {}}
    report['condition_mean_AULC'] = {}
    condition_rng = np.random.default_rng(20260916)
    for condition in CONDITIONS:
        values = scores[condition]
        boot = values[condition_rng.integers(len(values), size=(2000, len(values)))].mean(axis=1)
        report['condition_mean_AULC'][condition] = {
            'label': LABELS[condition], 'mean': float(values.mean()),
            'bootstrap_95_percent_interval': np.quantile(boot, [.025, .975]).tolist()}
    for left, right in itertools.combinations(CONDITIONS, 2):
        report['comparisons'][left + ' minus ' + right] = paired_statistics(scores[left] - scores[right], rng)
    test = {}
    test_mae = {}
    with open(run / 'test_history.csv', newline='') as stream:
        for row in csv.DictReader(stream):
            if row['tag'] == 'last':
                loss = float(row['test_loss'])
                if not np.isfinite(loss):
                    raise ValueError('Non-finite final test loss')
                metric = float(row['test_metric'])
                if not np.isfinite(metric):
                    raise ValueError('Non-finite final test MAE')
                test[row['variant']] = loss
                test_mae[row['variant']] = metric
    final_pairs = set()
    for variant in test:
        condition = next((c for c in CONDITIONS if c in variant), None)
        seed = re.search(r'(?:^|-)seed(\d+)(?:-|$)', variant)
        if condition is not None and seed is not None:
            final_pairs.add((condition, int(seed.group(1))))
    if final_pairs != set(itertools.product(CONDITIONS, seeds)):
        raise ValueError('Final test history must contain every condition and paired seed')
    report['final_test_loss_by_variant'] = test
    report['final_test_MAE_by_variant'] = test_mae
    report['final_test_condition_means'] = {
        condition: {'MSE': float(np.mean([value for name, value in test.items() if condition in name])),
                    'MAE': float(np.mean([value for name, value in test_mae.items() if condition in name]))}
        for condition in CONDITIONS}
    for filename in ('experiment_metadata.json', 'gpu_memory.json'):
        path = run / filename
        if path.exists():
            report[filename.removesuffix('.json')] = json.loads(path.read_text())
    (run / 'matlab_aligned_summary.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print(run / 'matlab_aligned_summary.json')


if __name__ == '__main__':
    main()
