"""Memory-bounded, version-2 checkpoint statistics for the SRNN run report.

Statistics describe distributions within each network. Population SD (ddof=0)
across neurons/active connections is distinct from uncertainty across seeds.
Only one complete checkpoint (including its serialized optimizer) is loaded at
once; optimizer state is immediately discarded. Never consume partial uploads.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from postprocess import (active_js, effective_a_0, effective_c, effective_taus,
                         effective_W, effective_W_in, effective_W_out)


def split_variant(name: str) -> tuple[str, int | None]:
    match = re.search(r"-seed(\d+)$", name)
    return (name[:match.start()], int(match[1])) if match else (name, None)


def _checkpoint_epoch(path: Path, ckpt: dict) -> int:
    if path.stem == 'init':
        return 0  # Historical init files stored epoch=0, before any training.
    match = re.fullmatch(r'epoch_(\d+)', path.stem)
    if match:
        epoch = int(match[1])
        if int(ckpt['epoch']) != epoch:
            raise ValueError(f'{path.name}: filename and checkpoint epoch disagree')
        return epoch + 1
    epoch = int(ckpt['epoch']) + 1
    if epoch < 0:
        raise ValueError(f'{path.name}: invalid completed epoch {epoch}')
    return epoch


def collect_run(run_dir: Path) -> dict:
    """Collect JSON-ready effective parameters; ``last`` wins duplicate epochs.

    Parameters ending ``_1``, ``_2``, ... identify one-based active timescale
    indices. W_E/I are signed presynaptic-column weights, excluding structural
    zeros. W_relative_change is RMS change / initial RMS over active connections.
    Dale_violations counts wrong-sign active recurrent connections, even when
    Dale enforcement is disabled. No adaptation rows are emitted for padding.
    """
    run_dir = Path(run_dir).resolve()
    paths = ([run_dir / 'init.pt'] if (run_dir / 'init.pt').exists() else [])
    paths += sorted(run_dir.glob('epoch_*.pt'))
    if (run_dir / 'last.pt').exists():
        paths.append(run_dir / 'last.pt')
    if not paths:
        raise ValueError(f'No completed checkpoint files in {run_dir}')
    canonical_config = canonical_names = None
    initial_weights = {}
    variant_settings = []
    by_epoch = {}
    provenance = []
    structural = None
    for path in paths:
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        ms = checkpoint.pop('model_state_dict')
        checkpoint.pop('optimizer_state_dict', None)
        checkpoint.pop('scheduler_state_dict', None)
        cfg = checkpoint['config']
        names = checkpoint.get('variant_names')
        if (checkpoint.get('model_version') != 2 or
                int(ms.get('cell.model_version', -1)) != 2 or
                cfg.get('model', {}).get('model_version') != 2):
            raise ValueError(f'{path.name}: report requires SRNN model version 2')
        if not names or len(names) != ms['cell.W_raw'].shape[0] or len(set(names)) != len(names):
            raise ValueError(f'{path.name}: invalid variant names or network count')
        if canonical_config is None:
            canonical_config = cfg
            canonical_names = names
            for k, name in enumerate(names):
                settings = dict(name=name, skip=bool(ms['cell.skip_flags'][k].item()),
                                dales=bool(ms['cell.dales_mask'][k].item()),
                                per_neuron=bool(ms['cell.per_neuron_mask'][k].item()))
                for side in ('E', 'I'):
                    settings[f'n_a_{side}'] = len(active_js(ms, k, side, 'sfa'))
                    settings[f'n_b_{side}'] = len(active_js(ms, k, side, 'std'))
                variant_settings.append(settings)
        elif names != canonical_names or cfg != canonical_config:
            raise ValueError(f'{path.name}: checkpoint variant/config mismatch')
        structure_keys = ('cell.sparsity_masks', 'cell.dales_signs', 'cell.dales_mask',
                          'cell.skip_flags', 'cell.per_neuron_mask',
                          'cell.sfa_E_mask', 'cell.sfa_I_mask', 'cell.std_E_mask', 'cell.std_I_mask')
        if structural is None:
            structural = {key: ms[key].clone() for key in structure_keys}
        elif any(not torch.equal(ms[key], structural[key]) for key in structure_keys):
            raise ValueError(f'{path.name}: checkpoint structural mask mismatch')
        epoch = _checkpoint_epoch(path, checkpoint)
        record = {'tag': path.stem, 'file': path.name, 'completed_epoch': epoch,
                  'bytes': path.stat().st_size}
        rows = []
        for k, name in enumerate(names):
            condition, seed = split_variant(name)
            def add(parameter, values):
                values = np.asarray(values, dtype=np.float64).ravel()
                if not values.size:
                    return
                if not np.isfinite(values).all():
                    raise ValueError(f'{path.name}: nonfinite {parameter} in {name}')
                rows.append(dict(variant=name, condition=condition, seed=seed,
                                 epoch=epoch, parameter=parameter,
                                 mean=float(values.mean()), std=float(values.std()), n=int(values.size)))
            taus = effective_taus(ms, k)
            add('tau_d', taus['tau_d'])
            for side in ('E', 'I'):
                sfa = active_js(ms, k, side, 'sfa')
                for j in sfa:
                    add(f'tau_a_{side}_{j+1}', taus[f'tau_a_{side}'][:, j])
                if sfa:
                    add(f'c_{side}', effective_c(ms, k, side))
                for j in active_js(ms, k, side, 'std'):
                    for kind in ('rec', 'rel'):
                        key = f'tau_b_{kind}_{side}'
                        add(f'{key}_{j+1}', taus[key][:, j])
            # a_0 shifts the rate nonlinearity, including without adaptation.
            a0 = effective_a_0(ms, k)
            add('a_0', a0)
            W, signs, sparsity = effective_W(ms, k)
            for side, sign in (('E', 1), ('I', -1)):
                columns = signs == sign
                selected = W[:, columns][sparsity[:, columns] != 0]
                add(f'W_{side}', selected)
                add(f'W_abs_{side}', np.abs(selected))
                add(f'a_0_{side}', a0[columns])
            input_weights = effective_W_in(ms, k)
            input_mask = ms.get('cell.W_in_mask', torch.ones(1, W.shape[0], 1))[0].numpy()
            add('W_in', input_weights[np.broadcast_to(input_mask != 0, input_weights.shape)])
            add('W_out', effective_W_out(ms, k))
            add('readout_bias', ms['readout_bias'][k].numpy())
            active = sparsity != 0
            add('Dale_violations', [np.count_nonzero((W * signs[None, :] < 0) & active)])
            if path.stem == 'init':
                initial_weights[k] = W.copy()
            if k in initial_weights:
                base = initial_weights[k][active].astype(np.float64)
                denom = np.linalg.norm(base)
                if denom:
                    add('W_relative_change', [np.linalg.norm(W[active] - base) / denom])
            del W
        prior = by_epoch.get(epoch)
        if prior is not None:
            provenance.append({'file': prior[0]['file'], 'replaced_by': path.name,
                               'reason': 'duplicate completed epoch'})
        by_epoch[epoch] = (record, rows)
        del ms, checkpoint
    ordered = [by_epoch[epoch] for epoch in sorted(by_epoch)]
    result = dict(run_dir=str(run_dir), model_version=2, config=canonical_config,
                  variant_names=canonical_names, variant_settings=variant_settings, checkpoints=[x[0] for x in ordered],
                  rows=[row for x in ordered for row in x[1]], provenance=provenance,
                  statistics='Within-network population SD; structural zeros and inactive timescales excluded.')
    json.dumps(result, allow_nan=False)  # Fail here rather than emitting invalid report JSON.
    return result
