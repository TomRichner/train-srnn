"""Launch an isolated MATLAB-aligned experiment or one-epoch capacity probe."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CONDITIONS = ['srnn-no-adapt', 'srnn-sfa1-std1', 'srnn-sfa3-std2']


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def overrides(args):
    suffix = ('-skip' if args.skip else '') + ('-no-dales' if args.no_dales else '')
    return ['task=cheetah100', 'model=srnn',
            'model.variants=[' + ','.join(c + suffix for c in CONDITIONS) + ']',
            'model.variant_seeds=[' + ','.join(map(str, range(1, args.seeds + 1))) + ']',
            'model.num_units=500', 'model.solver=sra1', 'model.ode_unfolds=4',
            'task.batch_size=24', 'task.bptt_chunk_len=250', 'burn_in=10',
            'epochs=100', 'warmup_epochs=3', 'lr=0.0005', 'cosine_decay=false',
            'grad_clip=1', 'amp=fp32', 'checkpoint_interval=5', 'log_interval=5',
            'device=' + args.device, 'run_name=' + args.run_name,
            'compile.enabled=' + str(not args.no_compile).lower()]



def active_parameter_counts(model):
    """Count optimized scalar entries, excluding structural masks and frozen IC.

    Counts parameter entries, not identifiable degrees of freedom: multiplicative
    gains can be redundant with the parameters they scale.
    """
    cell = model.cell
    reports = {}
    for k, cfg in enumerate(cell.configs):
        def count(name, number, owner=cell):
            parameter = getattr(owner, name, None)
            return int(number) if parameter is not None and parameter.requires_grad else 0
        mask = cell.W_in_mask
        input_entries = cell.N * cell.input_size if mask is None else int((mask != 0).sum()) * cell.input_size
        parts = {
            'recurrent_nonzero_weights': count('W_raw', int((cell.sparsity_masks[k] != 0).sum())) if not cfg.echo else 0,
            'recurrent_gain': count('log_W_raw_gain', 1),
            'input_weights': count('W_in', input_entries),
            'input_gain': count('W_in_gain', 1),
            'readout_weights': count('readout_weight', model.readout_weight[k].numel(), model),
            'readout_bias': count('readout_bias', model.output_size, model),
            'readout_gain': count('W_out_gain', 1, model),
            'setpoint': count('a_0_scalar', 1) + count('a_0_vec', cell.N if cfg.per_neuron else 0),
            'dendritic_time_constant': count('log_tau_d_gain', 1) + count('isp_tau_d_vec', cell.N if cfg.per_neuron else 0),
        }
        for side, n in [('E', cfg.n_E), ('I', cfg.n_I)]:
            A, M = getattr(cfg, 'n_a_' + side), getattr(cfg, 'n_b_' + side)
            parts['SFA_' + side] = (count('log_tau_a_' + side + '_gain', 1) + count('log_c_' + side + '_gain', 1)
                + count('isp_tau_a_' + side + '_vec', n * A if cfg.per_neuron else 0)
                + count('isp_c_' + side + '_vec', n if cfg.per_neuron else 0)) if A else 0
            parts['STD_' + side] = sum(count('log_tau_b_' + kind + '_' + side + '_gain', 1)
                + count('isp_tau_b_' + kind + '_' + side + '_vec', n * M if cfg.per_neuron else 0)
                for kind in ['rec', 'rel']) if M else 0
        reports[model.variant_names[k]] = {'total': sum(parts.values()), 'by_group': parts}
    return reports


def worker(args, cfg):
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from train import resolve_device, check_amp, freeze_params, maybe_compile
    from train_srnn.data import build_task
    from train_srnn.models.factory import build_model
    from train_srnn.training import TRAINERS
    from train_srnn.utils.history import load_checkpoint

    run = Path(cfg.output_dir)
    run.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, run / 'resolved_config.yaml', resolve=True)
    started = time.monotonic()
    metadata = {'status': 'running', 'profile': args.profile,
                'source_sha256': {str(p.relative_to(ROOT)): sha256(p)
                                  for p in sorted(ROOT.rglob('*.py'))
                                  if '.venv' not in p.parts and '.git' not in p.parts},
                'dataset_sha256': {p.name: sha256(p) for p in sorted(Path(cfg.task.data_dir).iterdir())
                                   if p.name in ('train.npz', 'valid.npz', 'test.npz', 'manifest.json')}}
    try:
        revision_file = ROOT / 'SOURCE_REVISION'
        metadata['git_revision'] = (revision_file.read_text().strip() if revision_file.exists() else
            subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        device = resolve_device(cfg.device)
        check_amp(cfg, device)
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        task = build_task(cfg)
        data = task.load(Path(cfg.task.data_dir))
        task.validate(data)
        model = build_model(cfg).to(device)
        freeze_params(model, list(cfg.freeze_params))
        torch.save({k: v.detach().cpu() for k, v in model.cell.effective_params().items()},
                   run / 'initial_effective_params.pt')
        metadata['parameter_elements'] = {n: p.numel() for n, p in model.named_parameters() if p.requires_grad}
        metadata['parameter_count_note'] = 'Allocated trainable elements; includes masked/inactive padding.'
        metadata['active_optimized_parameter_counts'] = active_parameter_counts(model)
        metadata['active_count_note'] = 'Nonzero recurrent mask, input mask, active adaptation slots/gains, optional per-neuron entries; excludes IC frozen by this protocol. Counts optimized entries, not identifiable degrees of freedom.'
        model = maybe_compile(cfg, model, device)
        trainer = TRAINERS[cfg.task.trainer](cfg, model, task, data, device, run)
        steps = trainer.steps_per_epoch()
        if steps != 20:
            raise ValueError(f'Protocol requires 20 steps per epoch; dataset/config gives {steps}')
        metadata['steps_per_epoch'] = steps
        metadata['variant_names'] = trainer.names
        # Fail immediately on bad losses/gradients, including after optimizer updates.
        original_step = trainer.optimizer_step
        def checked_step(loss):
            if not torch.isfinite(loss).all():
                raise FloatingPointError('Non-finite training loss')
            original_step(loss)
            checks = [torch.isfinite(p).all() for p in model.parameters()]
            checks.extend(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            if not torch.stack(checks).all():
                raise FloatingPointError('Non-finite parameter or gradient')
        trainer.optimizer_step = checked_step
        original_evaluate = trainer.evaluate
        def checked_evaluate(split):
            result = original_evaluate(split)
            if not np.isfinite(result.loss).all() or not np.isfinite(result.metric).all():
                raise FloatingPointError(f'Non-finite {split} evaluation')
            return result
        trainer.evaluate = checked_evaluate
        if args.profile:
            trainer.refresh_ic()
            if cfg.freeze_ic_after_burnin:
                model.ic.ic.requires_grad_(False)
            trainer.checkpoint(0, 'init')
            initial = trainer.evaluate('valid')
            model.train()
            train_stats = trainer.train_epoch(0)
            valid = trainer.evaluate('valid')
            test = trainer.test(0, 'profile')
            trainer.checkpoint(0, 'last')
            metadata['profile_losses'] = {'initial_valid': initial.loss, 'train': train_stats.loss,
                                           'valid': valid.loss, 'test': test.loss}
            if not all(np.isfinite(v).all() for v in metadata['profile_losses'].values()):
                raise FloatingPointError('Non-finite profile evaluation')
        else:
            trainer.fit()
        load_checkpoint(run / 'last.pt', model=model, device=str(device))
        final_effective = {k: v.detach().cpu() for k, v in model.cell.effective_params().items()}
        torch.save(final_effective, run / 'effective_params.pt')
        internal_dt = float(cfg.model.h) / int(cfg.model.ode_unfolds)
        tau_diagnostic = {}
        for k, condition in enumerate(model.cell.configs):
            values = {'tau_d': float(final_effective['tau_d'][k].min())}
            for side in ('E', 'I'):
                for family, count in [('tau_a', getattr(condition, 'n_a_' + side)),
                                      ('tau_b_rec', getattr(condition, 'n_b_' + side)),
                                      ('tau_b_rel', getattr(condition, 'n_b_' + side))]:
                    if count:
                        key = family + '_' + side
                        values[key] = float(final_effective[key][k, :, :count].min())
            tau_diagnostic[trainer.names[k]] = {
                'minimum_active_tau_seconds': values,
                'minimum_active_tau_over_internal_dt': {key: value / internal_dt for key, value in values.items()}}
        metadata['internal_dt_seconds'] = internal_dt
        metadata['final_time_constant_diagnostics'] = tau_diagnostic
        metadata['optimizer_steps'] = steps * (1 if args.profile else cfg.epochs)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
            metadata['cuda_peak_allocated_bytes'] = torch.cuda.max_memory_allocated(device)
            metadata['cuda_peak_reserved_bytes'] = torch.cuda.max_memory_reserved(device)
        metadata['status'] = 'complete'
    except BaseException as exc:
        metadata.update(status='failed', error=repr(exc))
        raise
    finally:
        metadata['wall_seconds'] = time.monotonic() - started
        (run / 'experiment_metadata.json').write_text(json.dumps(metadata, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--run-name', default=None)
    parser.add_argument('--profile', action='store_true', help='One full epoch + evaluation/checkpoint; retain 100-epoch scheduler')
    parser.add_argument('--skip', action='store_true')
    parser.add_argument('--no-dales', action='store_true')
    parser.add_argument('--no-compile', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error('--seeds must be positive')
    args.run_name = args.run_name or 'matlab-aligned-' + time.strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:6]
    from train_srnn.config import compose_config
    cfg = compose_config(overrides(args))
    if args.dry_run:
        from omegaconf import OmegaConf
        print(OmegaConf.to_yaml(cfg, resolve=True))
        return
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    if args.worker:
        worker(args, cfg)
        return
    run = Path(cfg.output_dir)
    run.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, str(Path(__file__).resolve()), '--worker', '--run-name', args.run_name,
               '--seeds', str(args.seeds), '--device', args.device]
    for flag in ('profile', 'skip', 'no_dales', 'no_compile'):
        if getattr(args, flag):
            command.append('--' + flag.replace('_', '-'))
    peak, total = 0, 0
    samples = 0
    with open(run / 'training.log', 'w') as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        while process.poll() is None:
            try:
                output = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'], text=True, timeout=5)
                for line in output.splitlines():
                    used, capacity = map(int, line.split(','))
                    peak, total = max(peak, used), max(total, capacity)
                samples += 1
            except (OSError, subprocess.SubprocessError, ValueError):
                pass
            time.sleep(1)
    (run / 'gpu_memory.json').write_text(json.dumps({'sampled_peak_total_used_mib': peak if samples else None,
        'device_total_mib': total if samples else None, 'sample_count': samples,
        'within_80_percent': peak / total < .8 if total else None,
        'note': 'One-second whole-device samples, including other processes; transient peaks may be missed.',
        'exit_code': process.returncode}, indent=2))
    print(run, flush=True)
    raise SystemExit(process.returncode)


if __name__ == '__main__':
    main()
