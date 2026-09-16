"""Checkpointing must run in the ring trainer and preserve its optimizer/state path."""
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.utils.checkpoint as checkpoint_module

from train_srnn.config import compose_config
from train_srnn.data import build_task
from train_srnn.models.factory import build_model
from train_srnn.training import continuous


@pytest.mark.parametrize('closed_loop', [False, True], ids=['teacher_forcing', 'closed_loop'])
def test_continuous_checkpoint_invoked_and_equivalent(tmp_path, closed_loop, monkeypatch):
    original_checkpoint = checkpoint_module.checkpoint
    with patch.object(checkpoint_module, 'checkpoint', wraps=original_checkpoint) as spy:
        # Support both `import checkpoint` and direct function-import styles.
        for name, value in vars(continuous).copy().items():
            if value is original_checkpoint:
                monkeypatch.setattr(continuous, name, spy)
        results = []
        for enabled in (False, True):
            spy.reset_mock()
            cfg = compose_config([
                'task=synthetic', 'task.trainer=continuous', 'model=srnn',
                'model.num_units=7', 'model.ode_unfolds=1',
                'model.variants=[srnn-no-adapt,srnn-sfa1-std1,srnn-sfa3-std2-skip]',
                'task.batch_size=2', 'task.bptt_chunk_len=5', 'epochs=2',
                'compile.enabled=false', 'device=cpu', 'burn_in=0',
                f'grad_checkpoint={str(enabled).lower()}', 'grad_checkpoint_segment_len=2',
                f'closed_loop.enabled={str(closed_loop).lower()}',
                'closed_loop.alpha_baseline=0.4',
                f'output_dir={tmp_path / str(enabled)}',
            ])
            task = build_task(cfg)
            data = task.load(None)
            data.train_trace = data.train_trace[:7]  # one chunk/epoch, wrap on epoch two
            model = build_model(cfg)
            with torch.no_grad():
                model.ic.ic.copy_(model.cell.init_state(1).squeeze(1))
            trainer = continuous.ContinuousTrainer(cfg, model, task, data,
                                                   torch.device('cpu'), tmp_path / str(enabled))
            logits, snapshots = [], []
            loss_and_metrics = trainer.loss_and_metrics

            def capture(output, target):
                logits.append(output.detach().clone())
                return loss_and_metrics(output, target)

            trainer.loss_and_metrics = capture
            for epoch in range(2):
                stats = trainer.train_epoch(epoch)
                snapshots.append(dict(
                    loss=stats.loss, metric=stats.metric,
                    state=trainer.state.clone(),
                    y_prev=None if trainer.y_prev is None else trainer.y_prev.clone(),
                    positions=trainer.positions.clone(),
                    parameters={n: p.detach().clone() for n, p in model.named_parameters()},
                    gradients={n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None},
                ))
            assert trainer.optimizer_steps == 2
            if enabled:
                assert spy.call_count > 0, 'grad_checkpoint=True must invoke activation checkpointing in the ring loop'
                assert all(call.kwargs.get('use_reentrant') is False for call in spy.call_args_list)
            else:
                assert spy.call_count == 0
            results.append((logits, snapshots))
    for a, b in zip(results[0][0], results[1][0]):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-7)
    for a, b in zip(results[0][1], results[1][1]):
        for key in ('loss', 'metric'):
            np.testing.assert_allclose(a[key], b[key], rtol=1e-5, atol=1e-7)
        for key in ('state', 'positions', 'y_prev'):
            if a[key] is not None:
                torch.testing.assert_close(a[key], b[key], rtol=1e-5, atol=1e-7)
        for group in ('parameters', 'gradients'):
            assert a[group].keys() == b[group].keys()
            for name in a[group]:
                torch.testing.assert_close(a[group][name], b[group][name], rtol=1e-4, atol=1e-7,
                                           msg=f'{group}: {name}')
