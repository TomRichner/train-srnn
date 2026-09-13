"""Task registry, the synthetic trace task, and batch wrapping."""
import numpy as np
import pytest
import torch

from train_srnn.config import SyntheticConfig, TASK_CONFIGS, compose_config
from train_srnn.data import TASKS, build_task
from train_srnn.data.synthetic import SyntheticTraceTask


def test_every_task_config_has_a_registered_task():
    for name in TASK_CONFIGS:
        assert name in TASKS, name


def test_unknown_task_name_lists_options():
    with pytest.raises(KeyError, match="Available"):
        TASKS["nope"]


def test_synthetic_trace_shapes_and_normalisation():
    d = SyntheticTraceTask(SyntheticConfig()).load(None)
    assert d.train_trace.shape == (3989, 4)
    assert np.abs(d.train_trace.mean(0)).max() < 1e-4
    x, y = d.train
    assert x.shape[1:] == (100, 4) and x.shape == y.shape
    assert np.array_equal(x[0, 1:], y[0, :-1])


def test_build_task_and_batches():
    cfg = compose_config(["task=synthetic"])
    task = build_task(cfg)
    d = task.load(None)
    task.validate(d)
    rng = np.random.RandomState(0)
    b = task.train_batch(*[a[:2] for a in d.train], rng)
    assert b.x.shape == (2, 100, 4) and b.readout_idx == slice(50, 100)
    assert b.y.shape == (2, 50, 4) and b.bptt_start_idx == 50
    e = task.eval_batch(*[a[:2] for a in d.valid])
    assert e.y.shape == (2, 50, 4) and e.bptt_start_idx is None


def test_loss_and_metric_regression():
    task = build_task(compose_config(["task=synthetic"]))
    logits = torch.zeros(2, 5, 4)
    y = torch.ones(2, 5, 4)
    assert task.loss(logits, y, task.criterion()).item() == pytest.approx(1.0)
    assert task.metric(logits, y) == pytest.approx(1.0)


def test_loss_and_metric_classification():
    task = build_task(compose_config(["task=smnist"]))
    logits = torch.tensor([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    y = torch.tensor([0, 0])
    assert task.metric(logits, y) == pytest.approx(0.5)
    assert task.loss(logits, y, task.criterion()).item() > 0
