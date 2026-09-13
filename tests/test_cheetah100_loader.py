"""cheetah100 loading: channel counts, train-only z-score, AR offset, transient removal."""
import json
from pathlib import Path

import numpy as np
import pytest

from train_srnn import paths
from train_srnn.config import Cheetah100Config
from train_srnn.data.cheetah100 import Cheetah100Task

DATA_DIR = paths.data_dir() / "cheetah100"
HZ = 100.0

pytestmark = pytest.mark.skipif(
    not (DATA_DIR / "train.npz").exists(),
    reason=f"cheetah100 dataset not staged at {DATA_DIR}")


def _load(**kw):
    return Cheetah100Task(Cheetah100Config(**kw)).load(DATA_DIR)


@pytest.mark.parametrize("include_actions, n_chan", [(False, 17), (True, 23)])
def test_shapes(include_actions, n_chan):
    d = _load(include_actions=include_actions)
    x, y = d.train
    assert d.input_size == d.output_size == n_chan
    assert x.shape[1:] == (1500, n_chan) and x.shape == y.shape
    assert d.train_trace.ndim == 2 and d.train_trace.shape[1] == n_chan


def test_truncation_gives_phase_drift():
    trace = _load().train_trace
    assert trace.shape[0] == 118973
    steps = -(-trace.shape[0] // (24 * 250))
    drift = steps * 24 * 250 - trace.shape[0]
    assert drift > 0 and np.gcd(drift, 250) == 1


def test_train_only_zscore():
    d = _load()
    trace = d.train_trace
    assert np.abs(trace.mean(0)).max() < 1e-4
    assert np.abs(trace.std(0) - 1).max() < 1e-4
    va_x, _ = d.valid
    assert np.abs(va_x.reshape(-1, va_x.shape[-1]).mean(0)).max() > 1e-3
    actions = _load(include_actions=True).train_trace[:, 17:]
    assert np.abs(actions.std(0) - 1).max() < 1e-4


def test_autoregressive_offset():
    d = _load()
    x, y = d.train
    assert np.array_equal(x[0, 1:], y[0, :-1])
    assert np.array_equal(x[0], d.train_trace[:1500])


def test_transient_removal_and_manifest():
    raw = np.load(DATA_DIR / "train.npz")["obs"]
    assert abs(raw[0, 8]) < 1.0                     # standstill
    kept = _load(skip_transient_s=0.0, normalize=False, train_trace_max_len=None).train_trace
    dropped = _load(skip_transient_s=10.0, normalize=False, train_trace_max_len=None).train_trace
    assert abs(kept[0, 8]) < 1.0 and dropped[0, 8] > 5.0
    assert kept.shape[0] - dropped.shape[0] == int(10.0 * HZ)
    manifest = json.loads((DATA_DIR / "manifest.json").read_text())
    ref = np.array(manifest["zscore_train_stats"]["obs_std"])
    assert np.allclose(kept.std(0), ref, rtol=1e-3)
