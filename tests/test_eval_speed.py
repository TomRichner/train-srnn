"""scripts/eval_speed.py: segmented evaluation keeps predictions aligned and in order."""
import sys
from pathlib import Path

import numpy as np
import torch

from train_srnn.config import compose_config
from train_srnn.models.factory import build_model

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import eval_speed  # noqa: E402


def _model():
    cfg = compose_config(["task=synthetic", "model=srnn", "model.num_units=12",
                          "model.variants=[srnn-sfa1-std1,srnn-no-adapt]", "device=cpu"])
    torch.manual_seed(0)
    return build_model(cfg).eval()


def _continuous(model, x):
    state = model.initial_state(1)
    res = model.unroll(x[None, :-1], state, hoisted=model.cell.hoist())
    pred = model.apply_readout(res.hidden, x[None, :-1])
    return ((pred[:, 0] - x[None, 1:]) ** 2).mean(-1)            # (K, T-1)


def test_segments_are_aligned_and_ordered():
    model = _model()
    x = torch.tensor(np.random.default_rng(0).standard_normal((230, 4)), dtype=torch.float32)
    warmup, seg = 20, 50
    err = eval_speed.sample_errors(model, x, warmup, seg_len=seg, chunk=17)
    assert err.shape == (2, 229 - warmup)
    full = _continuous(model, x)
    # The first segment starts from the IC at sample 0, exactly like a continuous pass.
    torch.testing.assert_close(err[:, :seg], full[:, warmup:warmup + seg])
    # One segment covering everything is the continuous pass itself.
    whole = eval_speed.sample_errors(model, x, warmup, seg_len=10_000)
    torch.testing.assert_close(whole, full[:, warmup:])


def test_summary_bins_and_persistence():
    rate = np.concatenate([np.full(60, 0.5), np.full(61, 2.0)])
    err = np.ones((2, 120 - 10))
    persistence = np.linspace(1, 2, 120)
    s = eval_speed.summarize(err, persistence, rate, warmup=10)
    assert s["samples"] == 110 and len(s["mse"]) == 2
    assert sum(b["samples"] for b in s["bins"]) == 110
