"""Replay the cell goldens: forward outputs and per-parameter grads must match bitwise."""
from __future__ import annotations

import pytest
import torch

from tests.golden import harness as api


def _load(name):
    path = api.GOLDEN_DIR / f"cell_{name}.pt"
    if not path.exists():
        pytest.skip(f"missing golden {path.name}")
    return torch.load(path)


def _check(model, rec):
    model.load_state_dict(rec["state_dict"])
    x, alpha = rec["x"], rec["alpha"]
    assert torch.equal(api.run_eval(model, x), rec["eval_last"])
    for mode in api.FORWARD_MODES:
        y, grads = api.run_forward(model, x, alpha, mode)
        assert torch.equal(y, rec[mode]["y"]), mode
        assert grads.keys() == rec[mode]["grads"].keys(), mode
        for n, g in grads.items():
            assert torch.equal(g, rec[mode]["grads"][n]), f"{mode}: {n}"


@pytest.mark.parametrize("name", api.SINGLE_MODELS)
def test_single_cell_golden(name):
    _check(api.build_single(name), _load(name))


def test_batched_srnn_golden():
    _check(api.build_batched(api.BATCHED_VARIANTS), _load("batched"))
