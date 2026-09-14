"""Replay the cell goldens: forward outputs and per-parameter grads must match.

The batched SRNN golden was recorded with gradient hooks that never fired
(``copy.deepcopy`` drops tensor hooks), so its ``*_vec`` rows for variants
without per-neuron parameters and the echo variant's ``W_raw`` carry
gradients. The unified cell cuts those gradients in the forward pass, so
those rows are checked to be exactly zero and every other value to be equal.
"""
from __future__ import annotations

import pytest
import torch

from tests.golden import legacy_api as api


def _load(name):
    path = api.GOLDEN_DIR / f"cell_{name}.pt"
    if not path.exists():
        pytest.skip(f"missing golden {path.name}")
    return torch.load(path)


def _linked_rows(name: str):
    """Per-variant mask of rows whose gradient the new cell cuts, or None."""
    if name.startswith("cell.") and name.endswith("_vec"):
        return torch.tensor(api.BATCHED_PER_NEURON)
    if name == "cell.W_raw":
        return ~torch.tensor(api.BATCHED_ECHO)
    return None


def _check(model, rec, linked: bool):
    sd = {k: v for k, v in rec["state_dict"].items()
          if not (k.startswith("cell._") and k.endswith("_mask"))}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected and set(missing) <= {"cell.per_neuron_mask"}
    x, alpha = rec["x"], rec["alpha"]
    assert torch.equal(api.run_eval(model, x), rec["eval_last"])
    for mode in api.FORWARD_MODES:
        y, grads = api.run_forward(model, x, alpha, mode)
        assert torch.equal(y, rec[mode]["y"]), mode
        assert grads.keys() == rec[mode]["grads"].keys(), mode
        for n, g in grads.items():
            want = rec[mode]["grads"][n]
            keep = _linked_rows(n) if linked else None
            if keep is None:
                assert torch.equal(g, want), f"{mode}: {n}"
            else:
                assert torch.equal(g[keep], want[keep]), f"{mode}: {n} (trained rows)"
                assert torch.count_nonzero(g[~keep]) == 0, f"{mode}: {n} (linked rows)"


@pytest.mark.parametrize("name", api.SINGLE_MODELS)
def test_single_cell_golden(name):
    _check(api.build_single(name), _load(name), linked=False)


def test_batched_srnn_golden():
    _check(api.build_batched(api.BATCHED_VARIANTS), _load("batched"), linked=True)
