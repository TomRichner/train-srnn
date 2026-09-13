"""Replay the trainer goldens: 3 optimizer steps must reproduce losses and weights."""
from __future__ import annotations

import pathlib

import pytest
import torch

from tests.golden import legacy_api as api


def _load(name):
    path = api.GOLDEN_DIR / f"{name}.pt"
    if not path.exists():
        pytest.skip(f"missing golden {path.name}")
    return torch.load(path)


def _assert_state_close(got, want):
    assert got.keys() == want.keys()
    for k in want:
        assert torch.allclose(got[k], want[k], atol=1e-6, rtol=1e-5), k


@pytest.mark.parametrize("closed_loop", [False, True], ids=["tf", "cl"])
def test_windowed_trainer_golden(closed_loop):
    want = _load(f"trainer_windowed_{'cl' if closed_loop else 'tf'}")
    got = api.run_windowed_epoch(closed_loop)
    for key in ("train_loss", "train_metric", "valid_loss", "valid_metric"):
        assert got[key] == pytest.approx(want[key], rel=1e-5), key
    assert got["alpha_stats"] == pytest.approx(want["alpha_stats"])
    _assert_state_close(got["state_dict"], want["state_dict"])


@pytest.mark.parametrize("closed_loop", [False, True], ids=["tf", "cl"])
def test_continuous_trainer_golden(closed_loop, tmp_path: pathlib.Path):
    want = _load(f"trainer_continuous_{'cl' if closed_loop else 'tf'}")
    got = api.run_continuous_epoch(closed_loop, tmp_path)
    assert got["training_history"] == want["training_history"]
    assert got["test_history"] == want["test_history"]
    _assert_state_close(got["state_dict"], want["state_dict"])
