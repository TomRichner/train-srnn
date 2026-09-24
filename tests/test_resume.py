"""resume=true continues a run bitwise: interrupted + resumed == uninterrupted, on CPU."""
import csv
from pathlib import Path

import numpy as np
import pytest
import torch

from train_srnn.config import compose_config
from train_srnn.data import build_task
from train_srnn.models.factory import build_model
from train_srnn.training import TRAINERS

EPOCHS = 4


class Interrupt(Exception):
    pass


def _trainer(run_dir: Path, *overrides: str):
    """Build a trainer the way train.py's main does (seeded globals, fresh model)."""
    cfg = compose_config(["task=synthetic", "model=srnn", "model.num_units=16",
                          "model.variants=[srnn-skip,srnn-no-adapt]", f"epochs={EPOCHS}",
                          "checkpoint_interval=1", "device=cpu", "compile.enabled=false",
                          "burn_in=0.5", "burn_in_every=2", "task.train_trace_max_len=101",
                          "task.window_len=40", "task.seq_len=40", "task.bptt_len=20",
                          f"output_dir={run_dir}", *overrides])
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    task = build_task(cfg)
    data = task.load(None)
    model = build_model(cfg)
    return TRAINERS[cfg.task.trainer](cfg, model, task, data, torch.device("cpu"), run_dir)


def _rows(path: Path):
    with open(path, newline="") as f:
        return [{k: v for k, v in r.items() if k != "timestamp"} for r in csv.DictReader(f)]


def _assert_same_run(a: Path, b: Path):
    assert _rows(a / "training_history.csv") == _rows(b / "training_history.csv")
    assert _rows(a / "test_history.csv") == _rows(b / "test_history.csv")
    names = sorted(p.name for p in a.glob("*.pt"))
    assert names == sorted(p.name for p in b.glob("*.pt"))
    for name in names:
        x = torch.load(a / name, weights_only=False)
        y = torch.load(b / name, weights_only=False)
        for key, value in x["model_state_dict"].items():
            assert torch.equal(value, y["model_state_dict"][key]), (name, key)
        assert x["trainer_state"]["optimizer_steps"] == y["trainer_state"]["optimizer_steps"]


def _interrupt_in(monkeypatch, cls, method: str, epoch: int):
    original = getattr(cls, method)

    def failing(self, *args, **kwargs):
        if (args[0] if args else kwargs.get("epoch")) == epoch:
            raise Interrupt(f"{method} at epoch {epoch}")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(cls, method, failing)


CASES = [
    pytest.param("continuous", False, id="continuous-tf"),
    pytest.param("continuous", True, id="continuous-cl"),
    pytest.param("windowed", False, id="windowed-tf"),
    pytest.param("windowed", True, id="windowed-cl"),
]


def _extra(trainer: str, closed_loop: bool) -> list[str]:
    extra = [f"task.trainer={trainer}"]
    if closed_loop:
        extra += ["closed_loop.enabled=true", "closed_loop.t_warm=10"]
    return extra


@pytest.mark.parametrize("trainer_name,closed_loop", CASES)
@pytest.mark.parametrize("where", ["train_epoch", "checkpoint"])
def test_interrupted_and_resumed_equals_uninterrupted(tmp_path, monkeypatch, trainer_name,
                                                      closed_loop, where):
    extra = _extra(trainer_name, closed_loop)
    _trainer(tmp_path / "straight", *extra).fit()

    resumed = tmp_path / "resumed"
    with monkeypatch.context() as patch:
        cls = TRAINERS[trainer_name]
        # "train_epoch": killed mid-epoch 2. "checkpoint": killed after epoch 2's history
        # and test rows were written but before its checkpoint, so they must be dropped.
        _interrupt_in(patch, cls, where, 2)
        with pytest.raises(Interrupt):
            _trainer(resumed, *extra, "resume=true").fit()
    assert not (resumed / "last.pt").exists()
    _trainer(resumed, *extra, "resume=true").fit()
    _assert_same_run(tmp_path / "straight", resumed)


def test_resume_on_empty_dir_starts_fresh_and_completed_run_is_left_alone(tmp_path):
    fresh = tmp_path / "fresh"
    _trainer(fresh, "resume=true").fit()
    _trainer(tmp_path / "plain").fit()
    _assert_same_run(tmp_path / "plain", fresh)
    before = (fresh / "training_history.csv").read_text()
    _trainer(fresh, "resume=true").fit()
    assert (fresh / "training_history.csv").read_text() == before


def test_resume_rejects_init_ckpt(tmp_path):
    trainer = _trainer(tmp_path, "resume=true", f"init_ckpt={tmp_path / 'x.pt'}")
    with pytest.raises(ValueError, match="init_ckpt"):
        trainer.fit()
