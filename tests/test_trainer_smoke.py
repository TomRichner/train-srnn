"""Both trainers run end to end on the synthetic task and write the run directory."""
from pathlib import Path

import pytest
import torch

from train_srnn.config import compose_config
from train_srnn.data import build_task
from train_srnn.models.factory import build_model
from train_srnn.training import TRAINERS

EXPECTED = {"init.pt", "last.pt", "epoch_000.pt", "training_history.csv",
            "test_history.csv", "progress.json"}


def _run(tmp_path: Path, *overrides: str):
    cfg = compose_config(["task=synthetic", "model=srnn", "model.num_units=16",
                          "model.variants=[srnn-skip,srnn-no-adapt]", "epochs=2",
                          "checkpoint_interval=1", "device=cpu", "compile.enabled=false",
                          "burn_in=0.5", "task.train_trace_max_len=101", "task.window_len=40", "task.seq_len=40",
                          "task.bptt_len=20", f"output_dir={tmp_path}", *overrides])
    torch.manual_seed(cfg.seed)
    task = build_task(cfg)
    data = task.load(None)
    model = build_model(cfg)
    trainer = TRAINERS[cfg.task.trainer](cfg, model, task, data, torch.device("cpu"), tmp_path)
    trainer.fit()
    return cfg, trainer


@pytest.mark.parametrize("trainer", ["continuous", "windowed"])
@pytest.mark.parametrize("closed_loop", [False, True], ids=["tf", "cl"])
def test_fit_writes_run_dir(tmp_path, trainer, closed_loop):
    extra = [f"task.trainer={trainer}"]
    if closed_loop:
        extra += ["closed_loop.enabled=true", "closed_loop.t_warm=10"]
    _run(tmp_path, *extra)
    assert EXPECTED <= {p.name for p in tmp_path.iterdir()}
    history = (tmp_path / "training_history.csv").read_text().splitlines()
    assert history[0].startswith("epoch,variant,train_loss")
    assert len(history) == 1 + 3 * 2                      # header + initialization and 2 epochs x K=2
    assert history[1].split(",")[-2] == "0"
    assert "srnn-skip" in history[1] and "srnn-no-adapt" in history[2]


def test_resume_from_checkpoint(tmp_path):
    cfg, first = _run(tmp_path / "a", "task.trainer=continuous")
    cfg2, second = _run(tmp_path / "b", "task.trainer=continuous", f"init_ckpt={tmp_path / 'a' / 'last.pt'}")
    a = torch.load(tmp_path / "a" / "last.pt", weights_only=False)
    b = torch.load(tmp_path / "b" / "init.pt", weights_only=False)
    for k in a["model_state_dict"]:
        if k != "ic.ic":                                    # burn-in rewrites the IC
            assert torch.equal(a["model_state_dict"][k], b["model_state_dict"][k]), k


def test_single_network_baseline(tmp_path):
    cfg = compose_config(["task=synthetic", "task.trainer=windowed", "model=ltc",
                          "model.num_units=8", "epochs=1", "device=cpu",
                          "compile.enabled=false", "burn_in=0", f"output_dir={tmp_path}"])
    task = build_task(cfg)
    data = task.load(None)
    trainer = TRAINERS["windowed"](cfg, build_model(cfg), task, data, torch.device("cpu"), tmp_path)
    trainer.fit()
    rows = (tmp_path / "training_history.csv").read_text().splitlines()
    assert rows[1].split(",")[1] == "ltc"
