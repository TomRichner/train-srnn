"""Input-output trace tasks: inputs at t map to target channels at t + target_shift."""
from pathlib import Path

import numpy as np
import torch

from train_srnn.config import compose_config
from train_srnn.data import build_task
from train_srnn.models.factory import build_model
from train_srnn.training import TRAINERS


def _dataset(root: Path):
    rng = np.random.default_rng(0)
    d = root / "io_demo"
    d.mkdir(parents=True)
    for split, n in (("train", 997), ("valid", 400), ("test", 400)):
        x = rng.standard_normal((n, 2)).cumsum(0)
        obs = np.concatenate([x, 2 * x[:, :1]], axis=1).astype(np.float32)
        np.savez(d / f"{split}.npz", obs=obs, act=np.zeros((n - 1, 1), np.float32))
    return root


def _cfg(root, *extra):
    return compose_config([
        "task=cheetah100", "task.dataset=io_demo", f"task.data_dir={root}/io_demo",
        "task.skip_transient_s=0", "task.train_trace_max_len=null", "task.input_size=2",
        "task.output_size=1", "task.input_channels=[0,1]", "task.target_channels=[2]",
        "task.seq_len=100", "task.stride=50", "task.window_len=100", "task.bptt_len=50",
        "task.bptt_chunk_len=10", "task.batch_size=4", "model=srnn", "model.num_units=12",
        "model.variants=[srnn-sfa1-std1]", "device=cpu", "compile.enabled=false", "epochs=1",
        "burn_in=0.1", "checkpoint_interval=1", *extra])


def test_io_windows_and_trace_alignment(tmp_path):
    root = _dataset(tmp_path)
    for shift in (0, 3):
        cfg = _cfg(root, f"task.target_shift={shift}")
        data = build_task(cfg).load(Path(cfg.task.data_dir))
        (xw, yw) = data.train
        assert xw.shape[-1] == 2 and yw.shape == xw.shape[:2]
        assert data.input_size == 2 and data.output_size == 1 and data.target_shift == shift
        # Target channel is 2 x channel 0 before z-scoring, so after z-scoring they coincide.
        np.testing.assert_allclose(yw[0, :50], xw[0, shift:shift + 50, 0], atol=1e-5)
        np.testing.assert_allclose(data.train_target[:, 0], data.train_trace[:, 0], atol=1e-5)


def test_ring_trainer_uses_the_target_trace(tmp_path):
    root = _dataset(tmp_path)
    cfg = _cfg(root, "task.target_shift=0", f"output_dir={tmp_path / 'run'}")
    task = build_task(cfg)
    data = task.load(Path(cfg.task.data_dir))
    trainer = TRAINERS["continuous"](cfg, build_model(cfg), task, data, torch.device("cpu"),
                                     tmp_path / "run")
    x, y = trainer._gather()
    assert x.shape[-1] == 2 and y.shape == x.shape[:2]
    torch.testing.assert_close(y, x[..., 0], atol=1e-5, rtol=0)
    trainer.fit()
    assert (tmp_path / "run" / "last.pt").exists()
