"""Builders and runners shared by ``make_golden.py`` (records snapshots) and the
``test_golden_*`` tests (replay them). Small shapes keep the snapshots small.
"""
from __future__ import annotations

import copy
import pathlib

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parents[2]
GOLDEN_DIR = pathlib.Path(__file__).resolve().parent

B, T, C, N = 2, 40, 4, 16
READOUT = slice(20, T)
BPTT_START = 10
CHUNK = 5
SEG = 5

SINGLE_MODELS = ["lstm", "ltc", "ltc_rk", "ltc_ex", "ctrnn", "node", "ctgru"]
BATCHED_VARIANTS = ["srnn-per-neuron", "srnn-no-adapt", "srnn-e-only-skip",
                    "srnn-echo"]
TRAINER_VARIANTS = ["srnn-no-dales-skip", "srnn-no-adapt-no-dales-skip"]


def make_cfg(model: str, *extra: str):
    from train_srnn.config import compose_config
    overrides = [
        f"model={model}", "task=cheetah100", "seed=0", f"model.num_units={N}",
        f"task.input_size={C}", f"task.output_size={C}",
        f"task.window_len={T}", f"task.bptt_len={T - BPTT_START}",
        f"task.bptt_chunk_len={CHUNK}", f"task.batch_size={B}",
        "task.seq_len=40", "task.stride=40",
        "device=cpu", "compile.enabled=false", "grad_checkpoint=false",
        "grad_checkpoint_segment_len=5", "epochs=1", "warmup_epochs=1",
        "burn_in=0", "burn_in_every=0",
        *extra,
    ]
    return compose_config(overrides)


def build_single(name: str):
    from train_srnn.models.factory import build_model
    torch.manual_seed(0)
    return build_model(make_cfg(name))


def build_batched(variants: list[str]):
    from train_srnn.models.factory import build_model
    torch.manual_seed(0)
    return build_model(make_cfg("srnn"), variants)


def inputs():
    g = torch.Generator().manual_seed(1234)
    x = torch.randn(B, T, C, generator=g)
    ramp = torch.linspace(0.0, 0.6, T).unsqueeze(1).expand(T, C).clone()
    ramp[0] = 0.0
    return x, ramp


FORWARD_MODES = {
    "open": dict(grad_checkpoint=False),
    "open_ckpt": dict(grad_checkpoint=True, grad_checkpoint_segment_len=SEG),
    "closed": dict(grad_checkpoint=False, closed=True),
    "closed_ckpt": dict(grad_checkpoint=True, grad_checkpoint_segment_len=SEG,
                        closed=True),
}


def run_forward(model, x, alpha, mode: str):
    """Forward + backward of a sum-of-squares loss; returns (y, grads)."""
    kw = dict(FORWARD_MODES[mode])
    closed = kw.pop("closed", False)
    m = copy.deepcopy(model)
    m.train()
    y = m(x, readout_idx=READOUT, bptt_start_idx=BPTT_START,
          bptt_chunk_len=CHUNK, alpha_schedule=alpha if closed else None, **kw)
    y.pow(2).sum().backward()
    grads = {n: p.grad.detach().clone() for n, p in m.named_parameters()
             if p.grad is not None}
    return y.detach(), grads


def run_eval(model, x):
    m = copy.deepcopy(model)
    m.eval()
    with torch.no_grad():
        return m(x, readout_idx=T - 1)


def strip_timestamps(csv_text: str) -> str:
    """Drop the trailing timestamp column so CSV text compares run to run."""
    return "\n".join(line.rsplit(",", 1)[0] for line in csv_text.splitlines())


def _windowed_data(rng: np.random.RandomState):
    x = rng.randn(3 * B, T, C).astype(np.float32)
    y = np.roll(x, -1, axis=1).astype(np.float32)
    return x, y


def run_windowed_epoch(closed_loop: bool):
    """One epoch (3 optimizer steps) of the windowed trainer on K=2."""
    from train_srnn.data import Dataset, build_task
    from train_srnn.training import WindowedTrainer

    extra = ["task.trainer=windowed", "task.no_augment=true",
             "task.loss_over_bptt=true"]
    if closed_loop:
        extra += ["closed_loop.enabled=true", "closed_loop.alpha_baseline=0.3",
                  "closed_loop.teacher_forcing_batch_frac=0.34",
                  "closed_loop.t_warm=10"]
    cfg = make_cfg("srnn", *extra)
    model = build_batched(TRAINER_VARIANTS)
    rng = np.random.RandomState(0)
    x, y = _windowed_data(rng)
    data = Dataset(train=(x, y), valid=(x, y), test=(x, y), input_size=C, output_size=C)
    trainer = WindowedTrainer(cfg, model, build_task(cfg), data, torch.device("cpu"),
                              pathlib.Path("/nonexistent"), rng=rng,
                              cl_gen=torch.Generator().manual_seed(7))
    train = trainer.train_epoch(0)
    valid = trainer.evaluate("valid")
    return {
        "train_loss": train.loss, "train_metric": train.metric,
        "valid_loss": valid.loss, "valid_metric": valid.metric,
        "alpha_stats": train.alpha or {},
        "state_dict": {k: v.detach().clone() for k, v in model.state_dict().items()},
    }


def run_continuous_epoch(closed_loop: bool, out_dir: pathlib.Path):
    """One epoch (3 chunks) of the ring trainer on K=2, with its end-of-epoch I/O."""
    from train_srnn.data import Dataset, build_task
    from train_srnn.training import ContinuousTrainer

    extra = ["checkpoint_interval=1", "log_interval=1"]
    if closed_loop:
        extra += ["closed_loop.enabled=true", "closed_loop.alpha_baseline=0.3",
                  "closed_loop.teacher_forcing_batch_frac=0.0",
                  "closed_loop.alpha_rnd_density=0.5",
                  "closed_loop.alpha_rnd_sigma=0.1"]
    cfg = make_cfg("srnn", *extra)
    model = build_batched(TRAINER_VARIANTS)
    rng = np.random.RandomState(0)
    g = torch.Generator().manual_seed(99)
    trace = torch.randn(29, C, generator=g)  # ceil(29 / (B*CHUNK)) = 3 steps
    vx, vy = _windowed_data(rng)
    data = Dataset(train=(vx, vy), valid=(vx, vy), test=(vx, vy), input_size=C,
                   output_size=C, train_trace=trace.numpy())
    trainer = ContinuousTrainer(cfg, model, build_task(cfg), data, torch.device("cpu"),
                                out_dir, rng=rng, cl_gen=torch.Generator().manual_seed(7))
    trainer.run_epoch_and_log(0)
    trainer.checkpoint(0, "last")
    trainer.test(0, "last")
    return {
        "training_history": strip_timestamps((out_dir / "training_history.csv").read_text()),
        "test_history": strip_timestamps((out_dir / "test_history.csv").read_text()),
        "state_dict": {k: v.detach().clone() for k, v in model.state_dict().items()},
    }
