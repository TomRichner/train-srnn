"""Builders and runners against the pre-refactor API.

Shared by ``make_golden.py`` (which records snapshots) and the
``test_golden_*`` tests (which replay them). Small shapes keep the
snapshots a few hundred KB.
"""
from __future__ import annotations

import copy
import pathlib

import numpy as np
import torch
from hydra import compose, initialize_config_dir

REPO = pathlib.Path(__file__).resolve().parents[2]
GOLDEN_DIR = pathlib.Path(__file__).resolve().parent

B, T, C, N = 2, 40, 4, 16
READOUT = slice(20, T)
BPTT_START = 10
CHUNK = 5
SEG = 5

SINGLE_MODELS = ["lstm", "ltc", "ltc_rk", "ltc_ex", "ctrnn", "node", "ctgru",
                 "srnn", "srnn_e_only"]
BATCHED_VARIANTS = ["srnn-per-neuron", "srnn-no-adapt", "srnn-e-only-skip",
                    "srnn-echo"]
TRAINER_VARIANTS = ["srnn-no-dales-skip", "srnn-no-adapt-no-dales-skip"]


def make_cfg(model: str, *extra: str):
    overrides = [
        f"model={model}", "task=cheetah100", "seed=0", f"size={N}",
        f"task.input_size={C}", f"task.output_size={C}",
        f"task.window_len={T}", f"task.bptt_len={T - BPTT_START}",
        f"task.bptt_chunk_len={CHUNK}", f"task.batch_size={B}",
        "task.seq_len=40", "task.stride=40",
        "device=cpu", "compile=false", "grad_checkpoint=false",
        "grad_checkpoint_segment_len=5", "epochs=1", "warmup_epochs=1",
        "burn_in=0", "burn_in_every=0",
        *extra,
    ]
    with initialize_config_dir(config_dir=str(REPO / "conf"), version_base=None):
        return compose("config", overrides=overrides)


def build_single(name: str):
    from train_srnn.models.factory import build_model
    torch.manual_seed(0)
    return build_model(make_cfg(name))


def build_batched(variants: list[str]):
    from train_srnn.models.factory import build_batched_model
    torch.manual_seed(0)
    return build_batched_model(make_cfg("srnn"), variants)


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
    """One epoch (3 optimizer steps) of the old windowed trainer on K=2."""
    import train  # root train.py
    from train_srnn.utils.lr_schedule import WarmupHoldCosineSchedule

    extra = ["task.continuous_train=false", "task.no_augment=true",
             "task.loss_over_bptt=true"]
    if closed_loop:
        extra += ["closed_loop.enabled=true", "closed_loop.alpha_baseline=0.3",
                  "closed_loop.teacher_forcing_batch_frac=0.34",
                  "closed_loop.t_warm=10"]
    cfg = make_cfg("srnn", *extra)
    model = build_batched(TRAINER_VARIANTS)
    rng = np.random.RandomState(0)
    x, y = _windowed_data(rng)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = WarmupHoldCosineSchedule(opt, total_steps=3, max_lr=cfg.lr,
                                     warmup_frac=0.2, cosine_decay=False)
    cl_cfg = train._build_closed_loop_cfg(cfg)
    gen = torch.Generator().manual_seed(7)
    model.train()
    loss, metric = train.run_epoch(
        model, x, y, opt, sched, torch.nn.MSELoss(), cfg, rng,
        torch.device("cpu"), training=True, K=2,
        closed_loop_cfg=cl_cfg, closed_loop_gen=gen,
    )
    alpha_stats = dict(getattr(train.run_epoch, "last_alpha_stats", {}) or {})
    model.eval()
    with torch.no_grad():
        vloss, vmetric = train.run_epoch(
            model, x, y, None, None, torch.nn.MSELoss(), cfg, rng,
            torch.device("cpu"), training=False, K=2,
        )
    return {
        "train_loss": loss, "train_metric": metric,
        "valid_loss": vloss, "valid_metric": vmetric,
        "alpha_stats": alpha_stats,
        "state_dict": {k: v.detach().clone() for k, v in model.state_dict().items()},
    }


def run_continuous_epoch(closed_loop: bool, out_dir: pathlib.Path):
    """One epoch (3 chunks) of the old ring trainer on K=2."""
    import train
    from train_srnn.training.continuous import run_continuous_training
    from train_srnn.utils.lr_schedule import WarmupHoldCosineSchedule

    extra = [f"output_dir={out_dir}", "checkpoint_interval=1", "log_interval=1"]
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
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = WarmupHoldCosineSchedule(opt, total_steps=3, max_lr=cfg.lr,
                                     warmup_frac=0.2, cosine_decay=False)
    cl_cfg = train._build_closed_loop_cfg(cfg)
    gen = torch.Generator().manual_seed(7)
    run_continuous_training(
        model, trace, vx, vy, vx, vy, opt, sched, torch.nn.MSELoss(), cfg,
        cl_cfg, gen, rng, torch.device("cpu"), 2, list(model.ablation_names),
        train.eval_and_log_test, train.run_epoch, train.amp_autocast,
    )
    return {
        "training_history": strip_timestamps((out_dir / "training_history.csv").read_text()),
        "test_history": strip_timestamps((out_dir / "test_history.csv").read_text()),
        "state_dict": {k: v.detach().clone() for k, v in model.state_dict().items()},
    }
