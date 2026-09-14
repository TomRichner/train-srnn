"""Training loop shared by the windowed and continuous trainers.

The base class owns everything that does not depend on how batches are
formed: optimizer and learning-rate schedule, loss and metric over the K
networks, the optimizer step with per-variant gradient clipping, windowed
evaluation, burn-in of the initial condition, checkpoints, and the history
files. Subclasses provide ``train_epoch``.
"""
from __future__ import annotations

import contextlib
import dataclasses
import logging
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from train_srnn.config import ClosedLoopConfig
from train_srnn.data.task import Dataset, Task
from train_srnn.models.sequence_model import SequenceModel
from train_srnn.utils.grad_clip import clip_grad_norm_per_variant
from train_srnn.utils.history import (append_history, append_test_history, load_checkpoint,
                                      save_checkpoint, write_progress)
from train_srnn.utils.lr_schedule import WarmupHoldCosineSchedule
from train_srnn.utils.trainable_ic import compute_burn_in

log = logging.getLogger(__name__)


@dataclass
class EpochStats:
    loss: list[float]                 # per network
    metric: list[float]               # per network: accuracy or MAE
    wall_s: float = 0.0
    alpha: Optional[dict] = None      # closed-loop summary, when active


class Trainer(ABC):
    def __init__(self, cfg: DictConfig, model: SequenceModel, task: Task, data: Dataset,
                 device: torch.device, run_dir: Path,
                 rng: Optional[np.random.RandomState] = None,
                 cl_gen: Optional[torch.Generator] = None):
        self.cfg, self.model, self.task, self.data = cfg, model, task, data
        self.device, self.run_dir = device, Path(run_dir)
        self.K = model.K or 1
        self.names = model.variant_names or [cfg.model.name]
        self.criterion = task.criterion()
        self.rng = rng if rng is not None else np.random.RandomState(cfg.seed)

        self.cl_cfg = ClosedLoopConfig(**OmegaConf.to_container(cfg.closed_loop, resolve=True))
        if self.cl_cfg.enabled and cfg.task.input_size != cfg.task.output_size:
            raise ValueError("closed_loop.enabled requires task.input_size == task.output_size")
        if cl_gen is not None:
            self.cl_gen = cl_gen
        else:
            self.cl_gen = (torch.Generator(device=device).manual_seed(int(cfg.seed) + 1)
                           if self.cl_cfg.enabled else None)

        self.optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        steps = self.steps_per_epoch()
        total_steps = cfg.epochs * steps
        warmup_epochs = cfg.warmup_epochs or self.default_warmup_epochs()
        warmup_frac = min(warmup_epochs * steps / max(1, total_steps), 0.2)
        self.scheduler = WarmupHoldCosineSchedule(
            self.optimizer, total_steps, max_lr=cfg.lr, warmup_frac=warmup_frac,
            cosine_decay=cfg.cosine_decay)
        log.info("LR schedule: warmup_epochs=%d (%d steps), total_steps=%d "
                 "(steps_per_epoch=%d), max_lr=%.3e, cosine_decay=%s",
                 warmup_epochs, int(warmup_frac * total_steps), total_steps, steps,
                 cfg.lr, cfg.cosine_decay)
        if cfg.grad_clip and model.K is not None:
            log.info("grad_clip=%g applied per variant across K=%d networks", cfg.grad_clip, model.K)

    # -- hooks for subclasses -----------------------------------------------

    @abstractmethod
    def train_epoch(self, epoch: int) -> EpochStats:
        ...

    @abstractmethod
    def steps_per_epoch(self) -> int:
        ...

    @abstractmethod
    def default_warmup_epochs(self) -> int:
        ...

    @abstractmethod
    def run_epoch_and_log(self, epoch: int) -> None:
        """One epoch of training plus whatever evaluation and I/O its cadence calls for."""

    @property
    def log_interval(self) -> int:
        return int(self.cfg.log_interval or self.default_log_interval())

    @property
    def checkpoint_interval(self) -> int:
        return int(self.cfg.checkpoint_interval or self.default_checkpoint_interval())

    def default_log_interval(self) -> int:
        return 1

    def default_checkpoint_interval(self) -> int:
        return 10

    # -- shared pieces --------------------------------------------------------

    def autocast(self):
        if self.cfg.amp == "bf16":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def loss_and_metrics(self, logits: torch.Tensor, y: torch.Tensor):
        """Sum of the K losses (for backward) plus per-network loss and metric values.

        ``logits`` is ``(K, B, ..., O)``; a single-network model is treated as K=1.
        """
        per_k = [logits[k] for k in range(self.K)] if self.model.K is not None else [logits]
        losses = [self.task.loss(l, y, self.criterion) for l in per_k]
        metrics = [self.task.metric(l, y) for l in per_k]
        return torch.stack(losses).sum(), [l.item() for l in losses], metrics

    def optimizer_step(self, loss: torch.Tensor) -> None:
        self.optimizer.zero_grad()
        loss.backward()
        clip = float(self.cfg.grad_clip or 0.0)
        if clip > 0:
            if self.model.K is not None:
                clip_grad_norm_per_variant(self.model.parameters(), clip, self.model.K)
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip)
        self.optimizer.step()
        self.scheduler.step()
        self.model.constrain_parameters()

    def batches(self, x: np.ndarray, training: bool):
        """Index batches of ``cfg.task.batch_size``; shuffled when training."""
        order = self.rng.permutation(len(x)) if training else np.arange(len(x))
        B = int(self.cfg.task.batch_size)
        for start in range(0, len(order), B):
            yield order[start:start + B]

    def evaluate(self, split: str) -> EpochStats:
        """Windowed evaluation of one split with teacher forcing."""
        x_all, y_all = getattr(self.data, split)
        loss_sum, metric_sum, n_total = [0.0] * self.K, [0.0] * self.K, 0
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            for idx in self.batches(x_all, training=False):
                b = self.task.eval_batch(x_all[idx], y_all[idx])
                x = torch.tensor(b.x, dtype=torch.float32, device=self.device)
                y = self.task.target_tensor(b.y, self.device)
                with self.autocast():
                    logits = self.model(x, readout_idx=b.readout_idx,
                                        bptt_chunk_len=self.cfg.task.bptt_chunk_len)
                    _, losses, metrics = self.loss_and_metrics(logits, y)
                n = len(idx)
                for k in range(self.K):
                    loss_sum[k] += losses[k] * n
                    metric_sum[k] += metrics[k] * n
                n_total += n
        if was_training:
            self.model.train()
        return EpochStats([s / n_total for s in loss_sum], [s / n_total for s in metric_sum])

    def refresh_ic(self) -> None:
        """Burn the cell in with zero input and copy the settled state into the IC."""
        with self.autocast():
            state = compute_burn_in(self.model.cell, self.cfg.task.input_size,
                                    self.cfg.burn_in, self.device)
        self.model.ic.ic.data.copy_(state)

    def checkpoint(self, epoch: int, tag: str) -> None:
        save_checkpoint(self.run_dir, tag, self.model, self.optimizer, self.scheduler, epoch, self.cfg)

    def test(self, epoch: int, tag: str) -> EpochStats:
        stats = self.evaluate("test")
        append_test_history(self.run_dir, epoch, tag, self.names, stats.loss, stats.metric)
        log.info("Test [%s @ epoch %d]:", tag, epoch)
        for k, name in enumerate(self.names):
            log.info("  [%s] test_loss=%.4f test_metric=%.4f", name, stats.loss[k], stats.metric[k])
        return stats

    def record(self, epoch: int, train: EpochStats, valid: Optional[EpochStats]) -> None:
        append_history(self.run_dir, epoch, self.names, train.loss, train.metric,
                       valid.loss if valid else None, valid.metric if valid else None,
                       lr=self.optimizer.param_groups[0]["lr"])
        write_progress(self.run_dir, epoch + 1, self.cfg.epochs)

    def log_epoch(self, epoch: int, train: EpochStats, valid: Optional[EpochStats]) -> None:
        log.info("Epoch %d (%.1fs):", epoch, train.wall_s)
        for k, name in enumerate(self.names):
            line = f"  [{name}] train_loss={train.loss[k]:.4f} train_metric={train.metric[k]:.4f}"
            if valid is not None:
                line += f" valid_loss={valid.loss[k]:.4f} valid_metric={valid.metric[k]:.4f}"
            log.info(line)
        if train.alpha:
            a = train.alpha
            log.info("  closed-loop: alpha_mean=%.3f alpha_max=%.3f pure_tf_frac=%.2f over %d batches",
                     a["alpha_mean"], a["alpha_max"], a["pure_tf_frac"], a["n_batches"])

    def epoch_closed_loop_cfg(self, epoch: int) -> ClosedLoopConfig:
        """Closed-loop config with the per-epoch baseline ramp applied."""
        from train_srnn.training.closed_loop import effective_alpha_baseline
        if not self.cl_cfg.enabled:
            return self.cl_cfg
        base = effective_alpha_baseline(self.cl_cfg, epoch, self.cfg.epochs)
        return dataclasses.replace(self.cl_cfg, alpha_baseline=base)

    def resume(self, init_ckpt: str) -> None:
        """Load model, optimizer, and scheduler from a local path or a gs:// URL."""
        local = init_ckpt
        if init_ckpt.startswith("gs://"):
            local = str(self.run_dir / "_init_ckpt.pt")
            self.run_dir.mkdir(parents=True, exist_ok=True)
            log.info("Downloading init_ckpt %s -> %s", init_ckpt, local)
            subprocess.run(["gcloud", "storage", "cp", init_ckpt, local], check=True)
        prev = load_checkpoint(local, model=self.model, optimizer=self.optimizer,
                               scheduler=self.scheduler, device=str(self.device))
        log.info("Resumed from epoch=%s (lr=%.3e)", prev.get("epoch"),
                 self.optimizer.param_groups[0]["lr"])

    # -- the run --------------------------------------------------------------

    def fit(self) -> None:
        cfg = self.cfg
        if cfg.init_ckpt:
            self.resume(cfg.init_ckpt)
        if cfg.burn_in > 0:
            self.refresh_ic()
            if cfg.freeze_ic_after_burnin:
                self.model.ic.ic.requires_grad_(False)
                log.info("Initial condition frozen after burn-in")
        self.checkpoint(0, "init")
        self.test(0, "init")
        if cfg.early_exit_after_init:
            self.checkpoint(0, "last")
            log.info("early_exit_after_init: wrote init.pt and last.pt to %s", self.run_dir)
            return
        self.model.train()
        for epoch in range(cfg.epochs):
            if (cfg.burn_in_every > 0 and epoch > 0 and epoch % cfg.burn_in_every == 0
                    and cfg.burn_in > 0 and self.rebuild_ic_each_epoch):
                self.refresh_ic()
            self.run_epoch_and_log(epoch)
        last = cfg.epochs - 1
        self.checkpoint(last, "last")
        self.test(last, "last")

    #: Whether periodic burn-in makes sense (windowed runs restart from the IC
    #: every batch; the ring trainer never resets its state).
    rebuild_ic_each_epoch = True


def timed(fn):
    """Run ``fn()`` and return ``(result, seconds)``."""
    t0 = time.time()
    out = fn()
    return out, time.time() - t0
