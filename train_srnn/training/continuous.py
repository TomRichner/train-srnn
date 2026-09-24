"""Continuous (ring) trainer for one long trace.

B readers sit at fixed phase offsets around the z-scored training trace,
treated as a ring. Every step each reader advances ``bptt_chunk_len``
samples, the target is the trace one sample ahead, gradients flow only
within the chunk, and the hidden state is carried across chunks and across
epochs without reset. A logical epoch is one pass of the readers through
the ring; with the ring length coprime to ``B * chunk_len`` the detach
boundaries drift, so every sample pair eventually sits inside a chunk.
"""
from __future__ import annotations

import contextlib
import logging
import time
from collections import defaultdict

import torch
import torch.utils.checkpoint

from train_srnn.models.sequence_model import Unrolled

from train_srnn.training.closed_loop import (init_channel_phases, sample_continuous_alpha,
                                             sample_per_reader_jitter, summarize_alpha)
from train_srnn.training.trainer import EpochStats, Trainer

log = logging.getLogger(__name__)


class ContinuousTimer:
    """Opt-in per-phase timer. CUDA events on GPU (one sync per report), perf_counter elsewhere.

    With grad checkpointing the forward is recomputed inside backward, so the
    "backward" phase includes that recompute.
    """

    def __init__(self, device: torch.device, enabled: bool = True):
        self.enabled = enabled
        self.device = device
        self.is_cuda = enabled and device.type == "cuda"
        self._cuda_events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = defaultdict(list)
        self._cpu_totals: dict[str, float] = defaultdict(float)
        self._epoch_start_wall: float | None = None

    def epoch_start(self) -> None:
        if not self.enabled:
            return
        self._cuda_events.clear()
        self._cpu_totals.clear()
        self._epoch_start_wall = time.perf_counter()

    @contextlib.contextmanager
    def section(self, name: str):
        if not self.enabled:
            yield
            return
        if self.is_cuda:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            try:
                yield
            finally:
                end.record()
                self._cuda_events[name].append((start, end))
        else:
            t0 = time.perf_counter()
            try:
                yield
            finally:
                self._cpu_totals[name] += time.perf_counter() - t0

    def report(self, logger: logging.Logger, *, label: str = "") -> None:
        if not self.enabled or self._epoch_start_wall is None:
            return
        if self.is_cuda:
            torch.cuda.synchronize()
            totals = {
                name: sum(s.elapsed_time(e) for s, e in evts) / 1000.0
                for name, evts in self._cuda_events.items()
            }
        else:
            totals = dict(self._cpu_totals)
        wall = time.perf_counter() - self._epoch_start_wall
        rows = sorted(totals.items(), key=lambda kv: -kv[1])
        backend = "cuda.Event" if self.is_cuda else "perf_counter"
        suffix = f" [{label}]" if label else ""
        logger.info(
            "Phase breakdown%s — total wall %.1fs (backend=%s; under "
            "grad_checkpoint=true, forward is recomputed during backward "
            "so backward includes recompute)",
            suffix, wall, backend,
        )
        for name, t in rows:
            logger.info("  %-18s %8.2fs  (%5.2f%%)",
                        name, t, 100.0 * t / max(wall, 1e-9))
        accounted = sum(totals.values())
        unaccounted = wall - accounted
        logger.info(
            "  %-18s %8.2fs  (%5.2f%%)",
            "(unaccounted)", unaccounted,
            100.0 * unaccounted / max(wall, 1e-9),
        )


class ContinuousTrainer(Trainer):
    rebuild_ic_each_epoch = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cfg = self.cfg
        if self.data.train_trace is None:
            raise ValueError(f"task {cfg.task.name} provides no train_trace; use the windowed trainer")
        self.trace = torch.tensor(self.data.train_trace, dtype=torch.float32, device=self.device)
        self.T, self.C = self.trace.shape
        self.B = int(cfg.task.batch_size)
        self.chunk_len = int(cfg.task.bptt_chunk_len or 250)
        self.positions = (torch.arange(self.B, device=self.device) * (self.T // self.B)) % self.T
        self.state = None
        self.y_prev = None
        self.channel_phases = init_channel_phases(self.C, seed=int(cfg.seed), device=self.device,
                                                  dtype=self.trace.dtype)
        # Compile only the cell, and only for this loop: evaluation runs the
        # eager cell through SequenceModel with different shapes and grad mode.
        self.cell = self.model.cell
        if cfg.compile.enabled and cfg.compile.cell_only and self.device.type == "cuda":
            kwargs = {}
            if cfg.compile.mode:
                kwargs["mode"] = str(cfg.compile.mode)
            if cfg.compile.dynamic is not None:
                kwargs["dynamic"] = bool(cfg.compile.dynamic)
            log.info("torch.compile(cell) for the ring loop; kwargs=%s", kwargs or "(defaults)")
            self.cell = torch.compile(self.cell, **kwargs)
        self.timer = ContinuousTimer(self.device, enabled=bool(cfg.profile))
        log.info("Ring trainer: T=%d, B=%d, chunk_len=%d, steps_per_epoch=%d, "
                 "checkpoint_interval=%d, log_interval=%d, K=%d",
                 self.T, self.B, self.chunk_len, self.steps_per_epoch(),
                 self.checkpoint_interval, self.log_interval, self.K)

    def extra_state(self) -> dict:
        """Carried hidden state, reader positions and closed-loop feedback; None before epoch 0."""
        return {"state": self.state, "positions": self.positions, "y_prev": self.y_prev}

    def load_extra_state(self, extra: dict) -> None:
        def on_device(t):
            return None if t is None else t.to(self.device)
        self.state, self.y_prev = on_device(extra["state"]), on_device(extra["y_prev"])
        self.positions = on_device(extra["positions"])

    def steps_per_epoch(self) -> int:
        T = int(self.data.train_trace.shape[0])
        B, chunk = int(self.cfg.task.batch_size), int(self.cfg.task.bptt_chunk_len or 250)
        return (T + B * chunk - 1) // (B * chunk)

    def default_warmup_epochs(self) -> int:
        return max(1, round(int(self.cfg.task.batch_size) / 2))

    def default_checkpoint_interval(self) -> int:
        return max(1, round(self.B / 4))

    def default_log_interval(self) -> int:
        return self.checkpoint_interval

    def _gather(self):
        """``(x, y)`` chunks of shape ``(B, chunk_len, C)``; ``y`` is ``x`` one sample ahead, mod T."""
        offsets = torch.arange(self.chunk_len, device=self.device)
        idx_x = (self.positions[:, None] + offsets[None, :]) % self.T
        return self.trace[idx_x], self.trace[(idx_x + 1) % self.T]

    def _start_state(self) -> None:
        """Broadcast the (frozen) initial condition to the B readers."""
        self.state = self.model.ic(self.B).detach().clone()
        self.model.ic.ic.requires_grad_(False)
        if self.cl_cfg.enabled:
            self.y_prev = torch.zeros(self.model.K or 1, self.B, self.C, device=self.device,
                                      dtype=self.trace.dtype)
            if self.model.K is None:
                self.y_prev = self.y_prev[0]

    def _unroll_chunk(self, x, alpha, hoisted):
        """Recompute segments during backward without changing BPTT boundaries."""
        if not self.cfg.grad_checkpoint:
            return self.model.unroll(x, self.state, alpha_seg=alpha, y_prev=self.y_prev,
                                     hoisted=hoisted, cell=self.cell)
        length = int(self.cfg.grad_checkpoint_segment_len or x.shape[1])
        if length < 1:
            raise ValueError("grad_checkpoint_segment_len must be positive")
        state, previous = self.state, self.y_prev
        outputs = []

        def segment(inputs, initial, schedule, prior, weights):
            result = self.model.unroll(inputs, initial, alpha_seg=schedule,
                                       y_prev=prior, hoisted=weights, cell=self.cell)
            values = result.hidden if schedule is None else result.y
            return values, result.state, result.y_prev

        for start in range(0, x.shape[1], length):
            end = min(start + length, x.shape[1])
            schedule = None if alpha is None else alpha[:, start:end, :]
            values, state, previous = torch.utils.checkpoint.checkpoint(
                segment, x[:, start:end], state, schedule, previous, hoisted,
                use_reentrant=False)
            outputs.append(values)
        values = torch.cat(outputs, dim=-2)
        return Unrolled(hidden=values if alpha is None else None,
                        y=None if alpha is None else values, state=state, y_prev=previous)

    def train_epoch(self, epoch: int) -> EpochStats:
        cfg, timer = self.cfg, self.timer
        if self.state is None:
            self._start_state()
        cl = self.cl_cfg.enabled
        t0 = time.time()
        timer.epoch_start()
        loss_sum, metric_sum = [0.0] * self.K, [0.0] * self.K
        alpha_sum, alpha_max, pure_tf, n_alpha = 0.0, 0.0, 0, 0
        jitter = sample_per_reader_jitter(self.cl_cfg, self.B, self.device, generator=self.cl_gen,
                                          dtype=self.trace.dtype) if cl else None
        log.info("Epoch %d start: positions[0]=%d  state.norm=%.4f",
                 epoch, int(self.positions[0].item()), float(self.state.flatten().norm()))
        self.model.train()
        steps = self.steps_per_epoch()
        for _ in range(steps):
            with timer.section("chunk_gather"):
                x, y = self._gather()
            alpha = None
            if cl:
                with timer.section("alpha_sample"):
                    alpha = sample_continuous_alpha(
                        self.cl_cfg, epoch=epoch, total_epochs=cfg.epochs, B=self.B,
                        chunk_len=self.chunk_len, C=self.C, channel_phases=self.channel_phases,
                        per_reader_jitter=jitter, device=self.device, generator=self.cl_gen,
                        dtype=self.trace.dtype)
                s = summarize_alpha(None if alpha is None else alpha.reshape(-1, self.C))
                alpha_sum += s["alpha_mean"]
                alpha_max = max(alpha_max, s["alpha_max"])
                pure_tf += int(s["is_pure_tf"])
                n_alpha += 1
            with timer.section("forward"), self.autocast():
                hoisted = self.cell.hoist()
                res = self._unroll_chunk(x, alpha, hoisted)
                if alpha is None:
                    logits = self.model.apply_readout(res.hidden, x)
                else:
                    logits = res.y
                    self.y_prev = res.y_prev
                self.state = res.state
            with timer.section("loss"), self.autocast():
                loss, losses, metrics = self.loss_and_metrics(logits, y)
            with timer.section("backward"):
                self.optimizer_step(loss)
            for k in range(self.K):
                loss_sum[k] += losses[k]
                metric_sum[k] += metrics[k]
            with timer.section("detach"):
                self.state = self.state.detach()
                if self.y_prev is not None:
                    self.y_prev = self.y_prev.detach()
                self.positions = (self.positions + self.chunk_len) % self.T
        alpha_stats = None
        if cl and n_alpha:
            alpha_stats = {"alpha_mean": alpha_sum / n_alpha, "alpha_max": alpha_max,
                           "pure_tf_frac": pure_tf / n_alpha, "n_batches": n_alpha}
        return EpochStats([s / steps for s in loss_sum], [s / steps for s in metric_sum],
                          wall_s=time.time() - t0, alpha=alpha_stats)

    def run_epoch_and_log(self, epoch: int) -> None:
        """Evaluation, history rows, and checkpoints happen every ``checkpoint_interval`` epochs and on the last."""
        train = self.train_epoch(epoch)
        is_eval_epoch = (epoch + 1) % self.checkpoint_interval == 0 or epoch == self.cfg.epochs - 1
        if not is_eval_epoch:
            self.log_epoch(epoch, train, None)
            self.timer.report(log, label=f"epoch {epoch}")
            return
        with self.timer.section("valid_eval"):
            valid = self.evaluate("valid")
        self.log_epoch(epoch, train, valid)
        with self.timer.section("io_log"):
            self.record(epoch, train, valid)
        tag = f"epoch_{epoch:03d}"
        with self.timer.section("test_eval"):
            self.test(epoch, tag)
        # Last, so a resumable checkpoint implies this epoch's history rows exist.
        with self.timer.section("checkpoint_save"):
            self.checkpoint(epoch, tag)
        self.timer.report(log, label=f"epoch {epoch}")
