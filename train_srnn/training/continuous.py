"""Stateful continuous-batching trainer for autoregressive trace data.

Place B parallel readers at fixed phase offsets in a circular ring of length
T (the train trace, truncated to a length coprime to chunk_len for phase
diversity). All readers advance chunk_len samples per training step;
gradients average across B before each optimizer.step(). State propagates
indefinitely across chunks AND across "logical epoch" boundaries (no reset).

This module implements the inner training loop only. Model construction,
optimizer/scheduler/criterion setup, init burn-in, init/last test eval,
and checkpointing are still handled by `train.py:main`. Eval (valid/test)
uses the existing windowed `run_epoch(training=False, ...)` unchanged.

See plan: ~/.claude/plans/1-i-don-t-care-quiet-melody.md
"""
from __future__ import annotations

import contextlib
import logging
import time
from collections import defaultdict
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig

from train_srnn.training.closed_loop import (
    ClosedLoopConfig,
    init_channel_phases,
    sample_continuous_alpha,
    sample_per_reader_jitter,
    summarize_alpha,
)
from train_srnn.utils.cell_loop import empty_time_buffer, mark_cudagraph_step
from train_srnn.utils.grad_clip import clip_grad_norm_per_variant
from train_srnn.utils.checkpoint import (
    append_history_row,
    append_test_history_row,
    save_checkpoint,
    write_progress,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Profiling helper
# ---------------------------------------------------------------------------

class ContinuousTimer:
    """Phase-wise wall-clock timer for the continuous trainer (opt-in).

    On CUDA, uses async ``torch.cuda.Event(enable_timing=True)`` pairs so the
    instrumentation doesn't force per-section host-GPU sync (which would
    defeat the async dispatch we're trying to measure). One global
    ``torch.cuda.synchronize()`` is issued only when ``report()`` is called,
    once per epoch.

    On CPU / MPS, falls back to ``time.perf_counter()`` accumulators. Note
    that on MPS this measures host-side dispatch latency, not actual MPS
    kernel execution — but the real measurement we care about is on CUDA.

    When ``enabled=False`` (the default off-state), every ``section()`` call
    is a thin no-op generator that yields once, and ``report()`` returns
    immediately. No CUDA events are allocated; no overhead.

    Usage per epoch:
        timer.epoch_start()
        for step in ...:
            with timer.section("forward"):
                ...
        with timer.section("valid_eval"):
            ...
        timer.report(log)

    Caveat: under ``grad_checkpoint=true`` the forward is recomputed during
    backward; the cuda.Event recorded in the ``forward`` section captures
    only the first pass, while ``backward`` captures backward + recompute.
    The report header notes this so the numbers are not misread.
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


# ---------------------------------------------------------------------------
# Position management
# ---------------------------------------------------------------------------

def _init_positions(B: int, T: int, device) -> torch.Tensor:
    """Even-spaced reader start positions in [0, T)."""
    return (torch.arange(B, device=device, dtype=torch.long) * (T // B)) % T


def _gather_chunks(
    train_trace: torch.Tensor,        # (T, C) on device
    positions: torch.Tensor,          # (B,) long
    chunk_len: int,
    T: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (chunk_x, chunk_y), each (B, chunk_len, C). y is x[t+1].

    Indexes wrap mod T (trace-circular).
    """
    offsets = torch.arange(chunk_len, device=positions.device, dtype=torch.long)
    idx_x = (positions[:, None] + offsets[None, :]) % T  # (B, chunk_len)
    idx_y = (idx_x + 1) % T
    return train_trace[idx_x], train_trace[idx_y]


# ---------------------------------------------------------------------------
# Forward / chunk step
# ---------------------------------------------------------------------------

def _readout_chunk(model, hidden_seq: torch.Tensor, x_in_seq: torch.Tensor) -> torch.Tensor:
    """Apply per-timestep readout across a chunk.

    hidden_seq : (B, T, N) or (K, B, T, N)
    x_in_seq   : (B, T, C) or (K, B, T, C)   — for skip residual
    Returns    : (B, T, O) or (K, B, T, O)

    SequenceModel._readout_one applies output_mask, readout_weight,
    readout_bias, and (for batched) skip flags. We call it per t because
    it's already designed for one timestep; loop is small (T ≤ chunk_len).
    """
    T = hidden_seq.shape[-2]
    outs = []
    for t in range(T):
        if hidden_seq.dim() == 4:    # (K, B, T, N)
            h_t = hidden_seq[:, :, t, :]
            x_t = x_in_seq[:, :, t, :] if x_in_seq.dim() == 4 else x_in_seq[:, t, :]
        else:                         # (B, T, N)
            h_t = hidden_seq[:, t, :]
            x_t = x_in_seq[:, t, :]
        outs.append(model._readout_one(h_t, x_t))
    return torch.stack(outs, dim=-2)


def _forward_chunk_pure_tf(
    model, cell, chunk_x: torch.Tensor, state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure teacher forcing forward over a chunk.

    chunk_x : (B, T, C)
    state   : (B, S) or (K, B, S)
    Returns : (logits, new_state) where logits has shape (B, T, O) or (K, B, T, O).

    Pre-allocates ``hidden_seq`` lazily on the first iteration (so it
    picks up the cell's output dtype, which differs from chunk_x under
    AMP autocast) and writes per-step outputs by slice-assign. This is
    the CUDA-graph-friendly equivalent of an ``append + torch.stack``
    accumulator: avoids holding refs to graph-owned output buffers
    across iterations under ``compile(mode='reduce-overhead')``.
    """
    T = chunk_x.shape[1]
    # Hoist effective recurrent weight: SRNN cells rebuild W_eff from W_raw
    # via softplus + Dale signs + sparsity_mask on every forward call. W_raw
    # is constant across this chunk (only changes at optimizer.step), so
    # build once and pass in. Other cell types (LSTM/LTC/CTRNN) don't have
    # _effective_W; fall through to the original signature.
    W_eff = cell.hoist()
    hidden_seq: torch.Tensor | None = None
    for t in range(T):
        # Tells the cudagraph trees allocator the previous step's outputs
        # are no longer in use (so it may recycle them). No-op outside
        # CUDA-graph capture.
        mark_cudagraph_step()
        if W_eff is not None:
            h_t, state = cell(chunk_x[:, t, :], state, W_eff=W_eff)
        else:
            h_t, state = cell(chunk_x[:, t, :], state)
        # Clone state: it's both an output of call t and the input of call
        # t+1, so it cannot be recycled by mark_cudagraph_step. The clone
        # produces a fresh non-graph-owned tensor with canonical strides
        # and breaks aliasing — required for reduce-overhead, and also
        # keeps Dynamo from recompiling on stride variation.
        state = state.clone()
        if hidden_seq is None:
            hidden_seq = empty_time_buffer(h_t, T)
        hidden_seq[..., t, :] = h_t
    # x_in_seq matches hidden_seq's leading dims for the skip residual.
    if hidden_seq.dim() == 4:  # K-batched -> hidden (K, B, T, N)
        x_in_seq = chunk_x.unsqueeze(0).expand(hidden_seq.shape[0], -1, -1, -1)
    else:
        x_in_seq = chunk_x
    logits = _readout_chunk(model, hidden_seq, x_in_seq)
    return logits, state


def _forward_chunk_closed_loop(
    model, cell, chunk_x: torch.Tensor, state: torch.Tensor,
    y_prev: torch.Tensor, alpha_chunk: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Closed-loop forward over a chunk, per-reader α.

    chunk_x      : (B, T, C)
    state        : (B, S) or (K, B, S)
    y_prev       : (B, C) or (K, B, C) — last-step prediction (zeros at t=0)
    alpha_chunk  : (B, T, C)              — per-reader, per-step, per-channel α
    Returns      : (logits, new_state, new_y_prev)

    Same pre-alloc + slice-assign pattern as _forward_chunk_pure_tf. The
    blended ``x_in_t`` must be retained for the per-step skip residual
    (unlike the pure-TF path where x_in_t == chunk_x[:, t, :]), so it
    gets its own pre-allocated buffer.
    """
    T = chunk_x.shape[1]
    # Hoist W_eff once per chunk (see _forward_chunk_pure_tf for rationale).
    W_eff = cell.hoist()
    hidden_seq: torch.Tensor | None = None
    x_in_seq: torch.Tensor | None = None
    for t in range(T):
        mark_cudagraph_step()
        alpha_t = alpha_chunk[:, t, :]          # (B, C)
        x_real_t = chunk_x[:, t, :]              # (B, C)
        # Blend broadcasts to whatever y_prev is: (B, C) or (K, B, C).
        x_in_t = (1.0 - alpha_t) * x_real_t + alpha_t * y_prev
        if W_eff is not None:
            h_t, state = cell(x_in_t, state, W_eff=W_eff)
        else:
            h_t, state = cell(x_in_t, state)
        state = state.clone()
        # _readout_one is einsum/linear (not graph-owned), so its output
        # is a fresh allocation — no clone needed for the y_prev carry.
        y_prev = model._readout_one(h_t, x_in_t)
        if hidden_seq is None:
            hidden_seq = empty_time_buffer(h_t, T)
            x_in_seq = empty_time_buffer(x_in_t, T)
        hidden_seq[..., t, :] = h_t
        x_in_seq[..., t, :] = x_in_t
    logits = _readout_chunk(model, hidden_seq, x_in_seq)
    return logits, state, y_prev


# ---------------------------------------------------------------------------
# Loss helper (mirrors train.py:179-188)
# ---------------------------------------------------------------------------

def _compute_loss(logits, target, criterion, cfg, K: Optional[int]):
    """Per-K loss + total. Returns (loss_for_backward, per_k_loss_list_or_none)."""
    if K is not None:
        losses = []
        for k in range(K):
            logits_k = logits[k]
            if cfg.task.task_type == "regression":
                logits_k = logits_k.squeeze(-1) if logits_k.shape[-1] == 1 else logits_k
            losses.append(criterion(logits_k, target))
        loss = torch.stack(losses).sum()
        per_k = [l.item() for l in losses]
        return loss, per_k
    else:
        if cfg.task.task_type == "regression" and logits.shape[-1] == 1:
            logits = logits.squeeze(-1)
        loss = criterion(logits, target)
        return loss, None


def _per_k_metric(logits, target, cfg, K: Optional[int]) -> list[float] | float:
    """Negative MAE per K (or scalar). Mirrors run_epoch's regression metric."""
    with torch.no_grad():
        if K is not None:
            # Parenthesise the squeeze: without it the conditional binds
            # looser than the subtraction, so a width-1 output computed
            # mean(|pred|) instead of mean(|pred - target|). See
            # KnownIssues §13.
            return [
                float(-torch.mean(torch.abs(
                    (logits[k].squeeze(-1) if logits[k].shape[-1] == 1
                     else logits[k]) - target
                )).item())
                for k in range(K)
            ]
        else:
            l = logits.squeeze(-1) if logits.shape[-1] == 1 else logits
            return float(-torch.mean(torch.abs(l - target)).item())


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_continuous_training(
    model: nn.Module,
    train_trace: torch.Tensor,        # (T, C) on device
    valid_x: np.ndarray,
    valid_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    optimizer,
    scheduler,
    criterion,
    cfg: DictConfig,
    closed_loop_cfg: ClosedLoopConfig,
    closed_loop_gen: Optional[torch.Generator],
    rng: np.random.RandomState,
    device: torch.device,
    K: Optional[int],
    variant_names: Optional[list[str]],
    eval_and_log_test_fn,         # train.py's eval_and_log_test, passed in to avoid circular import
    run_epoch_fn,                 # train.py's run_epoch, passed in for eval
    amp_autocast_fn,              # train.py's amp_autocast, passed in
):
    """Stateful continuous-batching trainer. Returns when cfg.epochs reached."""
    T = train_trace.shape[0]
    C = train_trace.shape[1]
    B = int(cfg.task.batch_size)
    chunk_len = int(cfg.task.bptt_chunk_len or 250)
    # Per-reader-sweep semantics (Mikolov RNNLM / Karpathy char-rnn / Keras
    # stateful LSTM convention): each reader covers ~T/B samples per epoch;
    # the B readers collectively cover one full pass through T per epoch.
    # After B epochs ("super-epoch"), each individual reader has visited the
    # whole trace once. With T coprime to B*chunk_len the detach boundaries
    # drift by (B*chunk_len*steps_per_epoch - T) samples per epoch.
    steps_per_epoch = (T + B * chunk_len - 1) // (B * chunk_len)
    total_epochs = int(cfg.epochs)
    # Default checkpoint cadence to round(B/4) so we get ~4 checkpoints per
    # full reader-sweep (B epochs). Floor to 1 for tiny B.
    # Honor explicit user values; only fall back when null/missing.
    default_ckpt = max(1, round(B / 4))
    checkpoint_interval = int(cfg.checkpoint_interval or default_ckpt)
    # Default log cadence to match checkpoint cadence — eval is expensive and
    # there's no value logging more often than we checkpoint.
    log_interval = int(cfg.log_interval or checkpoint_interval)

    log.info(
        "Continuous training (per-reader-sweep): T=%d, B=%d, chunk_len=%d, "
        "steps_per_epoch=%d, total_steps=%d, "
        "checkpoint_interval=%d, log_interval=%d, K=%s",
        T, B, chunk_len, steps_per_epoch, steps_per_epoch * total_epochs,
        checkpoint_interval, log_interval,
        K if K is not None else "None",
    )

    # ---- One-time setup ----
    cell = model.cell
    # Cell-level torch.compile lives here (not in train.py) so that eval — which
    # calls SequenceModel.forward → self.cell with different shapes (windowed
    # batch) and grad mode (no_grad) — sees the eager cell and doesn't trigger
    # extra recompiles in the cache. Only the trainer's hot loop sees compiled.
    if cfg.compile.enabled and cfg.compile.cell_only and device.type == "cuda":
        compile_kwargs = {}
        if cfg.compile.mode:
            compile_kwargs["mode"] = str(cfg.compile.mode)
        if cfg.compile.dynamic is not None:
            compile_kwargs["dynamic"] = bool(cfg.compile.dynamic)
        log.info("Compiling cell (continuous trainer scope only); kwargs=%s",
                 compile_kwargs or "(defaults)")
        cell = torch.compile(cell, **compile_kwargs)
    cl_active = bool(closed_loop_cfg is not None and closed_loop_cfg.enabled)

    # IC -> broadcast initial state. Then freeze IC (continuous: no grad
    # ever flows back to it; KnownIssues §4).
    if hasattr(model, "ic"):
        ic_state = model.ic(B)               # (B, S) or (K, B, S)
        # Take a clone to avoid aliasing the parameter.
        state = ic_state.detach().clone().requires_grad_(False)
        try:
            model.ic.ic.requires_grad_(False)
        except AttributeError:
            pass
    else:
        state = cell.init_state(B).to(device)
        if K is not None and state.dim() == 2:
            # init_state returned (B, S) for a single cell; not expected here
            # since K is only set for batched cells. Defensive.
            state = state.unsqueeze(0).expand(K, -1, -1).contiguous()

    # y_prev (closed-loop autoregressive feedback)
    if cl_active:
        if K is not None:
            y_prev = torch.zeros(K, B, C, device=device, dtype=train_trace.dtype)
        else:
            y_prev = torch.zeros(B, C, device=device, dtype=train_trace.dtype)
    else:
        y_prev = None

    # Channel phases (continuous-mode α rotation)
    channel_phases = init_channel_phases(C, seed=int(cfg.seed), device=device,
                                          dtype=train_trace.dtype)

    # Reader start positions
    positions = _init_positions(B, T, device)

    grad_clip = float(cfg.grad_clip or 0.0)
    # clip_grad_norm_ takes ONE norm over the whole model, so in batched-
    # ablation mode a single variant's gradient would throttle every other
    # variant's step. Clip each variant's slice independently instead — see
    # KnownIssues §14. Nothing else couples the variants.
    if grad_clip > 0 and K is not None:
        log.info("grad_clip=%g applied per-variant across the K=%d batched "
                 "variants (KnownIssues §14)", grad_clip, K)

    # Optional per-phase profiler (off by default).
    profile_enabled = bool(cfg.profile)
    timer = ContinuousTimer(device, enabled=profile_enabled)
    if profile_enabled:
        log.info("Profiling enabled: per-phase timing breakdown will print "
                 "after each epoch (backend=%s).",
                 "cuda.Event" if timer.is_cuda else "perf_counter")

    # ---- Per-epoch loop ----
    for epoch in range(total_epochs):
        epoch_start = time.time()
        timer.epoch_start()
        epoch_loss_sum = 0.0   # for logging (per-K when K is not None)
        epoch_loss_sum_k = [0.0] * K if K is not None else None
        epoch_metric_sum_k = [0.0] * K if K is not None else None
        epoch_metric_sum = 0.0
        n_steps_done = 0
        cl_alpha_sum = 0.0
        cl_alpha_max = 0.0
        cl_pure_tf = 0
        cl_total = 0

        # Per-epoch jitter draw (per-reader baseline offset, fixed for the epoch)
        per_reader_jitter = sample_per_reader_jitter(
            closed_loop_cfg, B, device,
            generator=closed_loop_gen, dtype=train_trace.dtype,
        ) if cl_active else None

        # Log positions[0] at epoch start for phase-drift sanity (verification 5)
        log.info("Epoch %d start: positions[0]=%d  state.norm=%.4f",
                 epoch, int(positions[0].item()), float(state.detach().flatten().norm()))

        model.train()

        for step in range(steps_per_epoch):
            with timer.section("chunk_gather"):
                chunk_x, chunk_y = _gather_chunks(train_trace, positions, chunk_len, T)

            # α schedule for this chunk
            alpha_chunk = None
            if cl_active:
                with timer.section("alpha_sample"):
                    alpha_chunk = sample_continuous_alpha(
                        closed_loop_cfg, epoch=epoch, total_epochs=total_epochs,
                        B=B, chunk_len=chunk_len, C=C,
                        channel_phases=channel_phases,
                        per_reader_jitter=per_reader_jitter,
                        device=device, generator=closed_loop_gen,
                        dtype=train_trace.dtype,
                    )
                stats = summarize_alpha(alpha_chunk if alpha_chunk is None else
                                        alpha_chunk.reshape(-1, C))
                cl_alpha_sum += stats["alpha_mean"]
                cl_alpha_max = max(cl_alpha_max, stats["alpha_max"])
                cl_pure_tf += int(stats["is_pure_tf"])
                cl_total += 1

            # Forward + loss
            with timer.section("forward"):
                with amp_autocast_fn(cfg):
                    if alpha_chunk is None:
                        logits, state = _forward_chunk_pure_tf(model, cell, chunk_x, state)
                    else:
                        logits, state, y_prev = _forward_chunk_closed_loop(
                            model, cell, chunk_x, state, y_prev, alpha_chunk,
                        )
            with timer.section("loss"):
                with amp_autocast_fn(cfg):
                    loss, per_k_loss = _compute_loss(logits, chunk_y, criterion, cfg, K)

            # Backward + step
            with timer.section("backward"):
                optimizer.zero_grad()
                loss.backward()
            with timer.section("optim_step"):
                if grad_clip > 0:
                    if K is not None:
                        clip_grad_norm_per_variant(model.parameters(), grad_clip, K)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                       max_norm=grad_clip)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            # Metrics
            if K is not None:
                metric_k = _per_k_metric(logits, chunk_y, cfg, K)
                for k in range(K):
                    epoch_loss_sum_k[k] += per_k_loss[k]
                    epoch_metric_sum_k[k] += metric_k[k]
            else:
                epoch_loss_sum += loss.item()
                epoch_metric_sum += _per_k_metric(logits, chunk_y, cfg, None)
            n_steps_done += 1

            with timer.section("detach"):
                # Detach state + y_prev for next chunk (truncated BPTT boundary)
                state = state.detach()
                if y_prev is not None:
                    y_prev = y_prev.detach()
                # Advance positions (wrap mod T)
                positions = (positions + chunk_len) % T

        # ---- Epoch-end ----
        train_loss = (
            [s / n_steps_done for s in epoch_loss_sum_k] if K is not None
            else epoch_loss_sum / n_steps_done
        )
        # Metric is MAE (positive) — _per_k_metric returns negative; flip.
        train_metric = (
            [-m / n_steps_done for m in epoch_metric_sum_k] if K is not None
            else -epoch_metric_sum / n_steps_done
        )

        # Eval/CSV/checkpoint cadence: every checkpoint_interval epochs, plus
        # the final epoch (so a run always lands a final CSV row + ckpt).
        is_eval_epoch = ((epoch + 1) % checkpoint_interval == 0
                        or epoch == total_epochs - 1)

        wall = time.time() - epoch_start

        # Eval (windowed) ----------------------------------------------------
        valid_loss = valid_metric = None
        if is_eval_epoch:
            with timer.section("valid_eval"):
                model.eval()
                with torch.no_grad():
                    valid_loss, valid_metric = run_epoch_fn(
                        model, valid_x, valid_y,
                        None, None, criterion,
                        cfg, rng, device, training=False, K=K,
                    )
                model.train()

        # Logging ------------------------------------------------------------
        if not is_eval_epoch:
            # Train-only epoch: terse one-line log, no eval columns.
            if K is not None:
                tl_str = " ".join(
                    f"[{variant_names[k]}]={train_loss[k]:.4f}/{train_metric[k]:.4f}"
                    for k in range(K)
                )
                log.info("Epoch %d (continuous, train-only, %.1fs): %s",
                         epoch, wall, tl_str)
            else:
                log.info(
                    "Epoch %d (continuous, train-only, %.1fs):"
                    " train_loss=%.4f train_metric=%.4f",
                    epoch, wall, train_loss, train_metric,
                )
        else:
            if K is not None:
                log.info("Epoch %d (continuous, %.1fs):", epoch, wall)
                for k in range(K):
                    vl = valid_loss[k] if valid_loss is not None else float("nan")
                    vm = valid_metric[k] if valid_metric is not None else float("nan")
                    log.info(
                        "  [%s] train_loss=%.4f train_metric=%.4f valid_loss=%.4f valid_metric=%.4f",
                        variant_names[k], train_loss[k], train_metric[k], vl, vm,
                    )
            else:
                log.info(
                    "Epoch %d (continuous, %.1fs): train_loss=%.4f"
                    " train_metric=%.4f valid_loss=%.4f valid_metric=%.4f",
                    epoch, wall, train_loss, train_metric,
                    valid_loss if valid_loss is not None else float("nan"),
                    valid_metric if valid_metric is not None else float("nan"),
                )
        if cl_active and cl_total > 0:
            log.info(
                "  closed-loop: alpha_mean=%.3f alpha_max=%.3f pure_tf_frac=%.3f",
                cl_alpha_sum / cl_total, cl_alpha_max, cl_pure_tf / cl_total,
            )

        # CSVs + checkpoint + test eval — only at checkpoint boundaries -----
        if is_eval_epoch:
            with timer.section("io_log"):
                cur_lr = optimizer.param_groups[0]["lr"]
                append_history_row(
                    cfg.output_dir, epoch,
                    train_loss=train_loss, train_metric=train_metric,
                    valid_loss=valid_loss, valid_metric=valid_metric,
                    lr=cur_lr, K=K, variant_names=variant_names,
                )
                write_progress(cfg.output_dir, epoch + 1, total_epochs)

            tag = f"epoch_{epoch:03d}"
            with timer.section("checkpoint_save"):
                save_checkpoint(model, optimizer, scheduler, epoch, cfg, tag)
            with timer.section("test_eval"):
                eval_and_log_test_fn(
                    model, test_x, test_y, criterion, cfg, rng, device,
                    epoch=epoch, tag=tag, K=K, variant_names=variant_names,
                )

        # Per-phase profiler report (no-op when profile=false)
        timer.report(log, label=f"epoch {epoch}")

    # ---- Final ----
    save_checkpoint(model, optimizer, scheduler, total_epochs - 1, cfg, "last")
    eval_and_log_test_fn(
        model, test_x, test_y, criterion, cfg, rng, device,
        epoch=total_epochs - 1, tag="last", K=K, variant_names=variant_names,
    )
