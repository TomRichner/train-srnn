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

import logging
import time
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
from train_srnn.utils.checkpoint import (
    append_history_row,
    append_test_history_row,
    save_checkpoint,
    write_progress,
)

log = logging.getLogger(__name__)


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
    """
    hidden_outs = []
    x_in_outs = []
    for t in range(chunk_x.shape[1]):
        x_t = chunk_x[:, t, :]
        h_t, state = cell(x_t, state)
        hidden_outs.append(h_t)
        x_in_outs.append(x_t)
    hidden_seq = torch.stack(hidden_outs, dim=-2)
    # x_in_seq matches hidden_seq's leading dims for skip residual
    if hidden_seq.dim() == 4:  # K-batched -> hidden (K, B, T, N), need (K, B, T, C)
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
    """
    hidden_outs = []
    x_in_outs = []
    for t in range(chunk_x.shape[1]):
        alpha_t = alpha_chunk[:, t, :]              # (B, C)
        x_real_t = chunk_x[:, t, :]                  # (B, C)
        # Blend: when y_prev is (K, B, C), result broadcasts to (K, B, C).
        # When y_prev is (B, C), result is (B, C).
        x_in_t = (1.0 - alpha_t) * x_real_t + alpha_t * y_prev
        h_t, state = cell(x_in_t, state)
        y_t = model._readout_one(h_t, x_in_t)
        hidden_outs.append(h_t)
        x_in_outs.append(x_in_t)
        y_prev = y_t
    hidden_seq = torch.stack(hidden_outs, dim=-2)
    x_in_seq = torch.stack(x_in_outs, dim=-2)
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
            return [
                float(-torch.mean(torch.abs(logits[k].squeeze(-1) if logits[k].shape[-1] == 1 else logits[k] - target)).item())
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
    ablation_names: Optional[list[str]],
    eval_and_log_test_fn,         # train.py's eval_and_log_test, passed in to avoid circular import
    run_epoch_fn,                 # train.py's run_epoch, passed in for eval
    amp_autocast_fn,              # train.py's amp_autocast, passed in
):
    """Stateful continuous-batching trainer. Returns when cfg.epochs reached."""
    T = train_trace.shape[0]
    C = train_trace.shape[1]
    B = int(cfg.batch_size)
    chunk_len = int(cfg.bptt_chunk_len) if cfg.bptt_chunk_len else 250
    # Use ceiling division so each logical epoch advances ≥ T samples per
    # reader. With T coprime to chunk_len (e.g. T=179,989 prime, chunk_len=250),
    # this overshoots T by `chunk_len - (T % chunk_len)` samples each epoch,
    # giving constant phase drift mod T. For seeg defaults: 720 steps × 250
    # = 180,000 samples advanced; T = 179,989; drift = 11 samples per epoch.
    steps_per_epoch = (T + chunk_len - 1) // chunk_len
    total_epochs = int(cfg.epochs)
    log_interval = int(cfg.get("log_interval", 1))
    checkpoint_interval = int(cfg.get("checkpoint_interval", 10))

    log.info(
        "Continuous training: T=%d, B=%d, chunk_len=%d, steps_per_epoch=%d, "
        "total_steps=%d, K=%s",
        T, B, chunk_len, steps_per_epoch, steps_per_epoch * total_epochs,
        K if K is not None else "None",
    )

    # ---- One-time setup ----
    cell = model.cell
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

    grad_clip = float(cfg.get("grad_clip", 0.0) or 0.0)

    # ---- Per-epoch loop ----
    for epoch in range(total_epochs):
        epoch_start = time.time()
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
            chunk_x, chunk_y = _gather_chunks(train_trace, positions, chunk_len, T)

            # α schedule for this chunk
            alpha_chunk = None
            if cl_active:
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
            with amp_autocast_fn(cfg):
                if alpha_chunk is None:
                    logits, state = _forward_chunk_pure_tf(model, cell, chunk_x, state)
                else:
                    logits, state, y_prev = _forward_chunk_closed_loop(
                        model, cell, chunk_x, state, y_prev, alpha_chunk,
                    )
                loss, per_k_loss = _compute_loss(logits, chunk_y, criterion, cfg, K)

            # Backward + step
            optimizer.zero_grad()
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
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

        # Eval (windowed) ----------------------------------------------------
        valid_loss = valid_metric = None
        if (epoch + 1) % log_interval == 0:
            model.eval()
            with torch.no_grad():
                valid_loss, valid_metric = run_epoch_fn(
                    model, valid_x, valid_y,
                    None, None, criterion,
                    cfg, rng, device, training=False, K=K,
                )
            model.train()

        # Logging ------------------------------------------------------------
        wall = time.time() - epoch_start
        if K is not None:
            log.info("Epoch %d (continuous, %.1fs):", epoch, wall)
            for k in range(K):
                vl = valid_loss[k] if valid_loss is not None else float("nan")
                vm = valid_metric[k] if valid_metric is not None else float("nan")
                log.info(
                    "  [%s] train_loss=%.4f train_metric=%.4f valid_loss=%.4f valid_metric=%.4f",
                    ablation_names[k], train_loss[k], train_metric[k], vl, vm,
                )
        else:
            log.info(
                "Epoch %d (continuous, %.1fs): train_loss=%.4f train_metric=%.4f"
                " valid_loss=%.4f valid_metric=%.4f",
                epoch, wall, train_loss, train_metric,
                valid_loss if valid_loss is not None else float("nan"),
                valid_metric if valid_metric is not None else float("nan"),
            )
        if cl_active and cl_total > 0:
            log.info(
                "  closed-loop: alpha_mean=%.3f alpha_max=%.3f pure_tf_frac=%.3f",
                cl_alpha_sum / cl_total, cl_alpha_max, cl_pure_tf / cl_total,
            )

        # CSVs ---------------------------------------------------------------
        cur_lr = optimizer.param_groups[0]["lr"]
        append_history_row(
            cfg.output_dir, epoch,
            train_loss=train_loss, train_metric=train_metric,
            valid_loss=valid_loss if valid_loss is not None else (
                [float("nan")] * K if K is not None else float("nan")
            ),
            valid_metric=valid_metric if valid_metric is not None else (
                [float("nan")] * K if K is not None else float("nan")
            ),
            lr=cur_lr, K=K, ablation_names=ablation_names,
        )
        write_progress(cfg.output_dir, epoch + 1, total_epochs)

        # Checkpoint + test eval ---------------------------------------------
        if epoch % checkpoint_interval == 0:
            tag = f"epoch_{epoch:03d}"
            save_checkpoint(model, optimizer, scheduler, epoch, cfg, tag)
            eval_and_log_test_fn(
                model, test_x, test_y, criterion, cfg, rng, device,
                epoch=epoch, tag=tag, K=K, ablation_names=ablation_names,
            )

    # ---- Final ----
    save_checkpoint(model, optimizer, scheduler, total_epochs - 1, cfg, "last")
    eval_and_log_test_fn(
        model, test_x, test_y, criterion, cfg, rng, device,
        epoch=total_epochs - 1, tag="last", K=K, ablation_names=ablation_names,
    )
