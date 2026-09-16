"""Sequence model: a recurrent cell unrolled over time with a linear readout.

Handles the neuron partition (only ~25% of units feed the readout), the
trainable initial condition, truncated BPTT with optional gradient
checkpointing, the residual skip for autoregressive variants, and
closed-loop (variable teacher forcing) unrolls. Works for single-network
cells (state ``(B, S)``) and K-batched cells (state ``(K, B, S)``).
"""
from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import Tensor

from train_srnn.models.base import RNNCell
from train_srnn.utils.cell_loop import empty_time_buffer, mark_cudagraph_step
from train_srnn.utils.io_masks import generate_neuron_partition, make_input_mask, make_output_mask
from train_srnn.utils.trainable_ic import TrainableIC


def _no_autocast_cache_ctx():
    """Re-enter an active autocast region with its cast cache disabled.

    Under ``torch.utils.checkpoint`` the cache state differs between the
    original forward and the recompute, which trips the saved-tensor
    checks; disabling it makes both passes identical.
    """
    for dev in ("cuda", "cpu"):
        if torch.is_autocast_enabled(dev):
            return torch.autocast(device_type=dev, dtype=torch.get_autocast_dtype(dev),
                                  cache_enabled=False)
    return contextlib.nullcontext()


@dataclass
class Unrolled:
    """Result of :meth:`SequenceModel.unroll` over one time segment."""
    hidden: Optional[Tensor]   # (..., T, E) cell outputs; None in closed loop
    y: Optional[Tensor]        # (..., T, O) predictions; only in closed loop
    state: Tensor
    y_prev: Optional[Tensor]   # last prediction, carried into the next segment


class SequenceModel(nn.Module):
    """``cell`` unrolled over ``(B, T, F)`` inputs with a masked linear readout.

    Args:
        cell: any :class:`RNNCell`.
        input_size, output_size: feature sizes.
        num_units: hidden size of the cell.
        use_io_masks: partition units into input / interneuron / output groups;
            the readout then sees only the output group.
        io_mask_seed: seed of the partition.
        trainable_ic: learn the initial state (otherwise it is frozen at zero
            or whatever burn-in wrote into it).
    """

    def __init__(self, cell: RNNCell, input_size: int, output_size: int, num_units: int,
                 use_io_masks: bool = True, io_mask_seed: int = 0, trainable_ic: bool = True,
                 task_type: str = "classification"):
        super().__init__()
        self.cell = cell
        self.input_size, self.output_size = input_size, output_size
        self.use_io_masks = use_io_masks
        self.task_type = task_type
        self._K = cell.K

        flags = cell.skip_mask()
        self._has_skip = bool(flags is not None and flags.any().item())
        if self._has_skip and input_size != output_size:
            raise ValueError(f"skip variants require input_size == output_size, "
                             f"got {input_size} != {output_size}")

        readout_size = num_units
        if use_io_masks:
            input_idx, _, output_idx = generate_neuron_partition(num_units, io_mask_seed)
            self.register_buffer("input_mask", torch.tensor(make_input_mask(num_units, input_idx), dtype=torch.float32))
            self.register_buffer("output_mask", torch.tensor(make_output_mask(num_units, output_idx), dtype=torch.float32))
            readout_size = len(output_idx)
        else:
            self.register_buffer("input_mask", None)
            self.register_buffer("output_mask", None)

        self.ic = TrainableIC(cell.state_size, K=self._K)
        self.ic.ic.requires_grad_(trainable_ic)

        if self._K is not None:
            self.readout_weight = nn.Parameter(torch.empty(self._K, output_size, readout_size))
            self.readout_bias = nn.Parameter(torch.zeros(self._K, 1, output_size))
            for k in range(self._K):
                generator = None
                if hasattr(cell, "configs"):
                    generator = torch.Generator().manual_seed(cell.configs[k].init_seed + 67867967)
                nn.init.kaiming_uniform_(self.readout_weight[k], a=math.sqrt(5), generator=generator)
            self.readout = None
            self.W_out_gain = nn.Parameter(torch.ones(self._K))
        else:
            self.readout = nn.Linear(readout_size, output_size)
            self.W_out_gain = nn.Parameter(torch.tensor(1.0))

    @property
    def K(self) -> Optional[int]:
        return self._K

    @property
    def variant_names(self) -> Optional[list[str]]:
        """Names of the K networks, or None for a single-network cell."""
        return getattr(self.cell, "variant_names", None) if self._K is not None else None

    # -- pieces ---------------------------------------------------------------

    def initial_state(self, batch_size: int) -> Tensor:
        return self.ic(batch_size)

    def apply_readout(self, h: Tensor, x_in: Tensor) -> Tensor:
        """Readout of cell outputs ``h`` with the skip residual from the cell input.

        ``h`` is ``(B, [T,] E)`` or ``(K, B, [T,] E)``; ``x_in`` has matching
        batch/time dims with the input features last, with or without a K axis.
        """
        if self.output_mask is not None:
            h = h * self.output_mask
            h = h[..., self.output_mask.bool()]
        if self._K is None:
            return F.linear(h, self.W_out_gain * self.readout.weight, self.readout.bias)
        W_out = self.W_out_gain.view(self._K, 1, 1) * self.readout_weight
        logits = torch.einsum("k...e,koe->k...o", h, W_out)
        logits = logits + (self.readout_bias.unsqueeze(1) if logits.ndim == 4 else self.readout_bias)
        if self._has_skip:
            flags = self.cell.skip_mask().view(self._K, 1, 1)
            if logits.ndim == 4:
                flags = flags.unsqueeze(-2)
            logits = logits + flags * x_in
        return logits

    def _step(self, cell: nn.Module, x_t: Tensor, state: Tensor, hoisted) -> tuple[Tensor, Tensor]:
        mark_cudagraph_step()
        out, state = cell(x_t, state) if hoisted is None else cell(x_t, state, hoisted)
        # state is both this step's output and the next step's input; the
        # clone breaks the aliasing that CUDA graphs and Dynamo dislike.
        return out, state.clone()

    def unroll(self, x_seg: Tensor, state: Tensor, *, alpha_seg: Optional[Tensor] = None,
               y_prev: Optional[Tensor] = None, hoisted=None,
               cell: Optional[nn.Module] = None) -> Unrolled:
        """Run the cell over one contiguous time segment ``x_seg`` of shape ``(B, T, C)``.

        Open loop (``alpha_seg`` None) returns the raw cell outputs so the
        caller applies the readout once. Closed loop blends each step's
        input, ``x_in = (1 - alpha) x_real + alpha y_prev``, and must read
        out every step; ``alpha_seg`` is ``(T, C)`` or per-batch ``(B, T, C)``.
        ``hoisted`` is the cell's per-pass constant (see ``RNNCell.hoist``);
        ``cell`` may be a compiled stand-in for ``self.cell``.
        """
        cell = self.cell if cell is None else cell
        T = x_seg.shape[1]
        buf: Optional[Tensor] = None
        with _no_autocast_cache_ctx():
            for t in range(T):
                x_t = x_seg[:, t, :]
                if alpha_seg is not None:
                    alpha_t = alpha_seg[:, t, :] if alpha_seg.dim() == 3 else alpha_seg[t]
                    x_t = (1.0 - alpha_t) * x_t + alpha_t * y_prev
                out, state = self._step(cell, x_t, state, hoisted)
                if alpha_seg is not None:
                    out = y_prev = self.apply_readout(out, x_t)
                if buf is None:
                    buf = empty_time_buffer(out, T)
                buf[..., t, :] = out
        if alpha_seg is None:
            return Unrolled(hidden=buf, y=None, state=state, y_prev=None)
        return Unrolled(hidden=None, y=buf, state=state, y_prev=y_prev)

    def _unroll_tuple(self, x_seg, alpha_seg, state, y_prev, hoisted):
        """Tuple-returning wrapper for ``torch.utils.checkpoint``."""
        r = self.unroll(x_seg, state, alpha_seg=alpha_seg, y_prev=y_prev, hoisted=hoisted)
        return (r.hidden if alpha_seg is None else r.y), r.state, r.y_prev

    # -- full pass ------------------------------------------------------------

    def forward(self, x: Tensor, readout_idx: int | slice | None = None,
                bptt_start_idx: Optional[int] = None, bptt_chunk_len: Optional[int] = None,
                grad_checkpoint: bool = False, grad_checkpoint_segment_len: Optional[int] = None,
                alpha_schedule: Optional[Tensor] = None) -> Tensor:
        """Unroll ``x`` of shape ``(B, T, F)`` and read out at ``readout_idx``.

        Steps before ``bptt_start_idx`` run without gradient; the grad region
        is detached every ``bptt_chunk_len`` steps and, with
        ``grad_checkpoint``, recomputed in backward in segments of
        ``grad_checkpoint_segment_len``. ``alpha_schedule`` ``(T, C)`` turns on
        the closed-loop blend (requires ``input_size == output_size``).

        Returns ``(B, O)`` / ``(K, B, O)`` for an int index, or with a time
        axis before the features for a slice; ``None`` reads the last step.
        """
        B, T, C = x.shape
        closed = alpha_schedule is not None
        if closed:
            if alpha_schedule.shape != (T, C):
                raise ValueError(f"alpha_schedule shape {tuple(alpha_schedule.shape)} "
                                 f"does not match (T={T}, C={C})")
            if C != self.output_size:
                raise ValueError(f"closed loop requires input_size == output_size, "
                                 f"got {C} != {self.output_size}")
            alpha_schedule = alpha_schedule.to(dtype=x.dtype, device=x.device)
            lead = (self._K, B, C) if self._K is not None else (B, C)
            y_prev: Optional[Tensor] = torch.zeros(*lead, device=x.device, dtype=x.dtype)
        else:
            y_prev = None

        state = self.initial_state(B)
        hoisted = self.cell.hoist()
        grad_start = bptt_start_idx or 0
        pieces: list[Tensor] = []

        def run(t0: int, t1: int, state, y_prev, checkpoint: bool):
            alpha = alpha_schedule[t0:t1] if closed else None
            if checkpoint:
                return torch.utils.checkpoint.checkpoint(
                    self._unroll_tuple, x[:, t0:t1, :], alpha, state, y_prev, hoisted,
                    use_reentrant=False)
            return self._unroll_tuple(x[:, t0:t1, :], alpha, state, y_prev, hoisted)

        if grad_start > 0:
            with torch.no_grad():
                out, state, y_prev = run(0, grad_start, state, y_prev, False)
            pieces.append(out)
            state = state.detach()
            y_prev = None if y_prev is None else y_prev.detach()

        if grad_checkpoint and grad_checkpoint_segment_len is not None:
            seg_len = grad_checkpoint_segment_len
        else:
            seg_len = bptt_chunk_len or (T - grad_start)
        seg_len = max(1, seg_len)

        t, since_detach = grad_start, 0
        while t < T:
            end = min(t + seg_len, T)
            out, state, y_prev = run(t, end, state, y_prev, grad_checkpoint)
            pieces.append(out)
            since_detach += end - t
            t = end
            if bptt_chunk_len is not None and since_detach >= bptt_chunk_len and t < T:
                state = state.detach()
                y_prev = None if y_prev is None else y_prev.detach()
                since_detach = 0

        full = torch.cat(pieces, dim=-2)
        idx = readout_idx if readout_idx is not None else -1
        out = full[..., idx, :]
        if closed:
            return out
        return self.apply_readout(out, x[:, idx, :])

    def constrain_parameters(self) -> None:
        self.cell.constrain_parameters()
