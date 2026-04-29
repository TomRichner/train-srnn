"""Sequence model wrapper for RNN cells.

Wraps any RNN cell into a full sequence-to-prediction model with
time-step unrolling, optional I/O masking, readout head, trainable
initial conditions, and truncated BPTT support.

Supports both single-variant cells (output shape ``(B, N)``) and
K-batched cells like ``BatchedSRNNCell`` (output shape ``(K, B, N)``).
"""

import math

import torch
import torch.nn as nn
import torch.utils.checkpoint

from train_srnn.utils.io_masks import (
    generate_neuron_partition,
    make_input_mask,
    make_output_mask,
)
from train_srnn.utils.trainable_ic import TrainableIC


class LSTMCellWrapper(nn.Module):
    """Wraps nn.LSTMCell to match our cell interface.

    The combined state is ``[h, c]`` concatenated along the last dimension,
    so ``state_size = num_units * 2``.
    """

    def __init__(self, input_size: int, num_units: int):
        super().__init__()
        self.cell = nn.LSTMCell(input_size, num_units)
        self.num_units = num_units
        self.state_size = num_units * 2  # h + c

    def forward(self, input: torch.Tensor, state: torch.Tensor):
        h, c = state.chunk(2, dim=-1)
        h_new, c_new = self.cell(input, (h, c))
        return h_new, torch.cat([h_new, c_new], dim=-1)


class SequenceModel(nn.Module):
    """Wraps an RNN cell into a full sequence-to-prediction model.

    Args:
        cell: RNN cell module (LTCCell, SRNNCell, CTRNNCell, etc.)
            Must implement: ``cell(input, state) -> (output, new_state)``
            Must have: ``cell.state_size`` (int)
        input_size: Number of input features per timestep.
        output_size: Number of output classes or regression dims.
        num_units: Hidden size of the RNN cell.
        use_io_masks: Whether to use neuron partitioning (input / inter / output).
        io_mask_seed: Seed for the neuron partition random generator.
        trainable_ic: Whether to learn initial conditions.
        task_type: ``"classification"`` or ``"regression"``.
    """

    def __init__(
        self,
        cell: nn.Module,
        input_size: int,
        output_size: int,
        num_units: int,
        use_io_masks: bool = True,
        io_mask_seed: int = 0,
        trainable_ic: bool = True,
        task_type: str = "classification",
    ):
        super().__init__()
        self.cell = cell
        self.use_io_masks = use_io_masks
        self.task_type = task_type

        # K-batched detection --------------------------------------------------
        self._K = getattr(cell, "K", None)

        # Skip-connection: per-variant residual y = readout(state) + α_k · x.
        # Only supported in batched mode for v1. The flag tensor lives on the
        # cell as `cell.skip_flags` (registered buffer), so it follows
        # model.to(device) automatically; forward views it on the fly.
        self._has_skip = bool(
            self._K is not None and getattr(cell, "any_skip", False)
        )
        if self._has_skip:
            if input_size != output_size:
                raise ValueError(
                    f"skip variants require input_size == output_size, got "
                    f"{input_size} != {output_size}"
                )
        elif self._K is None:
            cfg = getattr(cell, "config", None)
            if cfg is not None and getattr(cfg, "skip", False):
                raise NotImplementedError(
                    "skip is currently only supported in batched_ablations mode"
                )

        # I/O masks -----------------------------------------------------------
        effective_output_size = num_units
        if use_io_masks:
            input_idx, inter_idx, output_idx = generate_neuron_partition(
                num_units, io_mask_seed
            )
            self.register_buffer(
                "input_mask",
                torch.tensor(make_input_mask(num_units, input_idx), dtype=torch.float32),
            )
            self.register_buffer(
                "output_mask",
                torch.tensor(make_output_mask(num_units, output_idx), dtype=torch.float32),
            )
            effective_output_size = len(output_idx)

        # Trainable initial conditions -----------------------------------------
        self.trainable_ic = trainable_ic
        if trainable_ic:
            self.ic = TrainableIC(cell.state_size, K=self._K)

        # Readout head ---------------------------------------------------------
        if self._K is not None:
            # K independent readout heads stored as batched parameters for bmm
            self.readout_weight = nn.Parameter(
                torch.empty(self._K, output_size, effective_output_size)
            )
            self.readout_bias = nn.Parameter(
                torch.zeros(self._K, 1, output_size)
            )
            for k in range(self._K):
                nn.init.kaiming_uniform_(self.readout_weight[k], a=math.sqrt(5))
            self.readout = None  # sentinel: use batched readout path
        else:
            self.readout = nn.Linear(effective_output_size, output_size)

    # ------------------------------------------------------------------
    @staticmethod
    def _no_autocast_cache_ctx():
        """If autocast is currently active, return a context that re-applies
        it with ``cache_enabled=False`` so subsequent calls don't reuse cached
        bf16 casts. Otherwise return a no-op nullcontext.

        Why: PyTorch's autocast cache stores fp32->bf16 casts of weight
        tensors. Under ``torch.utils.checkpoint``, the cache state diverges
        between the original forward (cache hits) and the recompute (cache
        empty), producing a different count of saved ``aten._to_copy`` ops
        and triggering ``CheckpointError`` on saved-tensor metadata sanity
        checks. Disabling the cache makes both passes save the same count.
        See KnownIssues #7.
        """
        import contextlib
        for dev in ("cuda", "cpu"):
            try:
                if torch.is_autocast_enabled(dev):
                    return torch.autocast(
                        device_type=dev,
                        dtype=torch.get_autocast_dtype(dev),
                        cache_enabled=False,
                    )
            except (TypeError, RuntimeError):
                # Older PyTorch: no device argument.
                if dev == "cuda" and torch.is_autocast_enabled():
                    return torch.autocast(
                        device_type="cuda",
                        dtype=torch.get_autocast_gpu_dtype(),
                        cache_enabled=False,
                    )
                break
        return contextlib.nullcontext()

    def _run_segment(self, x_seg: torch.Tensor, state: torch.Tensor):
        """Run the cell over a contiguous time slice.

        Returns a stacked outputs tensor with a time axis inserted just
        before the feature axis: ``(B, T_seg, E)`` for single cells,
        ``(K, B, T_seg, E)`` for K-batched cells.
        """
        outputs = []
        with self._no_autocast_cache_ctx():
            for t in range(x_seg.shape[1]):
                out, state = self.cell(x_seg[:, t, :], state)
                outputs.append(out)
        return torch.stack(outputs, dim=-2), state

    # ------------------------------------------------------------------
    def _readout_one(self, out_t: torch.Tensor, x_in_t: torch.Tensor) -> torch.Tensor:
        """Apply output_mask + readout head + (optional) skip residual at one timestep.

        Mirrors the end-of-forward block but for a single timestep. Only used
        by the closed-loop forward path (the open-loop path keeps the inline
        end-of-forward block to guarantee byte-identical numerics).

        Args:
            out_t: hidden-units output from the cell at one timestep.
                ``(B, E)`` single mode, ``(K, B, E)`` K-batched.
            x_in_t: the actual input fed to the cell at this timestep
                (already including any closed-loop blend). Used for the
                skip residual ``y += alpha_skip * x_in_t``.
                ``(B, C)`` or ``(K, B, C)``.

        Returns:
            ``(B, O)`` single mode or ``(K, B, O)`` K-batched.
        """
        # Output mask
        if hasattr(self, "output_mask"):
            out_t = out_t * self.output_mask
            out_t = out_t[..., self.output_mask.bool()]

        # Readout head
        if self._K is not None:
            # einsum handles (K, B, E) -> (K, B, O); bias (K, 1, O) broadcasts.
            logits = torch.einsum("k...e,koe->k...o", out_t, self.readout_weight)
            logits = logits + self.readout_bias

            if self._has_skip:
                # skip_flags: (K,) -> (K, 1, 1); x_in_t may be (K, B, C) or (B, C).
                skip_flags = self.cell.skip_flags.view(self._K, 1, 1)
                if x_in_t.dim() == 2:  # (B, C) broadcast to all K variants
                    x_in_kb = x_in_t.unsqueeze(0)  # (1, B, C) -> broadcasts on K
                else:
                    x_in_kb = x_in_t  # (K, B, C)
                logits = logits + skip_flags * x_in_kb
            return logits
        else:
            return self.readout(out_t)

    # ------------------------------------------------------------------
    def _cl_run_segment(
        self,
        x_seg: torch.Tensor,         # (B, T_seg, C)
        alpha_seg: torch.Tensor,     # (T_seg, C)
        state: torch.Tensor,         # (B, S) or (K, B, S)
        y_prev: torch.Tensor,        # (B, C) or (K, B, C); zeros at t=0
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Closed-loop unroll over one contiguous time slice.

        Returns ``(y_outs, state, y_prev)`` where ``y_outs`` is in OUTPUT
        space (post-readout, post-skip), shape ``(B, T_seg, O)`` for single
        cells or ``(K, B, T_seg, O)`` for K-batched cells.

        This is the closed-loop counterpart to ``_run_segment``; the extra
        carry ``y_prev`` is the last-step prediction, which serves as the
        autoregressive feedback for the next step's input blend.

        Wrapped in ``torch.utils.checkpoint.checkpoint(...)`` from the
        forward path when ``grad_checkpoint=True``. Pure function of inputs:
        no RNG, no side effects.
        """
        y_outs: list[torch.Tensor] = []
        with self._no_autocast_cache_ctx():
            for t in range(x_seg.shape[1]):
                x_real_t = x_seg[:, t, :]
                alpha_t = alpha_seg[t]
                x_in_t = (1.0 - alpha_t) * x_real_t + alpha_t * y_prev
                out_t, state = self.cell(x_in_t, state)
                y_t = self._readout_one(out_t, x_in_t)
                y_outs.append(y_t)
                y_prev = y_t
        y_stacked = torch.stack(y_outs, dim=-2)
        return y_stacked, state, y_prev

    # ------------------------------------------------------------------
    def _forward_closed_loop(
        self,
        x: torch.Tensor,
        *,
        alpha_schedule: torch.Tensor,
        readout_idx: int | slice | None,
        bptt_start_idx: int | None,
        bptt_chunk_len: int | None,
        grad_checkpoint: bool,
        grad_checkpoint_segment_len: int | None,
    ) -> torch.Tensor:
        """Closed-loop unroll: cell input is a per-channel blend of real
        input and the model's previous prediction.

        See ``forward`` docstring for the alpha_schedule semantics. The
        ``grad_checkpoint`` flag is supported and behaves identically to
        the open-loop branch — each segment is wrapped in
        ``torch.utils.checkpoint.checkpoint(...)`` with ``use_reentrant=False``.
        """
        batch_size, seq_len, n_features = x.shape

        # Validation ----------------------------------------------------------
        if alpha_schedule.shape != (seq_len, n_features):
            raise ValueError(
                f"alpha_schedule shape {tuple(alpha_schedule.shape)} does not "
                f"match (T={seq_len}, C={n_features})"
            )
        # input_size == output_size is required because we feed y_pred back as x.
        if self._K is not None:
            out_size = self.readout_weight.shape[1]
        else:
            out_size = self.readout.out_features
        if n_features != out_size:
            raise ValueError(
                f"alpha_schedule (closed-loop) requires input_size == "
                f"output_size, got {n_features} != {out_size}"
            )

        # Match dtype/device to x ---------------------------------------------
        if alpha_schedule.dtype != x.dtype or alpha_schedule.device != x.device:
            alpha_schedule = alpha_schedule.to(dtype=x.dtype, device=x.device)

        # Initial state + y_prev (zeros; alpha[0,:]=0 by construction makes the
        # blend at t=0 trivially x_real[:, 0, :], regardless of y_prev's value).
        if hasattr(self, "ic"):
            state = self.ic(batch_size)
        elif self._K is not None:
            state = torch.zeros(
                self._K, batch_size, self.cell.state_size, device=x.device
            )
        else:
            state = torch.zeros(
                batch_size, self.cell.state_size, device=x.device
            )

        if self._K is not None:
            y_prev = torch.zeros(self._K, batch_size, n_features,
                                 device=x.device, dtype=x.dtype)
        else:
            y_prev = torch.zeros(batch_size, n_features,
                                 device=x.device, dtype=x.dtype)

        all_y: list[torch.Tensor] = []  # per-segment y in OUTPUT space

        grad_start = bptt_start_idx if bptt_start_idx is not None else 0

        # 1) Warmup region (no_grad) — never checkpointed (no graph anyway).
        if grad_start > 0:
            with torch.no_grad():
                outs, state, y_prev = self._cl_run_segment(
                    x[:, :grad_start, :], alpha_schedule[:grad_start],
                    state, y_prev,
                )
            all_y.append(outs)
            state = state.detach()
            y_prev = y_prev.detach()

        # 2) Grad region: choose seg_len matching open-loop logic ------------
        if grad_checkpoint:
            seg_len = (grad_checkpoint_segment_len
                       if grad_checkpoint_segment_len is not None
                       else (bptt_chunk_len or (seq_len - grad_start)))
        else:
            seg_len = bptt_chunk_len or (seq_len - grad_start)
        seg_len = max(1, seg_len)

        t = grad_start
        steps_since_detach = 0
        while t < seq_len:
            end = min(t + seg_len, seq_len)
            x_seg = x[:, t:end, :]
            alpha_seg = alpha_schedule[t:end]

            if grad_checkpoint:
                outs, state, y_prev = torch.utils.checkpoint.checkpoint(
                    self._cl_run_segment,
                    x_seg, alpha_seg, state, y_prev,
                    use_reentrant=False,
                )
            else:
                outs, state, y_prev = self._cl_run_segment(
                    x_seg, alpha_seg, state, y_prev
                )

            all_y.append(outs)
            steps_since_detach += (end - t)
            t = end

            # Detach at chunk boundary (cap grad horizon to bptt_chunk_len).
            if (bptt_chunk_len is not None
                    and steps_since_detach >= bptt_chunk_len
                    and t < seq_len):
                state = state.detach()
                y_prev = y_prev.detach()
                steps_since_detach = 0

        # Concatenate along the time axis (-2). full_y shape:
        #   single:  (B, T, O)
        #   batched: (K, B, T, O)
        full_y = torch.cat(all_y, dim=-2)

        # Select readout timestep(s) -----------------------------------------
        if isinstance(readout_idx, slice):
            return full_y[..., readout_idx, :]
        elif readout_idx is not None:
            return full_y[..., readout_idx, :]
        else:
            return full_y[..., -1, :]

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        readout_idx: int | slice | None = None,
        bptt_start_idx: int | None = None,
        bptt_chunk_len: int | None = None,
        grad_checkpoint: bool = False,
        grad_checkpoint_segment_len: int | None = None,
        alpha_schedule: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: ``(batch, seq_len, features)`` -- batch-first input.
            readout_idx: Which timestep(s) to read output from. ``int`` selects
                one timestep (classic single-readout behavior); ``slice`` stacks
                outputs over that range along a new time axis so multi-step
                loss can be computed; ``None`` = last timestep.
            bptt_start_idx: Detach gradients before this index (truncated BPTT
                warmup: steps < bptt_start_idx run under ``torch.no_grad()``).
            bptt_chunk_len: If set, call ``state.detach()`` every this many
                steps inside the grad region so each loss term's backward path
                is capped to at most ``bptt_chunk_len`` cell applications.
                ``None`` = one contiguous graph over the grad region.
            grad_checkpoint: If True, wrap each segment of the grad region in
                ``torch.utils.checkpoint.checkpoint`` so its forward
                intermediates are *recomputed* during backward instead of
                saved. Trades ~30-50% wall-clock for 5-10x activation memory.
            grad_checkpoint_segment_len: Length of each checkpoint segment
                within the grad region. Only consulted when
                ``grad_checkpoint=True``. ``None`` defaults to
                ``bptt_chunk_len`` (one checkpoint per detach-chunk).
            alpha_schedule: Optional ``(T, C)`` tensor enabling **closed-loop**
                (variable teacher-forcing) unroll. When set, at each step ``t``
                the cell sees ``x_in[t,c] = (1 - alpha[t,c]) * x_real[t,c] +
                alpha[t,c] * y_pred[t-1,c]``. ``alpha[0,:]`` should be 0 (the
                schedule sampler enforces this) since ``y_prev`` at t=0 is
                zeros. Requires ``input_size == output_size``. Composes with
                ``grad_checkpoint=True``: each segment is wrapped in
                ``torch.utils.checkpoint.checkpoint`` and the carried
                ``y_prev`` is threaded through the checkpointed callable.
                Per-step readout cost is paid every timestep (unavoidable).

        Returns:
            logits: ``(batch, output_size)`` for single cells, or
                    ``(K, batch, output_size)`` for K-batched cells. When
                    ``readout_idx`` is a slice, an extra time axis is inserted
                    before the output axis: ``(batch, T, output_size)`` or
                    ``(K, batch, T, output_size)``.
        """
        # Closed-loop dispatch ------------------------------------------------
        if alpha_schedule is not None:
            return self._forward_closed_loop(
                x,
                alpha_schedule=alpha_schedule,
                readout_idx=readout_idx,
                bptt_start_idx=bptt_start_idx,
                bptt_chunk_len=bptt_chunk_len,
                grad_checkpoint=grad_checkpoint,
                grad_checkpoint_segment_len=grad_checkpoint_segment_len,
            )

        batch_size, seq_len, _ = x.shape

        # Initial state --------------------------------------------------------
        if hasattr(self, "ic"):
            state = self.ic(batch_size)
        elif self._K is not None:
            state = torch.zeros(
                self._K, batch_size, self.cell.state_size, device=x.device
            )
        else:
            state = torch.zeros(
                batch_size, self.cell.state_size, device=x.device
            )

        # Segmented unroll -----------------------------------------------------
        # Runs the warmup region (no_grad) as one segment, then the grad region
        # in segments of size `seg_len`, optionally wrapped in checkpoint().
        all_outputs: list[torch.Tensor] = []

        # 1. Warmup region (no_grad) — never needs checkpointing.
        grad_start = bptt_start_idx if bptt_start_idx is not None else 0
        if grad_start > 0:
            with torch.no_grad():
                outs, state = self._run_segment(x[:, :grad_start, :], state)
            all_outputs.append(outs)
            state = state.detach()

        # 2. Grad region.
        if grad_checkpoint:
            seg_len = (grad_checkpoint_segment_len
                       if grad_checkpoint_segment_len is not None
                       else (bptt_chunk_len or (seq_len - grad_start)))
        else:
            seg_len = bptt_chunk_len or (seq_len - grad_start)
        seg_len = max(1, seg_len)

        t = grad_start
        steps_since_detach = 0
        while t < seq_len:
            end = min(t + seg_len, seq_len)
            x_seg = x[:, t:end, :]

            if grad_checkpoint:
                outs, state = torch.utils.checkpoint.checkpoint(
                    self._run_segment, x_seg, state, use_reentrant=False,
                )
            else:
                outs, state = self._run_segment(x_seg, state)

            all_outputs.append(outs)
            steps_since_detach += (end - t)
            t = end

            # Detach at chunk boundary (existing semantics): cap the grad
            # horizon to bptt_chunk_len cell applications.
            if (bptt_chunk_len is not None
                    and steps_since_detach >= bptt_chunk_len
                    and t < seq_len):
                state = state.detach()
                steps_since_detach = 0

        # Concatenate along the time axis (-2). full_outputs shape:
        #   single:  (B, T, E)
        #   batched: (K, B, T, E)
        full_outputs = torch.cat(all_outputs, dim=-2)

        # Select readout timestep(s) -------------------------------------------
        if isinstance(readout_idx, slice):
            out = full_outputs[..., readout_idx, :]
        elif readout_idx is not None:
            out = full_outputs[..., readout_idx, :]
        else:
            out = full_outputs[..., -1, :]

        # Apply output mask ----------------------------------------------------
        if hasattr(self, "output_mask"):
            out = out * self.output_mask
            # [..., mask] indexes last dim for any leading shape.
            out = out[..., self.output_mask.bool()]

        # Readout head ---------------------------------------------------------
        if self._K is not None:
            # einsum handles both (K, B, E) and (K, B, T, E) uniformly.
            # readout_weight: (K, O, E); readout_bias: (K, 1, O).
            logits = torch.einsum(
                "k...e,koe->k...o", out, self.readout_weight
            )
            if logits.ndim == 4:
                # (K, B, T, O) needs (K, 1, 1, O) bias for broadcasting.
                logits = logits + self.readout_bias.unsqueeze(1)
            else:
                logits = logits + self.readout_bias

            # Skip / residual term: y_pred += α_k · x_at_readout. α_k is 0 for
            # non-skip variants, so this adds exactly zero for them. Skip is
            # added in output-space (post-readout), so the input-feature dim F
            # must equal output dim O — checked at __init__.
            if self._has_skip:
                skip_flags = self.cell.skip_flags.view(self._K, 1, 1)  # (K, 1, 1)
                if isinstance(readout_idx, slice):
                    x_at_readout = x[:, readout_idx, :]               # (B, T, F)
                    logits = logits + skip_flags.unsqueeze(-2) * x_at_readout
                elif readout_idx is not None:
                    x_at_readout = x[:, readout_idx, :]               # (B, F)
                    logits = logits + skip_flags * x_at_readout
                else:
                    x_at_readout = x[:, -1, :]                         # (B, F)
                    logits = logits + skip_flags * x_at_readout
            return logits
        else:
            # nn.Linear broadcasts over leading dims: (B, E) or (B, T, E).
            return self.readout(out)

    # ------------------------------------------------------------------
    def constrain_parameters(self):
        """Apply parameter constraints (e.g. LTC weight clipping)."""
        if hasattr(self.cell, "constrain_parameters"):
            self.cell.constrain_parameters()
