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
    def forward(
        self,
        x: torch.Tensor,
        readout_idx: int | slice | None = None,
        bptt_start_idx: int | None = None,
        bptt_chunk_len: int | None = None,
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

        Returns:
            logits: ``(batch, output_size)`` for single cells, or
                    ``(K, batch, output_size)`` for K-batched cells. When
                    ``readout_idx`` is a slice, an extra time axis is inserted
                    before the output axis: ``(batch, T, output_size)`` or
                    ``(K, batch, T, output_size)``.
        """
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

        # Unroll ---------------------------------------------------------------
        outputs: list[torch.Tensor] = []
        steps_in_chunk = 0
        for t in range(seq_len):
            inp = x[:, t, :]

            if bptt_start_idx is not None and t < bptt_start_idx:
                with torch.no_grad():
                    output, state = self.cell(inp, state)
                state = state.detach()
            else:
                output, state = self.cell(inp, state)
                steps_in_chunk += 1
                if (bptt_chunk_len is not None
                        and steps_in_chunk >= bptt_chunk_len
                        and t < seq_len - 1):
                    state = state.detach()
                    steps_in_chunk = 0

            outputs.append(output)

        # Select readout timestep(s) -------------------------------------------
        if isinstance(readout_idx, slice):
            # Stack along a new time axis just before the feature axis so the
            # shape is (B, T, E) for single cells and (K, B, T, E) for batched.
            out = torch.stack(outputs[readout_idx], dim=-2)
        elif readout_idx is not None:
            out = outputs[readout_idx]
        else:
            out = outputs[-1]

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
