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
        readout_idx: int | None = None,
        bptt_start_idx: int | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: ``(batch, seq_len, features)`` -- batch-first input.
            readout_idx: Which timestep to read output from (``None`` = last).
            bptt_start_idx: Detach gradients before this index for truncated BPTT.

        Returns:
            logits: ``(batch, output_size)`` for single cells, or
                    ``(K, batch, output_size)`` for K-batched cells.
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
        for t in range(seq_len):
            inp = x[:, t, :]

            if bptt_start_idx is not None and t < bptt_start_idx:
                with torch.no_grad():
                    output, state = self.cell(inp, state)
                state = state.detach()
            else:
                output, state = self.cell(inp, state)

            outputs.append(output)

        # Select readout timestep ----------------------------------------------
        if readout_idx is not None:
            out = outputs[readout_idx]
        else:
            out = outputs[-1]

        # Apply output mask ----------------------------------------------------
        if hasattr(self, "output_mask"):
            out = out * self.output_mask
            # [..., mask] indexes last dim for both (B, N) and (K, B, N)
            out = out[..., self.output_mask.bool()]

        # Readout head ---------------------------------------------------------
        if self._K is not None:
            # out: (K, B, E), weight: (K, O, E) -> bmm needs (K, B, E) @ (K, E, O)
            logits = torch.bmm(
                out, self.readout_weight.transpose(-1, -2)
            ) + self.readout_bias
            return logits  # (K, B, O)
        else:
            return self.readout(out)

    # ------------------------------------------------------------------
    def constrain_parameters(self):
        """Apply parameter constraints (e.g. LTC weight clipping)."""
        if hasattr(self.cell, "constrain_parameters"):
            self.cell.constrain_parameters()
