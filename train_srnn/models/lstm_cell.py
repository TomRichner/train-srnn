"""LSTM baseline behind the flat-state cell interface."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from train_srnn.models.base import RNNCell


class LSTMCell(RNNCell):
    """``nn.LSTMCell`` with state stored as ``[h, c]`` along the last axis."""

    def __init__(self, input_size: int, num_units: int):
        super().__init__(input_size, num_units, state_size=2 * num_units)
        self.cell = nn.LSTMCell(input_size, num_units)

    def forward(self, inputs: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
        h, c = state.chunk(2, dim=-1)
        h_new, c_new = self.cell(inputs, (h, c))
        return h_new, torch.cat([h_new, c_new], dim=-1)

    def init_state(self, batch_size: int, device: torch.device | None = None) -> Tensor:
        return torch.zeros(batch_size, self.state_size, device=device)
