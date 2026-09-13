"""Common interface for every recurrent cell."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import torch
import torch.nn as nn
from torch import Tensor


class RNNCell(nn.Module, ABC):
    """``cell(x, state) -> (output, new_state)`` with a flat state vector.

    Single-network cells carry state as ``(B, state_size)`` and leave ``K``
    at ``None``. Batched cells run ``K`` networks side by side with state
    ``(K, B, state_size)`` and input ``(B, I)`` or ``(K, B, I)``.

    Attributes:
        input_size, num_units, state_size: sizes of the flat interfaces.
        dt: seconds of simulated time per call, or None for discrete cells.
        W_in_mask: optional buffer zeroing input rows of non-input neurons,
            stored pre-broadcast to the shape the subclass multiplies with.
    """

    K: Optional[int] = None

    def __init__(
        self,
        input_size: int,
        num_units: int,
        state_size: int,
        *,
        dt: Optional[float] = None,
        W_in_mask: Optional[Tensor] = None,
        mask_shape: tuple[int, ...] = (1, -1),
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.num_units = num_units
        self.state_size = state_size
        self.dt = dt
        self.register_buffer(
            "W_in_mask", None if W_in_mask is None else W_in_mask.reshape(mask_shape))

    @abstractmethod
    def forward(self, inputs: Tensor, state: Tensor, *args: Any) -> tuple[Tensor, Tensor]:
        ...

    @abstractmethod
    def init_state(self, batch_size: int, device: torch.device | None = None) -> Tensor:
        ...

    def hoist(self) -> Any:
        """Per-forward-pass constants the caller may compute once and pass back in."""
        return None

    def skip_mask(self) -> Optional[Tensor]:
        """``(K,)`` residual-skip flags for batched cells; None otherwise."""
        return None

    @torch.no_grad()
    def constrain_parameters(self) -> None:
        """Project parameters back into their valid ranges after an optimizer step."""
