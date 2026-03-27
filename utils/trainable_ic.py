"""Trainable initial conditions for RNN cells."""

import torch
import torch.nn as nn


class TrainableIC(nn.Module):
    """Trainable initial state for RNN cells.

    Stores a learnable (state_dim,) parameter that gets tiled to
    (batch, state_dim) at forward time.
    """

    def __init__(self, state_dim: int):
        super().__init__()
        self.ic = nn.Parameter(torch.zeros(state_dim))

    def forward(self, batch_size: int) -> torch.Tensor:
        """Expand the learned IC to a full batch.

        Returns:
            Tensor of shape (batch_size, state_dim).
        """
        return self.ic.unsqueeze(0).expand(batch_size, -1)


def compute_burn_in(cell, input_size, burn_in_seconds=30.0, device="cpu"):
    """Run *cell* with zero input to compute a stable initial condition.

    The number of burn-in timesteps is estimated from the cell's
    ``dt_per_step`` attribute (seconds of simulated time per call).
    If the attribute is absent a default of 0.04 s is assumed.

    Args:
        cell: An RNN cell module whose forward signature is
            ``cell(input, state) -> (output, new_state)``.
        input_size: Dimensionality of the cell's input.
        burn_in_seconds: How many simulated seconds to burn in.
        device: Torch device to use.

    Returns:
        A (state_dim,) tensor suitable for initialising a TrainableIC.
    """
    dt = getattr(cell, "dt_per_step", 0.04)
    n_steps = max(1, int(burn_in_seconds / dt))

    cell = cell.to(device)
    cell.eval()

    with torch.no_grad():
        # Initialise state -- try cell's own method first, fall back to zeros.
        if hasattr(cell, "initial_state"):
            state = cell.initial_state(1, device=device)
        else:
            state = torch.zeros(1, cell.num_units, device=device)

        zero_input = torch.zeros(1, input_size, device=device)
        for _ in range(n_steps):
            _, state = cell(zero_input, state)

    # Squeeze out the batch dimension and return on CPU.
    if isinstance(state, tuple):
        state = state[0]
    return state.squeeze(0).cpu()
