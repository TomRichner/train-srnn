"""Trainable initial conditions for RNN cells."""

import torch
import torch.nn as nn

from train_srnn.utils.cell_loop import mark_cudagraph_step


class TrainableIC(nn.Module):
    """Trainable initial state for RNN cells.

    For single-variant cells, stores a learnable (state_dim,) parameter
    that gets tiled to (batch, state_dim) at forward time.

    For K-batched cells (SRNNCell), stores (K, state_dim) and
    returns (K, batch, state_dim).
    """

    def __init__(self, state_dim: int, K: int | None = None):
        super().__init__()
        self.K = K
        if K is not None:
            self.ic = nn.Parameter(torch.zeros(K, state_dim))
        else:
            self.ic = nn.Parameter(torch.zeros(state_dim))

    def forward(self, batch_size: int) -> torch.Tensor:
        """Expand the learned IC to a full batch.

        Returns:
            (batch_size, state_dim) if non-batched, or
            (K, batch_size, state_dim) if K-batched.
        """
        if self.K is not None:
            return self.ic.unsqueeze(1).expand(self.K, batch_size, -1)
        return self.ic.unsqueeze(0).expand(batch_size, -1)


def compute_burn_in(cell, input_size, burn_in_seconds=30.0, device="cpu"):
    """Run *cell* with zero input to compute a stable initial condition.

    The number of steps follows the cell's ``dt`` (seconds per call);
    discrete cells without a ``dt`` are run at 0.04 s per step.

    Args:
        cell: An RNN cell module whose forward signature is
            ``cell(input, state) -> (output, new_state)``.
        input_size: Dimensionality of the cell's input.
        burn_in_seconds: How many simulated seconds to burn in.
        device: Torch device to use.

    Returns:
        A (state_dim,) tensor for single cells, or (K, state_dim) for
        batched cells, suitable for initialising a TrainableIC.
    """
    dt = cell.dt if cell.dt is not None else 0.04
    n_steps = max(1, int(burn_in_seconds / dt))

    cell = cell.to(device)
    cell.eval()

    with torch.no_grad():
        state = cell.init_state(1, device=device)
        zero_input = torch.zeros(1, input_size, device=device)
        for _ in range(n_steps):
            mark_cudagraph_step()
            _, state = cell(zero_input, state)
            # Clone for compatibility with torch.compile(mode="reduce-overhead"):
            # state is both an output of the compiled cell and the next call's
            # input, so it cannot be recycled by mark_cudagraph_step.
            state = state.clone()

    # Drop the batch axis: (K, 1, S) -> (K, S) or (1, S) -> (S,).
    return (state.squeeze(1) if cell.K is not None else state.squeeze(0)).cpu()
