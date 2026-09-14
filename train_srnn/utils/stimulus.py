"""Step stimulus for open-loop simulations of a cell."""
from __future__ import annotations

import numpy as np


def step_stimulus(n_units: int, n_E: int, n_steps_total: int, *, n_steps: int = 3,
                  amp: float = 0.5, density_E: float = 0.15, density_I: float = 0.0,
                  silent: tuple[bool, ...] = (True, False, True), seed: int = 8) -> np.ndarray:
    """``(n_steps_total, n_units)`` piecewise-constant input.

    Time is split into ``n_steps`` equal blocks; each block drives a random
    sparse subset of neurons (``density_E`` of the E population, ``density_I``
    of the I population) with Gaussian amplitudes of scale ``amp``. Blocks
    flagged in ``silent`` carry no input.
    """
    rng = np.random.default_rng(seed)
    block = n_steps_total // n_steps
    amps = amp * rng.standard_normal((n_units, n_steps))
    mask = np.zeros((n_units, n_steps), dtype=bool)
    mask[:n_E] = rng.random((n_E, n_steps)) < density_E
    mask[n_E:] = rng.random((n_units - n_E, n_steps)) < density_I
    amps *= mask
    for k, off in enumerate(silent):
        if off:
            amps[:, k] = 0.0
    u = np.zeros((n_steps_total, n_units), dtype=np.float32)
    for k in range(n_steps):
        u[k * block:(k + 1) * block] = amps[:, k]
    return u
