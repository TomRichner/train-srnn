"""A small deterministic trace for smoke tests that need no data on disk."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from train_srnn.data.task import TASKS, TraceTask


@TASKS.register("synthetic")
class SyntheticTraceTask(TraceTask):
    """Sum of incommensurate sinusoids with a little coupling between channels."""

    def _trace(self, n: int, seed: int) -> np.ndarray:
        c = self.cfg
        rng = np.random.RandomState(seed)
        t = np.arange(n) / c.sample_rate_hz
        freqs = rng.uniform(0.2, 2.0, size=c.input_size)
        phases = rng.uniform(0, 2 * np.pi, size=c.input_size)
        x = np.sin(2 * np.pi * freqs[None, :] * t[:, None] + phases[None, :])
        x = x + 0.3 * np.roll(x, 1, axis=1) ** 2
        return x.astype(np.float32)

    def read_traces(self, data_dir: Path):
        c = self.cfg
        return (self._trace(c.train_samples, c.seed),
                self._trace(c.eval_samples, c.seed + 1),
                self._trace(c.eval_samples, c.seed + 2))
