"""Task interface: how a dataset is loaded, batched, scored.

A ``Task`` is built from its config (``cfg.task``) and knows how to load the
arrays, wrap a training or evaluation batch, and compute the loss and metric.
Tasks register themselves in ``TASKS`` by name; ``build_task`` looks them up.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from numpy.lib.stride_tricks import sliding_window_view

from train_srnn.data.transforms import wrap_eval_batch, wrap_train_batch
from train_srnn.registry import Registry

log = logging.getLogger(__name__)

TASKS: Registry[type["Task"]] = Registry("task")


@dataclass
class Dataset:
    """Batch-first ``(N, T, F)`` arrays per split, plus the unbroken train trace for ring training."""
    train: tuple[np.ndarray, np.ndarray]
    valid: tuple[np.ndarray, np.ndarray]
    test: tuple[np.ndarray, np.ndarray]
    input_size: int
    output_size: int
    train_trace: Optional[np.ndarray] = None   # (T, C), only for trace tasks
    # Input-output trace tasks: the target trace (T, O) aligned with train_trace; the
    # target of input sample t is train_target[t + target_shift]. None = next-step.
    train_target: Optional[np.ndarray] = None
    target_shift: int = 1


@dataclass
class Batch:
    x: np.ndarray                     # (B, window_len, F)
    y: np.ndarray                     # labels at the readout steps
    readout_idx: int | slice
    bptt_start_idx: Optional[int]     # steps before this run without grad; None at eval


class Task(ABC):
    def __init__(self, cfg: Any):
        self.cfg = cfg

    @property
    def name(self) -> str:
        return self.cfg.name

    @abstractmethod
    def load(self, data_dir: Path) -> Dataset:
        ...

    def validate(self, data: Dataset) -> None:
        if data.input_size != self.cfg.input_size or data.output_size != self.cfg.output_size:
            raise ValueError(
                f"Dataset has input/output size {data.input_size}/{data.output_size} "
                f"but task config says {self.cfg.input_size}/{self.cfg.output_size}. "
                f"Override with task.input_size=... task.output_size=...")

    # -- loss and metric ----------------------------------------------------

    @property
    def is_classification(self) -> bool:
        return self.cfg.task_type == "classification"

    def criterion(self) -> nn.Module:
        return nn.CrossEntropyLoss() if self.is_classification else nn.MSELoss()

    def target_tensor(self, y: np.ndarray, device: torch.device) -> torch.Tensor:
        dtype = torch.long if self.is_classification else torch.float32
        return torch.tensor(y, dtype=dtype, device=device)

    def loss(self, logits: torch.Tensor, y: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
        """Loss for one network's logits ``(B, [T,] O)``."""
        if not self.is_classification and logits.shape[-1] == 1:
            logits = logits.squeeze(-1)
        return criterion(logits, y)

    def metric(self, logits: torch.Tensor, y: torch.Tensor) -> float:
        """Accuracy for classification, mean absolute error for regression."""
        with torch.no_grad():
            if self.is_classification:
                return float((logits.argmax(dim=-1) == y).float().mean())
            if logits.shape[-1] == 1:
                logits = logits.squeeze(-1)
            return float(torch.mean(torch.abs(logits - y)))

    # -- batching -----------------------------------------------------------

    def train_batch(self, x: np.ndarray, y: np.ndarray, rng: np.random.RandomState) -> Batch:
        c = self.cfg
        x, y, readout_idx, bptt_start = wrap_train_batch(
            x, y, rng, c.stretch_lo, c.stretch_hi, c.window_len, c.bptt_len,
            c.per_timestep_labels, no_augment=c.no_augment, loss_over_bptt=c.loss_over_bptt)
        if c.per_timestep_labels:
            y = y[:, readout_idx]
        return Batch(x, y, readout_idx, bptt_start)

    def eval_batch(self, x: np.ndarray, y: np.ndarray) -> Batch:
        c = self.cfg
        x, y, readout_idx = wrap_eval_batch(
            x, y, c.window_len, c.per_timestep_labels, no_augment=c.no_augment,
            loss_over_bptt=c.loss_over_bptt, bptt_len=c.bptt_len)
        return Batch(x, y, readout_idx, None)


class TraceTask(Task):
    """One long multichannel trace per split; the target is the trace one sample ahead.

    Subclasses read the raw traces; this class truncates the train trace,
    z-scores every split with train statistics, and cuts strided windows for
    the windowed evaluation path. The z-scored train trace is kept whole for
    the continuous trainer.
    """

    @abstractmethod
    def read_traces(self, data_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(train, valid, test)`` traces, each ``(T, C)`` float32."""

    def load(self, data_dir: Path) -> Dataset:
        c = self.cfg
        train, valid, test = self.read_traces(data_dir)
        if c.train_trace_max_len is not None and train.shape[0] > c.train_trace_max_len:
            train = train[:c.train_trace_max_len]
        if c.normalize:
            mu = train.mean(axis=0, keepdims=True)
            sd = train.std(axis=0, keepdims=True)
            sd[sd < 1e-8] = 1.0
            train, valid, test = (train - mu) / sd, (valid - mu) / sd, (test - mu) / sd
        n_chan = train.shape[1]
        log.info("%s: %d channels, train/valid/test = %d/%d/%d samples, normalize=%s",
                 self.name, n_chan, len(train), len(valid), len(test), c.normalize)
        if c.target_channels is not None:
            return self._io_dataset(train, valid, test)
        return Dataset(
            train=self._windows(train), valid=self._windows(valid), test=self._windows(test),
            input_size=n_chan, output_size=n_chan, train_trace=train,
        )

    def _io_dataset(self, train, valid, test) -> Dataset:
        """Map ``input_channels`` at t to ``target_channels`` at ``t + target_shift``."""
        c = self.cfg
        ins, outs, shift = list(c.input_channels), list(c.target_channels), int(c.target_shift)
        log.info("%s: input channels %s -> target channels %s, shift %d", self.name, ins, outs, shift)

        def windows(trace):
            seq_len, stride = c.seq_len, c.stride
            x_all, y_all = trace[:len(trace) - shift, ins], trace[shift:, outs]
            xw = sliding_window_view(x_all, (seq_len, len(ins)))[:, 0][::stride]
            yw = sliding_window_view(y_all, (seq_len, len(outs)))[:, 0][::stride]
            return xw, (yw[..., 0] if len(outs) == 1 else yw)   # single output: (N, T), like Task.loss

        return Dataset(train=windows(train), valid=windows(valid), test=windows(test),
                       input_size=len(ins), output_size=len(outs), train_trace=train[:, ins],
                       train_target=train[:, outs], target_shift=shift)

    def _windows(self, trace: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        seq_len, stride, n_chan = self.cfg.seq_len, self.cfg.stride, trace.shape[1]
        if trace.shape[0] < seq_len + 1:
            raise ValueError(f"{self.name}: split has {trace.shape[0]} samples, "
                             f"need at least seq_len + 1 = {seq_len + 1}")
        xw = sliding_window_view(trace[:-1], (seq_len, n_chan))[:, 0][::stride]
        yw = sliding_window_view(trace[1:], (seq_len, n_chan))[:, 0][::stride]
        return xw, yw


def build_task(cfg: Any) -> Task:
    """Instantiate the task named by ``cfg.task.name``."""
    return TASKS[cfg.task.name](cfg.task)
