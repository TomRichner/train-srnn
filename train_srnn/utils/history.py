"""Run-directory artifacts: checkpoints, progress.json, and the history CSVs."""
from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

HISTORY_HEADER = ["epoch", "variant", "train_loss", "train_metric", "valid_loss",
                  "valid_metric", "lr", "optimizer_step", "timestamp"]
TEST_HEADER = ["epoch", "tag", "variant", "test_loss", "test_metric", "timestamp"]


def _timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _append_rows(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        w.writerows(rows)


def append_history(run_dir: Path, epoch: int, names: Sequence[str],
                   train_loss: Sequence[float], train_metric: Sequence[float],
                   valid_loss: Sequence[float] | None, valid_metric: Sequence[float] | None,
                   lr: float, optimizer_step: int | None = None) -> None:
    """One row per network in ``training_history.csv``."""
    ts = _timestamp()
    nan = float("nan")
    rows = [[epoch, name, f"{train_loss[k]:.6f}", f"{train_metric[k]:.6f}",
             f"{(valid_loss[k] if valid_loss is not None else nan):.6f}",
             f"{(valid_metric[k] if valid_metric is not None else nan):.6f}",
             f"{lr:.8f}", optimizer_step, ts]
            for k, name in enumerate(names)]
    _append_rows(Path(run_dir) / "training_history.csv", HISTORY_HEADER, rows)


def append_test_history(run_dir: Path, epoch: int, tag: str, names: Sequence[str],
                        test_loss: Sequence[float], test_metric: Sequence[float]) -> None:
    ts = _timestamp()
    rows = [[epoch, tag, name, f"{test_loss[k]:.6f}", f"{test_metric[k]:.6f}", ts]
            for k, name in enumerate(names)]
    _append_rows(Path(run_dir) / "test_history.csv", TEST_HEADER, rows)


def write_progress(run_dir: Path, epoch: int, total_epochs: int) -> None:
    """Atomic ``progress.json`` for the cloud upload watcher."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    tmp = run_dir / ".progress.json.tmp"
    with open(tmp, "w") as f:
        json.dump({"epoch": epoch, "total_epochs": total_epochs, "timestamp": _timestamp()}, f)
    os.replace(tmp, run_dir / "progress.json")


def save_checkpoint(run_dir: Path, tag: str, model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer | None, scheduler,
                    epoch: int, cfg: DictConfig) -> Path:
    """``<run_dir>/<tag>.pt`` with everything needed to resume or to rebuild the model."""
    path = Path(run_dir) / f"{tag}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_version": getattr(model.cell, "MODEL_VERSION", None),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "config": OmegaConf.to_container(cfg, resolve=True),
        "variant_names": model.variant_names,
        "torch_rng_state": torch.random.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
    }, path)
    return path


def load_checkpoint(path: str | Path, model: torch.nn.Module | None = None,
                    optimizer=None, scheduler=None, device: str = "cpu") -> dict:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if model is not None:
        model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt
