"""Checkpoint, progress, and training history utilities."""

from __future__ import annotations

import csv
import json
import os
import tempfile
import time

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: object | None,
    epoch: int,
    cfg: DictConfig,
    tag: str,
    extra: dict | None = None,
) -> str:
    """Save a training checkpoint to ``{cfg.output_dir}/{tag}.pt``.

    The checkpoint dict contains everything needed to resume training or
    reconstruct the model for analysis (FTLE, parameter inspection, etc.).

    Returns:
        Path to the saved checkpoint file.
    """
    path = os.path.join(cfg.output_dir, f"{tag}.pt")
    os.makedirs(os.path.dirname(path), exist_ok=True)

    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": (
            optimizer.state_dict() if optimizer else None
        ),
        "scheduler_state_dict": (
            scheduler.state_dict() if scheduler and hasattr(scheduler, "state_dict") else None
        ),
        "config": OmegaConf.to_container(cfg, resolve=True),
        "ablation_names": getattr(model, "ablation_names", None),
        "torch_rng_state": torch.random.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
    }
    if extra:
        ckpt.update(extra)

    torch.save(ckpt, path)
    return path


def load_checkpoint(
    path: str,
    model: torch.nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: object | None = None,
    device: str = "cpu",
) -> dict:
    """Load a checkpoint and optionally restore model/optimizer/scheduler.

    Args:
        path: Path to the ``.pt`` checkpoint file.
        model: If provided, calls ``model.load_state_dict()``.
        optimizer: If provided, restores optimizer state.
        scheduler: If provided, restores scheduler state.
        device: Device to map tensors to.

    Returns:
        The full checkpoint dict.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)

    if model is not None:
        model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    return ckpt


def write_progress(output_dir: str, epoch: int, total_epochs: int) -> None:
    """Write progress.json atomically for the GCS upload watcher."""
    data = {
        "epoch": epoch,
        "total_epochs": total_epochs,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    os.makedirs(output_dir, exist_ok=True)
    tmp_path = os.path.join(output_dir, ".progress.json.tmp")
    final_path = os.path.join(output_dir, "progress.json")
    with open(tmp_path, "w") as f:
        json.dump(data, f)
    os.replace(tmp_path, final_path)


# ---------------------------------------------------------------------------
# Training history CSV
# ---------------------------------------------------------------------------

_HISTORY_HEADER_SINGLE = [
    "epoch", "train_loss", "train_metric", "valid_loss", "valid_metric",
    "lr", "timestamp",
]
_HISTORY_HEADER_BATCHED = [
    "epoch", "variant", "train_loss", "train_metric", "valid_loss",
    "valid_metric", "lr", "timestamp",
]


def append_history_row(
    output_dir: str,
    epoch: int,
    train_loss: float | list[float],
    train_metric: float | list[float],
    valid_loss: float | list[float],
    valid_metric: float | list[float],
    lr: float,
    K: int | None = None,
    ablation_names: list[str] | None = None,
) -> None:
    """Append one (or K) rows to ``training_history.csv``."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "training_history.csv")
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    write_header = not os.path.exists(path)

    with open(path, "a", newline="") as f:
        writer = csv.writer(f)

        if K is not None:
            if write_header:
                writer.writerow(_HISTORY_HEADER_BATCHED)
            for k in range(K):
                writer.writerow([
                    epoch,
                    ablation_names[k] if ablation_names else f"variant_{k}",
                    f"{train_loss[k]:.6f}",
                    f"{train_metric[k]:.6f}",
                    f"{valid_loss[k]:.6f}",
                    f"{valid_metric[k]:.6f}",
                    f"{lr:.8f}",
                    ts,
                ])
        else:
            if write_header:
                writer.writerow(_HISTORY_HEADER_SINGLE)
            writer.writerow([
                epoch,
                f"{train_loss:.6f}",
                f"{train_metric:.6f}",
                f"{valid_loss:.6f}",
                f"{valid_metric:.6f}",
                f"{lr:.8f}",
                ts,
            ])


# ---------------------------------------------------------------------------
# Test history CSV
# ---------------------------------------------------------------------------

_TEST_HISTORY_HEADER_SINGLE = [
    "epoch", "tag", "test_loss", "test_metric", "timestamp",
]
_TEST_HISTORY_HEADER_BATCHED = [
    "epoch", "tag", "variant", "test_loss", "test_metric", "timestamp",
]


def append_test_history_row(
    output_dir: str,
    epoch: int,
    tag: str,
    test_loss: float | list[float],
    test_metric: float | list[float],
    K: int | None = None,
    ablation_names: list[str] | None = None,
) -> None:
    """Append test metrics to ``test_history.csv`` (one row per variant)."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "test_history.csv")
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    write_header = not os.path.exists(path)

    with open(path, "a", newline="") as f:
        writer = csv.writer(f)

        if K is not None:
            if write_header:
                writer.writerow(_TEST_HISTORY_HEADER_BATCHED)
            for k in range(K):
                writer.writerow([
                    epoch,
                    tag,
                    ablation_names[k] if ablation_names else f"variant_{k}",
                    f"{test_loss[k]:.6f}",
                    f"{test_metric[k]:.6f}",
                    ts,
                ])
        else:
            if write_header:
                writer.writerow(_TEST_HISTORY_HEADER_SINGLE)
            writer.writerow([
                epoch,
                tag,
                f"{test_loss:.6f}",
                f"{test_metric:.6f}",
                ts,
            ])
