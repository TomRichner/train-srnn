"""Main Hydra training entry point for the PyTorch refactor."""

from __future__ import annotations

import csv
import logging
import os

import hydra
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig

from pytorch_refactor.data.datasets import load_dataset
from pytorch_refactor.data.transforms import wrap_eval_batch, wrap_train_batch
from pytorch_refactor.models.factory import build_batched_model, build_model
from pytorch_refactor.utils.lr_schedule import WarmupHoldCosineSchedule
from pytorch_refactor.utils.trainable_ic import compute_burn_in

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Training / evaluation loop
# ---------------------------------------------------------------------------

def run_epoch(
    model: nn.Module,
    data_x: np.ndarray,
    data_y: np.ndarray,
    optimizer: torch.optim.Optimizer | None,
    scheduler: object | None,
    criterion: nn.Module,
    cfg: DictConfig,
    rng: np.random.RandomState,
    device: torch.device,
    training: bool = True,
) -> tuple[float, float]:
    """Run one epoch of training or evaluation.

    Returns:
        ``(avg_loss, avg_metric)`` where *metric* is accuracy for
        classification or negative MAE for regression.
    """
    total_loss = 0.0
    total_correct = 0.0
    total_samples = 0
    batch_size: int = cfg.batch_size

    indices = rng.permutation(len(data_x)) if training else np.arange(len(data_x))

    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start : start + batch_size]
        batch_x = data_x[batch_idx]
        batch_y = data_y[batch_idx]

        # Augmentation / wrapping ---------------------------------------------
        if training:
            batch_x, batch_y, readout_idx, bptt_start = wrap_train_batch(
                batch_x,
                batch_y,
                rng,
                cfg.stretch_lo,
                cfg.stretch_hi,
                cfg.min_loops,
                cfg.min_loop_len,
                cfg.task.per_timestep_labels,
            )
        else:
            batch_x, batch_y, readout_idx = wrap_eval_batch(
                batch_x,
                batch_y,
                cfg.min_loops,
                cfg.min_loop_len,
                cfg.task.per_timestep_labels,
            )
            bptt_start = None

        # To device -----------------------------------------------------------
        batch_x_t = torch.tensor(batch_x, dtype=torch.float32, device=device)
        if cfg.task.task_type == "classification":
            batch_y_t = torch.tensor(batch_y, dtype=torch.long, device=device)
        else:
            batch_y_t = torch.tensor(batch_y, dtype=torch.float32, device=device)

        # Forward -------------------------------------------------------------
        logits = model(batch_x_t, readout_idx=readout_idx, bptt_start_idx=bptt_start)

        if cfg.task.task_type == "regression":
            logits = logits.squeeze(-1)

        loss = criterion(logits, batch_y_t)

        # Backward + step -----------------------------------------------------
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        # Metrics -------------------------------------------------------------
        n = len(batch_idx)
        total_loss += loss.item() * n

        if cfg.task.task_type == "classification":
            preds = logits.argmax(dim=-1)
            total_correct += (preds == batch_y_t).sum().item()
        else:
            # Accumulate negative MAE (higher is better, consistent with "best")
            total_correct += (
                -torch.mean(torch.abs(logits - batch_y_t)).item() * n
            )

        total_samples += n

    avg_loss = total_loss / total_samples
    if cfg.task.task_type == "classification":
        avg_metric = total_correct / total_samples
    else:
        avg_metric = -total_correct / total_samples  # positive MAE
    return avg_loss, avg_metric


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def resolve_device(device_str: str) -> torch.device:
    """Resolve device string (``"auto"``, ``"cpu"``, ``"cuda"``, ``"mps"``)."""
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    cfg: DictConfig,
    tag: str,
) -> None:
    """Save a training checkpoint to ``cfg.output_dir``."""
    path = os.path.join(cfg.output_dir, f"checkpoint_{tag}.pt")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": (
                optimizer.state_dict() if optimizer else None
            ),
        },
        path,
    )


def save_results_csv(
    cfg: DictConfig,
    best_epoch: int,
    train_loss: float,
    train_metric: float,
    valid_loss: float,
    valid_metric: float,
    test_loss: float,
    test_metric: float,
) -> None:
    """Write a single-row results CSV next to the checkpoints."""
    path = os.path.join(cfg.output_dir, f"{cfg.model.name}_{cfg.size}.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    metric_name = (
        "accuracy" if cfg.task.task_type == "classification" else "mae"
    )
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "best_epoch",
                "train_loss",
                f"train_{metric_name}",
                "valid_loss",
                f"valid_{metric_name}",
                "test_loss",
                f"test_{metric_name}",
            ]
        )
        writer.writerow(
            [
                best_epoch,
                f"{train_loss:.6f}",
                f"{train_metric:.6f}",
                f"{valid_loss:.6f}",
                f"{valid_metric:.6f}",
                f"{test_loss:.6f}",
                f"{test_metric:.6f}",
            ]
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # 1. Seeds ----------------------------------------------------------------
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # 2. Device ---------------------------------------------------------------
    device = resolve_device(cfg.device)
    log.info("Using device: %s", device)

    # 3. Load data ------------------------------------------------------------
    dataset = load_dataset(cfg.task.name, cfg.task.data_dir)
    train_x, train_y = dataset["train"]
    valid_x, valid_y = dataset["valid"]
    test_x, test_y = dataset["test"]

    # 4. Build model ----------------------------------------------------------
    if cfg.batched_ablations:
        model = build_batched_model(cfg, cfg.batched_ablations)
    else:
        model = build_model(cfg)
    model = model.to(device)
    log.info("Model parameters: %d", sum(p.numel() for p in model.parameters()))

    # 5. Optional torch.compile -----------------------------------------------
    if cfg.compile and device.type == "cuda":
        model = torch.compile(model)

    # 6. Optimizer + LR schedule ----------------------------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    total_steps = cfg.epochs * (len(train_x) // cfg.batch_size + 1)
    scheduler = WarmupHoldCosineSchedule(
        optimizer, total_steps, max_lr=cfg.lr
    )

    # 7. Loss function --------------------------------------------------------
    if cfg.task.task_type == "classification":
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()

    # 8. Optional burn-in for IC ----------------------------------------------
    if cfg.burn_in > 0 and hasattr(model, "ic"):
        burn_in_state = compute_burn_in(
            model.cell, cfg.task.input_size, cfg.burn_in, device
        )
        model.ic.ic.data.copy_(burn_in_state)

    # 9. Training loop --------------------------------------------------------
    rng = np.random.RandomState(cfg.seed)
    best_metric: float | None = None
    best_epoch = 0

    for epoch in range(cfg.epochs):
        model.train()
        train_loss, train_metric = run_epoch(
            model,
            train_x,
            train_y,
            optimizer,
            scheduler,
            criterion,
            cfg,
            rng,
            device,
            training=True,
        )

        # Constrain parameters (e.g. LTC weight clipping)
        model.constrain_parameters()

        # Validation
        model.eval()
        with torch.no_grad():
            valid_loss, valid_metric = run_epoch(
                model,
                valid_x,
                valid_y,
                None,
                None,
                criterion,
                cfg,
                rng,
                device,
                training=False,
            )

        # Logging
        if epoch % cfg.log_interval == 0:
            log.info(
                "Epoch %d: train_loss=%.4f train_metric=%.4f "
                "valid_loss=%.4f valid_metric=%.4f",
                epoch,
                train_loss,
                train_metric,
                valid_loss,
                valid_metric,
            )

        # Checkpointing
        is_best = best_metric is None or valid_metric > best_metric
        if is_best:
            best_metric = valid_metric
            best_epoch = epoch
            save_checkpoint(model, optimizer, epoch, cfg, "best")

        if epoch % cfg.checkpoint_interval == 0:
            save_checkpoint(model, optimizer, epoch, cfg, f"epoch{epoch}")

    # 10. Final test evaluation -----------------------------------------------
    model.eval()
    with torch.no_grad():
        test_loss, test_metric = run_epoch(
            model,
            test_x,
            test_y,
            None,
            None,
            criterion,
            cfg,
            rng,
            device,
            training=False,
        )

    log.info(
        "Test: loss=%.4f metric=%.4f (best_epoch=%d)",
        test_loss,
        test_metric,
        best_epoch,
    )

    # 11. Save results CSV ----------------------------------------------------
    save_results_csv(
        cfg,
        best_epoch,
        train_loss,
        train_metric,
        valid_loss,
        valid_metric,
        test_loss,
        test_metric,
    )


if __name__ == "__main__":
    main()
