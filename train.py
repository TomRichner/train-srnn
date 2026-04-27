"""Main Hydra training entry point for the PyTorch refactor."""

from __future__ import annotations

import contextlib
import logging

import hydra
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from train_srnn.data.datasets import load_dataset
from train_srnn.data.transforms import wrap_eval_batch, wrap_train_batch
from train_srnn.models.factory import build_batched_model, build_model
from train_srnn.utils.checkpoint import (
    append_history_row,
    append_test_history_row,
    save_checkpoint,
    write_progress,
)
from train_srnn.utils.lr_schedule import WarmupHoldCosineSchedule
from train_srnn.utils.trainable_ic import compute_burn_in

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AMP / autocast
# ---------------------------------------------------------------------------

def amp_autocast(cfg: DictConfig):
    """Context manager that enables AMP autocast per cfg.amp.

    `off` -> no-op nullcontext (bit-identical to pre-AMP code path).
    `bf16` -> `torch.autocast(device_type='cuda', dtype=torch.bfloat16)`.

    Capability + device validation happens once at startup in main();
    this helper is fast-path only.
    """
    amp = cfg.get("amp", "off")
    if amp == "off":
        return contextlib.nullcontext()
    if amp == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    raise ValueError(f"Unknown amp mode: {amp!r}. Expected 'off' or 'bf16'.")


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
    K: int | None = None,
) -> tuple[float | list[float], float | list[float]]:
    """Run one epoch of training or evaluation.

    Returns:
        ``(avg_loss, avg_metric)`` — scalars for single models, or
        lists of length K for batched ablation models.
    """
    if K is not None:
        total_loss_k = [0.0] * K
        total_correct_k = [0.0] * K
    else:
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
                cfg.window_len,
                cfg.bptt_len,
                cfg.task.per_timestep_labels,
                no_augment=cfg.get("no_augment", False),
                loss_over_bptt=cfg.get("loss_over_bptt", False),
            )
            # Extract label at readout timestep(s) for per-timestep tasks.
            # readout_idx is int (shape -> (B, F)) or slice (shape -> (B, T, F)).
            if cfg.task.per_timestep_labels:
                batch_y = batch_y[:, readout_idx]
        else:
            batch_x, batch_y, readout_idx = wrap_eval_batch(
                batch_x,
                batch_y,
                cfg.window_len,
                cfg.task.per_timestep_labels,
                no_augment=cfg.get("no_augment", False),
                loss_over_bptt=cfg.get("loss_over_bptt", False),
                bptt_len=cfg.bptt_len,
            )
            # wrap_eval_batch already extracts labels_at_readout
            bptt_start = None

        # To device -----------------------------------------------------------
        batch_x_t = torch.tensor(batch_x, dtype=torch.float32, device=device)
        if cfg.task.task_type == "classification":
            batch_y_t = torch.tensor(batch_y, dtype=torch.long, device=device)
        else:
            batch_y_t = torch.tensor(batch_y, dtype=torch.float32, device=device)

        # Forward + loss (under AMP autocast when cfg.amp != off) -------------
        with amp_autocast(cfg):
            logits = model(
                batch_x_t,
                readout_idx=readout_idx,
                bptt_start_idx=bptt_start,
                bptt_chunk_len=cfg.get("bptt_chunk_len", None),
                grad_checkpoint=cfg.get("grad_checkpoint", False),
                grad_checkpoint_segment_len=cfg.get("grad_checkpoint_segment_len", None),
            )

            if K is not None:
                # logits: (K, B, C) — compute K independent losses, sum for backward
                losses = []
                for k in range(K):
                    logits_k = logits[k]
                    if cfg.task.task_type == "regression":
                        logits_k = logits_k.squeeze(-1)
                    losses.append(criterion(logits_k, batch_y_t))
                loss = torch.stack(losses).sum()
                per_k_loss = [l.item() for l in losses]
            else:
                if cfg.task.task_type == "regression":
                    logits = logits.squeeze(-1)
                loss = criterion(logits, batch_y_t)

        # Backward + step -----------------------------------------------------
        if training:
            optimizer.zero_grad()
            loss.backward()
            grad_clip = cfg.get("grad_clip", 0.0)
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        # Metrics -------------------------------------------------------------
        n = len(batch_idx)

        if K is not None:
            for k in range(K):
                total_loss_k[k] += per_k_loss[k] * n
                if cfg.task.task_type == "classification":
                    preds_k = logits[k].argmax(dim=-1)
                    total_correct_k[k] += (preds_k == batch_y_t).sum().item()
                else:
                    logits_k = logits[k].squeeze(-1)
                    total_correct_k[k] += (
                        -torch.mean(torch.abs(logits_k - batch_y_t)).item() * n
                    )
        else:
            total_loss += loss.item() * n
            if cfg.task.task_type == "classification":
                preds = logits.argmax(dim=-1)
                total_correct += (preds == batch_y_t).sum().item()
            else:
                total_correct += (
                    -torch.mean(torch.abs(logits - batch_y_t)).item() * n
                )

        total_samples += n

    if K is not None:
        avg_losses = [tl / total_samples for tl in total_loss_k]
        if cfg.task.task_type == "classification":
            avg_metrics = [tc / total_samples for tc in total_correct_k]
        else:
            avg_metrics = [-tc / total_samples for tc in total_correct_k]
        return avg_losses, avg_metrics
    else:
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


def eval_and_log_test(
    model: nn.Module,
    test_x: np.ndarray,
    test_y: np.ndarray,
    criterion: nn.Module,
    cfg: DictConfig,
    rng: np.random.RandomState,
    device: torch.device,
    epoch: int,
    tag: str,
    K: int | None,
    ablation_names: list[str] | None,
) -> None:
    """Evaluate test set and append a row to test_history.csv."""
    was_training = model.training
    model.eval()
    with torch.no_grad():
        test_loss, test_metric = run_epoch(
            model, test_x, test_y,
            None, None, criterion,
            cfg, rng, device, training=False, K=K,
        )
    if was_training:
        model.train()
    append_test_history_row(
        cfg.output_dir, epoch, tag,
        test_loss, test_metric,
        K=K, ablation_names=ablation_names,
    )
    if K is not None:
        log.info("Test [%s @ epoch %d]:", tag, epoch)
        for k in range(K):
            log.info(
                "  [%s] test_loss=%.4f test_metric=%.4f",
                ablation_names[k], test_loss[k], test_metric[k],
            )
    else:
        log.info(
            "Test [%s @ epoch %d]: loss=%.4f metric=%.4f",
            tag, epoch, test_loss, test_metric,
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

    # AMP capability check (fail fast at startup, not mid-training).
    amp = cfg.get("amp", "off")
    if amp != "off":
        if amp != "bf16":
            raise ValueError(f"Unknown amp mode: {amp!r}. Expected 'off' or 'bf16'.")
        if device.type != "cuda":
            raise RuntimeError(
                f"amp=bf16 requires a CUDA device, got device={device}. "
                f"Set amp=off or run on a GPU."
            )
        if not torch.cuda.is_bf16_supported():
            cap = torch.cuda.get_device_capability(0)
            name = torch.cuda.get_device_name(0)
            raise RuntimeError(
                f"amp=bf16 requires an Ampere+ GPU (compute capability >= 8.0). "
                f"Detected: {name} (cap {cap}). "
                f"Use L4, A100, H100, RTX 30/40 series, or set amp=off."
            )
        log.info("AMP enabled: bf16 (autocast dtype=torch.bfloat16)")

    # 3. Load data ------------------------------------------------------------
    # Forward any extra task-level loader kwargs (used by seeg for
    # subject_id/block/sleep/cond/decimate/seq_len/stride).
    _loader_reserved = {"name", "data_dir", "input_size", "output_size",
                        "task_type", "per_timestep_labels", "batch_size"}
    loader_kwargs = {
        k: v for k, v in OmegaConf.to_container(cfg.task, resolve=True).items()
        if k not in _loader_reserved
    }
    dataset = load_dataset(cfg.task.name, cfg.task.data_dir, **loader_kwargs)
    train_x, train_y = dataset["train"]
    valid_x, valid_y = dataset["valid"]
    test_x, test_y = dataset["test"]

    # Assert channel count (loaders that report input_size in meta) matches
    # task YAML so the model builds against the right feature count.
    meta = dataset.get("meta", {})
    meta_in = meta.get("input_size")
    if meta_in is not None and meta_in != cfg.task.input_size:
        raise ValueError(
            f"Dataset input_size={meta_in} does not match task YAML "
            f"input_size={cfg.task.input_size}. Override with "
            f"task.input_size={meta_in} task.output_size={meta_in}.")

    # 4. Build model ----------------------------------------------------------
    if cfg.batched_ablations:
        model = build_batched_model(cfg, cfg.batched_ablations)
    else:
        model = build_model(cfg)
    model = model.to(device)
    log.info("Model parameters: %d", sum(p.numel() for p in model.parameters()))

    # Extract K and ablation names before possible torch.compile wrapping
    K = getattr(model, "_K", None)
    ablation_names = getattr(model, "ablation_names", None)

    # 5. Optional torch.compile -----------------------------------------------
    if cfg.compile and device.type == "cuda":
        model = torch.compile(model)

    # 6. Optimizer + LR schedule ----------------------------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    steps_per_epoch = len(train_x) // cfg.batch_size + 1
    total_steps = cfg.epochs * steps_per_epoch
    # Warmup over min(2 epochs, 20% of training), whichever is shorter.
    warmup_frac = min(2 * steps_per_epoch / max(1, total_steps), 0.2)
    scheduler = WarmupHoldCosineSchedule(
        optimizer, total_steps, max_lr=cfg.lr, warmup_frac=warmup_frac,
        cosine_decay=cfg.get("cosine_decay", False),
    )

    # 7. Loss function --------------------------------------------------------
    if cfg.task.task_type == "classification":
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()

    # 8. Optional burn-in for IC ----------------------------------------------
    def _refresh_ic_from_burn_in():
        """Re-run compute_burn_in and copy the resulting state into model.ic.ic.

        Safe to call at any training epoch; the copy bypasses autograd and does
        not depend on whether the IC parameter is currently frozen.
        """
        with amp_autocast(cfg):
            burn_in_state = compute_burn_in(
                model.cell, cfg.task.input_size, cfg.burn_in, device
            )
        model.ic.ic.data.copy_(burn_in_state)

    if cfg.burn_in > 0 and hasattr(model, "ic"):
        _refresh_ic_from_burn_in()
        if cfg.get("freeze_ic_after_burnin", True):
            model.ic.ic.requires_grad_(False)
            log.info("TrainableIC frozen after burn-in (freeze_ic_after_burnin=True)")

    # 9. Training loop --------------------------------------------------------
    rng = np.random.RandomState(cfg.seed)
    burn_in_every = cfg.get("burn_in_every", 0)

    # Save init checkpoint + test eval (before any training)
    save_checkpoint(model, optimizer, scheduler, epoch=0, cfg=cfg, tag="init")
    eval_and_log_test(
        model, test_x, test_y, criterion, cfg, rng, device,
        epoch=0, tag="init", K=K, ablation_names=ablation_names,
    )

    for epoch in range(cfg.epochs):
        # Periodic re-burn-in: track the moving unforced fixed point as
        # network parameters drift during training. Skip epoch 0 since we
        # already ran burn-in at init. See KnownIssues §4.
        if (burn_in_every and burn_in_every > 0 and epoch > 0
                and epoch % burn_in_every == 0
                and cfg.burn_in > 0 and hasattr(model, "ic")):
            _refresh_ic_from_burn_in()

        model.train()
        train_loss, train_metric = run_epoch(
            model, train_x, train_y,
            optimizer, scheduler, criterion,
            cfg, rng, device, training=True, K=K,
        )

        # Constrain parameters (e.g. LTC weight clipping)
        model.constrain_parameters()

        # Validation
        model.eval()
        with torch.no_grad():
            valid_loss, valid_metric = run_epoch(
                model, valid_x, valid_y,
                None, None, criterion,
                cfg, rng, device, training=False, K=K,
            )

        # Logging
        if epoch % cfg.log_interval == 0:
            if K is not None:
                log.info("Epoch %d:", epoch)
                for k in range(K):
                    log.info(
                        "  [%s] train_loss=%.4f train_metric=%.4f "
                        "valid_loss=%.4f valid_metric=%.4f",
                        ablation_names[k],
                        train_loss[k], train_metric[k],
                        valid_loss[k], valid_metric[k],
                    )
            else:
                log.info(
                    "Epoch %d: train_loss=%.4f train_metric=%.4f "
                    "valid_loss=%.4f valid_metric=%.4f",
                    epoch, train_loss, train_metric,
                    valid_loss, valid_metric,
                )

        # Periodic checkpoint + test eval
        if epoch % cfg.checkpoint_interval == 0:
            tag = f"epoch_{epoch:03d}"
            save_checkpoint(model, optimizer, scheduler, epoch, cfg, tag)
            eval_and_log_test(
                model, test_x, test_y, criterion, cfg, rng, device,
                epoch=epoch, tag=tag, K=K, ablation_names=ablation_names,
            )

        # Training history + progress
        append_history_row(
            cfg.output_dir, epoch,
            train_loss, train_metric, valid_loss, valid_metric,
            lr=scheduler.get_last_lr()[0],
            K=K, ablation_names=ablation_names,
        )
        write_progress(cfg.output_dir, epoch, cfg.epochs)

    # 10. Save last checkpoint + final test eval -------------------------------
    last_epoch = cfg.epochs - 1
    save_checkpoint(
        model, optimizer, scheduler, epoch=last_epoch, cfg=cfg, tag="last",
    )
    eval_and_log_test(
        model, test_x, test_y, criterion, cfg, rng, device,
        epoch=last_epoch, tag="last", K=K, ablation_names=ablation_names,
    )


if __name__ == "__main__":
    main()
