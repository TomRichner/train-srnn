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
from train_srnn.training.closed_loop import (
    ClosedLoopConfig,
    effective_alpha_baseline,
    sample_alpha_schedule,
    summarize_alpha,
)
from train_srnn.utils.checkpoint import (
    append_history_row,
    append_test_history_row,
    load_checkpoint,
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

    `fp32` -> no-op nullcontext (bit-identical to pre-AMP code path).
    `bf16` -> `torch.autocast(device_type='cuda', dtype=torch.bfloat16)`.

    Capability + device validation happens once at startup in main();
    this helper is fast-path only.
    """
    amp = cfg.get("amp", "fp32")
    if amp == "fp32":
        return contextlib.nullcontext()
    if amp == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    raise ValueError(f"Unknown amp mode: {amp!r}. Expected 'fp32' or 'bf16'.")


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
    closed_loop_cfg: ClosedLoopConfig | None = None,
    closed_loop_gen: torch.Generator | None = None,
) -> tuple[float | list[float], float | list[float]]:
    """Run one epoch of training or evaluation.

    Returns:
        ``(avg_loss, avg_metric)`` — scalars for single models, or
        lists of length K for batched ablation models.

    When ``training=True`` and ``closed_loop_cfg.enabled`` is True, each
    minibatch samples an alpha schedule from ``closed_loop_cfg`` and passes
    it to the model so the unroll runs in closed-loop mode. The mean of the
    schedule (excluding the t=0 forced-zero) and the fraction of pure-TF
    batches across the epoch are accumulated into per-epoch summary stats
    available via ``run_epoch.last_alpha_stats`` after the call.
    """
    if K is not None:
        total_loss_k = [0.0] * K
        total_correct_k = [0.0] * K
    else:
        total_loss = 0.0
        total_correct = 0.0
    total_samples = 0
    batch_size: int = cfg.batch_size

    cl_active = bool(training and closed_loop_cfg is not None
                     and closed_loop_cfg.enabled)
    cl_alpha_sum = 0.0
    cl_alpha_max = 0.0
    cl_pure_tf_batches = 0
    cl_total_batches = 0

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

        # Closed-loop alpha schedule (training only; None = pure teacher forcing)
        alpha_schedule = None
        if cl_active:
            alpha_schedule = sample_alpha_schedule(
                closed_loop_cfg,
                T=batch_x_t.shape[1],
                C=batch_x_t.shape[2],
                device=device,
                generator=closed_loop_gen,
                dtype=batch_x_t.dtype,
            )
            stats = summarize_alpha(alpha_schedule)
            cl_alpha_sum += stats["alpha_mean"]
            cl_alpha_max = max(cl_alpha_max, stats["alpha_max"])
            cl_pure_tf_batches += int(stats["is_pure_tf"])
            cl_total_batches += 1

        # Forward + loss (under AMP autocast when cfg.amp != fp32) -----------
        with amp_autocast(cfg):
            logits = model(
                batch_x_t,
                readout_idx=readout_idx,
                bptt_start_idx=bptt_start,
                bptt_chunk_len=cfg.get("bptt_chunk_len", None),
                grad_checkpoint=cfg.get("grad_checkpoint", False),
                grad_checkpoint_segment_len=cfg.get("grad_checkpoint_segment_len", None),
                alpha_schedule=alpha_schedule,
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

    # Stash closed-loop summary for the caller to log (epoch-level).
    if cl_active and cl_total_batches > 0:
        run_epoch.last_alpha_stats = {  # type: ignore[attr-defined]
            "alpha_mean": cl_alpha_sum / cl_total_batches,
            "alpha_max": cl_alpha_max,
            "pure_tf_frac": cl_pure_tf_batches / cl_total_batches,
            "n_batches": cl_total_batches,
        }
    else:
        run_epoch.last_alpha_stats = None  # type: ignore[attr-defined]

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

def _build_closed_loop_cfg(cfg: DictConfig) -> ClosedLoopConfig:
    """Construct a typed ClosedLoopConfig from the Hydra `closed_loop` block.

    Missing or absent block -> default disabled config.
    """
    cl_block = cfg.get("closed_loop", None)
    if cl_block is None:
        return ClosedLoopConfig()
    raw = OmegaConf.to_container(cl_block, resolve=True)
    return ClosedLoopConfig(**raw)


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
    amp = cfg.get("amp", "fp32")
    if amp != "fp32":
        if amp != "bf16":
            raise ValueError(f"Unknown amp mode: {amp!r}. Expected 'fp32' or 'bf16'.")
        if device.type != "cuda":
            raise RuntimeError(
                f"amp=bf16 requires a CUDA device, got device={device}. "
                f"Set amp=fp32 or run on a GPU."
            )
        if not torch.cuda.is_bf16_supported():
            cap = torch.cuda.get_device_capability(0)
            name = torch.cuda.get_device_name(0)
            raise RuntimeError(
                f"amp=bf16 requires an Ampere+ GPU (compute capability >= 8.0). "
                f"Detected: {name} (cap {cap}). "
                f"Use L4, A100, H100, RTX 30/40 series, or set amp=fp32."
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
    # Cell-level compile is deferred to the continuous trainer (so that eval,
    # which calls SequenceModel.forward → self.cell, sees the eager cell and
    # doesn't pollute the compile cache with eval-only shape/grad-mode
    # configurations). Model-level compile (cfg.compile_cell=false) still
    # happens here.
    if cfg.compile and device.type == "cuda":
        if bool(cfg.get("compile_log_recompiles", False)):
            import torch._logging as _torch_logging
            _torch_logging.set_logs(recompiles=True)
            log.info("Dynamo recompile logging enabled")
        if not bool(cfg.get("compile_cell", False)):
            compile_kwargs = {}
            cm = cfg.get("compile_mode", None)
            if cm:
                compile_kwargs["mode"] = str(cm)
            cd = cfg.get("compile_dynamic", None)
            if cd is not None:
                compile_kwargs["dynamic"] = bool(cd)
            log.info("torch.compile (model-level) kwargs: %s",
                     compile_kwargs or "(defaults)")
            model = torch.compile(model, **compile_kwargs)
        else:
            log.info("torch.compile (cell-level) deferred to continuous trainer")

    # 6. Optimizer + LR schedule ----------------------------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # Steps-per-epoch must match the trainer that will actually fire
    # ``scheduler.step()``. The continuous trainer steps once per BPTT chunk
    # (15 calls/epoch at seeg defaults); the windowed trainer steps once per
    # mini-batch (~3 calls/epoch at seeg). Sizing the warmup against the
    # wrong cadence — as we previously did with the windowed formula — made
    # warmup absurdly short for continuous runs (~0.4 epochs at B=48).
    if cfg.get("continuous_train", False) and "train_trace" in dataset:
        T_train = int(dataset["train_trace"].shape[0])
        chunk_len = int(cfg.bptt_chunk_len) if cfg.bptt_chunk_len else 250
        steps_per_epoch = (T_train + cfg.batch_size * chunk_len - 1) \
                          // (cfg.batch_size * chunk_len)
        # Default to round(B/2) epochs of warmup — long enough that Adam's
        # moment estimates have time to settle before the model sees full
        # max_lr, capped at 20% of the run for very short jobs.
        default_warmup_epochs = max(1, round(cfg.batch_size / 2))
    else:
        steps_per_epoch = len(train_x) // cfg.batch_size + 1
        # Windowed trainer historical default: warmup over 2 epochs.
        default_warmup_epochs = 2

    total_steps = cfg.epochs * steps_per_epoch
    warmup_epochs = cfg.get("warmup_epochs", None) or default_warmup_epochs
    warmup_frac = min(warmup_epochs * steps_per_epoch / max(1, total_steps), 0.2)
    scheduler = WarmupHoldCosineSchedule(
        optimizer, total_steps, max_lr=cfg.lr, warmup_frac=warmup_frac,
        cosine_decay=cfg.get("cosine_decay", False),
    )
    log.info(
        "LR schedule: warmup_epochs=%d (warmup_frac=%.4f, %d scheduler steps), "
        "total_steps=%d (steps_per_epoch=%d), max_lr=%.3e, cosine_decay=%s",
        warmup_epochs, warmup_frac,
        int(warmup_frac * total_steps), total_steps, steps_per_epoch,
        cfg.lr, cfg.get("cosine_decay", False),
    )

    # 7. Loss function --------------------------------------------------------
    if cfg.task.task_type == "classification":
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()

    # 7b. Optional resume: full-state restore from a prior checkpoint ---------
    # Loads model + optimizer + scheduler. RNG state is intentionally NOT
    # restored so the continuation epochs see fresh batch order. Burn-in
    # below will overwrite model.ic.ic if cfg.burn_in > 0; set burn_in=0
    # to keep the restored IC verbatim.
    init_ckpt_cfg = cfg.get("init_ckpt", None)
    if init_ckpt_cfg:
        import os as _os
        import subprocess as _subprocess
        local_init_ckpt = init_ckpt_cfg
        if str(init_ckpt_cfg).startswith("gs://"):
            local_init_ckpt = _os.path.join(cfg.output_dir, "_init_ckpt.pt")
            _os.makedirs(cfg.output_dir, exist_ok=True)
            log.info(f"Downloading init_ckpt from {init_ckpt_cfg} -> {local_init_ckpt}")
            _subprocess.run(
                ["gsutil", "cp", str(init_ckpt_cfg), local_init_ckpt], check=True,
            )
        log.info(f"Loading init_ckpt: model + optimizer + scheduler from {local_init_ckpt}")
        prev_ckpt = load_checkpoint(
            local_init_ckpt,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=str(device),
        )
        log.info(
            f"Resumed from epoch={prev_ckpt.get('epoch')} "
            f"(scheduler step={getattr(scheduler, 'last_epoch', '?')}, "
            f"current lr={optimizer.param_groups[0]['lr']:.3e})"
        )

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

    # Closed-loop variable teacher forcing: build a typed config from the
    # Hydra block and a dedicated torch.Generator (independent of `rng` so
    # batch sampling order is unaffected when toggling closed-loop on/off).
    cl_cfg = _build_closed_loop_cfg(cfg)
    cl_gen = (torch.Generator(device=device).manual_seed(int(cfg.seed) + 1)
              if cl_cfg.enabled else None)
    if cl_cfg.enabled:
        # Validate task dim constraint up front (fail fast).
        if cfg.task.input_size != cfg.task.output_size:
            raise ValueError(
                f"closed_loop.enabled=true requires task.input_size == "
                f"task.output_size, got {cfg.task.input_size} != "
                f"{cfg.task.output_size}."
            )
        ramp_str = (f" (ramp from {cl_cfg.alpha_baseline_start:.3f})"
                    if cl_cfg.alpha_baseline_start is not None else "")
        log.info(
            "Closed-loop enabled: baseline=%.3f%s (jitter=%.3f), rnd "
            "density=%.2f sigma=%.2f, t_warm=%d, pure-TF batch frac=%.2f",
            cl_cfg.alpha_baseline, ramp_str, cl_cfg.alpha_baseline_jitter,
            cl_cfg.alpha_rnd_density, cl_cfg.alpha_rnd_sigma,
            cl_cfg.t_warm, cl_cfg.teacher_forcing_batch_frac,
        )

    # Save init checkpoint + test eval (before any training)
    save_checkpoint(model, optimizer, scheduler, epoch=0, cfg=cfg, tag="init")
    eval_and_log_test(
        model, test_x, test_y, criterion, cfg, rng, device,
        epoch=0, tag="init", K=K, ablation_names=ablation_names,
    )

    if cfg.get("early_exit_after_init", False):
        # Mirror init.pt -> last.pt so postprocess.py "last" lookups
        # (param_table, --replay-checkpoints last) keep working.
        save_checkpoint(model, optimizer, scheduler, epoch=0, cfg=cfg, tag="last")
        log.info("early_exit_after_init=true → exiting after init/last "
                 "checkpoints written to %s", cfg.output_dir)
        return

    # Continuous trace-circular trainer (SEEG only). Bypasses the windowed
    # epoch loop below; uses its own inner step structure with B parallel
    # readers around the trace ring. See plan + continuous.py.
    if cfg.get("continuous_train", False):
        if cfg.task.name != "seeg":
            log.warning(
                "continuous_train=true is only supported for SEEG; task=%s "
                "falls back to the windowed loop.", cfg.task.name)
        elif "train_trace" not in dataset:
            log.warning(
                "continuous_train=true but dataset has no 'train_trace' key; "
                "falls back to the windowed loop. (Did you set "
                "task.train_trace_max_len in seeg.yaml?)")
        else:
            from train_srnn.training.continuous import run_continuous_training
            import time as _time
            _t0 = _time.perf_counter()
            train_trace_t = torch.tensor(dataset["train_trace"],
                                          dtype=torch.float32, device=device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            _trace_xfer_s = _time.perf_counter() - _t0
            log.info("Dispatching to continuous trainer "
                     "(train_trace shape=%s, GPU upload %.3fs)",
                     tuple(train_trace_t.shape), _trace_xfer_s)
            run_continuous_training(
                model=model,
                train_trace=train_trace_t,
                valid_x=valid_x, valid_y=valid_y,
                test_x=test_x, test_y=test_y,
                optimizer=optimizer, scheduler=scheduler, criterion=criterion,
                cfg=cfg,
                closed_loop_cfg=cl_cfg,
                closed_loop_gen=cl_gen,
                rng=rng,
                device=device,
                K=K, ablation_names=ablation_names,
                eval_and_log_test_fn=eval_and_log_test,
                run_epoch_fn=run_epoch,
                amp_autocast_fn=amp_autocast,
            )
            return

    for epoch in range(cfg.epochs):
        # Periodic re-burn-in: track the moving unforced fixed point as
        # network parameters drift during training. Skip epoch 0 since we
        # already ran burn-in at init. See KnownIssues §4.
        if (burn_in_every and burn_in_every > 0 and epoch > 0
                and epoch % burn_in_every == 0
                and cfg.burn_in > 0 and hasattr(model, "ic")):
            _refresh_ic_from_burn_in()

        # Per-epoch effective alpha_baseline (linear ramp if alpha_baseline_start
        # is set; constant otherwise).
        if cl_cfg.enabled:
            import dataclasses as _dc
            eff_baseline = effective_alpha_baseline(cl_cfg, epoch, cfg.epochs)
            epoch_cl_cfg = _dc.replace(cl_cfg, alpha_baseline=eff_baseline)
        else:
            epoch_cl_cfg = cl_cfg

        model.train()
        train_loss, train_metric = run_epoch(
            model, train_x, train_y,
            optimizer, scheduler, criterion,
            cfg, rng, device, training=True, K=K,
            closed_loop_cfg=epoch_cl_cfg, closed_loop_gen=cl_gen,
        )
        if getattr(run_epoch, "last_alpha_stats", None):
            s = run_epoch.last_alpha_stats
            log.info(
                "  closed-loop[ep=%d]: eff_baseline=%.3f alpha_mean=%.3f "
                "alpha_max=%.3f pure_tf_frac=%.2f over %d batches",
                epoch, epoch_cl_cfg.alpha_baseline,
                s["alpha_mean"], s["alpha_max"], s["pure_tf_frac"],
                s["n_batches"],
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
