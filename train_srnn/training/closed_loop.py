"""Closed-loop teacher-forcing schedule sampling.

Produces a per-batch (T, C) alpha tensor that controls how strongly the
model's previous prediction is fed back as input at each timestep:

    x_in[t, c] = (1 - alpha[t, c]) * x_real[t, c] + alpha[t, c] * y_pred[t-1, c]

alpha = 0 -> pure teacher forcing (current behavior).
alpha = 1 -> pure free-run (closed-loop autoregressive).

Schedule structure:
    alpha[t, c] = envelope(t) * clip(baseline + rnd[c], 0, 1)

Where envelope is a half-cosine ramp 0 -> 1 over `t_warm` samples, then 1.
`baseline` is per-batch (optionally jittered). `rnd[c]` is a sparse
zero-mean Gaussian over channels: with probability `alpha_rnd_density`,
draw N(0, alpha_rnd_sigma^2); otherwise 0.

A fraction `teacher_forcing_batch_frac` of batches return `None`
(treat as alpha = 0 everywhere -> pure teacher forcing batch).

alpha[0, :] is forced to 0 by construction -- there is no y_prev at t=0.
"""
from __future__ import annotations

import math

import torch


from train_srnn.config import ClosedLoopConfig  # re-exported for callers


def effective_alpha_baseline(cfg: ClosedLoopConfig, epoch: int,
                             total_epochs: int) -> float:
    """Compute the per-epoch effective alpha_baseline.

    With ``alpha_baseline_start`` unset (None) or with ``total_epochs <= 1``,
    returns ``cfg.alpha_baseline`` unchanged (constant-baseline behavior).
    Otherwise linearly interpolates from ``alpha_baseline_start`` at epoch 0
    to ``alpha_baseline`` at epoch ``total_epochs - 1``.
    """
    if cfg.alpha_baseline_start is None or total_epochs <= 1:
        return cfg.alpha_baseline
    frac = max(0.0, min(1.0, epoch / (total_epochs - 1)))
    return cfg.alpha_baseline_start + frac * (
        cfg.alpha_baseline - cfg.alpha_baseline_start
    )


def sample_alpha_schedule(
    cfg: ClosedLoopConfig,
    T: int,
    C: int,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor | None:
    """Sample one (T, C) alpha schedule for a single batch.

    Returns ``None`` for batches that should run pure teacher forcing.

    Args:
        cfg: closed-loop hyperparameters.
        T: window length (number of timesteps).
        C: number of input channels (== output channels for closed-loop).
        device: device to place the result on.
        generator: torch RNG. Required when cfg.enabled is True for determinism;
            if None, falls back to the default RNG.
        dtype: floating dtype for the schedule.

    Returns:
        (T, C) tensor on `device`, or None when this batch is pure-TF.
        alpha[0, :] == 0 always (t=0 boundary).
    """
    if not cfg.enabled:
        return None

    # Pure-TF batch coin flip --------------------------------------------------
    if cfg.teacher_forcing_batch_frac > 0.0:
        u = torch.rand((), device=device, generator=generator)
        if u.item() < cfg.teacher_forcing_batch_frac:
            return None

    # Baseline (optionally jittered) -------------------------------------------
    if cfg.alpha_baseline_jitter > 0.0:
        j = (torch.rand((), device=device, generator=generator) * 2.0 - 1.0)
        a_base = float(cfg.alpha_baseline) + float(cfg.alpha_baseline_jitter) * j.item()
    else:
        a_base = float(cfg.alpha_baseline)

    # Per-channel sparse Gaussian perturbation, mu = 0 -------------------------
    a_rnd = torch.zeros(C, device=device, dtype=dtype)
    if cfg.alpha_rnd_density > 0.0 and cfg.alpha_rnd_sigma > 0.0:
        support = torch.rand(C, device=device, generator=generator) < cfg.alpha_rnd_density
        if support.any():
            noise = torch.randn(C, device=device, generator=generator, dtype=dtype) * cfg.alpha_rnd_sigma
            a_rnd = torch.where(support, noise, a_rnd)

    # Per-channel raw alpha, clipped to [0, 1] ---------------------------------
    alpha_raw = (a_base + a_rnd).clamp_(0.0, 1.0)  # (C,)

    # Envelope: half-cosine ramp 0 -> 1 over t_warm samples, then 1 ------------
    if cfg.t_warm > 0:
        ts = torch.arange(T, device=device, dtype=dtype)
        ramp = torch.clamp(ts / float(cfg.t_warm), max=1.0)
        envelope = 0.5 * (1.0 - torch.cos(math.pi * ramp))  # (T,)
    else:
        envelope = torch.ones(T, device=device, dtype=dtype)

    # Outer product -> (T, C) --------------------------------------------------
    alpha = envelope.unsqueeze(1) * alpha_raw.unsqueeze(0)

    # Force alpha[0, :] = 0 by construction (t=0 boundary, no y_prev) ----------
    alpha[0].zero_()

    return alpha


def init_channel_phases(C: int, seed: int, device, dtype=torch.float32) -> torch.Tensor:
    """Per-channel phase ∈ [0, 1) drawn once per training run.

    Used by the continuous-mode α sampler's per-channel rotation envelope.
    Deterministic given (C, seed) so repeat training runs use the same
    rotation pattern.
    """
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    return torch.rand(C, generator=g, dtype=dtype).to(device=device)


def sample_per_reader_jitter(
    cfg: ClosedLoopConfig,
    B: int,
    device,
    generator: torch.Generator | None = None,
    dtype=torch.float32,
) -> torch.Tensor:
    """One per-epoch draw: (B,) uniform in [-jitter, +jitter].

    Continuous-mode helper: each reader gets its own jitter offset, fixed
    for the duration of the epoch. Returns zeros if jitter is disabled.
    """
    if cfg.alpha_baseline_jitter <= 0.0:
        return torch.zeros(B, device=device, dtype=dtype)
    u = torch.rand(B, device=device, generator=generator, dtype=dtype) * 2.0 - 1.0
    return u * float(cfg.alpha_baseline_jitter)


def sample_continuous_alpha(
    cfg: ClosedLoopConfig,
    epoch: int,
    total_epochs: int,
    B: int,
    chunk_len: int,
    C: int,
    channel_phases: torch.Tensor,    # (C,) ∈ [0, 1) — fixed per training run
    per_reader_jitter: torch.Tensor, # (B,) — fixed per epoch
    device,
    generator: torch.Generator | None = None,
    dtype=torch.float32,
) -> torch.Tensor | None:
    """Sample one (B, chunk_len, C) α schedule for a continuous-mode chunk.

    Differences vs. the windowed sampler:
      - Drops `t_warm` (no within-window envelope).
      - Per-reader baseline (jitter is per-reader, fixed for the epoch).
      - Per-channel rotation envelope:
            chan_env[c] = α_rnd_sigma · sin²(π · (epoch/period − φ_c))
        with φ_c drawn once at training start.
      - Per-step Gaussian noise scaled by chan_env, fresh each chunk.

    Returns None for pure-TF chunks (when enabled and the coin flip hits).
    """
    if not cfg.enabled:
        return None

    # Pure-TF chunk coin flip -------------------------------------------------
    if cfg.teacher_forcing_batch_frac > 0.0:
        u = torch.rand((), device=device, generator=generator)
        if u.item() < cfg.teacher_forcing_batch_frac:
            return None

    # Per-reader baseline (epoch-ramp + jitter) -------------------------------
    target_scalar = effective_alpha_baseline(cfg, epoch, total_epochs)
    alpha_target = (torch.full((B,), target_scalar, device=device, dtype=dtype)
                    + per_reader_jitter.to(device=device, dtype=dtype))

    # Per-channel rotation envelope (deterministic from epoch + phases) -------
    period = max(1, int(cfg.alpha_rnd_period_epochs))
    phase = channel_phases.to(device=device, dtype=dtype)
    rot = math.pi * (epoch / period - phase)        # (C,)
    chan_env = float(cfg.alpha_rnd_sigma) * torch.sin(rot) ** 2  # (C,)

    # Per-step Gaussian rnd, scaled by channel envelope -----------------------
    if float(cfg.alpha_rnd_sigma) > 0.0:
        alpha_rnd = (torch.randn(B, chunk_len, C, generator=generator,
                                 device=device, dtype=dtype)
                     * chan_env.view(1, 1, C))
    else:
        alpha_rnd = torch.zeros(B, chunk_len, C, device=device, dtype=dtype)

    alpha = alpha_target.view(B, 1, 1) + alpha_rnd
    return alpha.clamp_(0.0, 1.0)


def summarize_alpha(alpha: torch.Tensor | None) -> dict[str, float]:
    """Compact summary stats for logging. Safe on None.

    Returns a dict with mean (over t>=1, i.e. excluding the forced-zero
    boundary), max, fraction of (t, c) entries that are >0.
    """
    if alpha is None:
        return {"alpha_mean": 0.0, "alpha_max": 0.0, "alpha_active_frac": 0.0,
                "is_pure_tf": 1.0}
    body = alpha[1:] if alpha.shape[0] > 1 else alpha
    return {
        "alpha_mean": float(body.mean().item()),
        "alpha_max": float(body.max().item()),
        "alpha_active_frac": float((body > 0).float().mean().item()),
        "is_pure_tf": 0.0,
    }
