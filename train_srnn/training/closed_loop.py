"""Alpha schedules for closed-loop teacher forcing.

The cell input is ``x_in = (1 - alpha) * x_real + alpha * y_pred[t-1]``;
alpha = 0 is teacher forcing, alpha = 1 a free run. The windowed sampler
draws one ``(T, C)`` schedule per batch: a half-cosine ramp over ``t_warm``
samples times a per-batch baseline plus a sparse per-channel perturbation.
The continuous sampler draws ``(B, T, C)`` per chunk with a per-reader
baseline and a per-channel rotating noise envelope. A fraction of batches
are pure teacher forcing and return None. ``alpha[0]`` is always 0.
"""
from __future__ import annotations

import math

import torch


from train_srnn.config import ClosedLoopConfig


def effective_alpha_baseline(cfg: ClosedLoopConfig, epoch: int,
                             total_epochs: int) -> float:
    """Baseline for this epoch: linear ramp from ``alpha_baseline_start`` when set."""
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
    """One ``(T, C)`` schedule for a windowed batch, or None for a teacher-forced batch."""
    if not cfg.enabled:
        return None

    # Pure-TF batch coin flip
    if cfg.teacher_forcing_batch_frac > 0.0:
        u = torch.rand((), device=device, generator=generator)
        if u.item() < cfg.teacher_forcing_batch_frac:
            return None

    # Baseline (optionally jittered)
    if cfg.alpha_baseline_jitter > 0.0:
        j = (torch.rand((), device=device, generator=generator) * 2.0 - 1.0)
        a_base = float(cfg.alpha_baseline) + float(cfg.alpha_baseline_jitter) * j.item()
    else:
        a_base = float(cfg.alpha_baseline)

    # Per-channel sparse Gaussian perturbation, mu = 0
    a_rnd = torch.zeros(C, device=device, dtype=dtype)
    if cfg.alpha_rnd_density > 0.0 and cfg.alpha_rnd_sigma > 0.0:
        support = torch.rand(C, device=device, generator=generator) < cfg.alpha_rnd_density
        if support.any():
            noise = torch.randn(C, device=device, generator=generator, dtype=dtype) * cfg.alpha_rnd_sigma
            a_rnd = torch.where(support, noise, a_rnd)

    # Per-channel raw alpha, clipped to [0, 1]
    alpha_raw = (a_base + a_rnd).clamp_(0.0, 1.0)  # (C,)

    # Envelope: half-cosine ramp 0 -> 1 over t_warm samples, then 1
    if cfg.t_warm > 0:
        ts = torch.arange(T, device=device, dtype=dtype)
        ramp = torch.clamp(ts / float(cfg.t_warm), max=1.0)
        envelope = 0.5 * (1.0 - torch.cos(math.pi * ramp))  # (T,)
    else:
        envelope = torch.ones(T, device=device, dtype=dtype)

    # Outer product -> (T, C)
    alpha = envelope.unsqueeze(1) * alpha_raw.unsqueeze(0)

    # Force alpha[0, :] = 0 by construction (t=0 boundary, no y_prev)
    alpha[0].zero_()

    return alpha


def init_channel_phases(C: int, seed: int, device, dtype=torch.float32) -> torch.Tensor:
    """Per-channel phases in [0, 1) for the rotating noise envelope; fixed per run."""
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    return torch.rand(C, generator=g, dtype=dtype).to(device=device)


def sample_per_reader_jitter(
    cfg: ClosedLoopConfig,
    B: int,
    device,
    generator: torch.Generator | None = None,
    dtype=torch.float32,
) -> torch.Tensor:
    """Per-reader baseline offsets, uniform in [-jitter, jitter], fixed for one epoch."""
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
    """One ``(B, chunk_len, C)`` schedule for a ring chunk, or None for a teacher-forced chunk.

    Per-reader baseline (epoch ramp plus jitter) plus Gaussian noise whose
    per-channel amplitude is ``alpha_rnd_sigma * sin^2(pi (epoch / period - phase_c))``.
    """
    if not cfg.enabled:
        return None

    # Pure-TF chunk coin flip
    if cfg.teacher_forcing_batch_frac > 0.0:
        u = torch.rand((), device=device, generator=generator)
        if u.item() < cfg.teacher_forcing_batch_frac:
            return None

    # Per-reader baseline (epoch-ramp + jitter)
    target_scalar = effective_alpha_baseline(cfg, epoch, total_epochs)
    alpha_target = (torch.full((B,), target_scalar, device=device, dtype=dtype)
                    + per_reader_jitter.to(device=device, dtype=dtype))

    # Per-channel rotation envelope (deterministic from epoch + phases)
    period = max(1, int(cfg.alpha_rnd_period_epochs))
    phase = channel_phases.to(device=device, dtype=dtype)
    rot = math.pi * (epoch / period - phase)        # (C,)
    chan_env = float(cfg.alpha_rnd_sigma) * torch.sin(rot) ** 2  # (C,)

    # Per-step Gaussian rnd, scaled by channel envelope
    if float(cfg.alpha_rnd_sigma) > 0.0:
        alpha_rnd = (torch.randn(B, chunk_len, C, generator=generator,
                                 device=device, dtype=dtype)
                     * chan_env.view(1, 1, C))
    else:
        alpha_rnd = torch.zeros(B, chunk_len, C, device=device, dtype=dtype)

    alpha = alpha_target.view(B, 1, 1) + alpha_rnd
    return alpha.clamp_(0.0, 1.0)


def summarize_alpha(alpha: torch.Tensor | None) -> dict[str, float]:
    """Mean (excluding t = 0), max, and active fraction of a schedule; safe on None."""
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
