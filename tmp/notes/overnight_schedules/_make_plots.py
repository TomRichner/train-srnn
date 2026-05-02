"""Generate the LR + alpha schedule plots for the overnight 700-epoch runs.

Run from repo root:  python tmp/notes/overnight_schedules/_make_plots.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/../../.."))
from train_srnn.utils.lr_schedule import WarmupHoldCosineSchedule

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


# -----------------------------------------------------------------------------
# Run config (matches conf/config.yaml defaults + overnight_run.sh overrides)
# -----------------------------------------------------------------------------
EPOCHS = 700
B = 48
T = 179989                                # full seeg train trace
chunk_len = 250
LR_BOTH = 5e-4

# train.py:425 — uses the *windowed* len(train_x), not the continuous step count.
# At seq_len=3750, stride=1250, T=179989  →  len(train_x) = 141
LEN_TRAIN_X_WINDOWED = 141
STEPS_PER_EPOCH_TRAINPY = LEN_TRAIN_X_WINDOWED // B + 1   # = 3
TOTAL_STEPS_TRAINPY = EPOCHS * STEPS_PER_EPOCH_TRAINPY    # = 2100
WARMUP_FRAC = min(2 * STEPS_PER_EPOCH_TRAINPY / TOTAL_STEPS_TRAINPY, 0.2)

STEPS_PER_EPOCH_CONTINUOUS = (T + B * chunk_len - 1) // (B * chunk_len)   # = 15
TOTAL_SCHED_STEPS = EPOCHS * STEPS_PER_EPOCH_CONTINUOUS                   # = 10500


# -----------------------------------------------------------------------------
# 1. LR schedule
# -----------------------------------------------------------------------------
def make_lr_plot():
    """Reproduce the actual scheduler trajectory by rolling forward
    `WarmupHoldCosineSchedule` exactly as `train.py` constructs it, then
    stepping `TOTAL_SCHED_STEPS` times (the continuous trainer's true cadence)."""
    dummy = torch.nn.Linear(1, 1)
    opt = torch.optim.Adam(dummy.parameters(), lr=LR_BOTH)
    sched = WarmupHoldCosineSchedule(
        opt,
        total_steps=TOTAL_STEPS_TRAINPY,            # < TOTAL_SCHED_STEPS — known shadow bug
        max_lr=LR_BOTH,
        warmup_frac=WARMUP_FRAC,
        cosine_decay=False,                          # current default
    )
    lrs = []
    for _ in range(TOTAL_SCHED_STEPS):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    lrs = np.asarray(lrs)
    epochs = np.arange(TOTAL_SCHED_STEPS) / STEPS_PER_EPOCH_CONTINUOUS

    fig, axs = plt.subplots(1, 2, figsize=(11, 4))

    # Panel A: zoom into the warmup
    ax = axs[0]
    warm_steps = int(TOTAL_STEPS_TRAINPY * WARMUP_FRAC)
    show_steps = max(warm_steps + 5, 30)
    ax.plot(np.arange(show_steps), lrs[:show_steps],
            marker="o", lw=1.5, ms=4, color="C0")
    ax.axvline(warm_steps, color="k", ls=":", lw=1)
    ax.text(warm_steps + 0.5, LR_BOTH * 0.5,
            f"warmup_end = {warm_steps} steps",
            rotation=90, va="center", fontsize=9)
    ax.set_xlabel("scheduler.step() count")
    ax.set_ylabel("learning rate")
    ax.set_title(f"Warmup phase (first {show_steps} scheduler steps)")
    ax.grid(alpha=0.3)
    secax = ax.secondary_xaxis(
        "top",
        functions=(
            lambda s: s / STEPS_PER_EPOCH_CONTINUOUS,
            lambda e: e * STEPS_PER_EPOCH_CONTINUOUS,
        ),
    )
    secax.set_xlabel("epoch")

    # Panel B: full run
    ax = axs[1]
    ax.plot(epochs, lrs, lw=1.0, color="C0")
    ax.set_xlabel("epoch")
    ax.set_ylabel("learning rate")
    ax.set_title(f"Full {EPOCHS}-epoch run")
    ax.set_ylim(0, LR_BOTH * 1.1)
    ax.grid(alpha=0.3)

    fig.suptitle(
        f"LR schedule — overnight runs (lr={LR_BOTH:g}, cosine_decay=false)\n"
        f"WarmupHoldCosineSchedule sees total_steps={TOTAL_STEPS_TRAINPY} "
        f"(windowed bug); actual scheduler.step() count = {TOTAL_SCHED_STEPS}",
        fontsize=10,
    )
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "lr_schedule.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"wrote {out}")
    print(f"  warmup completes after {warm_steps} scheduler.step() = "
          f"{warm_steps / STEPS_PER_EPOCH_CONTINUOUS:.3f} epochs")
    print(f"  steady-state lr after warmup = {lrs[warm_steps]:g}")


# -----------------------------------------------------------------------------
# 2. Closed-loop α schedule for run 2
# -----------------------------------------------------------------------------
# Run 2 used closed_loop.enabled=true with all conf/config.yaml defaults:
#   alpha_baseline:           0.3
#   alpha_baseline_start:     null  (no run-time ramp)
#   alpha_baseline_jitter:    0.0   (no per-reader scatter)
#   alpha_rnd_density:        0.0   (no per-channel sparse Gaussian)
#   alpha_rnd_sigma:          0.0
#   alpha_rnd_period_epochs:  10    (irrelevant when sigma=0)
#   teacher_forcing_batch_frac: 0.2 (20% of chunks forced to α=0)
#   t_warm:                   2500  (windowed-only — IGNORED in continuous mode)
#
# Net effect (per-step, per-reader, per-channel):
#   With prob 0.2:  alpha = 0   (pure teacher forcing for that chunk)
#   Otherwise:      alpha = 0.3 (constant, every reader, every channel, every step)
#
# Logged stats from epoch 699:
#   alpha_mean = 0.240   ≈ 0.3 * (1 - 0.2)
#   alpha_max  = 0.300
#   pure_tf_frac = 0.200

def make_alpha_plot():
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))

    # Panel A: per-chunk α value over a representative window of training.
    # Show 50 chunks; mark which are pure-TF (α=0) and which are α=0.3.
    rng = np.random.default_rng(0)
    n_chunks = 50
    is_pure_tf = rng.random(n_chunks) < 0.2
    alpha_per_chunk = np.where(is_pure_tf, 0.0, 0.3)

    ax = axs[0]
    ax.bar(np.arange(n_chunks), alpha_per_chunk,
           color=np.where(is_pure_tf, "#cc4444", "#4488cc"),
           width=0.85)
    ax.axhline(0.3, color="#4488cc", ls=":", lw=1, label=r"$\alpha_\mathrm{baseline}=0.3$")
    ax.axhline(0.24, color="k", ls="--", lw=1,
               label=r"empirical mean $=0.24$")
    ax.set_xlabel("chunk index (within an epoch)")
    ax.set_ylabel(r"$\alpha$")
    ax.set_title(r"Per-chunk $\alpha$ (50-chunk sample)")
    ax.set_ylim(-0.02, 0.35)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    # Panel B: empirical α distribution histogram across many chunks.
    n_total = 10000
    is_pure_tf_big = rng.random(n_total) < 0.2
    alpha_big = np.where(is_pure_tf_big, 0.0, 0.3)
    ax = axs[1]
    ax.hist(alpha_big, bins=[-0.025, 0.025, 0.275, 0.325],
            color="#4488cc", edgecolor="k", rwidth=0.9)
    ax.set_xlim(-0.05, 0.4)
    ax.set_xlabel(r"$\alpha$ value")
    ax.set_ylabel("count (out of 10,000 chunks)")
    ax.set_title(r"Empirical $\alpha$ distribution")
    ax.grid(alpha=0.3)
    ax.text(0.0, n_total * 0.22, f"~{int(n_total*0.2)}\n(pure-TF)",
            ha="center", fontsize=9)
    ax.text(0.3, n_total * 0.85, f"~{int(n_total*0.8)}\n(α=0.3)",
            ha="center", fontsize=9, color="white")

    fig.suptitle(
        r"Closed-loop $\alpha$ schedule — run 2 (overnight-run2-Whoist-cl)" "\n"
        r"$\alpha_\mathrm{baseline}=0.3$, jitter/sparse-rnd disabled, "
        r"$P(\mathrm{pure\ TF})=0.2$", fontsize=10,
    )
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "alpha_schedule.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    make_lr_plot()
    make_alpha_plot()
