"""Forward-replay an SRNN batched-ablation checkpoint and plot per-variant
time-series in the style of `simulate_srnn_eonly.py`.

Three input modes are supported:
    no_input : zeros for the entire t_range
    step     : zeros on [t_start, 0); on [0, t_end] divide into thirds —
               zero / 0.1*randn(input_size) per channel / zero
    trace    : zeros on [t_start, 0); on [0, t_end] the first
               int(t_end / cell.h) samples of the task's z-scored train trace

`t_start < 0` provides additional zero-input warm-up beyond the trained IC.

Outputs per variant per mode at:
    <out_dir>/<variant_name>/timeseries_<ckpt_tag>_<mode>.png
    <out_dir>/<variant_name>/lyapunov_<ckpt_tag>_<mode>.npz   (when LLE enabled)
where <ckpt_tag> is derived from the checkpoint filename:
    init.pt -> 'init', last.pt -> 'last', epoch_050.pt -> 'ep050'.

CLI:
    python scripts/plot_srnn_timeseries.py <ckpt> <out_dir> \\
        --mode no_input --t-start -15 --t-end 30 --plot-fs 25
    python scripts/plot_srnn_timeseries.py <ckpt> <out_dir> --mode all
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from _runs import load_train_trace, rebuild_model  # noqa: E402
from train_srnn.utils.history import load_checkpoint  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────
# Color palettes (ported from simulate_srnn_eonly.py)
# ─────────────────────────────────────────────────────────────────────────

EXCITATORY_COLORS = np.array([
    [1.00, 0.00, 0.00], [1.00, 0.75, 0.00], [0.85, 0.20, 0.45],
    [0.90, 0.10, 0.60], [0.90, 0.55, 0.00], [0.55, 0.27, 0.27],
    [0.86, 0.08, 0.24], [0.60, 0.15, 0.45],
])
INHIBITORY_COLORS = np.array([
    [0.00, 0.45, 0.74], [0.00, 0.75, 1.00], [0.20, 0.47, 0.62],
    [0.00, 0.50, 0.50], [0.30, 0.75, 0.93], [0.25, 0.62, 0.75],
    [0.00, 0.80, 0.80], [0.15, 0.55, 0.65],
])


def get_color(neuron_idx: int, n_E: int):
    if neuron_idx < n_E:
        return EXCITATORY_COLORS[neuron_idx % len(EXCITATORY_COLORS)]
    return INHIBITORY_COLORS[(neuron_idx - n_E) % len(INHIBITORY_COLORS)]


def plot_lines(ax, t, data, n_E, ylabel_text, ylim_range=None):
    """data shape: (T, n_neurons)."""
    n_neurons = data.shape[1]
    for i in range(n_neurons):
        ax.plot(t, data[:, i], color=get_color(i, n_E), linewidth=0.7, alpha=0.8)
    ax.set_ylabel(ylabel_text, fontsize=10)
    if ylim_range is not None:
        ax.set_ylim(ylim_range)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ─────────────────────────────────────────────────────────────────────────
# Input builders
# ─────────────────────────────────────────────────────────────────────────

def _task_trace(cfg, n_samples_needed: int) -> np.ndarray:
    """First ``n_samples_needed`` rows of the task's z-scored train trace, ``(T, C)``."""
    trace = load_train_trace(cfg)
    if trace is None or trace.shape[0] < n_samples_needed:
        raise RuntimeError(f"task {cfg.task.name} has no train trace of at least "
                           f"{n_samples_needed} samples; reduce --t-end")
    return trace[:n_samples_needed]


def build_inputs(
    mode: str,
    cfg,
    h: float,
    input_size: int,
    t_start: float,
    t_end: float,
    device: torch.device,
) -> tuple[torch.Tensor, int, int]:
    """Return (inputs, T_warm, T_main).

    inputs : (T_total, input_size) float32 on `device`
    T_warm : number of leading zero-input steps (from t_start < 0)
    T_main : number of mode-driven steps on [0, t_end]
    """
    if t_end <= 0:
        raise ValueError(f"t_end must be > 0, got {t_end}")
    T_warm = int(round(max(0.0, -t_start) / h))
    T_main = int(round(t_end / h))
    T_total = T_warm + T_main

    inputs = torch.zeros(T_total, input_size, dtype=torch.float32)

    if mode == "no_input":
        pass

    elif mode == "step":
        third = T_main // 3
        amp = 0.1 * torch.randn(input_size)
        # Middle third only
        inputs[T_warm + third : T_warm + 2 * third] = amp.unsqueeze(0)

    elif mode == "trace":
        trace = _task_trace(cfg, T_main)  # (T_main, n_chan)
        if trace.shape[1] != input_size:
            raise RuntimeError(f"trace has {trace.shape[1]} channels but input_size is {input_size}")
        inputs[T_warm:T_warm + T_main] = torch.from_numpy(trace)

    else:
        raise ValueError(f"Unknown mode {mode!r}")

    return inputs.to(device), T_warm, T_main


# ─────────────────────────────────────────────────────────────────────────
# Forward replay
# ─────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def replay(model, inputs: torch.Tensor, plot_step: int):
    """Roll cell forward over inputs (T_total, input_size). Capture
    diagnostics every plot_step steps.

    Returns a dict of CPU numpy arrays of shape (T_plot, K, ...) plus
    a (T_plot,) time-step index array.
    """
    cell = model.cell
    K = cell.K
    device = next(cell.parameters()).device

    # Initial state
    if hasattr(model, "ic"):
        state = model.ic(1)  # (K, 1, state_dim)
    else:
        warnings.warn(
            "Model has no TrainableIC (trainable_ic=False). Using the cell initial state — "
            "consider running with a more negative --t-start to allow longer "
            "zero-input warm-up.",
            stacklevel=2,
        )
        state = cell.init_state(1, device=device)

    state = state.to(device)
    T_total = inputs.shape[0]
    n_E = cell.n_E
    n_I = cell.n_I

    # Pre-allocate plot buffers
    n_plot = (T_total + plot_step - 1) // plot_step
    bufs = {
        "u":      np.zeros((n_plot, K, n_E + n_I), dtype=np.float32),
        "x":      np.zeros((n_plot, K, n_E + n_I), dtype=np.float32),
        "br":     np.zeros((n_plot, K, n_E + n_I), dtype=np.float32),
        "a_E":    np.zeros((n_plot, K, n_E, cell.max_n_a_E or 1), dtype=np.float32),
        "a_I":    np.zeros((n_plot, K, n_I, cell.max_n_a_I or 1), dtype=np.float32),
        "b_full": np.zeros((n_plot, K, n_E + n_I), dtype=np.float32),
    }
    t_idx = np.zeros(n_plot, dtype=np.int64)

    pi = 0
    for t in range(T_total):
        u_t = inputs[t : t + 1]  # (1, input_size) -> auto-broadcast to (K, 1, .)
        _, state = cell(u_t, state)
        if t % plot_step == 0:
            diag = cell.get_diagnostics(state, u_t)
            bufs["u"][pi]      = diag["u"].squeeze(1).cpu().numpy()
            bufs["x"][pi]      = diag["x"].squeeze(1).cpu().numpy()
            bufs["br"][pi]     = diag["br"].squeeze(1).cpu().numpy()
            bufs["b_full"][pi] = diag["b_full"].squeeze(1).cpu().numpy()
            if cell.max_n_a_E > 0:
                bufs["a_E"][pi] = diag["a_E"].squeeze(1).cpu().numpy()
            if cell.max_n_a_I > 0:
                bufs["a_I"][pi] = diag["a_I"].squeeze(1).cpu().numpy()
            t_idx[pi] = t
            pi += 1

    # Trim trailing slack if any
    for k in bufs:
        bufs[k] = bufs[k][:pi]
    t_idx = t_idx[:pi]
    return bufs, t_idx


# ─────────────────────────────────────────────────────────────────────────
# Benettin largest Lyapunov exponent
# ─────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def benettin_replay(
    model,
    inputs: torch.Tensor,
    T_warm: int,
    *,
    lya_M: int = 5,
    d0: float = 1e-3,
    seed: int = 0,
):
    """Streaming Benettin largest-Lyapunov pass for the K batched variants.

    Mirrors `benettin_algorithm.m`: every `lya_M` cell-forward steps, fork a
    perturbation of size `d0` from the fiducial state, integrate the same
    `lya_M` steps with the same input slice, measure divergence, and
    renormalise the perturbation direction. Local Lyapunov exponents are
    computed throughout the entire input window (including warm-up) so the
    transient settling is visible; the running-average `finite_lya` only
    accumulates `log(d_k/d0)` for intervals starting at real time t >= 0.

    Returns dict with arrays shaped (n_lya,) and (n_lya, K):
        t_lya, local_lya, finite_lya, LLE (K,), plus scalar metadata.
    `t_lya[k]` is the START of the k-th interval, in seconds, with t=0
    aligned at the end of the warm-up (so warm-up samples have t<0).
    """
    cell = model.cell
    h = float(cell.h)
    K = cell.K
    state_size = cell.state_size
    device = next(cell.parameters()).device
    tau_lya = lya_M * h

    if hasattr(model, "ic"):
        state_fid = model.ic(1).to(device)
    else:
        state_fid = cell.init_state(1, device=device)

    # Padded coordinates are not dynamical variables and must not enter the norm.
    a_e, a_i, b_e, b_i, x = cell.unpack_state(torch.ones_like(state_fid))
    active = cell.pack_state(a_e * cell.sfa_E_mask.unsqueeze(1),
                             a_i * cell.sfa_I_mask.unsqueeze(1),
                             b_e * cell.std_E_mask.unsqueeze(1),
                             b_i * cell.std_I_mask.unsqueeze(1), x)
    g = torch.Generator(device="cpu").manual_seed(seed)
    d_unit = torch.randn(K, 1, state_size, generator=g).to(state_fid) * active
    d_unit = d_unit / d_unit.flatten(1).norm(dim=-1).clamp_min(1e-30).view(K, 1, 1)

    T_total = inputs.shape[0]

    t = 0
    t_lya_list: list[float] = []
    local_list: list[np.ndarray] = []
    finite_list: list[np.ndarray] = []
    sum_log = torch.zeros(K, device=device)
    finite_t = 0.0  # accumulated time post-warm-up (for the running average)
    diverged = torch.zeros(K, dtype=torch.bool, device=device)
    last_finite = torch.zeros(K, device=device)

    while t + lya_M <= T_total:
        # Build perturbed state from current fiducial
        state_pert = state_fid + d_unit * d0
        # Roll fiducial AND perturbed forward by lya_M steps using the same inputs
        for s in range(lya_M):
            u_t = inputs[t + s : t + s + 1]
            _, state_fid = cell(u_t, state_fid)
            _, state_pert = cell(u_t, state_pert)
        t += lya_M

        delta = (state_pert - state_fid) * active
        d_k = delta.flatten(1).norm(dim=-1).clamp_min(1e-30)  # (K,)
        log_ratio = torch.log(d_k / d0)
        local_lya = log_ratio / tau_lya
        new_diverged = ~torch.isfinite(local_lya) | ~torch.isfinite(d_k)

        # Freeze diverged variants at their last good values for plotting
        local_out = torch.where(diverged | new_diverged, last_finite, local_lya)

        # Real-time interval bounds (t=0 at end of warm-up)
        t_seg_start = (t - lya_M - T_warm) * h
        t_seg_end = (t - T_warm) * h

        # Running average only accumulates for intervals starting at t>=0
        if t_seg_start >= 0.0:
            log_ratio_safe = torch.where(diverged | new_diverged,
                                         torch.zeros_like(log_ratio), log_ratio)
            sum_log = sum_log + log_ratio_safe
            finite_t = finite_t + tau_lya
            finite_lya = sum_log / max(finite_t, 1e-12)
            finite_lya = torch.where(diverged, last_finite, finite_lya)
            last_finite = torch.where(diverged, last_finite, finite_lya)
        else:
            # Pre-transient: keep finite_lya at NaN so the plot shows nothing
            finite_lya = torch.full_like(local_lya, float("nan"))

        diverged = diverged | new_diverged

        t_lya_list.append(t_seg_start)
        local_list.append(local_out.detach().cpu().numpy())
        finite_list.append(finite_lya.detach().cpu().numpy())

        # Renormalise perturbation direction (keep old direction for diverged)
        d_unit_new = delta / d_k.view(K, 1, 1)
        d_unit = torch.where(diverged.view(K, 1, 1), d_unit, d_unit_new)

    if not t_lya_list:
        return {
            "t_lya": np.zeros(0),
            "local_lya": np.zeros((0, K)),
            "finite_lya": np.zeros((0, K)),
            "LLE": np.zeros(K),
            "lya_dt": tau_lya, "lya_M": int(lya_M), "d0": float(d0), "h": h,
        }
    return {
        "t_lya": np.asarray(t_lya_list, dtype=np.float64),
        "local_lya": np.stack(local_list, axis=0),
        "finite_lya": np.stack(finite_list, axis=0),
        "LLE": last_finite.detach().cpu().numpy(),
        "lya_dt": tau_lya,
        "lya_M": int(lya_M),
        "d0": float(d0),
        "h": h,
    }


# ─────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────

def render_variant(
    bufs: dict,
    t_seconds: np.ndarray,
    k: int,
    variant_name: str,
    cfg_k,
    n_E: int,
    n_I: int,
    out_path: Path,
    mode: str,
    t_warm_seconds: float,
    lya: dict | None = None,
    ckpt_tag: str | None = None,
):
    """Build the timeseries figure for variant k and save to out_path.

    If `lya` is provided, append a 6th panel showing the local + finite
    Lyapunov exponent for this variant.
    """
    n_panels = 7 if lya is not None else 5
    fig, axes = plt.subplots(n_panels, 1, figsize=(16, 2.2 * n_panels), sharex=True)

    u = bufs["u"][:, k, :]              # (T_plot, N)
    x = bufs["x"][:, k, :]
    br = bufs["br"][:, k, :]
    b_full = bufs["b_full"][:, k, :]

    plot_lines(axes[0], t_seconds, u, n_E, "u(t)\n(post-W_in)")
    plot_lines(axes[1], t_seconds, x, n_E, "x(t)\ndendritic")
    plot_lines(axes[2], t_seconds, br, n_E, "r(t) Π b_m(t)\nsynaptic out", ylim_range=(-0.05, 1.05))

    # Mean SFA state: only over the variant's actual n_a_E timescales
    n_a_E_k = cfg_k.n_a_E
    n_a_I_k = cfg_k.n_a_I
    if n_a_E_k > 0 or n_a_I_k > 0:
        a_E = bufs["a_E"][:, k, :, :n_a_E_k] if n_a_E_k > 0 else None
        a_I = bufs["a_I"][:, k, :, :n_a_I_k] if n_a_I_k > 0 else None
        a_sum_E = a_E.mean(axis=-1) if a_E is not None else np.zeros((len(t_seconds), n_E))
        a_sum_I = a_I.mean(axis=-1) if a_I is not None else np.zeros((len(t_seconds), n_I))
        a_sum = np.concatenate([a_sum_E, a_sum_I], axis=-1)
        plot_lines(axes[3], t_seconds, a_sum, n_E, "mean a(t)\nSFA")
    else:
        axes[3].set_ylabel("mean a(t)\n(inactive)", fontsize=10)
        axes[3].text(0.5, 0.5, "no SFA in this variant",
                     transform=axes[3].transAxes, ha="center", va="center",
                     color="0.5")

    plot_lines(axes[4], t_seconds, b_full, n_E, "Π b_m(t)\nSTD gain",
               ylim_range=(-0.05, 1.1))

    lle_text = ""
    if lya is not None and lya["t_lya"].size > 0:
        t_l = lya["t_lya"]
        loc = lya["local_lya"][:, k]
        fin = lya["finite_lya"][:, k]
        lle_k = float(lya["LLE"][k])

        # Panel 5: local Lyapunov exponent
        ax_loc = axes[5]
        ax_loc.plot(t_l, loc, color="C0", lw=0.7)
        ax_loc.axhline(0.0, color="k", lw=0.5, ls="--")
        ax_loc.axhline(lle_k, color="C3", lw=1.0, ls="-",
                       label=f"LLE = {lle_k:+.4f}")
        ax_loc.set_ylabel("local λ (1/s)\nBenettin", fontsize=10)
        ax_loc.legend(fontsize=8, loc="best", frameon=False)
        ax_loc.spines["top"].set_visible(False)
        ax_loc.spines["right"].set_visible(False)

        # Panel 6: finite-time (running average) Lyapunov exponent
        ax_fin = axes[6]
        ax_fin.plot(t_l, fin, color="C0", lw=1.5)
        ax_fin.axhline(0.0, color="k", lw=0.5, ls="--")
        ax_fin.axhline(lle_k, color="C3", lw=1.0, ls="-",
                       label=f"LLE = {lle_k:+.4f}")
        ax_fin.set_ylabel("finite-time λ (1/s)\nrunning average", fontsize=10)
        ax_fin.legend(fontsize=8, loc="best", frameon=False)
        ax_fin.spines["top"].set_visible(False)
        ax_fin.spines["right"].set_visible(False)

        lle_text = f"  LLE={lle_k:+.4f}"

    # Mark the t=0 boundary (end of zero-input warm-up) when present
    if t_warm_seconds > 0:
        for ax in axes:
            ax.axvline(0.0, color="0.4", linestyle=":", linewidth=0.8)

    axes[-1].set_xlabel("time (s)")
    title = f"{variant_name} — mode={mode}"
    if ckpt_tag:
        title += f" — ckpt={ckpt_tag}"
    fig.suptitle(f"{title}{lle_text}", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────

def _default_ckpt_tag(ckpt_path: Path) -> str:
    """Derive a short tag from a checkpoint filename:
        init.pt        -> 'init'
        last.pt        -> 'last'
        epoch_050.pt   -> 'ep050'
        anything else  -> the bare filename stem.
    """
    stem = Path(ckpt_path).stem
    if stem.startswith("epoch_"):
        return "ep" + stem[len("epoch_"):]
    return stem


def plot_replay(
    ckpt_path: Path,
    out_dir: Path,
    mode: str,
    t_range: tuple[float, float] = (-15.0, 30.0),
    plot_fs: float = 25.0,
    device: str = "cpu",
    compute_lyapunov: bool = True,
    lya_M: int = 5,
    lya_d0: float = 1e-3,
    lya_seed: int = 0,
    ckpt_tag: str | None = None,
):
    """Replay one mode + checkpoint; write tagged figures into <out_dir>/<variant>/.

    Outputs (per variant):
        timeseries_<ckpt_tag>_<mode>.png
        lyapunov_<ckpt_tag>_<mode>.npz   (when compute_lyapunov=True)
    `ckpt_tag` defaults to a short tag derived from the checkpoint filename
    via _default_ckpt_tag (e.g. 'init', 'last', 'ep050').
    """
    ckpt_path = Path(ckpt_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if ckpt_tag is None:
        ckpt_tag = _default_ckpt_tag(ckpt_path)

    print(f"[replay] ckpt={ckpt_tag} loading {ckpt_path}")
    ckpt = load_checkpoint(str(ckpt_path), device=device)
    model, cfg, variant_names = rebuild_model(ckpt, device)

    cell = model.cell
    h = float(cell.h)
    input_size = int(cfg.task.input_size)

    plot_step = max(1, int(round((1.0 / plot_fs) / h)))
    print(f"[replay] ckpt={ckpt_tag} mode={mode} t_range={t_range} h={h} plot_step={plot_step} "
          f"(plot_fs ≈ {1.0/(plot_step*h):.2f} Hz)")

    inputs, T_warm, T_main = build_inputs(
        mode, cfg, h, input_size, t_range[0], t_range[1], torch.device(device),
    )
    print(f"[replay] inputs: T_warm={T_warm}, T_main={T_main}, total={inputs.shape[0]}")

    bufs, t_idx = replay(model, inputs, plot_step)
    # t_idx is in steps; convert to seconds with t=0 at end of warm-up
    t_seconds = (t_idx - T_warm) * h
    t_warm_seconds = T_warm * h

    n_E = cell.n_E
    n_I = cell.n_I

    lya = None
    if compute_lyapunov:
        print(f"[replay] benettin: lya_M={lya_M} (lya_dt={lya_M*h:.4f}s) d0={lya_d0:g}")
        lya = benettin_replay(
            model, inputs, T_warm,
            lya_M=lya_M, d0=lya_d0, seed=lya_seed,
        )
        for k, name in enumerate(variant_names):
            variant_dir = out_dir / name
            variant_dir.mkdir(parents=True, exist_ok=True)
            np.savez(
                variant_dir / f"lyapunov_{ckpt_tag}_{mode}.npz",
                t_lya=lya["t_lya"],
                local_lya=lya["local_lya"][:, k],
                finite_lya=lya["finite_lya"][:, k],
                LLE=lya["LLE"][k],
                lya_dt=lya["lya_dt"],
                lya_M=lya["lya_M"],
                d0=lya["d0"],
                h=lya["h"],
                mode=mode,
                ckpt=ckpt_tag,
                variant=name,
            )
        print(f"[replay] ckpt={ckpt_tag} LLE per variant: " + ", ".join(
            f"{n}={lya['LLE'][k]:+.4f}" for k, n in enumerate(variant_names)
        ))

    for k, name in enumerate(variant_names):
        variant_dir = out_dir / name
        variant_dir.mkdir(parents=True, exist_ok=True)
        out_path = variant_dir / f"timeseries_{ckpt_tag}_{mode}.png"
        render_variant(
            bufs, t_seconds, k, name, cell.configs[k],
            n_E, n_I, out_path, mode, t_warm_seconds,
            lya=lya,
            ckpt_tag=ckpt_tag,
        )
        print(f"[replay]   wrote {out_path.relative_to(out_dir.parent)}")


# ─────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ckpt_path", help="Path to last.pt (or any *.pt with config + variant names)")
    p.add_argument("out_dir", help="Per-variant subdirs are created under this path")
    p.add_argument("--mode", default="all",
                   choices=["no_input", "step", "trace", "all"],
                   help="Input mode (default: all → runs all three)")
    p.add_argument("--t-start", type=float, default=-15.0,
                   help="Start time in seconds (negative = zero-input warm-up; default: -15)")
    p.add_argument("--t-end", type=float, default=30.0,
                   help="End time in seconds (default: 30)")
    p.add_argument("--plot-fs", type=float, default=25.0,
                   help="Plot decimation rate in Hz (default: 25)")
    p.add_argument("--device", default="cpu", help="torch device (default: cpu)")
    p.add_argument("--no-lyapunov", action="store_true",
                   help="Skip the Benettin LLE pass and the lyapunov panel")
    p.add_argument("--lya-M", type=int, default=5,
                   help="Cell forward steps between Benettin rescalings (default: 5 → lya_dt=5h)")
    p.add_argument("--lya-d0", type=float, default=1e-3,
                   help="Benettin perturbation magnitude (default: 1e-3)")
    p.add_argument("--lya-seed", type=int, default=0,
                   help="RNG seed for the initial perturbation direction (default: 0)")
    return p.parse_args()


def main():
    args = parse_args()
    modes = ["no_input", "step", "trace"] if args.mode == "all" else [args.mode]
    for m in modes:
        plot_replay(
            ckpt_path=Path(args.ckpt_path),
            out_dir=Path(args.out_dir),
            mode=m,
            t_range=(args.t_start, args.t_end),
            plot_fs=args.plot_fs,
            device=args.device,
            compute_lyapunov=not args.no_lyapunov,
            lya_M=args.lya_M,
            lya_d0=args.lya_d0,
            lya_seed=args.lya_seed,
        )


if __name__ == "__main__":
    main()
