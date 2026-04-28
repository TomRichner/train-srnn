"""Forward-replay an SRNN batched-ablation checkpoint and plot per-variant
time-series in the style of `simulate_srnn_eonly.py`.

Three input modes are supported:
    no_input : zeros for the entire t_range
    step     : zeros on [t_start, 0); on [0, t_end] divide into thirds —
               zero / 0.1*randn(input_size) per channel / zero
    seeg     : zeros on [t_start, 0); on [0, t_end] use the first
               int(t_end / cell.h) samples of the SEEG train trace
               (decimated + z-scored exactly as `load_seeg` does)

`t_start < 0` provides additional zero-input warm-up beyond the trained IC.

Outputs one PNG per variant per mode at:
    <out_dir>/<variant_name>/timeseries_<mode>.png

CLI:
    python scripts/plots/plot_srnn_timeseries.py <ckpt> <out_dir> \\
        --mode no_input --t-start -15 --t-end 30 --plot-fs 25
    python scripts/plots/plot_srnn_timeseries.py <ckpt> <out_dir> --mode all
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
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from train_srnn.models.factory import build_batched_model  # noqa: E402
from train_srnn.utils.checkpoint import load_checkpoint  # noqa: E402


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

def _seeg_trace(cfg, n_samples_needed: int) -> np.ndarray:
    """Read a continuous SEEG train trace (decimated + z-scored) and return
    the first n_samples_needed rows.  Replicates the head of `load_seeg`
    without windowing.  Returns shape (T, n_chan)."""
    import h5py

    task = cfg.task
    data_dir = task.get("data_dir", "train_srnn/data/seeg")
    if not os.path.isabs(data_dir):
        data_dir = str(REPO / data_dir)
    fname = (f"seeg_{task.subject_id}_b{task.block}_"
             f"{task.sleep}_{task.cond}.mat")
    path = os.path.join(data_dir, fname)
    with h5py.File(path, "r") as f:
        data = np.array(f["data_filt"]).T.astype(np.float32)  # (T, C)

    decimate = int(task.get("decimate", 1))
    if decimate > 1:
        T = (data.shape[0] // decimate) * decimate
        data = data[:T].reshape(-1, decimate, data.shape[1]).sum(axis=1) / decimate

    n_samples = data.shape[0]
    t1 = int(0.75 * n_samples)
    train_trace = data[:t1]
    mu = train_trace.mean(axis=0, keepdims=True)
    sd = train_trace.std(axis=0, keepdims=True)
    sd[sd < 1e-8] = 1.0
    train_trace = (train_trace - mu) / sd

    if train_trace.shape[0] < n_samples_needed:
        raise RuntimeError(
            f"SEEG train trace has {train_trace.shape[0]} samples "
            f"after decimation; need {n_samples_needed} for the requested "
            f"t_end. Reduce --t-end or use a longer recording."
        )
    return train_trace[:n_samples_needed]


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

    elif mode == "seeg":
        trace = _seeg_trace(cfg, T_main)  # (T_main, n_chan)
        if trace.shape[1] != input_size:
            raise RuntimeError(
                f"SEEG channel count {trace.shape[1]} != input_size "
                f"{input_size}. Check task config."
            )
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
            "Model has no TrainableIC (trainable_ic=False). Using zeros — "
            "consider running with a more negative --t-start to allow longer "
            "zero-input warm-up.",
            stacklevel=2,
        )
        state = torch.zeros(K, 1, cell.state_size, device=device)

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
):
    """Build the 5-panel figure for variant k and save to out_path."""
    fig, axes = plt.subplots(5, 1, figsize=(16, 12), sharex=True)

    u = bufs["u"][:, k, :]              # (T_plot, N)
    x = bufs["x"][:, k, :]
    br = bufs["br"][:, k, :]
    b_full = bufs["b_full"][:, k, :]

    plot_lines(axes[0], t_seconds, u, n_E, "u(t)\n(post-W_in)")
    plot_lines(axes[1], t_seconds, x, n_E, "x(t)\ndendritic")
    plot_lines(axes[2], t_seconds, br, n_E, "b·r(t)\nsynaptic out", ylim_range=(-0.05, 1.05))

    # a_sum: only over the variant's actual n_a_E timescales
    n_a_E_k = cfg_k.n_a_E
    n_a_I_k = cfg_k.n_a_I
    if n_a_E_k > 0 or n_a_I_k > 0:
        a_E = bufs["a_E"][:, k, :, :n_a_E_k] if n_a_E_k > 0 else None
        a_I = bufs["a_I"][:, k, :, :n_a_I_k] if n_a_I_k > 0 else None
        a_sum_E = a_E.sum(axis=-1) if a_E is not None else np.zeros((len(t_seconds), n_E))
        a_sum_I = a_I.sum(axis=-1) if a_I is not None else np.zeros((len(t_seconds), n_I))
        a_sum = np.concatenate([a_sum_E, a_sum_I], axis=-1)
        plot_lines(axes[3], t_seconds, a_sum, n_E, "Σ a(t)\nSFA")
    else:
        axes[3].set_ylabel("Σ a(t)\n(inactive)", fontsize=10)
        axes[3].text(0.5, 0.5, "no SFA in this variant",
                     transform=axes[3].transAxes, ha="center", va="center",
                     color="0.5")

    plot_lines(axes[4], t_seconds, b_full, n_E, "b(t)\nSTD",
               ylim_range=(-0.05, 1.1))

    # Mark the t=0 boundary (end of zero-input warm-up) when present
    if t_warm_seconds > 0:
        for ax in axes:
            ax.axvline(0.0, color="0.4", linestyle=":", linewidth=0.8)

    axes[-1].set_xlabel("time (s)")
    fig.suptitle(f"{variant_name} — mode={mode}", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────

def plot_replay(
    ckpt_path: Path,
    out_dir: Path,
    mode: str,
    t_range: tuple[float, float] = (-15.0, 30.0),
    plot_fs: float = 25.0,
    device: str = "cpu",
):
    """Replay one mode and write per-variant figures into <out_dir>/<variant>/."""
    ckpt_path = Path(ckpt_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[replay] loading {ckpt_path}")
    ckpt = load_checkpoint(str(ckpt_path), device=device)
    cfg = OmegaConf.create(ckpt["config"])
    ablation_names = ckpt.get("ablation_names") or []
    if not ablation_names:
        raise RuntimeError(
            f"{ckpt_path} has no ablation_names — not a batched-ablation run."
        )

    model = build_batched_model(cfg, ablation_names)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()

    cell = model.cell
    h = float(cell.h)
    input_size = int(cfg.task.input_size)

    plot_step = max(1, int(round((1.0 / plot_fs) / h)))
    print(f"[replay] mode={mode} t_range={t_range} h={h} plot_step={plot_step} "
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

    for k, name in enumerate(ablation_names):
        variant_dir = out_dir / name
        variant_dir.mkdir(parents=True, exist_ok=True)
        out_path = variant_dir / f"timeseries_{mode}.png"
        render_variant(
            bufs, t_seconds, k, name, cell.configs[k],
            n_E, n_I, out_path, mode, t_warm_seconds,
        )
        print(f"[replay]   wrote {out_path.relative_to(out_dir.parent)}")


# ─────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ckpt_path", help="Path to last.pt (or any *.pt with config + ablation_names)")
    p.add_argument("out_dir", help="Per-variant subdirs are created under this path")
    p.add_argument("--mode", default="all",
                   choices=["no_input", "step", "seeg", "all"],
                   help="Input mode (default: all → runs all three)")
    p.add_argument("--t-start", type=float, default=-15.0,
                   help="Start time in seconds (negative = zero-input warm-up; default: -15)")
    p.add_argument("--t-end", type=float, default=30.0,
                   help="End time in seconds (default: 30)")
    p.add_argument("--plot-fs", type=float, default=25.0,
                   help="Plot decimation rate in Hz (default: 25)")
    p.add_argument("--device", default="cpu", help="torch device (default: cpu)")
    return p.parse_args()


def main():
    args = parse_args()
    modes = ["no_input", "step", "seeg"] if args.mode == "all" else [args.mode]
    for m in modes:
        plot_replay(
            ckpt_path=Path(args.ckpt_path),
            out_dir=Path(args.out_dir),
            mode=m,
            t_range=(args.t_start, args.t_end),
            plot_fs=args.plot_fs,
            device=args.device,
        )


if __name__ == "__main__":
    main()
