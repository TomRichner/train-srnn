"""plot_cheetah_traces.py — raw cheetah rollouts, one figure per 50 s episode.

17 stacked subplots (one per observation channel), shared x-axis, independent
y-axes. Red verticals mark where load_cheetah currently starts each training
window (seq_len=32, stride inc=10), so the 69% overlap is visible.

Channel names / units follow the HalfCheetah-v2 observation spec:
qpos[1:] = channels 0-7, qvel = channels 8-16.

Usage:
    python scripts/plot_cheetah_traces.py                  # all traces
    python scripts/plot_cheetah_traces.py --traces 0 15 30
"""
import argparse
import glob
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DT = 0.05          # HalfCheetah-v2: frame_skip=5 x 0.01 s physics timestep
SEQ_LEN = 32       # load_cheetah
INC = 10           # load_cheetah stride

CHANNELS = [
    ("z of front tip", "m"),
    ("angle front tip", "rad"),
    ("angle back thigh", "rad"),
    ("angle back shin", "rad"),
    ("angle back foot", "rad"),
    ("angle front thigh", "rad"),
    ("angle front shin", "rad"),
    ("angle front foot", "rad"),
    ("vel x front tip", "m/s"),
    ("vel z front tip", "m/s"),
    ("ang vel front tip", "rad/s"),
    ("ang vel back thigh", "rad/s"),
    ("ang vel back shin", "rad/s"),
    ("ang vel back foot", "rad/s"),
    ("ang vel front thigh", "rad/s"),
    ("ang vel front shin", "rad/s"),
    ("ang vel front foot", "rad/s"),
]


def plot_trace(arr: np.ndarray, name: str, out: pathlib.Path,
               seq_len: int, inc: int,
               tmin: float | None = None, tmax: float | None = None) -> int:
    n_frames, n_ch = arr.shape
    t = np.arange(n_frames) * DT
    # window starts, mirroring load_cheetah's range(0, T - seq_len - 1, inc).
    # Computed on the FULL trace so the phase is right, then clipped to view.
    starts = np.arange(0, n_frames - seq_len - 1, inc)

    lo = t[0] if tmin is None else tmin
    hi = t[-1] if tmax is None else tmax
    vis = (t >= lo) & (t <= hi)
    starts_vis = starts[(starts * DT >= lo) & (starts * DT <= hi)]

    fig, axes = plt.subplots(n_ch, 1, figsize=(15, 20), sharex=True)
    for c, ax in enumerate(axes):
        label, unit = CHANNELS[c] if c < len(CHANNELS) else (f"ch{c}", "")
        group_color = "#1f77b4" if c < 8 else "#d62728"
        ax.plot(t[vis], arr[vis, c], color=group_color, lw=0.9)
        # thicker/darker when few enough to resolve individually
        lw, al = (0.8, 0.8) if len(starts_vis) <= 40 else (0.35, 0.35)
        for s in starts_vis:
            ax.axvline(s * DT, color="red", lw=lw, alpha=al, zorder=0)
        ax.set_ylabel(f"ch{c}\n{unit}", fontsize=7, rotation=0,
                      ha="right", va="center", labelpad=22)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.15)
        ax.text(0.004, 0.90, f"{label}  (std={arr[vis, c].std():.3f})",
                transform=ax.transAxes, fontsize=7, va="top",
                bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.2))
    axes[-1].set_xlabel("time (s)   —   dt = 0.05 s (20 Hz)", fontsize=9)
    axes[-1].set_xlim(lo, hi)

    # shade the first visible window on the top axis to show its extent
    if len(starts_vis):
        s0 = starts_vis[0] * DT
        axes[0].axvspan(s0, s0 + seq_len * DT, color="red", alpha=0.18, zorder=0)

    fig.suptitle(
        f"{name} — {int(vis.sum())} frames = {hi - lo:.1f} s shown "
        f"(t = {lo:.1f}–{hi:.1f} s of {n_frames * DT:.0f} s)   |   "
        f"blue = qpos (ch0-7), red = qvel (ch8-16)   |   "
        f"{len(starts_vis)} window starts in view, seq_len={seq_len} ({seq_len * DT:.1f} s), "
        f"stride={inc} ({inc * DT:.1f} s) → {100 * (1 - inc / seq_len):.0f}% overlap",
        fontsize=10, y=0.998)
    fig.tight_layout(rect=(0, 0, 1, 0.992))
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return len(starts_vis)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=pathlib.Path,
                    default=pathlib.Path("train_srnn/data/cheetah"))
    ap.add_argument("--out-dir", type=pathlib.Path,
                    default=pathlib.Path("tmp/cheetah_traces"))
    ap.add_argument("--traces", type=int, nargs="*", default=None,
                    help="trace indices to plot (default: all)")
    ap.add_argument("--seq-len", type=int, default=SEQ_LEN)
    ap.add_argument("--inc", type=int, default=INC)
    ap.add_argument("--tmin", type=float, default=None, help="start of view (s)")
    ap.add_argument("--tmax", type=float, default=None, help="end of view (s)")
    ap.add_argument("--middle", type=float, default=None, metavar="SECONDS",
                    help="show only the middle SECONDS of each trace "
                         "(overrides --tmin/--tmax)")
    args = ap.parse_args()

    files = sorted(glob.glob(str(args.data_dir / "*.npy")))
    if not files:
        raise SystemExit(f"no .npy files under {args.data_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    idxs = args.traces if args.traces is not None else range(len(files))
    for i in idxs:
        f = pathlib.Path(files[i])
        arr = np.load(f).astype(np.float32)
        tmin, tmax = args.tmin, args.tmax
        if args.middle is not None:
            mid = arr.shape[0] * DT / 2.0
            tmin, tmax = mid - args.middle / 2.0, mid + args.middle / 2.0
        out = args.out_dir / f"{f.stem}.png"
        n = plot_trace(arr, f.stem, out, args.seq_len, args.inc, tmin, tmax)
        print(f"  {f.stem}  {arr.shape}  {n} windows  ->  {out}")


if __name__ == "__main__":
    main()
