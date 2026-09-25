"""Synthetic input-output dataset: map band-limited noise to its causal fractional derivative.

    uv run python scripts/make_fractional_dataset.py --out "$SRNN_HOME/data/frac_d05" [--alpha 0.5]

A fractional derivative of order alpha has a constant phase advance of alpha * 90 degrees
at every frequency (gain proportional to f^alpha), and its causal kernel decays as a
power law, which a ladder of exponential timescales approximates. Inputs are ``C``
independent channels of white noise band-passed to ``--band`` Hz; targets are the causal
Grunwald-Letnikov derivative of each input channel at the same time step,

    D^alpha x(t) ~ h^-alpha sum_{k=0}^{L} w_k x(t - k h),  w_0 = 1, w_k = w_{k-1} (1 - (alpha+1)/k),

with a ``--kernel-s`` long kernel; the first kernel length is discarded so every target is
fully formed. Each split is one continuous trace saved like the cheetah100 data
(``obs`` = [inputs, targets], plus a dummy ``act``), so it trains with

    task=cheetah100 task.dataset=<name> task.skip_transient_s=0 task.input_size=C
    task.output_size=C task.input_channels=[0..C-1] task.target_channels=[C..2C-1]
    task.target_shift=0
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.signal import butter, fftconvolve, sosfiltfilt

FS = 100.0
SPLITS = {"train": (1200.0, 9000), "valid": (180.0, 9001), "test": (180.0, 9002)}


def gl_weights(alpha: float, n: int) -> np.ndarray:
    w = np.empty(n)
    w[0] = 1.0
    for k in range(1, n):
        w[k] = w[k - 1] * (1.0 - (alpha + 1.0) / k)
    return w


def make_split(seconds: float, seed: int, channels: int, band: tuple[float, float],
               alpha: float, kernel_s: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    L = int(kernel_s * FS)
    n = int(seconds * FS) + L
    sos = butter(4, band, btype="bandpass", fs=FS, output="sos")
    x = sosfiltfilt(sos, rng.standard_normal((n + 2000, channels)), axis=0)[1000:1000 + n]
    x /= x.std(axis=0, keepdims=True)
    w = gl_weights(alpha, L) * FS ** alpha
    y = np.stack([fftconvolve(x[:, c], w)[:n] for c in range(channels)], axis=1)
    return np.concatenate([x, y], axis=1)[L:].astype(np.float32)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--channels", type=int, default=3)
    p.add_argument("--band", default="0.05,5", help="input band-pass edges in Hz")
    p.add_argument("--kernel-s", type=float, default=30.0)
    args = p.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"{args.out} is not empty")
    args.out.mkdir(parents=True, exist_ok=True)
    band = tuple(float(b) for b in args.band.split(","))
    manifest = {"generator": "scripts/make_fractional_dataset.py", "alpha": args.alpha,
                "channels": args.channels, "band_hz": band, "kernel_s": args.kernel_s,
                "sampling_hz": FS, "layout": "obs = [inputs (C), targets D^alpha inputs (C)]",
                "splits": {}, "sha256": {}}
    for split, (seconds, seed) in SPLITS.items():
        obs = make_split(seconds, seed, args.channels, band, args.alpha, args.kernel_s)
        path = args.out / f"{split}.npz"
        np.savez(path, obs=obs, act=np.zeros((len(obs) - 1, 1), np.float32))
        manifest["splits"][split] = {"seed": seed, "samples": int(len(obs))}
        manifest["sha256"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        print(f"{split}: {obs.shape}")
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
