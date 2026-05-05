"""Offline zero-phase Butterworth bandpass for SEEG .mat traces.

Reads the bespoke v7.3 HDF5 .mat (keys: data_filt (C, N) float64, SR scalar),
applies a 3rd-order Butterworth HPF + LPF via sosfiltfilt (forward+backward,
MATLAB-filtfilt-style), and writes a tagged copy with the same two keys.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import h5py
import numpy as np
from scipy.signal import butter, sosfiltfilt, welch


def _filter_trace(data: np.ndarray, sr: float, hpf: float, lpf: float, order: int) -> np.ndarray:
    nyq = sr / 2.0
    # Why: SOS form is required — at 0.1/(500/2)=4e-4 normalized, ba coeffs are numerically singular.
    sos_hp = butter(order, hpf / nyq, btype="highpass", output="sos")
    sos_lp = butter(order, lpf / nyq, btype="lowpass", output="sos")
    y = sosfiltfilt(sos_hp, data, axis=1)
    y = sosfiltfilt(sos_lp, y, axis=1)
    return y


def _band_rms(x: np.ndarray, sr: float, lo: float, hi: float) -> float:
    f, p = welch(x, fs=sr, nperseg=min(8192, x.shape[-1]), axis=-1)
    mask = (f >= lo) & (f < hi)
    if not mask.any():
        return float("nan")
    return float(np.sqrt(np.mean(p[..., mask].sum(axis=-1) * (f[1] - f[0]))))


def _process_file(in_path: str, out_path: str, hpf: float, lpf: float, order: int) -> None:
    with h5py.File(in_path, "r") as f:
        data = np.asarray(f["data_filt"], dtype=np.float64)
        sr = float(np.asarray(f["SR"]).squeeze())

    print(f"[seeg-filter] in:  {in_path}")
    print(f"  data_filt shape={data.shape} dtype={data.dtype} SR={sr} Hz")

    n_min = 3 * (max(len(butter(order, hpf / (sr / 2), btype="highpass", output="sos")),
                     len(butter(order, lpf / (sr / 2), btype="lowpass", output="sos"))))
    if data.shape[1] < n_min:
        raise ValueError(f"trace too short for sosfiltfilt: {data.shape[1]} < {n_min}")

    pre_lo = _band_rms(data, sr, 0.0, hpf)
    pre_mid = _band_rms(data, sr, hpf, lpf)
    pre_hi = _band_rms(data, sr, lpf, sr / 2)

    y = _filter_trace(data, sr, hpf, lpf, order)

    post_lo = _band_rms(y, sr, 0.0, hpf)
    post_mid = _band_rms(y, sr, hpf, lpf)
    post_hi = _band_rms(y, sr, lpf, sr / 2)

    print(f"  band RMS  pre  -> [DC..{hpf}]={pre_lo:.4g}  [{hpf}..{lpf}]={pre_mid:.4g}  [{lpf}..Nyq]={pre_hi:.4g}")
    print(f"  band RMS  post -> [DC..{hpf}]={post_lo:.4g}  [{hpf}..{lpf}]={post_mid:.4g}  [{lpf}..Nyq]={post_hi:.4g}")
    print(f"  per-ch std    pre={data.std(axis=1).mean():.4g}  post={y.std(axis=1).mean():.4g}")

    tmp = out_path + ".tmp"
    with h5py.File(tmp, "w") as f:
        f.create_dataset("data_filt", data=y.astype(np.float64), dtype="float64")
        f.create_dataset("SR", data=np.array([[sr]], dtype=np.float64), dtype="float64")
    os.replace(tmp, out_path)
    print(f"[seeg-filter] out: {out_path}\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="train_srnn/data/seeg")
    ap.add_argument("--hpf", type=float, default=0.1)
    ap.add_argument("--lpf", type=float, default=50.0)
    ap.add_argument("--order", type=int, default=3)
    ap.add_argument("--tag", default="bp0p1-50")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    pattern = os.path.join(args.in_dir, "seeg_*.mat")
    inputs = sorted(p for p in glob.glob(pattern) if f"_{args.tag}.mat" not in p)
    if not inputs:
        print(f"no input .mat files matched {pattern}", file=sys.stderr)
        return 1

    for in_path in inputs:
        stem, _ = os.path.splitext(in_path)
        out_path = f"{stem}_{args.tag}.mat"
        if os.path.exists(out_path) and not args.overwrite:
            print(f"[seeg-filter] skip (exists): {out_path}")
            continue
        _process_file(in_path, out_path, args.hpf, args.lpf, args.order)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
