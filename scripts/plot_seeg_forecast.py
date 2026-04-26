"""Forward a trained checkpoint on a SEEG test window and plot pred vs actual.

Usage:
    python scripts/plot_seeg_forecast.py --ckpt tmp/last.pt --out tmp/forecast.png

Loads a BatchedSRNNCell checkpoint (config + state dict embedded), reconstructs
the model via factory.build_batched_model, runs one test window through it, and
plots per-channel autoregressive predictions for every ablation variant.
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from train_srnn.data.datasets import load_dataset
from train_srnn.models.factory import build_batched_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", default="tmp/forecast.png")
    p.add_argument("--window", type=int, default=0,
                   help="Index into test set windows.")
    p.add_argument("--channels", type=int, nargs="+",
                   default=[0, 10, 20, 40, 60, 80])
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    print(f"Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    cfg = OmegaConf.create(ckpt["config"])
    ablation_names = ckpt["ablation_names"]
    print(f"  ablations: {ablation_names}")
    print(f"  epoch:     {ckpt['epoch']}")

    model = build_batched_model(cfg, ablation_names).to(args.device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Load SEEG test set with same loader kwargs the trainer used.
    task = cfg.task
    print(f"Loading {task.name} test set...")
    data = load_dataset(
        task.name,
        data_dir=task.data_dir,
        subject_id=task.subject_id,
        block=task.block,
        sleep=task.sleep,
        cond=task.cond,
        decimate=task.decimate,
        seq_len=task.seq_len,
        stride=task.stride,
    )
    te_x, te_y = data["test"]   # (N_windows, T, C) views
    sr = data["meta"]["sr_hz"]
    T = task.seq_len

    win = args.window
    if win >= len(te_x):
        raise SystemExit(f"--window {win} >= test windows {len(te_x)}")
    x = torch.from_numpy(np.ascontiguousarray(te_x[win:win+1])).to(args.device)  # (1, T, C)
    y = np.ascontiguousarray(te_y[win])                                          # (T, C)

    print(f"Forward window {win}: x.shape={tuple(x.shape)}, sr={sr} Hz")
    with torch.no_grad():
        # readout slice spanning the whole window → (K, B=1, T, C)
        logits = model(x, readout_idx=slice(0, T))
    pred = logits.squeeze(1).cpu().numpy()   # (K, T, C)

    K = pred.shape[0]
    chans = [c for c in args.channels if c < y.shape[-1]]
    n_ch = len(chans)
    fig, axes = plt.subplots(n_ch, 1, figsize=(12, 1.6 * n_ch + 1.0),
                             sharex=True, squeeze=False)
    t_axis = np.arange(T) / sr

    for i, c in enumerate(chans):
        ax = axes[i, 0]
        ax.plot(t_axis, y[:, c], color="black", lw=1.2, label="actual (z-scored)")
        for k in range(K):
            ax.plot(t_axis, pred[k, :, c], lw=0.9, alpha=0.85,
                    label=ablation_names[k])
        ax.set_ylabel(f"ch {c}", rotation=0, ha="right", va="center")
        ax.grid(True, alpha=0.25)
        if i == 0:
            ax.legend(loc="upper right", ncol=1 + K, fontsize=8)
    axes[-1, 0].set_xlabel("time (s)")
    fig.suptitle(
        f"SEEG 1-step forecast — window {win}, epoch {ckpt['epoch']} "
        f"({os.path.basename(args.ckpt)})",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"Saved {args.out}")

    # Per-variant per-channel MSE summary
    mse = ((pred - y[None]) ** 2).mean(axis=1)   # (K, C)
    print("\nPer-variant test MSE (mean over channels):")
    for k, name in enumerate(ablation_names):
        print(f"  {name:30s}  mse={mse[k].mean():.4f}  "
              f"min={mse[k].min():.4f} max={mse[k].max():.4f}")

    # Magnitude diagnostics — are predictions actually scaling, or near-flat?
    print("\nSignal vs prediction magnitude (std over time, mean over channels):")
    print(f"  actual                          std={y.std(axis=0).mean():.4f}")
    for k, name in enumerate(ablation_names):
        print(f"  {name:30s}  std={pred[k].std(axis=1).mean():.4f}  "
              f"mean={pred[k].mean():+.4f}")
    # MSE of predicting zero (z-score mean) baseline
    zero_mse = (y ** 2).mean()
    pers_mse = ((y[1:] - y[:-1]) ** 2).mean()
    print(f"\nBaselines:  predict-zero MSE={zero_mse:.4f}   "
          f"1-step persistence MSE={pers_mse:.4f}")


if __name__ == "__main__":
    main()
