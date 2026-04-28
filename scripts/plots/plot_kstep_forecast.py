"""1/2/3-step-ahead SEEG forecast for the 8 overnight9h variants.

Loads a BatchedSRNNCell checkpoint, warms up on 15 s of test SEEG, then
plots 1/2/3-step autoregressive predictions versus actual for 6 chosen
channels and computes Pearson correlation on the forecast region.

The k-step trajectory is built by chained teacher-forcing: a single
forward pass per k, where x_in_k[t] = pred_{k-1}[t-1] for t >= warmup.
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from train_srnn.data.datasets import load_dataset
from train_srnn.models.factory import build_batched_model


def stitch_windows(te_x, te_y, stride, seq_len, n_windows=2):
    """Concatenate `n_windows` non-overlapping test windows into one trace.

    Test windows are sampled with `stride`; windows at indices
    0, seq_len/stride, 2*seq_len/stride, ... are non-overlapping when
    seq_len is a multiple of stride.
    """
    step = seq_len // stride  # window index step for non-overlap
    if step * stride != seq_len:
        raise ValueError(f"seq_len {seq_len} not a multiple of stride {stride}")
    needed = (n_windows - 1) * step
    if needed >= len(te_x):
        raise SystemExit(f"need test window index {needed}, only have {len(te_x)}")
    xs = np.concatenate([te_x[i * step] for i in range(n_windows)], axis=0)
    ys = np.concatenate([te_y[i * step] for i in range(n_windows)], axis=0)
    return xs, ys  # each (n_windows*seq_len, C)


def forward_full(model, x_np, device):
    T = x_np.shape[0]
    x = torch.from_numpy(np.ascontiguousarray(x_np[None])).to(device)  # (1, T, C)
    with torch.no_grad():
        logits = model(x, readout_idx=slice(0, T))
    return logits.squeeze(1).cpu().numpy()  # (K, T, C)


def pearson_r(a, b):
    """Per-channel Pearson r between (T,C) arrays."""
    a = a - a.mean(axis=0, keepdims=True)
    b = b - b.mean(axis=0, keepdims=True)
    num = (a * b).sum(axis=0)
    den = np.sqrt((a ** 2).sum(axis=0) * (b ** 2).sum(axis=0)) + 1e-12
    return num / den


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="tmp/overnight9h/last.pt")
    p.add_argument("--out-dir", default="tmp/overnight9h")
    p.add_argument("--warmup-sec", type=float, default=15.0)
    p.add_argument("--forecast-sec", type=float, default=15.0)
    p.add_argument("--channels", type=int, nargs="+",
                   default=[0, 14, 28, 42, 56, 70])
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    print(f"Loading {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    cfg = OmegaConf.create(ckpt["config"])
    ablation_names = ckpt["ablation_names"]
    print(f"  variants ({len(ablation_names)}): {ablation_names}")
    print(f"  epoch:    {ckpt['epoch']}")

    model = build_batched_model(cfg, ablation_names).to(args.device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    task = cfg.task
    data = load_dataset(
        task.name, data_dir=task.data_dir,
        subject_id=task.subject_id, block=task.block,
        sleep=task.sleep, cond=task.cond,
        decimate=task.decimate, seq_len=task.seq_len, stride=task.stride,
    )
    te_x, te_y = data["test"]
    sr = data["meta"]["sr_hz"]
    print(f"Test set: {len(te_x)} windows, sr={sr} Hz, seq_len={task.seq_len}")

    warmup = int(round(args.warmup_sec * sr))
    forecast = int(round(args.forecast_sec * sr))
    total = warmup + forecast
    n_windows = int(np.ceil(total / task.seq_len))
    x_full, y_full = stitch_windows(te_x, te_y, task.stride, task.seq_len, n_windows)
    x_full = x_full[:total]
    y_full = y_full[:total]
    print(f"Stitched {n_windows} windows: total={total} ({total/sr:.1f}s), "
          f"warmup={warmup}, forecast={forecast}")

    # k = 1: teacher-forced
    pred_1 = forward_full(model, x_full, args.device)  # (K, T, C)
    K, T, C = pred_1.shape

    # k = 2: x_in[t] = pred_1[t-1] for t >= warmup; teacher-forced before
    # We need a per-variant input → run K separate forwards via a single
    # batched forward of shape (K, T, C). SequenceModel handles
    # batched K via the cell directly when input is (K, B, T, C)?
    # Simpler: loop over K (K=8) — only 2 extra forwards per variant.
    pred_2 = np.zeros_like(pred_1)
    pred_3 = np.zeros_like(pred_1)
    for k_idx in range(K):
        # Build x_in_2: copy real x; replace forecast region with shifted pred_1
        x_in_2 = x_full.copy()
        x_in_2[warmup:] = pred_1[k_idx, warmup - 1:total - 1]
        out2 = forward_full(model, x_in_2, args.device)  # (K, T, C)
        pred_2[k_idx] = out2[k_idx]

        x_in_3 = x_full.copy()
        x_in_3[warmup:] = pred_2[k_idx, warmup - 1:total - 1]
        out3 = forward_full(model, x_in_3, args.device)
        pred_3[k_idx] = out3[k_idx]

    preds = [pred_1, pred_2, pred_3]

    # Correlation on forecast region
    chans = [c for c in args.channels if c < C]
    n_ch = len(chans)
    print("\nMean Pearson r over selected channels (forecast region):")
    print(f"  {'variant':30s}  k=1     k=2     k=3")
    corr = np.zeros((K, len(preds), len(chans)))
    for ki, pred in enumerate(preds):
        for k in range(K):
            r = pearson_r(y_full[warmup:][:, chans], pred[k, warmup:][:, chans])
            corr[k, ki] = r
    for k in range(K):
        means = [corr[k, ki].mean() for ki in range(3)]
        print(f"  {ablation_names[k]:30s}  "
              f"{means[0]:+.3f}  {means[1]:+.3f}  {means[2]:+.3f}")

    os.makedirs(args.out_dir, exist_ok=True)

    # ---- Time-series plot: one figure per variant, 1 s window ----
    t_axis = np.arange(T) / sr
    win_sec = 1.0
    t0 = warmup
    t1 = min(T, warmup + int(round(win_sec * sr)))

    for k in range(K):
        fig, axes = plt.subplots(n_ch, 3, figsize=(13, 1.5 * n_ch + 1.0),
                                 sharex=True, squeeze=False)
        for ki, (k_label, pred) in enumerate(
                zip(["1-step", "2-step", "3-step"], preds)):
            for i, c in enumerate(chans):
                ax = axes[i, ki]
                ax.plot(t_axis[t0:t1], y_full[t0:t1, c],
                        color="black", lw=1.2, label="actual")
                ax.plot(t_axis[t0:t1], pred[k, t0:t1, c],
                        color="C1", lw=1.0, label="pred")
                ax.grid(True, alpha=0.25)
                if ki == 0:
                    ax.set_ylabel(f"ch {c}", rotation=0, ha="right", va="center")
                if i == 0:
                    ax.set_title(f"{k_label}  (r={corr[k, ki, i]:.2f})")
                else:
                    ax.set_title(f"r={corr[k, ki, i]:.2f}", fontsize=8)
            axes[-1, ki].set_xlabel("time (s)")
        axes[0, 0].legend(loc="upper right", fontsize=8)
        fig.suptitle(
            f"{ablation_names[k]} — k-step forecast (epoch {ckpt['epoch']}, "
            f"first {win_sec:.0f}s of forecast region)",
            fontsize=11,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        safe = ablation_names[k].replace("/", "_")
        out1 = os.path.join(args.out_dir, f"forecast_kstep_{safe}.png")
        fig.savefig(out1, dpi=130)
        plt.close(fig)
        print(f"wrote {out1}")

    # ---- Correlation heatmap ----
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4 + 0.2 * K), squeeze=False)
    vmin, vmax = -1.0, 1.0
    for ki, k_label in enumerate(["k=1", "k=2", "k=3"]):
        ax = axes2[0, ki]
        im = ax.imshow(corr[:, ki, :], aspect="auto", cmap="RdBu_r",
                       vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(chans)))
        ax.set_xticklabels([str(c) for c in chans])
        ax.set_yticks(range(K))
        ax.set_yticklabels(ablation_names, fontsize=8)
        ax.set_title(f"{k_label}: Pearson r")
        ax.set_xlabel("channel")
        for i in range(K):
            for j in range(len(chans)):
                ax.text(j, i, f"{corr[i, ki, j]:.2f}", ha="center", va="center",
                        fontsize=7,
                        color="white" if abs(corr[i, ki, j]) > 0.6 else "black")
    fig2.colorbar(im, ax=axes2[0, :].tolist(), shrink=0.8)
    fig2.suptitle(f"Per-channel forecast correlation (forecast region only)",
                  fontsize=11)
    out2 = os.path.join(args.out_dir, "forecast_corr.png")
    fig2.savefig(out2, dpi=130, bbox_inches="tight")
    print(f"wrote {out2}")


if __name__ == "__main__":
    main()
