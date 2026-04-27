"""Analysis of the overnight9h SEEG run (8 batched skip variants, 90 epochs).

Reads CSVs and per-epoch checkpoints from tmp/overnight9h/ and writes:
  - tmp/overnight9h/curves.png         loss/metric curves per variant
  - tmp/overnight9h/weight_evolution.png  param-norm trajectories from epoch_*.pt
"""
import csv
import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

RUN_DIR = Path(__file__).resolve().parents[2] / "tmp" / "overnight9h"


def load_history(path):
    rows = list(csv.DictReader(open(path)))
    by_variant = defaultdict(list)
    for r in rows:
        by_variant[r["variant"]].append(r)
    return by_variant


def plot_curves():
    th = load_history(RUN_DIR / "training_history.csv")
    test = list(csv.DictReader(open(RUN_DIR / "test_history.csv")))
    variants = sorted(th.keys())
    cmap = plt.get_cmap("tab10")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for i, v in enumerate(variants):
        rows = th[v]
        ep = [int(r["epoch"]) + 1 for r in rows]
        tr = [float(r["train_loss"]) for r in rows]
        va = [float(r["valid_loss"]) for r in rows]
        trm = [float(r["train_metric"]) for r in rows]
        vam = [float(r["valid_metric"]) for r in rows]
        c = cmap(i % 10)
        axes[0, 0].plot(ep, tr, color=c, label=v, lw=1)
        axes[0, 1].plot(ep, va, color=c, label=v, lw=1)
        axes[1, 0].plot(ep, trm, color=c, label=v, lw=1)
        axes[1, 1].plot(ep, vam, color=c, label=v, lw=1)

    axes[0, 0].set_title("train_loss");  axes[0, 1].set_title("valid_loss")
    axes[1, 0].set_title("train_metric"); axes[1, 1].set_title("valid_metric")
    for ax in axes.flat:
        ax.set_xlabel("epoch"); ax.grid(alpha=0.3)
    axes[0, 1].legend(fontsize=7, loc="upper right")

    # final test losses as text
    final = {r["variant"]: float(r["test_loss"]) for r in test if r["tag"] == "last"}
    init = {r["variant"]: float(r["test_loss"]) for r in test if r["tag"] == "init"}
    txt = "Final test_loss (init → last):\n" + "\n".join(
        f"  {v}: {init.get(v, float('nan')):.5f} → {final.get(v, float('nan')):.5f}"
        for v in variants
    )
    fig.text(0.01, 0.01, txt, family="monospace", fontsize=8, va="bottom")
    plt.tight_layout(rect=[0, 0.18, 1, 1])
    out = RUN_DIR / "curves.png"
    plt.savefig(out, dpi=120)
    print(f"wrote {out}")
    for ax in axes.flat:
        ax.set_yscale("log")
    out2 = RUN_DIR / "semilogy_curves.png"
    plt.savefig(out2, dpi=120)
    print(f"wrote {out2}")
    for ax in axes.flat:
        ax.set_xscale("log")
        ax.set_xlabel("epoch + 1 (log)")
    out3 = RUN_DIR / "log_log_curves.png"
    plt.savefig(out3, dpi=120)
    print(f"wrote {out3}")


def plot_semilogy_direct():
    """Same panel as plot_curves but using ax.semilogy(x, y) directly."""
    th = load_history(RUN_DIR / "training_history.csv")
    variants = sorted(th.keys())
    cmap = plt.get_cmap("tab10")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for i, v in enumerate(variants):
        rows = th[v]
        ep = [int(r["epoch"]) + 1 for r in rows]
        c = cmap(i % 10)
        axes[0, 0].semilogy(ep, [float(r["train_loss"]) for r in rows], color=c, label=v, lw=1)
        axes[0, 1].semilogy(ep, [float(r["valid_loss"]) for r in rows], color=c, label=v, lw=1)
        axes[1, 0].semilogy(ep, [float(r["train_metric"]) for r in rows], color=c, label=v, lw=1)
        axes[1, 1].semilogy(ep, [float(r["valid_metric"]) for r in rows], color=c, label=v, lw=1)

    axes[0, 0].set_title("train_loss");  axes[0, 1].set_title("valid_loss")
    axes[1, 0].set_title("train_metric"); axes[1, 1].set_title("valid_metric")
    for ax in axes.flat:
        ax.set_xlabel("epoch + 1"); ax.grid(alpha=0.3, which="both")
    axes[0, 1].legend(fontsize=7, loc="upper right")
    plt.tight_layout()
    out = RUN_DIR / "semilogy_direct_curves.png"
    plt.savefig(out, dpi=120)
    print(f"wrote {out}")


def plot_semilogy_top6():
    """Semilogy panel excluding the two worst variants."""
    excluded = {"srnn-no-adapt-no-dales-skip", "srnn-sfa-e-only-skip"}
    th = load_history(RUN_DIR / "training_history.csv")
    variants = sorted(v for v in th if v not in excluded)
    cmap = plt.get_cmap("tab10")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for i, v in enumerate(variants):
        rows = th[v]
        ep = [int(r["epoch"]) + 1 for r in rows]
        c = cmap(i % 10)
        axes[0, 0].semilogy(ep, [float(r["train_loss"]) for r in rows], color=c, label=v, lw=1)
        axes[0, 1].semilogy(ep, [float(r["valid_loss"]) for r in rows], color=c, label=v, lw=1)
        axes[1, 0].semilogy(ep, [float(r["train_metric"]) for r in rows], color=c, label=v, lw=1)
        axes[1, 1].semilogy(ep, [float(r["valid_metric"]) for r in rows], color=c, label=v, lw=1)

    axes[0, 0].set_title("train_loss");  axes[0, 1].set_title("valid_loss")
    axes[1, 0].set_title("train_metric"); axes[1, 1].set_title("valid_metric")
    for ax in axes.flat:
        ax.set_xlabel("epoch + 1"); ax.grid(alpha=0.3, which="both")
    axes[0, 1].legend(fontsize=7, loc="upper right")
    fig.suptitle(f"semilogy curves (excluding: {', '.join(sorted(excluded))})", fontsize=10)
    plt.tight_layout()
    out = RUN_DIR / "semilogy_top6_curves.png"
    plt.savefig(out, dpi=120)
    print(f"wrote {out}")


def plot_weight_evolution():
    """Track L2 norm of each parameter group across epochs."""
    pts = sorted(RUN_DIR.glob("epoch_*.pt"))
    if not pts:
        print("no epoch_*.pt found"); return
    epochs, snapshots = [], []
    for p in pts:
        epochs.append(int(p.stem.split("_")[1]))
        sd = torch.load(p, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "model_state_dict" in sd:
            sd = sd["model_state_dict"]
        snapshots.append(sd)

    # group params by trailing key, only those that vary across snapshots
    keys = list(snapshots[0].keys())
    norms = {}
    for k in keys:
        try:
            ts = [s[k].float() for s in snapshots]
        except (KeyError, AttributeError):
            continue
        if ts[0].ndim == 0:
            continue
        n = np.array([t.norm().item() for t in ts])
        if np.allclose(n, n[0]):
            continue
        norms[k] = n

    if not norms:
        print("no varying params"); return

    # show top 16 by relative change
    rel = sorted(norms.items(), key=lambda kv: abs(kv[1][-1] - kv[1][0]) / (abs(kv[1][0]) + 1e-9), reverse=True)[:16]
    fig, ax = plt.subplots(figsize=(13, 8))
    for k, n in rel:
        ax.plot(epochs, n, marker="o", ms=3, lw=1, label=k)
    ax.set_xlabel("epoch"); ax.set_ylabel("|param| (L2)")
    ax.set_title(f"Top-16 most-changed param L2 norms (overnight9h, {len(snapshots)} snapshots)")
    ax.legend(fontsize=7, loc="best", ncol=2)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = RUN_DIR / "weight_evolution.png"
    plt.savefig(out, dpi=120)
    print(f"wrote {out}")

    # also report relative changes as text
    print("\nTop-16 most-changed params (final / initial L2 norm):")
    for k, n in rel:
        print(f"  {k:60s}  {n[0]:.4g} -> {n[-1]:.4g}  ({100*(n[-1]-n[0])/(n[0]+1e-9):+.2f}%)")


if __name__ == "__main__":
    plot_curves()
    plot_semilogy_direct()
    plot_weight_evolution()
