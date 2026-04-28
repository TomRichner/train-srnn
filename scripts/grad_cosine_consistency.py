"""Gradient direction-consistency probe.

For variant srnn-e-only-skip, measure for each parameter:
  - mean ‖∇θ‖ across N batches
  - mean pairwise cosine similarity of ∇θ across batches
  - "concentration" r = ‖mean(∇θ)‖ / mean(‖∇θ‖)  (1 = perfectly aligned, ~0 = noise)

High mean-cosine → consistent direction → Adam can accumulate signal → param will move.
Low mean-cosine → direction noise → Adam normalizes the magnitude but the running mean
  stays near zero → param drifts much more slowly than the per-step magnitude suggests.

Output:
  tmp/grad_probe/cosine_consistency_srnn-e-only-skip.csv
  tmp/grad_probe/cosine_consistency_srnn-e-only-skip.txt
  tmp/grad_probe/cosine_consistency_srnn-e-only-skip.png
"""
import os
import sys
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from hydra import compose, initialize_config_dir

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from train_srnn.models.factory import build_batched_model  # noqa
from train_srnn.data.datasets import load_dataset           # noqa

OUT_DIR = REPO / "tmp" / "grad_probe"
OUT_DIR.mkdir(parents=True, exist_ok=True)
VARIANT = "srnn-e-only-skip"
N_BATCHES = 16
CHUNK_LEN = 128
WINDOW_LEN = 1024
BPTT_LEN = 768
BATCH_SIZE = 4


def build_cfg():
    with initialize_config_dir(config_dir=str(REPO / "conf"), version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[
                "task=seeg", "model=srnn",
                "size=300", "model.h=0.004", "model.ode_unfolds=1",
                f"batch_size={BATCH_SIZE}",
                f"window_len={WINDOW_LEN}", f"bptt_len={BPTT_LEN}",
                "stretch_lo=1.0", "stretch_hi=1.0", "no_augment=true", "loss_over_bptt=true",
                "seed=1", "device=cpu",
            ],
        )
    return cfg


def get_data(cfg):
    task_kwargs = OmegaConf.to_container(cfg.task, resolve=True)
    name = task_kwargs.pop("name"); ddir = task_kwargs.pop("data_dir", None)
    ds = load_dataset(name, data_dir=ddir, **task_kwargs)
    return ds["train"]


def make_batch(train_x, train_y, rng, cfg):
    """Random contiguous window from a random subset of recordings."""
    N = len(train_x)
    idx = rng.choice(N, size=cfg.batch_size, replace=False)
    T_full = train_x.shape[1]
    start = rng.randint(0, T_full - cfg.window_len + 1)
    bx = train_x[idx][:, start:start + cfg.window_len, :]
    by = train_y[idx][:, start:start + cfg.window_len, :]
    bptt_start = cfg.window_len - cfg.bptt_len
    readout_idx = slice(bptt_start, cfg.window_len)
    by = by[:, readout_idx]
    return (
        torch.tensor(bx, dtype=torch.float32),
        torch.tensor(by, dtype=torch.float32),
        readout_idx, bptt_start,
    )


def grads_one_batch(model, bx, by, readout_idx, bptt_start, K):
    model.zero_grad(set_to_none=True)
    logits = model(bx, readout_idx=readout_idx, bptt_start_idx=bptt_start,
                   bptt_chunk_len=CHUNK_LEN, grad_checkpoint=False)
    crit = nn.MSELoss()
    losses = [crit(logits[k].squeeze(-1), by) for k in range(K)]
    torch.stack(losses).sum().backward()
    out = {}
    for name, p in model.named_parameters():
        if p.grad is None: continue
        g = p.grad.detach().float()
        # Slice to variant 0 if leading dim is K
        if g.ndim >= 1 and g.shape[0] == K:
            g = g[0]
        out[name] = g.flatten().cpu().numpy().astype(np.float64)
    return out


def main():
    print("Building cfg + model (K=1)...")
    cfg = build_cfg()
    model = build_batched_model(cfg, [VARIANT]).to("cpu")
    K = 1

    print("Loading dataset...")
    train_x, train_y = get_data(cfg)

    rng = np.random.RandomState(42)
    print(f"Computing gradients on {N_BATCHES} batches at cl={CHUNK_LEN}...")
    all_grads = []  # list of dict: name -> 1D array
    for b in range(N_BATCHES):
        bx, by, ridx, bs = make_batch(train_x, train_y, rng, cfg)
        gd = grads_one_batch(model, bx, by, ridx, bs, K)
        all_grads.append(gd)
        if (b + 1) % 4 == 0:
            print(f"  batch {b+1}/{N_BATCHES}")

    # Aggregate per-param statistics
    names = sorted(all_grads[0].keys())
    stats = []
    for name in names:
        G = np.stack([g[name] for g in all_grads], axis=0)   # (N_batches, P)
        norms = np.linalg.norm(G, axis=1)                    # (N_batches,)
        mean_norm = float(norms.mean())
        if mean_norm == 0:
            stats.append((name, 0.0, 0.0, 0.0, 0.0))
            continue
        # Concentration: ‖mean(g)‖ / mean(‖g‖)  ∈ [0, 1] for IID centered, can be larger if signal
        mean_g = G.mean(axis=0)
        concentration = float(np.linalg.norm(mean_g) / mean_norm)
        # Mean pairwise cosine (off-diag)
        Gn = G / (norms[:, None] + 1e-30)
        sim = Gn @ Gn.T   # (Nb, Nb)
        N = sim.shape[0]
        off = (sim.sum() - np.trace(sim)) / (N * (N - 1))
        # Std of grad norm (relative noise indicator)
        std_norm = float(norms.std())
        stats.append((name, mean_norm, std_norm, float(off), concentration))

    # Sort by mean grad norm descending
    stats.sort(key=lambda r: -r[1])

    # CSV
    csv_path = OUT_DIR / f"cosine_consistency_{VARIANT}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["param", "mean_grad_norm", "std_grad_norm", "mean_pairwise_cosine", "concentration"])
        w.writerows(stats)
    print(f"wrote {csv_path}")

    # Pretty table
    txt_path = OUT_DIR / f"cosine_consistency_{VARIANT}.txt"
    lines = [
        f"# Gradient direction consistency for variant: {VARIANT}",
        f"# {N_BATCHES} batches, bptt_chunk_len={CHUNK_LEN}, batch_size={BATCH_SIZE}, "
        f"window_len={WINDOW_LEN}, bptt_len={BPTT_LEN}, fresh init.",
        "#",
        "# mean_pairwise_cosine: average of cos(g_i, g_j) for i≠j; close to 1 = consistent direction,",
        "#                       close to 0 = direction is noise (Adam can't accumulate momentum).",
        "# concentration:        ‖<g>‖ / <‖g‖>; 1 = all batches point the same way, 0 = pure noise.",
        "",
    ]
    header = (f"{'param':<32s}{'<|g|>':>12s}{'std|g|':>12s}{'<cos>':>10s}{'concentration':>16s}")
    lines.append(header); lines.append("-" * len(header))
    for name, mn, sn, mc, conc in stats:
        if mn == 0:
            continue
        lines.append(f"{name.replace('cell.',''):<32s}{mn:>12.3e}{sn:>12.3e}{mc:>10.3f}{conc:>16.3f}")
    txt_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {txt_path}")
    print()
    print("\n".join(lines))

    # Plot: mean grad norm vs mean cosine, color by concentration
    fig, ax = plt.subplots(figsize=(10, 7))
    short = [n.replace("cell.", "") for n, *_ in stats if _[0] > 0]
    norms = [r[1] for r in stats if r[1] > 0]
    cosines = [r[3] for r in stats if r[1] > 0]
    sc = ax.scatter(norms, cosines, c=[r[4] for r in stats if r[1] > 0],
                    cmap="viridis", s=70, edgecolor="k", linewidths=0.5, vmin=0, vmax=1)
    for x, y, lbl in zip(norms, cosines, short):
        ax.annotate(lbl, (x, y), fontsize=7, alpha=0.85,
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xscale("log")
    ax.axhline(0, color="k", lw=0.5)
    ax.axhline(1.0 / np.sqrt(N_BATCHES - 1), color="r", ls="--", lw=0.7,
               label=f"noise-floor cosine ≈ ±1/√(N−1) = {1/np.sqrt(N_BATCHES-1):.2f}")
    ax.set_xlabel("mean ‖∇θ‖₂ across batches (log)")
    ax.set_ylabel("mean pairwise cosine across batches")
    ax.set_title(f"Gradient direction consistency — {VARIANT}\n"
                 f"({N_BATCHES} batches, cl={CHUNK_LEN})")
    plt.colorbar(sc, ax=ax, label="concentration ‖<g>‖ / <‖g‖>")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3, which="both")
    plt.tight_layout()
    out = OUT_DIR / f"cosine_consistency_{VARIANT}.png"
    plt.savefig(out, dpi=120); plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
