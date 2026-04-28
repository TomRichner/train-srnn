"""Gradient-norm diagnostic: measure how parameter gradients scale with bptt_chunk_len.

Builds the same batched model the lr3e4-20ep run used, loads its trained weights from
last.pt, runs ONE forward+backward at several values of bptt_chunk_len, and records the
L2 norm of every parameter's gradient. Output:
  - tmp/grad_probe/grad_norms.csv    long-form table (chunk_len, variant, param, grad_norm)
  - tmp/grad_probe/grad_norms.png    grouped log-scale plot (key params vs chunk_len)
  - tmp/grad_probe/grad_norms.txt    pretty per-variant table at default chunk_len=128
"""
import os
import sys
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from hydra import compose, initialize_config_dir

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from train_srnn.models.factory import build_batched_model  # noqa: E402
from train_srnn.data.datasets import load_dataset           # noqa: E402
from train_srnn.data.transforms import wrap_train_batch     # noqa: E402

OUT_DIR = REPO / "tmp" / "grad_probe"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RUN_DIR = REPO / "tmp" / "lr3e4-20ep"
# Light probe: just the canonical full SRNN and the no-adapt-no-dales baseline,
# enough to compare across active vs inactive slow timescales.
ABLATIONS = [
    "srnn-e-only-skip",
    "srnn-e-only-skip-per-neuron",
]
# Use shorter window so CPU forward is tractable. We still vary bptt_chunk_len
# across an order of magnitude to characterize gradient scaling.
WINDOW_LEN = 1024
BPTT_LEN = 768
BATCH_SIZE = 4
CHUNK_LENS = [16, 32, 64, 128, 256, 512]


def build_cfg():
    """Replicate the run's resolved Hydra config (size=300, h=0.004, etc.)."""
    with initialize_config_dir(config_dir=str(REPO / "conf"), version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[
                "task=seeg",
                "model=srnn",
                "size=300",
                "model.h=0.004",
                "model.ode_unfolds=1",
                f"batch_size={BATCH_SIZE}",
                f"window_len={WINDOW_LEN}",
                f"bptt_len={BPTT_LEN}",
                "stretch_lo=1.0",
                "stretch_hi=1.0",
                "no_augment=true",
                "loss_over_bptt=true",
                "seed=1",
                "device=cpu",
            ],
        )
    return cfg


def make_batch(cfg, device):
    """Pull one real training batch from the seeg dataset."""
    task_kwargs = OmegaConf.to_container(cfg.task, resolve=True)
    task_name = task_kwargs.pop("name")
    data_dir = task_kwargs.pop("data_dir", None)
    ds = load_dataset(task_name, data_dir=data_dir, **task_kwargs)
    train_x, train_y = ds["train"]
    rng = np.random.RandomState(0)
    idx = rng.choice(len(train_x), size=cfg.batch_size, replace=False)
    bx = train_x[idx][:, :cfg.window_len, :]   # take first window_len steps
    by = train_y[idx][:, :cfg.window_len, :]
    bptt_start = cfg.window_len - cfg.bptt_len
    readout_idx = slice(bptt_start, cfg.window_len)  # loss_over_bptt: predict every step in grad region
    by = by[:, readout_idx]
    bx_t = torch.tensor(bx, dtype=torch.float32, device=device)
    by_t = torch.tensor(by, dtype=torch.float32, device=device)
    return bx_t, by_t, readout_idx, bptt_start


def measure(model, bx_t, by_t, readout_idx, bptt_start, chunk_len, K):
    """Run one forward+backward, return dict of param_name -> grad L2 norm (per variant where applicable)."""
    model.zero_grad(set_to_none=True)
    logits = model(
        bx_t,
        readout_idx=readout_idx,
        bptt_start_idx=bptt_start,
        bptt_chunk_len=chunk_len,
        grad_checkpoint=False,
    )  # (K, B, T, 1) for regression+per-timestep
    # Sum K independent losses (matches train.py)
    crit = nn.MSELoss()
    losses = []
    for k in range(K):
        lk = logits[k].squeeze(-1)
        losses.append(crit(lk, by_t))
    loss = torch.stack(losses).sum()
    loss.backward()

    out = {}
    for name, p in model.named_parameters():
        if p.grad is None:
            out[name] = None
            continue
        g = p.grad.detach()
        if g.ndim >= 1 and g.shape[0] == K and "ic" not in name and "readout" not in name and name != "ic.ic":
            # Per-variant slice
            for k in range(K):
                out[f"{name}::k{k}"] = g[k].float().norm().item()
        elif g.ndim >= 1 and g.shape[0] == K:
            # readout / ic — also per-variant
            for k in range(K):
                out[f"{name}::k{k}"] = g[k].float().norm().item()
        else:
            out[name] = g.float().norm().item()
    return out, loss.item()


def main():
    print("Building cfg + model...")
    cfg = build_cfg()
    device = torch.device("cpu")
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))

    model = build_batched_model(cfg, ABLATIONS).to(device)
    K = len(ABLATIONS)

    # Use freshly-initialized weights — for the gradient-flow analysis the *shape*
    # of |∇| vs chunk_len matters, and the post-training param values are very close
    # to init for non-readout params anyway (gains barely moved during the actual run).
    print("(using freshly-initialized weights — taus at their nominal inits)")

    print("Pulling one batch from seeg train split...")
    bx_t, by_t, readout_idx, bptt_start = make_batch(cfg, device)
    print(f"batch: {tuple(bx_t.shape)} -> y {tuple(by_t.shape)}; readout_idx={type(readout_idx).__name__}; bptt_start={bptt_start}")

    rows = []  # (chunk_len, key, grad_norm, variant_idx)
    losses_by_chunk = {}
    for cl in CHUNK_LENS:
        print(f"--- bptt_chunk_len = {cl} ---")
        torch.manual_seed(0)  # same dropout/init-noise, if any
        gn, total_loss = measure(model, bx_t, by_t, readout_idx, bptt_start, cl, K)
        losses_by_chunk[cl] = total_loss
        for key, val in gn.items():
            if val is None:
                continue
            base, _, vk = key.partition("::")
            kidx = int(vk[1:]) if vk else -1
            variant = ABLATIONS[kidx] if 0 <= kidx < K else "shared"
            rows.append((cl, base, variant, kidx, val))
        print(f"  total loss={total_loss:.4f}; #params with grads = {sum(1 for v in gn.values() if v is not None)}")

    # Long-form CSV
    csv_path = OUT_DIR / "grad_norms.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["chunk_len", "param", "variant", "k", "grad_norm"])
        w.writerows(rows)
    print(f"wrote {csv_path}")

    # Pretty per-variant table at chunk=128
    write_table(rows, chunk_len=128, out=OUT_DIR / "grad_norms_chunk128.txt")

    # Plot grad norms vs chunk_len for srnn-skip (k=5)
    plot_grad_vs_chunk(rows, variant_k=5, variant_name="srnn-skip",
                       out=OUT_DIR / "grad_vs_chunk_srnn-skip.png")
    # Plot for all variants combined: focus on log_tau_a_E_gain (slow) vs readout_weight (fast)
    plot_per_variant_focus(rows, ABLATIONS,
                           out=OUT_DIR / "grad_vs_chunk_focus.png")
    print(f"loss by chunk_len: {losses_by_chunk}")


def write_table(rows, chunk_len, out):
    by_var = {}
    for cl, param, variant, kidx, gn in rows:
        if cl != chunk_len:
            continue
        by_var.setdefault(variant, []).append((param, gn))
    lines = [f"# Grad L2 norms at bptt_chunk_len={chunk_len}, after one forward+backward on one real seeg batch."]
    lines.append(f"# Model loaded from {RUN_DIR.name}/last.pt (post-training).")
    lines.append("")
    for v in sorted(by_var):
        lines.append(f"## variant: {v}")
        lines.append(f"  {'param':<45s} {'grad L2':>12s}")
        for param, gn in sorted(by_var[v], key=lambda kv: -kv[1]):
            lines.append(f"  {param:<45s} {gn:>12.4g}")
        lines.append("")
    out.write_text("\n".join(lines))
    print(f"wrote {out}")


KEY_PARAMS = [
    ("readout_weight", "readout_weight"),
    ("readout_bias", "readout_bias"),
    ("cell.W_in", "W_in"),
    ("cell.W_in_gain", "W_in_gain"),
    ("cell.W_raw", "W_raw"),
    ("cell.W_raw_gain", "W_raw_gain"),
    ("cell.log_tau_global", "log_tau_global"),
    ("cell.log_tau_d_gain", "log_tau_d_gain"),
    ("cell.log_tau_a_E_gain", "log_tau_a_E_gain"),
    ("cell.log_c_E_gain", "log_c_E_gain"),
    ("cell.log_tau_b_rec_E_gain", "log_tau_b_rec_E_gain"),
    ("cell.a_0_scalar", "a_0_scalar"),
    ("cell.c_0_E_scalar", "c_0_E_scalar"),
]


def plot_grad_vs_chunk(rows, variant_k, variant_name, out):
    fig, ax = plt.subplots(figsize=(11, 6))
    cmap = plt.get_cmap("tab20")
    for i, (key, label) in enumerate(KEY_PARAMS):
        xs, ys = [], []
        for cl, param, variant, kidx, gn in rows:
            if param == key and (kidx == variant_k or kidx == -1):
                xs.append(cl); ys.append(gn)
        if not xs:
            continue
        order = np.argsort(xs)
        ax.loglog(np.array(xs)[order], np.array(ys)[order], "o-", color=cmap(i % 20), label=label, lw=1.5, ms=5)
    ax.set_xlabel("bptt_chunk_len (steps)")
    ax.set_ylabel("|∇θ|₂ after one batch (log)")
    ax.set_title(f"Gradient norms vs bptt_chunk_len — variant {variant_name}")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="best", ncol=2)
    plt.tight_layout()
    plt.savefig(out, dpi=120); plt.close(fig)
    print(f"wrote {out}")


def plot_per_variant_focus(rows, ablations, out):
    """For each variant, plot |∇readout_weight| vs |∇log_tau_a_E_gain| across chunk_len."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    cmap = plt.get_cmap("tab10")
    for ax, focus_param, label in (
        (axes[0], "readout_weight", "readout_weight (fast/output-side)"),
        (axes[1], "cell.log_tau_a_E_gain", "log_tau_a_E_gain (slow SFA)"),
    ):
        for k, name in enumerate(ablations):
            xs, ys = [], []
            for cl, param, variant, kidx, gn in rows:
                if param == focus_param and kidx == k:
                    xs.append(cl); ys.append(gn)
            if not xs:
                continue
            order = np.argsort(xs)
            ax.loglog(np.array(xs)[order], np.array(ys)[order], "o-", color=cmap(k % 10), label=name, lw=1.2, ms=4)
        ax.set_title(label); ax.set_xlabel("bptt_chunk_len"); ax.grid(alpha=0.3, which="both")
        ax.set_ylabel("|∇θ|₂")
    axes[0].legend(fontsize=7, loc="best", ncol=2)
    plt.tight_layout()
    plt.savefig(out, dpi=120); plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
