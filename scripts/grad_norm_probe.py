"""Gradient-norm diagnostic: how parameter gradients scale with bptt_chunk_len.

Builds a small batch of variants at their initial weights, runs one
forward+backward on one real cheetah100 batch at several bptt_chunk_len
values, and records the L2 norm of every parameter's gradient. Outputs go
to $SRNN_CACHE_DIR/grad_probe/: grad_norms.csv, grad_norms_chunk128.txt,
grad_vs_chunk_<variant>.png, grad_vs_chunk_focus.png.
"""
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from _runs import cache_dir  # noqa: E402
from train_srnn.config import compose_config  # noqa: E402
from train_srnn.data import build_task  # noqa: E402
from train_srnn.models.factory import build_model  # noqa: E402

OUT_DIR = cache_dir() / "grad_probe"
OUT_DIR.mkdir(parents=True, exist_ok=True)
ABLATIONS = ["srnn-e-only-skip", "srnn-e-only-skip-per-neuron"]
WINDOW_LEN = 1024
BPTT_LEN = 768
BATCH_SIZE = 4
CHUNK_LENS = [16, 32, 64, 128, 256, 512]


def build_cfg():
    return compose_config([
        "task=cheetah100", "model=srnn", "model.num_units=300", "seed=1", "device=cpu",
        f"task.batch_size={BATCH_SIZE}", f"task.window_len={WINDOW_LEN}", f"task.bptt_len={BPTT_LEN}",
        "task.seq_len=1024",
        "model.variants=[" + ",".join(ABLATIONS) + "]",
    ])


def make_batch(cfg, device):
    """One real training batch: the first window_len steps of a few training windows."""
    task = build_task(cfg)
    train_x, train_y = task.load(Path(cfg.task.data_dir)).train
    rng = np.random.RandomState(0)
    idx = rng.choice(len(train_x), size=BATCH_SIZE, replace=False)
    bx = train_x[idx][:, :WINDOW_LEN, :]
    by = train_y[idx][:, :WINDOW_LEN, :]
    bptt_start = WINDOW_LEN - BPTT_LEN
    readout_idx = slice(bptt_start, WINDOW_LEN)
    by = by[:, readout_idx]
    return (torch.tensor(bx, dtype=torch.float32, device=device),
            torch.tensor(by, dtype=torch.float32, device=device), readout_idx, bptt_start)


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

    model = build_model(cfg).to(device)
    K = len(ABLATIONS)

    print("Pulling one batch from the cheetah100 train split...")
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

    plot_grad_vs_chunk(rows, variant_k=0, variant_name=ABLATIONS[0],
                       out=OUT_DIR / f"grad_vs_chunk_{ABLATIONS[0]}.png")
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
    lines = [f"# Grad L2 norms at bptt_chunk_len={chunk_len}, one forward+backward on one cheetah100 batch, initial weights."]
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
    ("cell.isp_tau_global", "isp_tau_global"),
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
