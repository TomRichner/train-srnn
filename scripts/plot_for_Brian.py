"""plot_for_Brian.py — presentation figure: adaptation vs none, first 50 epochs.

Single-purpose companion to plot_adaptation_comparison.py, kept separate so the
presentation styling (open axes, hand-set ticks) does not complicate the
general-purpose script.

Compares the two no-Dale's, skip variants from a batched-ablation run:
    srnn-no-dales-skip           multi-timescale SFA (n_a=3) + STD (n_b=1)
    srnn-no-adapt-no-dales-skip  same network, adaptation off

Everything else is identical between them — Dale's law off, skip residual on,
same size and seed, and BatchedSRNNCell exports both from one RMTMatrix so
they start from the same recurrent weights.

Usage:
    python scripts/plot_for_Brian.py tmp/ring6-400e
"""
import argparse
import csv
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

VARIANTS = [
    ("srnn-no-adapt-no-dales-skip", "No adaptation", "#B8860B"),
    ("srnn-no-dales-skip", "Multiple timescale adaptation", "#1f77b4"),
]
MAX_EPOCH = 60
YTICKS = [1e-1, 5e-2]
XTICKS = [1, 1e1, 5e1]


def init_valid_losses(run_dir: pathlib.Path) -> dict:
    """Per-variant validation loss at epoch 0, from init.pt.

    The trainer only runs a TEST eval before training (eval_and_log_test at
    epoch 0), so there is no epoch-0 validation row in training_history.csv.
    Recompute it here with the same split and criterion as the curve, and
    cache it — the forward pass over the whole validation set is slow.
    """
    cache = run_dir / "init_valid_loss.json"
    if cache.exists():
        return json.loads(cache.read_text())
    if not (run_dir / "init.pt").exists():
        return {}

    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    import numpy as _np
    import torch
    from omegaconf import OmegaConf
    from train_srnn.models.factory import build_batched_model
    from train_srnn.data.datasets import load_dataset
    import train as _train

    ck = torch.load(run_dir / "init.pt", map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(ck["config"]); names = list(ck["ablation_names"])
    model = build_batched_model(cfg, names)
    model.load_state_dict(ck["model_state_dict"]); model.eval()
    loader_keys = ("include_actions", "seq_len", "stride", "skip_transient_s",
                   "sample_rate_hz", "train_trace_max_len", "normalize")
    d = load_dataset(cfg.task.name, cfg.task.data_dir,
                     **{k: cfg.task[k] for k in loader_keys if k in cfg.task})
    vx, vy = d["valid"]
    with torch.no_grad():
        vl, _ = _train.run_epoch(model, vx, vy, None, None, torch.nn.MSELoss(),
                                 cfg, _np.random.RandomState(int(cfg.seed)),
                                 torch.device("cpu"), training=False,
                                 K=len(names))
    out = {n: float(vl[k]) for k, n in enumerate(names)}
    cache.write_text(json.dumps(out, indent=2))
    return out


def series(rows, variant, col, max_epoch):
    """Epochs are returned 1-BASED, i.e. the checkpoint's 0-indexed epoch + 1.

    This puts the untrained model (0-indexed epoch 0) at x=1, which is a real
    position on a log axis, and keeps every other point at its true spacing —
    unlike pinning epoch 0 to x=1 while leaving the rest 0-indexed, which
    stretches the first interval.
    """
    sel = [r for r in rows if r["variant"] == variant]
    e = np.array([int(r["epoch"]) for r in sel]) + 1
    y = np.array([float(r[col]) for r in sel])
    keep = (y > 0) & (e <= max_epoch)
    return e[keep], y[keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=pathlib.Path)
    ap.add_argument("-o", "--out", type=pathlib.Path, default=None)
    ap.add_argument("--max-epoch", type=int, default=MAX_EPOCH)
    ap.add_argument("--no-init", action="store_true",
                    help="omit the epoch-0 point")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.run_dir / "training_history.csv")))
    out = args.out or args.run_dir / "plot_for_Brian.png"
    init = {} if args.no_init else init_valid_losses(args.run_dir)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    for key, label, colour in VARIANTS:
        e, y = series(rows, key, "valid_loss", args.max_epoch)
        if key in init:
            # The untrained model is the 1st epoch on this axis, so it joins
            # the curve as an ordinary point.
            e = np.concatenate([[1], e])
            y = np.concatenate([[init[key]], y])
        ax.plot(e, y, "o-", color=colour, lw=2.0, ms=5, label=label)

    ax.set_xscale("log")
    ax.set_yscale("log")

    # Hand-set ticks. x is 1-based, so the tick at 1 is the untrained model.
    ax.set_xticks(XTICKS)
    ax.set_xticklabels([r"$1$", r"$10^{1}$", r"$5\times10^{1}$"])
    ax.set_xlim(0.93, args.max_epoch * 1.05)
    ax.set_yticks(YTICKS)
    ax.set_yticklabels([r"$10^{-1}$", r"$5\times10^{-2}$"])
    ax.minorticks_off()

    # Open axes: drop the top and right spines.
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation loss")
    ax.legend(frameon=False, fontsize=10, loc="lower left")

    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")

    for key, label, _ in VARIANTS:
        e, y = series(rows, key, "valid_loss", args.max_epoch)
        i = f"epoch 1 (untrained): {init[key]:.5f}  ->  " if key in init else ""
        print(f"  {label:32s} {i}epoch {e[-1]}: {y[-1]:.5f}")


if __name__ == "__main__":
    main()
