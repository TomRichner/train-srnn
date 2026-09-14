"""Adaptation vs no adaptation across the paired seeds of one run.

Plots every ``<variant>-seed<n>`` curve of the two compared variants plus the
across-seed mean, on log-log axes with 1-based epochs so the untrained model
sits at x=1. The two variants at a given seed share one recurrent matrix.

    python scripts/plot_adaptation_seeds.py $SRNN_HOME/cache/ring2x5-100e
    python scripts/plot_adaptation_seeds.py $SRNN_HOME/cache/ring2x5-100e --loss valid
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Editable vector output: embed TrueType (42) and keep SVG text as text.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["svg.fonttype"] = "none"

VARIANTS = [("srnn-no-adapt-no-dales-skip", "No adaptation", "#B8860B"),
            ("srnn-no-dales-skip", "Multiple timescale adaptation", "#1f77b4")]
SEEDS = [1, 2, 3, 4, 5]


def series(rows, variant, col, max_epoch):
    """``(epochs, values)`` for one variant; epochs are 1-based so epoch 0 lands at x=1."""
    sel = [r for r in rows if r["variant"] == variant]
    e = np.array([int(r["epoch"]) for r in sel]) + 1
    y = np.array([float(r[col]) for r in sel])
    keep = (y > 0) & (e <= max_epoch)
    return e[keep], y[keep]


def init_valid_losses(run_dir: pathlib.Path) -> dict:
    """Validation loss of the untrained model per variant, cached in init_valid_loss.json."""
    cache = run_dir / "init_valid_loss.json"
    if cache.exists():
        return json.loads(cache.read_text())
    if not (run_dir / "init.pt").exists():
        return {}
    import torch
    from _runs import rebuild_model
    from train_srnn.data import build_task
    from train_srnn.training import WindowedTrainer
    ckpt = torch.load(run_dir / "init.pt", map_location="cpu", weights_only=False)
    model, cfg, names = rebuild_model(ckpt)
    task = build_task(cfg)
    data = task.load(pathlib.Path(cfg.task.data_dir))
    trainer = WindowedTrainer(cfg, model, task, data, torch.device("cpu"), run_dir)
    stats = trainer.evaluate("valid")
    out = {n: float(stats.loss[k]) for k, n in enumerate(names)}
    cache.write_text(json.dumps(out, indent=2))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=pathlib.Path)
    ap.add_argument("--loss", choices=["test", "valid"], default="test")
    ap.add_argument("--max-epoch", type=int, default=100)
    ap.add_argument("-o", "--out", type=pathlib.Path, default=None)
    args = ap.parse_args()

    if args.loss == "test":
        rows = list(csv.DictReader(open(args.run_dir / "test_history.csv")))
        col, init = "test_loss", {}
        rows = [r for r in rows if r["tag"] != "last"]      # the last row duplicates the final epoch
    else:
        rows = list(csv.DictReader(open(args.run_dir / "training_history.csv")))
        col, init = "valid_loss", init_valid_losses(args.run_dir)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    for key, label, colour in VARIANTS:
        curves = []
        for s in SEEDS:
            e, y = series(rows, f"{key}-seed{s}", col, args.max_epoch)
            if f"{key}-seed{s}" in init:
                e, y = np.concatenate([[1], e]), np.concatenate([[init[f"{key}-seed{s}"]], y])
            ax.plot(e, y, "-", color=colour, lw=0.8, alpha=0.45)
            curves.append((e, y))
        common = sorted(set.intersection(*[set(e) for e, _ in curves]))
        mean = [np.mean([y[list(e).index(x)] for e, y in curves]) for x in common]
        ax.plot(common, mean, "o-", color=colour, lw=2.2, ms=4, label=label)
        print(f"{label:32s} final mean {mean[-1]:.4f} (n={len(curves)} seeds)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_xlabel("Epoch", fontsize=15)
    ax.set_ylabel(f"{'Test' if args.loss == 'test' else 'Validation'} loss", fontsize=15)
    ax.tick_params(labelsize=13)
    ax.legend(frameon=False, fontsize=11, loc="lower left")
    fig.tight_layout()
    out = args.out or args.run_dir / f"adaptation_seeds_{args.loss}.svg"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"wrote {out} and {out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
