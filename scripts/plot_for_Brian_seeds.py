"""plot_for_Brian_seeds.py — presentation figure: adaptation vs none, 5 seeds each.

Multi-seed companion to plot_for_Brian.py. Same styling (open axes, hand-set
ticks, 1-based epochs) but plots every seed of a run whose variant names carry
the ``-seed<n>`` suffix written by ``build_batched_model`` when
``batched_ablation_seeds`` is set.

Compares the two no-Dale's, skip variants:
    srnn-no-dales-skip           multi-timescale SFA (n_a=3) + STD (n_b=1)
    srnn-no-adapt-no-dales-skip  same network, adaptation off

Variants sharing a seed are exported from one RMTMatrix, so each of the five
pairs starts from the same recurrent weights.

By default it plots the held-out TEST loss, read from test_history.csv, which
carries a genuine epoch-0 ``init`` row logged before training. Pass
``--loss valid`` for the validation curve instead; that path has no epoch-0 row
(the trainer only runs a test eval before training) so it recomputes one from
init.pt via plot_for_Brian.init_valid_losses.

Usage:
    python scripts/plot_for_Brian_seeds.py tmp/ring2x5-100e
    python scripts/plot_for_Brian_seeds.py tmp/ring2x5-100e --loss valid
"""
import argparse
import csv
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Vector output that stays editable in Illustrator. fonttype 42 embeds
# TrueType rather than matplotlib's default Type 3, which Illustrator cannot
# edit as text; svg.fonttype "none" leaves <text> elements referencing the
# system font instead of converting glyphs to paths.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["svg.fonttype"] = "none"
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from plot_for_Brian import init_valid_losses, series  # noqa: E402

VARIANTS = [
    ("srnn-no-adapt-no-dales-skip", "No adaptation", "#B8860B"),
    ("srnn-no-dales-skip", "Multiple timescale adaptation", "#1f77b4"),
]
SEEDS = [1, 2, 3, 4, 5]
MAX_EPOCH = 100
YTICKS = [1e-1, 1e-2]
YTICKLABELS = [r"$10^{-1}$", r"$10^{-2}$"]
YMIN = 1e-2
XTICKS = [1, 1e1, 1e2]
XTICKLABELS = [r"$1$", r"$10^{1}$", r"$10^{2}$"]
LABEL_FONTSIZE = 15
TICK_FONTSIZE = 13


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=pathlib.Path)
    ap.add_argument("-o", "--out", type=pathlib.Path, default=None)
    ap.add_argument("--max-epoch", type=int, default=MAX_EPOCH)
    ap.add_argument("--seeds", type=str, default=",".join(map(str, SEEDS)))
    ap.add_argument("--formats", default="png,pdf,svg",
                    help="comma-separated output formats (default: png,pdf,svg). "
                         "pdf/svg are vector and keep text editable in Illustrator.")
    ap.add_argument("--loss", choices=("test", "valid"), default="test",
                    help="which curve to plot (default: test)")
    ap.add_argument("--no-init", action="store_true",
                    help="omit the epoch-0 point")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    if args.loss == "test":
        # test_history.csv already carries epoch 0 as tag 'init'. Epoch 99 is
        # written twice (tag 'epoch_099' and tag 'last') with identical values;
        # drop the duplicate so the final point is not plotted on top of itself.
        rows = [r for r in csv.DictReader(open(args.run_dir / "test_history.csv"))
                if r["tag"] != "last"]
        col, init = "test_loss", {}
    else:
        rows = list(csv.DictReader(open(args.run_dir / "training_history.csv")))
        col = "valid_loss"
        init = {} if args.no_init else init_valid_losses(args.run_dir)
    default_name = ("plot_for_Brian_seeds.png" if args.loss == "test"
                    else "plot_for_Brian_seeds_valid.png")
    out = args.out or args.run_dir / default_name

    fig, ax = plt.subplots(figsize=(6.5, 5))
    for base, label, colour in VARIANTS:
        curves = []
        for s in seeds:
            key = f"{base}-seed{s}"
            e, y = series(rows, key, col, args.max_epoch)
            if key in init:
                # The untrained model is the 1st epoch on this axis, so it
                # joins the curve as an ordinary point.
                e = np.concatenate([[1], e])
                y = np.concatenate([[init[key]], y])
            ax.plot(e, y, "-", color=colour, lw=1.0, alpha=0.40, zorder=1)
            curves.append((e, y))
        # Mean across seeds (arithmetic; all seeds share the same epoch grid).
        e0 = curves[0][0]
        assert all(np.array_equal(e, e0) for e, _ in curves), \
            "seeds do not share an epoch grid"
        ax.plot(e0, np.mean([y for _, y in curves], axis=0), "o-",
                color=colour, lw=2.2, ms=4.5, label=label, zorder=2)

    ax.set_xscale("log")
    ax.set_yscale("log")

    # Hand-set ticks. x is 1-based, so the tick at 1 is the untrained model.
    ax.set_xticks(XTICKS)
    ax.set_xticklabels(XTICKLABELS)
    ax.set_xlim(0.93, args.max_epoch * 1.05)
    ax.set_yticks(YTICKS)
    ax.set_yticklabels(YTICKLABELS)
    ax.set_ylim(bottom=YMIN)
    ax.minorticks_off()
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)

    # Open axes: drop the top and right spines.
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    ax.set_xlabel("Epoch", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Test loss" if args.loss == "test" else "Validation loss",
                  fontsize=LABEL_FONTSIZE)
    ax.legend(frameon=False, fontsize=10, loc="lower left")

    fig.tight_layout()
    for fmt in [f.strip().lstrip(".") for f in args.formats.split(",") if f.strip()]:
        path = out.with_suffix(f".{fmt}")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"wrote {path}")

    for base, label, _ in VARIANTS:
        finals = []
        for s in seeds:
            e, y = series(rows, f"{base}-seed{s}", col, args.max_epoch)
            finals.append(y[-1])
        print(f"  {label:32s} epoch {e[-1]}: "
              f"mean={np.mean(finals):.5f} sd={np.std(finals, ddof=1):.5f} "
              f"min={min(finals):.5f} max={max(finals):.5f}")


if __name__ == "__main__":
    main()
