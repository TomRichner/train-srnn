"""plot_adaptation_comparison.py — adaptation vs no-adaptation, log-log.

Two variants only, on log-log axes: `srnn` (SFA n_a=3 + STD n_b=1, both E and
I) against `srnn-no-adapt` (all adaptation off). Everything else — Dale's law,
no skip, size, seed, and the shared recurrent matrix from BatchedSRNNCell — is
identical, so the difference is the adaptation states alone.

Log-log matters here: on linear axes the two curves appear to converge and sit
on top of each other, when in fact they are power laws with different
exponents that cross over past ~100 epochs.

Usage:
    python scripts/plot_adaptation_comparison.py tmp/ring6-400e
"""
import argparse
import csv
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Default pair: the Dale's-law, no-skip variants. --variants / --labels /
# --colors override, so the same script covers any adaptation-vs-none pair.
DEFAULT_VARIANTS = ["srnn-no-adapt", "srnn"]
DEFAULT_LABELS = ["No adaptation", "Multiple timescale adaptation"]
DEFAULT_COLORS = ["#ff7f0e", "#9467bd"]


def series(rows, variant, col, max_epoch=None):
    sel = [r for r in rows if r["variant"] == variant]
    e = np.array([int(r["epoch"]) for r in sel])
    y = np.array([float(r[col]) for r in sel])
    keep = (e > 0) & (y > 0)          # epoch 0 has no place on a log axis
    if max_epoch is not None:
        keep &= e <= max_epoch
    return e[keep], y[keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=pathlib.Path)
    ap.add_argument("-o", "--out", type=pathlib.Path, default=None)
    ap.add_argument("--fit-from", type=int, default=150,
                    help="first epoch of the power-law fit reported in the legend")
    ap.add_argument("--variants", nargs="*", default=DEFAULT_VARIANTS,
                    help="variant keys, in legend order")
    ap.add_argument("--labels", nargs="*", default=None,
                    help="display labels, same order as --variants")
    ap.add_argument("--colors", nargs="*", default=None)
    ap.add_argument("--metrics", nargs="*", default=["valid_loss", "train_loss"],
                    choices=["valid_loss", "train_loss"])
    ap.add_argument("--max-epoch", type=int, default=None,
                    help="truncate the x-axis at this epoch")
    args = ap.parse_args()

    # A fit window past the truncation point would be empty; fall back to the
    # second half of what is actually shown so alpha still means something.
    if args.max_epoch is not None and args.fit_from >= args.max_epoch:
        args.fit_from = max(1, args.max_epoch // 2)
        print(f"  --fit-from past --max-epoch; fitting from {args.fit_from}")

    labels = args.labels or (DEFAULT_LABELS if args.variants == DEFAULT_VARIANTS
                             else args.variants)
    colors = args.colors or (DEFAULT_COLORS if args.variants == DEFAULT_VARIANTS
                             else [f"C{i}" for i in range(len(args.variants))])
    if not (len(args.variants) == len(labels) == len(colors)):
        raise SystemExit("--variants, --labels and --colors must be the same length")
    variants = dict(zip(args.variants, zip(labels, colors)))

    rows = list(csv.DictReader(open(args.run_dir / "training_history.csv")))
    out = args.out or args.run_dir / "adaptation_comparison_loglog.png"

    n = len(args.metrics)
    fig, axes = plt.subplots(1, n, figsize=(6.5 * n, 5), sharey=True,
                             squeeze=False)
    axes = axes[0]
    for ax, col in zip(axes, args.metrics):
        for key, (label, colour) in variants.items():
            e, y = series(rows, key, col, args.max_epoch)
            m = e >= args.fit_from
            alpha = np.polyfit(np.log(e[m]), np.log(y[m]), 1)[0]
            ax.plot(e, y, "o-", color=colour, lw=2.0, ms=4.5,
                    label=f"{label}  (α={alpha:.2f})")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(alpha=0.3, which="major")
        ax.grid(alpha=0.12, which="minor")
        ax.set_xlabel("epoch")
        ax.set_title(col.replace("_", " ").title())
        ax.legend(fontsize=9, loc="lower left", frameon=True)
    axes[0].set_ylabel("loss")

    fig.suptitle(
        f"{args.run_dir.name} — adaptation vs none  |  log-log; "
        f"α = slope of loss ∝ epoch^α fitted from epoch {args.fit_from}",
        fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")

    # crossover epoch, if the curves swap order
    adapt_key = args.variants[-1]; none_key = args.variants[0]
    e1, y1 = series(rows, adapt_key, "valid_loss", args.max_epoch)
    e2, y2 = series(rows, none_key, "valid_loss", args.max_epoch)
    n = min(len(e1), len(e2))
    ahead = y1[:n] < y2[:n]
    flips = np.flatnonzero(ahead[1:] != ahead[:-1])
    if len(flips):
        print(f"valid-loss crossover near epoch {e1[flips[-1] + 1]}: "
              f"adaptation goes from behind to ahead")
    print(f"final valid loss — {adapt_key} {y1[-1]:.5f} vs {none_key} {y2[-1]:.5f} "
          f"({100 * (y1[-1] - y2[-1]) / y2[-1]:+.1f}%)")


if __name__ == "__main__":
    main()
