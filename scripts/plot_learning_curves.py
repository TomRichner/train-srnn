"""plot_learning_curves.py — learning curves for a batched-ablation run.

Reads training_history.csv / test_history.csv from a results directory and
plots per-variant loss vs epoch.

Skip and non-skip variants sit on very different baselines on autoregressive
tasks (non-skip predicts ~0 and lands on E||y||^2; skip predicts ~x and lands
on the persistence baseline E||y-x||^2), so plotting them on shared axes hides
all within-group motion. They get separate panels with independent y-scales.

Usage:
    python scripts/plot_learning_curves.py tmp/dales-skip-4s [-o out.png]
"""
import argparse
import csv
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_history(run_dir: pathlib.Path):
    tr = list(csv.DictReader(open(run_dir / "training_history.csv")))
    te_path = run_dir / "test_history.csv"
    te = list(csv.DictReader(open(te_path))) if te_path.exists() else []
    # dict.fromkeys preserves first-seen order (= the batched_ablations order)
    variants = list(dict.fromkeys(r["variant"] for r in tr))
    return tr, te, variants


def series(rows, variant, col):
    sel = [r for r in rows if r["variant"] == variant]
    return ([int(r["epoch"]) for r in sel],
            [float(r[col]) for r in sel])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=pathlib.Path)
    ap.add_argument("-o", "--out", type=pathlib.Path, default=None)
    args = ap.parse_args()

    tr, te, variants = read_history(args.run_dir)
    out = args.out or args.run_dir / "learning_curves.png"

    no_skip = [v for v in variants if "skip" not in v]
    skip = [v for v in variants if "skip" in v]
    groups = [("no-skip", no_skip), ("skip", skip)]

    cmap = plt.get_cmap("tab10")
    colors = {v: cmap(i % 10) for i, v in enumerate(variants)}

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))

    for row, col_name in enumerate(("valid_loss", "train_loss")):
        for col, (gname, gvars) in enumerate(groups):
            ax = axes[row, col]
            for v in gvars:
                e, y = series(tr, v, col_name)
                ax.plot(e, y, "o-", color=colors[v], label=v, lw=1.7, ms=5)
            ax.set_xlabel("epoch")
            ax.set_ylabel(col_name.replace("_", " "))
            ax.set_title(f"{col_name.replace('_', ' ').title()} — {gname}")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8, loc="best")

    fig.suptitle(f"{args.run_dir.name} — learning curves", y=0.995)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")

    if te:
        print("\nTest loss: init -> last")
        for v in variants:
            d = {r["tag"]: float(r["test_loss"]) for r in te if r["variant"] == v}
            i, l = d.get("init"), d.get("last")
            if i is None or l is None:
                continue
            print(f"  {v:32s} {i:9.4f} -> {l:9.4f}  "
                  f"({100.0 * (l - i) / i:+6.3f}%)")


if __name__ == "__main__":
    main()
