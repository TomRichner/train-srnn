"""Experiment report for the time-warped cheetah100 comparison.

    uv run python scripts/report_timewarp.py --out <report_dir> --data-root "$SRNN_HOME/data" \
        --run control=<run_dir> --run slow=<run_dir> --run multi=<run_dir> [--notes notes.md]

Each ``--run label=dir`` is a downloaded run directory (``training_history.csv``,
``test_history.csv``, ``run_metadata.json``, optional ``speed_eval/speed_eval.json`` from
``scripts/eval_speed.py`` and ``report/`` from ``scripts/report_srnn.py``). The runs are
expected to share their variants: conditions crossed with paired seeds.

Writes ``report.md`` with figures in ``figures/`` and ``summary.json``. ``--notes`` is a
Markdown file inserted after the introduction (the interpretation, written by hand).
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import re
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

CONDITIONS = ["no-adapt", "sfa1-std1", "sfa3-std2", "sfa3-std1", "sfa1-std2"]
LABELS = {"no-adapt": "No adaptation", "sfa1-std1": "SFA1 / STD1", "sfa3-std2": "SFA3 / STD2",
          "sfa3-std1": "SFA3 / STD1", "sfa1-std2": "SFA1 / STD2"}
COLORS = {"no-adapt": "#555555", "sfa1-std1": "#0072B2", "sfa3-std2": "#DAA520",
          "sfa3-std1": "#009E73", "sfa1-std2": "#CC79A7"}
VARIANT = re.compile(r"^srnn-(.+)-seed(\d+)$")
RATE_TICKS = ([0.4, 0.5, 0.71, 1, 1.41, 2, 2.5], ["0.4", "0.5", "0.71", "1", "1.41", "2", "2.5"])


# -- data ---------------------------------------------------------------------

RUN_WIDE = ("-no-dales", "-skip")   # tokens shared by a whole run; stripped from condition names


def parse(variant: str) -> tuple[str, int]:
    m = VARIANT.match(variant)
    cond = m.group(1)
    for tok in RUN_WIDE:
        cond = cond.replace(tok, "")
    return cond, int(m.group(2))


def register_conditions(finals: dict) -> None:
    """Extend CONDITIONS, LABELS and COLORS with any condition found in the runs."""
    palette = plt.get_cmap("tab10")
    for fin in finals.values():
        for cond in fin:
            if cond not in CONDITIONS:
                CONDITIONS.append(cond)
            if cond not in LABELS:
                base = next((b for b in ("sfa3-std2", "sfa1-std2", "sfa1-std1", "no-adapt")
                             if cond.startswith(b)), None)
                LABELS[cond] = (LABELS[base] + " " + cond[len(base):].strip("-")) if base else cond
            if cond not in COLORS:
                COLORS[cond] = matplotlib.colors.to_hex(palette(len(COLORS) % 10))


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_curves(run: Path) -> dict:
    """{condition: {seed: [(epoch, test_mse), ...]}} from epoch checkpoints (init as epoch -1)."""
    out = defaultdict(lambda: defaultdict(list))
    for r in read_csv(run / "test_history.csv"):
        if r["tag"] == "last":
            continue
        cond, seed = parse(r["variant"])
        epoch = -1 if r["tag"] == "init" else int(r["epoch"])
        out[cond][seed].append((epoch, float(r["test_loss"])))
    return out


def final_test(run: Path) -> dict:
    """{condition: {seed: test MSE}} at the ``last`` checkpoint."""
    out = defaultdict(dict)
    for r in read_csv(run / "test_history.csv"):
        if r["tag"] == "last":
            cond, seed = parse(r["variant"])
            out[cond][seed] = float(r["test_loss"])
    return out


def load_speed(run: Path) -> dict | None:
    path = run / "speed_eval" / "speed_eval.json"
    return json.loads(path.read_text()) if path.exists() else None


def by_condition(names: list[str], values: list[float]) -> dict:
    out = defaultdict(dict)
    for n, v in zip(names, values):
        cond, seed = parse(n)
        out[cond][seed] = v
    return out


# -- statistics ---------------------------------------------------------------

def sign_flip_p(diffs: list[float]) -> float:
    """Exact two-sided paired sign-flip p-value for the mean difference."""
    d = np.asarray(diffs, float)
    observed = abs(d.mean())
    if len(d) > 20:
        raise ValueError("exact enumeration limited to 20 pairs")
    signs = np.array(list(itertools.product([1, -1], repeat=len(d))))
    return float(np.mean(np.abs((signs * d).mean(axis=1)) >= observed - 1e-15))


def paired(a: dict, b: dict) -> dict:
    """Compare condition b with a over shared seeds using log MSE ratios (b/a)."""
    seeds = sorted(set(a) & set(b))
    lr = [math.log(b[s] / a[s]) for s in seeds]
    return {"seeds": len(seeds), "b_better": int(sum(x < 0 for x in lr)),
            "mean_log_ratio": float(np.mean(lr)), "ratio_geomean": float(math.exp(np.mean(lr))),
            "p": sign_flip_p(lr)}


def factorial(final: dict) -> dict | None:
    """Log-MSE main effects of 3 vs 1 SFA and 2 vs 1 STD timescales, and their interaction."""
    need = ["sfa1-std1", "sfa3-std1", "sfa1-std2", "sfa3-std2"]
    if not all(c in final for c in need):
        return None
    seeds = sorted(set.intersection(*(set(final[c]) for c in need)))
    g = {c: np.log([final[c][s] for s in seeds]) for c in need}
    effects = {
        "sfa3_vs_sfa1": (g["sfa3-std1"] + g["sfa3-std2"] - g["sfa1-std1"] - g["sfa1-std2"]) / 2,
        "std2_vs_std1": (g["sfa1-std2"] + g["sfa3-std2"] - g["sfa1-std1"] - g["sfa3-std1"]) / 2,
        "interaction": (g["sfa3-std2"] - g["sfa1-std2"]) - (g["sfa3-std1"] - g["sfa1-std1"]),
    }
    return {k: {"mean_log": float(v.mean()), "ratio": float(math.exp(v.mean())),
                "p": sign_flip_p(v.tolist())} for k, v in effects.items()}


# -- figures ------------------------------------------------------------------

def fig_rates(data_root: Path, datasets: dict, out: Path) -> str | None:
    warped = {k: v for k, v in datasets.items() if (data_root / v / "train.npz").exists()
              and "rate" in np.load(data_root / v / "train.npz").files}
    if not warped:
        return None
    fig, axes = plt.subplots(len(warped), 2, figsize=(11, 2.6 * len(warped)), squeeze=False,
                             gridspec_kw={"width_ratios": [3, 1]})
    for row, (label, name) in zip(axes, warped.items()):
        for split, alpha in (("train", 1.0), ("valid", 0.7), ("test", 0.7)):
            rate = np.load(data_root / name / f"{split}.npz")["rate"]
            if split == "train":
                t = np.arange(len(rate)) / 100.0
                keep = t < 600
                row[0].plot(t[keep], rate[keep], lw=0.8, color="k")
            row[1].hist(np.log2(rate), bins=60, range=(-1, 1), histtype="step", density=True,
                        label=split, alpha=alpha)
        row[0].axhline(1, color="0.6", lw=0.6)
        row[0].set_yscale("log", base=2)
        row[0].set_yticks([0.5, 0.71, 1, 1.41, 2], ["0.5", "0.71", "1", "1.41", "2"])
        row[0].set_ylim(0.45, 2.2)
        row[0].set_ylabel("playback rate r")
        row[0].set_title(f"{label} ({name}): first 10 min of train", fontsize=9)
        row[1].set_xlabel("log2 r")
        row[1].legend(fontsize=7)
    axes[-1][0].set_xlabel("warped time (s)")
    fig.tight_layout()
    fig.savefig(out / "rates.png", dpi=130)
    plt.close(fig)
    return "figures/rates.png"


def fig_snippet(data_root: Path, fixed: str, out: Path) -> str | None:
    base = data_root / fixed
    files = {r: base / f"test_r{r:.2f}.npz" for r in (0.5, 1.0, 2.0)}
    if not all(p.exists() for p in files.values()):
        return None
    fig, axes = plt.subplots(1, 2, figsize=(11, 2.8))
    for r, p in files.items():
        z = np.load(p)
        t = np.arange(300) / 100.0
        axes[0].plot(t, z["obs"][2000:2300, 2], label=f"r = {r:g}", lw=1)
        axes[1].plot(t, z["obs"][2000:2300, 11], lw=1)
    axes[0].set_title("back-thigh angle (ch 2), 3 s", fontsize=9)
    axes[1].set_title("back-thigh angular velocity (ch 11, scaled by r)", fontsize=9)
    for ax in axes:
        ax.set_xlabel("time (s)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "snippet.png", dpi=130)
    plt.close(fig)
    return "figures/snippet.png"


def fig_learning(curves: dict, out: Path) -> str:
    fig, axes = plt.subplots(1, len(curves), figsize=(4.2 * len(curves), 3.4), squeeze=False,
                             sharey=True)
    for ax, (label, cur) in zip(axes[0], curves.items()):
        for cond in CONDITIONS:
            if cond not in cur:
                continue
            seeds = sorted(cur[cond])
            epochs = [e for e, _ in sorted(cur[cond][seeds[0]])]
            vals = np.array([[v for _, v in sorted(cur[cond][s])] for s in seeds])
            x = np.array(epochs) + 1
            m, sd = vals.mean(0), vals.std(0, ddof=1)
            ax.plot(x, m, "-o", ms=3, color=COLORS[cond], label=LABELS[cond])
            ax.fill_between(x, m - sd, m + sd, color=COLORS[cond], alpha=0.15, lw=0)
        ax.set_yscale("log")
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("completed epochs (0 = init)")
        ax.grid(alpha=0.3, which="both")
    axes[0][0].set_ylabel("test MSE (windowed, z-scored)")
    axes[0][-1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "learning.png", dpi=130)
    plt.close(fig)
    return "figures/learning.png"


def fig_final(finals: dict, out: Path) -> str:
    fig, ax = plt.subplots(figsize=(10, 3.6))
    width = 0.8 / len(CONDITIONS)
    for i, (label, fin) in enumerate(finals.items()):
        for j, cond in enumerate(CONDITIONS):
            if cond not in fin:
                continue
            v = np.array(list(fin[cond].values()))
            x = i + (j - (len(CONDITIONS) - 1) / 2) * width
            ax.scatter(np.full(len(v), x) + np.random.default_rng(j).uniform(-0.25, 0.25, len(v)) * width,
                       v, s=7, color=COLORS[cond], alpha=0.6, label=LABELS[cond] if i == 0 else None)
            ax.hlines(np.exp(np.log(v).mean()), x - width / 2, x + width / 2, color="k", lw=1.5)
    ax.set_xticks(range(len(finals)), list(finals))
    ax.set_yscale("log")
    ax.set_ylabel("final test MSE")
    ax.grid(alpha=0.3, axis="y", which="both")
    ax.legend(fontsize=7, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    fig.tight_layout()
    fig.savefig(out / "final.png", dpi=130)
    plt.close(fig)
    return "figures/final.png"


def condition_bins(test: dict, names: list[str]) -> dict:
    """{cond: (centers, mean over seeds of MSE / persistence per bin)} for one test trace."""
    out = {}
    for cond in CONDITIONS:
        idx = [i for i, n in enumerate(names) if parse(n)[0] == cond]
        if not idx:
            continue
        centers, ratios = [], []
        for b in test["bins"]:
            if b["samples"] < 500:
                continue
            centers.append((b["log2_rate_lo"] + b["log2_rate_hi"]) / 2)
            ratios.append(np.mean([b["mse"][i] for i in idx]) / b["persistence_mse"])
        out[cond] = (np.array(centers), np.array(ratios))
    return out


def fig_speed_bins(speeds: dict, test_name: str, out: Path) -> str | None:
    panels = [(label, sp, test_name) for label, sp in speeds.items()
              if sp and test_name in sp["tests"]]
    if not panels:
        return None
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.4), squeeze=False,
                             sharey=True)
    for ax, (label, sp, test_name) in zip(axes[0], panels):
        test = sp["tests"].get(test_name)
        if test is None:
            continue
        for cond, (c, r) in condition_bins(test, sp["variants"]).items():
            ax.plot(2.0 ** c, r, "-o", ms=3, color=COLORS[cond], label=LABELS[cond])
        ax.set_xscale("log", base=2)
        ax.set_xticks(*RATE_TICKS)
        ax.set_yscale("log")
        ax.axhline(1, color="0.5", lw=0.8, ls="--")
        ax.set_title(f"trained on {label}", fontsize=9)
        ax.set_xlabel("local playback rate r")
        ax.grid(alpha=0.3, which="both")
    axes[0][0].set_ylabel("MSE / persistence MSE")
    axes[0][-1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "speed_bins.png", dpi=130)
    plt.close(fig)
    return "figures/speed_bins.png"


def fixed_speed_table(sp: dict) -> dict:
    """{cond: {rate: (mean MSE, mean MSE/persistence)}} over the fixed-speed test files."""
    out = defaultdict(dict)
    for name, test in sp["tests"].items():
        m = re.search(r"test_r(\d+\.\d+)$", name)
        if not m:
            continue
        rate = float(m.group(1))
        for cond, vals in by_condition(sp["variants"], test["mse"]).items():
            mse = float(np.mean(list(vals.values())))
            out[cond][rate] = (mse, mse / test["persistence_mse"])
    return out


def train_rate_range(data_root: Path, dataset: str | None) -> tuple[float, float]:
    """Observed playback-rate range of a dataset's training split (1.0 for unwarped data)."""
    path = data_root / dataset / "train.npz" if dataset else None
    if path is None or not path.exists():
        return (1.0, 1.0)
    z = np.load(path)
    if "rate" not in z.files:
        return (1.0, 1.0)
    return float(z["rate"].min()), float(z["rate"].max())


def fig_fixed(speeds: dict, ranges: dict, persistence: dict, out: Path) -> str | None:
    panels = [(label, fixed_speed_table(sp)) for label, sp in speeds.items() if sp]
    panels = [(l, t) for l, t in panels if t]
    if not panels:
        return None
    fig, axes = plt.subplots(2, len(panels), figsize=(4.2 * len(panels), 6.2), squeeze=False,
                             sharex=True, sharey="row")
    for col, (label, table) in enumerate(panels):
        for row, (which, ylabel) in enumerate(((1, "MSE / persistence MSE"), (0, "MSE"))):
            ax = axes[row][col]
            for cond in CONDITIONS:
                if cond not in table:
                    continue
                rates = sorted(table[cond])
                ax.plot(rates, [table[cond][r][which] for r in rates], "-o", ms=3,
                        color=COLORS[cond], label=LABELS[cond])
            lo, hi = ranges.get(label, (1.0, 1.0))
            if hi > lo:
                ax.axvspan(lo, hi, color="0.9", zorder=0)
            else:
                ax.axvline(1.0, color="0.8", lw=6, zorder=0)
            if which == 0 and persistence:
                rates = sorted(persistence)
                ax.plot(rates, [persistence[r] for r in rates], "k--", lw=1, label="persistence")
            ax.set_xscale("log", base=2)
            ax.set_xticks(*RATE_TICKS)
            ax.set_yscale("log")
            if which == 1:
                ax.axhline(1, color="0.5", lw=0.8, ls="--")
                ax.set_title(f"trained on {label}", fontsize=9)
            ax.grid(alpha=0.3, which="both")
            if col == 0:
                ax.set_ylabel(ylabel)
        axes[1][col].set_xlabel("constant playback rate r")
    axes[1][-1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "fixed_speeds.png", dpi=130)
    plt.close(fig)
    return "figures/fixed_speeds.png"


# -- report -------------------------------------------------------------------

def fmt_p(p: float) -> str:
    return f"{p:.4f}" if p >= 1e-4 else f"{p:.1e}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--run", action="append", required=True, help="label=run_dir")
    p.add_argument("--dataset", action="append", default=[], help="label=dataset folder name")
    p.add_argument("--fixed", default="cheetah100_fixed_speeds")
    p.add_argument("--bins-test", default="cheetah100_warp_multi/test",
                   help="speed_eval test whose per-rate bins are plotted for every run")
    p.add_argument("--title", default="Time-warped cheetah100: do multiple adaptation timescales help?")
    p.add_argument("--notes", type=Path)
    args = p.parse_args()

    runs = dict(r.split("=", 1) for r in args.run)
    runs = {k: Path(v) for k, v in runs.items()}
    datasets = dict(d.split("=", 1) for d in args.dataset)
    figs = args.out / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    curves = {k: test_curves(v) for k, v in runs.items()}
    finals = {k: final_test(v) for k, v in runs.items()}
    speeds = {k: load_speed(v) for k, v in runs.items()}
    register_conditions(finals)
    meta = {k: json.loads((v / "run_metadata.json").read_text()) for k, v in runs.items()
            if (v / "run_metadata.json").exists()}

    summary = {"runs": {k: str(v) for k, v in runs.items()}, "datasets": datasets,
               "final": {}, "paired": {}, "factorial": {}, "fixed_speeds": {}, "speed_tests": {}}
    lines = [f"# {args.title}", ""]
    lines += ["Runs: " + ", ".join(f"`{k}` = `{v.name if v.name != 'seed1' else v.parent.parent.parent.name}`"
                                   for k, v in runs.items()) + ".", ""]
    for k, m in meta.items():
        attempts = m.get("attempts", 1)
        last = " (last attempt)" if attempts and attempts > 1 else ""
        lines.append(f"- `{k}`: exit {m.get('exit_code')}, {m.get('duration_seconds')} s{last} on "
                     f"{m.get('hardware')}, commit `{str(m.get('commit'))[:10]}`, attempts {attempts}.")
    lines.append("")
    if args.notes and args.notes.exists():
        lines += [args.notes.read_text().rstrip(), ""]

    lines += ["## Datasets", ""]
    rates = fig_rates(args.data_root, datasets, figs)
    if rates:
        lines += [f"![Playback rate of the warped training traces and the rate distribution "
                  f"of each split]({rates})", ""]
    snip = fig_snippet(args.data_root, args.fixed, figs)
    if snip:
        lines += [f"![The same 3 s of the test trace at 0.5x, 1x and 2x; velocities scale with r]"
                  f"({snip})", ""]

    lines += ["## Learning curves", "",
              f"![Test MSE (trainer's windowed evaluation) at each checkpoint; mean and SD over "
              f"seeds]({fig_learning(curves, figs)})", "",
              "## Final test error", "",
              f"![Final test MSE per network; bars are geometric means]({fig_final(finals, figs)})", ""]
    header = "| Condition | " + " | ".join(finals) + " |"
    lines += [header, "|---|" + "---:|" * len(finals)]
    for cond in CONDITIONS:
        cells = []
        for label, fin in finals.items():
            if cond in fin:
                v = np.array(list(fin[cond].values()))
                cells.append(f"{np.exp(np.log(v).mean()):.4f} ({v.mean():.4f} ± {v.std(ddof=1):.4f})")
                summary["final"].setdefault(label, {})[cond] = {
                    "geomean": float(np.exp(np.log(v).mean())), "mean": float(v.mean()),
                    "sd": float(v.std(ddof=1)), "n": int(len(v))}
            else:
                cells.append("–")
        lines.append(f"| {LABELS[cond]} | " + " | ".join(cells) + " |")
    lines += ["", "Geometric mean over seeds, then arithmetic mean ± SD in parentheses.", ""]

    lines += ["### Paired comparisons", "",
              "Ratio = geometric mean of per-seed MSE ratios (second / first); wins = seeds where "
              "the second condition has lower MSE; p = exact two-sided paired sign-flip test on log "
              "ratios, uncorrected.", "",
              "| Dataset | Comparison | Ratio | Wins | p |", "|---|---|---:|---:|---:|"]
    pairs = [("sfa1-std1", "sfa3-std2"), ("no-adapt", "sfa1-std1"), ("no-adapt", "sfa3-std2"),
             ("sfa1-std1", "sfa3-std2-std-geo"), ("sfa1-std1", "sfa3-std2-std-usage"),
             ("sfa1-std1", "sfa3-std2-std-scale"), ("sfa3-std2", "sfa3-std2-std-geo"),
             ("sfa3-std2", "sfa1-std1-std-strong"), ("sfa1-std1", "sfa1-std2-std-geo"),
             ("no-adapt", "no-adapt-w-matched")]
    for label, fin in finals.items():
        for a, b in pairs:
            if a in fin and b in fin:
                r = paired(fin[a], fin[b])
                summary["paired"].setdefault(label, {})[f"{b}_vs_{a}"] = r
                lines.append(f"| {label} | {LABELS[b]} vs {LABELS[a]} | {r['ratio_geomean']:.3f} | "
                             f"{r['b_better']}/{r['seeds']} | {fmt_p(r['p'])} |")
    lines.append("")
    fac_rows = []
    for label, fin in finals.items():
        f = factorial(fin)
        if f:
            summary["factorial"][label] = f
            fac_rows.append(f"| {label} | {f['sfa3_vs_sfa1']['ratio']:.3f} ({fmt_p(f['sfa3_vs_sfa1']['p'])}) | "
                            f"{f['std2_vs_std1']['ratio']:.3f} ({fmt_p(f['std2_vs_std1']['p'])}) | "
                            f"{f['interaction']['ratio']:.3f} ({fmt_p(f['interaction']['p'])}) |")
    if fac_rows:
        lines += ["### Factorial: SFA count versus STD count", "",
                  "MSE ratios from the 2 × 2 design (SFA 1 or 3 timescales × STD 1 or 2), averaged "
                  "over the other factor; below 1 favors more timescales. The interaction is the "
                  "ratio of the SFA effect with two STD timescales to that with one. p in parentheses.",
                  "", "| Dataset | SFA 3 vs 1 | STD 2 vs 1 | Interaction |", "|---|---:|---:|---:|",
                  *fac_rows, ""]

    lines += ["## Error versus playback speed", "",
              "`scripts/eval_speed.py` runs each network continuously over whole test traces and "
              "bins its squared error by the local playback rate. Faster segments change more per "
              "sample, so each bin is divided by the persistence baseline (predicting the current "
              "sample) in the same bin; below 1 beats persistence. The first 5 s are excluded.", ""]
    sb = fig_speed_bins(speeds, args.bins_test, figs)
    if sb:
        lines += [f"![Error relative to persistence versus local rate on the `{args.bins_test}` "
                  f"split, whose rate spans about 0.6-1.6x]({sb})", ""]
    ranges = {k: train_rate_range(args.data_root, datasets.get(k)) for k in runs}
    persist = {}
    for sp in speeds.values():
        for name, t in (sp or {}).get("tests", {}).items():
            m = re.search(r"test_r(\d+\.\d+)$", name)
            if m:
                persist[float(m.group(1))] = t["persistence_mse"]
    fx = fig_fixed(speeds, ranges, persist, figs)
    if fx:
        rng = ", ".join(f"{k} {lo:.2f}-{hi:.2f}x" for k, (lo, hi) in ranges.items())
        lines += [f"![Constant-speed test traces, relative to persistence (top) and absolute "
                  f"(bottom, dashed = persistence). Grey: the playback rates present in each "
                  f"training set ({rng}); rates outside it are extrapolation]({fx})", ""]
        for label, sp in speeds.items():
            if not sp:
                continue
            table = fixed_speed_table(sp)
            rates = sorted(next(iter(table.values())))
            summary["fixed_speeds"][label] = {c: {str(r): v for r, v in t.items()} for c, t in table.items()}
            lines += [f"**Trained on {label}**: MSE / persistence at constant rates", "",
                      "| Condition | " + " | ".join(f"{r:g}x" for r in rates) + " |",
                      "|---|" + "---:|" * len(rates)]
            for cond in CONDITIONS:
                if cond in table:
                    lines.append(f"| {LABELS[cond]} | " + " | ".join(f"{table[cond][r][1]:.3f}"
                                                                      for r in rates) + " |")
            lines.append("| *Persistence MSE (absolute)* | " + " | ".join(
                f"*{persist[r]:.4f}*" for r in rates) + " |")
            lines.append("")
    test_names = sorted({n for sp in speeds.values() if sp for n in sp["tests"]
                         if "fixed_speeds" not in n})
    if test_names:
        lines += ["### Every model on every warped test split", "",
                  "Mean continuous-evaluation MSE / persistence over seeds.", "",
                  "| Trained on | Condition | " + " | ".join(f"`{n}`" for n in test_names) + " |",
                  "|---|---|" + "---:|" * len(test_names)]
        for label, sp in speeds.items():
            if not sp:
                continue
            for cond in CONDITIONS:
                cells = []
                for n in test_names:
                    t = sp["tests"].get(n)
                    if t is None:
                        cells.append("–")
                        continue
                    vals = by_condition(sp["variants"], t["mse"]).get(cond)
                    ratio = np.mean(list(vals.values())) / t["persistence_mse"] if vals else float("nan")
                    summary["speed_tests"].setdefault(label, {}).setdefault(cond, {})[n] = ratio
                    cells.append(f"{ratio:.3f}")
                if any(c != "–" for c in cells):
                    lines.append(f"| {label} | {LABELS[cond]} | " + " | ".join(cells) + " |")
        lines.append("")

    lines += ["## Weights and parameters during training", "",
              "Per-run reports from `scripts/report_srnn.py` (all networks in the aggregates; "
              "detail pages for the lowest seed of each condition):", ""]
    for label, run in runs.items():
        rep = run / "report"
        if not (rep / "report.md").exists():
            continue
        dest = args.out / f"run_{label}"
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(rep, dest)
        lines += [f"### {label}", "", f"Full report: [run_{label}/report.md](run_{label}/report.md).", ""]
        for fig in sorted((dest / "report_figures").glob("aggregate_*.png")):
            lines += [f"![{label}: {fig.stem}](run_{label}/report_figures/{fig.name})", ""]
        for fig in sorted((dest / "report_figures").glob("taus_*.png")):
            lines += [f"![{label}: {fig.stem}](run_{label}/report_figures/{fig.name})", ""]

    (args.out / "report.md").write_text("\n".join(lines) + "\n")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(args.out / "report.md")


if __name__ == "__main__":
    main()
