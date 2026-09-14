"""Run Benettin's method on every saved checkpoint of a run + plot FTLE-vs-epoch.

Wraps `scripts.plots.plot_srnn_timeseries.plot_replay` to do:
  1. Phase A: per-checkpoint replay with trace-driven input. Each
     checkpoint produces (per variant): a `timeseries_*.png` and a
     `lyapunov_*.npz` containing `t_lya`, `local_lya`, `finite_lya`,
     `LLE`. The runtime knobs (`t_warm`, `t_end`, `lya_*`) match what
     postprocess.py uses for the same mode.
  2. Phase B: aggregate every variant's `LLE` across checkpoints into
     a single `ftle_vs_epoch_trace_w<W>e<E>.png` summary plot, one line
     per variant.

To avoid clobbering the long-window LLE NPZs that postprocess.py writes
(default `t_range=(-15, 30)`), this script tags its output filenames
with a window suffix: `lyapunov_<ckpt>_trace_w<warm>e<end>.npz` and
`timeseries_<ckpt>_trace_w<warm>e<end>.png`. `--skip-replay` reads only
those suffixed NPZs, so legacy postprocess outputs are never picked up.

Usage:
    python scripts/lyapunov_evolution.py <run_name>
    python scripts/lyapunov_evolution.py ring6-400e
    python scripts/lyapunov_evolution.py myrun --checkpoints init,last
    python scripts/lyapunov_evolution.py myrun --skip-replay
    python scripts/lyapunov_evolution.py myrun --variants srnn-e-only-skip-per-neuron
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from _runs import cache_dir  # noqa: E402
from postprocess import _resolve_replay_checkpoints  # noqa: E402
from plot_srnn_timeseries import _default_ckpt_tag, plot_replay  # noqa: E402

DEFAULT_TMP = cache_dir()


def _window_suffix(t_warm: float, t_end: float) -> str:
    """Filename-safe encoding of a (warm, end) seconds pair, e.g. (5, 15) -> 'w5e15'.

    Floats are rendered as integers when they round cleanly; otherwise
    decimal points are replaced with 'p' so the suffix stays
    filename-safe (e.g. 2.5s warm-up -> 'w2p5').
    """
    def _fmt(x: float) -> str:
        if abs(x - round(x)) < 1e-6:
            return str(int(round(x)))
        return f"{x:g}".replace(".", "p")
    return f"w{_fmt(t_warm)}e{_fmt(t_end)}"


def _epoch_for_tag(tag: str, run_dir: Path) -> int:
    """Map a ckpt-tag back to its epoch number for the x-axis.

    `init` → -1 (matches postprocess.load_snapshots convention).
    `last` → epoch from last.pt's saved `epoch` key.
    `epNNN` → NNN.
    """
    if tag == "init":
        return -1
    m = re.match(r"^ep(\d+)$", tag)
    if m:
        return int(m.group(1))
    if tag == "last":
        # Defer to load_snapshots' approach: read epoch from last.pt.
        last = run_dir / "last.pt"
        if last.exists():
            import torch
            sd = torch.load(last, map_location="cpu", weights_only=False)
            if isinstance(sd, dict) and "epoch" in sd:
                return int(sd["epoch"])
        return 10**9  # sentinel: place at far right if unknown
    raise ValueError(f"unrecognised ckpt tag: {tag!r}")


LEGACY_POSTPROCESS_SUFFIX = "w15e30"
"""Window suffix to retag legacy unsuffixed outputs with.

`scripts/postprocess.py` writes its replay outputs without any window
suffix and uses the defaults `t_range=(-15, 30)`. We retag those legacy
files to ``..._{LEGACY_POSTPROCESS_SUFFIX}.{ext}`` before this script's
first `plot_replay` call, so the original long-window LLEs aren't
silently overwritten when `plot_replay` reuses the unsuffixed filenames.

If a particular legacy file came from a *non-default* postprocess run,
the suffix label will be wrong but the file is still preserved (just
mislabelled). Acceptable — the alternative is silent loss.
"""


def migrate_legacy_outputs(run_dir: Path, mode: str) -> int:
    """One-time migration: retag any unsuffixed `lyapunov_*_<mode>.npz`
    or `timeseries_*_<mode>.png` (from postprocess.py defaults) with the
    `w15e30` suffix so the next `plot_replay` doesn't clobber them.

    Idempotent — only renames a legacy file if the destination doesn't
    already exist. Returns the count of files renamed.
    """
    n = 0
    npz_re = re.compile(rf"^lyapunov_(.+)_{re.escape(mode)}\.npz$")
    png_re = re.compile(rf"^timeseries_(.+)_{re.escape(mode)}\.png$")
    for variant_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
        for f in sorted(variant_dir.iterdir()):
            for regex in (npz_re, png_re):
                m = regex.match(f.name)
                if not m:
                    continue
                # Skip if already suffixed (defensive — unsuffixed regex
                # should not match a `..._w<W>e<E>` filename, but keep
                # this guard for clarity).
                tag = m.group(1)
                if re.search(r"_w\d+(p\d+)?e\d+(p\d+)?$", tag):
                    continue
                stem = f.stem  # e.g. lyapunov_init_trace
                dst = variant_dir / f"{stem}_{LEGACY_POSTPROCESS_SUFFIX}{f.suffix}"
                if dst.exists():
                    continue  # already migrated previously; leave both alone
                f.rename(dst)
                n += 1
    return n


def _rename_outputs_for_window(run_dir: Path, ckpt_tag: str, mode: str,
                                window_suffix: str) -> None:
    """After plot_replay writes <variant>/<file>_<tag>_<mode>.{png,npz},
    rename those files to ..._<tag>_<mode>_<suffix>.{png,npz}.

    Walks every variant subdir under run_dir. Idempotent — silently
    skips files that have already been renamed (or never existed).
    """
    base_pairs = [
        (f"lyapunov_{ckpt_tag}_{mode}.npz",
         f"lyapunov_{ckpt_tag}_{mode}_{window_suffix}.npz"),
        (f"timeseries_{ckpt_tag}_{mode}.png",
         f"timeseries_{ckpt_tag}_{mode}_{window_suffix}.png"),
    ]
    for variant_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
        for src_name, dst_name in base_pairs:
            src = variant_dir / src_name
            if not src.exists():
                continue
            dst = variant_dir / dst_name
            if dst.exists():
                dst.unlink()  # overwrite stale tagged file
            src.rename(dst)


def collect_lle_table(run_dir: Path, mode: str, window_suffix: str,
                      variants_filter: list[str] | None
                      ) -> tuple[list[str], dict[str, list[tuple[int, float]]]]:
    """Walk <run_dir>/<variant>/lyapunov_*_<mode>_<suffix>.npz and pull
    out (epoch, LLE) per variant.

    Returns (variant_names_in_seen_order, {variant: [(epoch, LLE), ...]}).
    """
    variants_seen: list[str] = []
    by_variant: dict[str, list[tuple[int, float]]] = {}
    pattern = f"lyapunov_*_{mode}_{window_suffix}.npz"
    # Tag extraction regex: lyapunov_<TAG>_<MODE>_<SUFFIX>.npz
    tag_re = re.compile(rf"^lyapunov_(.+)_{re.escape(mode)}_{re.escape(window_suffix)}\.npz$")
    for variant_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
        npzs = sorted(variant_dir.glob(pattern))
        if not npzs:
            continue
        if variants_filter and variant_dir.name not in variants_filter:
            continue
        v_name = variant_dir.name
        if v_name not in variants_seen:
            variants_seen.append(v_name)
            by_variant[v_name] = []
        for npz in npzs:
            m = tag_re.match(npz.name)
            if not m:
                print(f"  [aggregate] could not parse tag from {npz.name!r}, skipping")
                continue
            tag = m.group(1)
            try:
                epoch = _epoch_for_tag(tag, run_dir)
            except ValueError as e:
                print(f"  [aggregate] {e}, skipping {npz.name}")
                continue
            data = np.load(npz, allow_pickle=False)
            lle = float(data["LLE"])
            by_variant[v_name].append((epoch, lle))
        # Sort by epoch
        by_variant[v_name].sort(key=lambda p: p[0])
    return variants_seen, by_variant


def make_aggregate_plot(run_dir: Path, mode: str, window_suffix: str,
                        t_warm: float, t_end: float,
                        variants: list[str],
                        by_variant: dict[str, list[tuple[int, float]]]
                        ) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5))
    cmap = plt.get_cmap("tab10")
    n_points_total = 0
    for i, v in enumerate(variants):
        eps, lles = zip(*by_variant[v]) if by_variant[v] else ([], [])
        if not eps:
            continue
        n_points_total += len(eps)
        ax.plot(eps, lles, "o-", lw=1.4, ms=4, color=cmap(i % 10), label=v)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("epoch  (init plotted at -1)")
    ax.set_ylabel(f"FTLE [1/s]   ({t_warm:g} s warm-up + {t_end:g} s trace-driven)")
    ax.set_title(f"{run_dir.name} — Lyapunov stability across training "
                 f"(trace input, {window_suffix})")
    if variants:
        ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = run_dir / f"ftle_vs_epoch_{mode}_{window_suffix}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  wrote {out.name}  ({n_points_total} points across {len(variants)} variants)")
    return out


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("run_name",
                   help="run name; data is read from <tmp_dir>/<run_name>/")
    p.add_argument("--task", default="cheetah100")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--tmp-dir", default=str(DEFAULT_TMP),
                   help=f"local tmp root (default: {DEFAULT_TMP})")
    p.add_argument("--checkpoints", default="all",
                   help="checkpoint specifier: 'all' (default), 'init,last', "
                        "'epoch_NNN', or comma-list. See "
                        "scripts/postprocess.py:_resolve_replay_checkpoints.")
    p.add_argument("--variants", default=None,
                   help="comma-separated variant filter (default: all)")
    p.add_argument("--mode", default="trace",
                   help="replay drive mode (default: trace)")
    p.add_argument("--t-warm", type=float, default=5.0,
                   help="warm-up duration in seconds (zero-input; the "
                        "Benettin perturbation aligns with the most-expanding "
                        "direction during this window). Default: 5")
    p.add_argument("--t-end", type=float, default=15.0,
                   help="post-warmup driven simulation length in seconds. "
                        "FTLE is accumulated over this window. Default: 15")
    p.add_argument("--lya-M", type=int, default=5,
                   help="Benettin rescaling stride in cell-forward steps "
                        "(lya_dt = lya_M * h). Default: 5")
    p.add_argument("--lya-d0", type=float, default=1e-3,
                   help="Benettin perturbation amplitude. Default: 1e-3")
    p.add_argument("--lya-seed", type=int, default=0,
                   help="seed for Benettin's initial perturbation direction. "
                        "Default: 0")
    p.add_argument("--device", default="cpu",
                   help="device for plot_replay (default: cpu)")
    p.add_argument("--skip-replay", action="store_true",
                   help="skip the per-checkpoint replay phase; only run the "
                        "aggregation. Useful for re-rendering the summary "
                        "after the replays have already been computed.")
    return p.parse_args()


def main():
    args = parse_args()

    run_dir = Path(args.tmp_dir) / args.run_name
    if not run_dir.exists():
        sys.exit(f"ERROR: {run_dir} does not exist. Did you postprocess "
                 f"this run yet (or pass an existing local dir)?")

    suffix = _window_suffix(args.t_warm, args.t_end)
    t_range = (-float(args.t_warm), float(args.t_end))
    print(f"=== Lyapunov evolution: {args.run_name} ===")
    print(f"  run_dir = {run_dir}")
    print(f"  mode = {args.mode}, t_range = {t_range}, suffix = {suffix}")

    if not args.skip_replay:
        ckpt_paths = _resolve_replay_checkpoints(run_dir, args.checkpoints)
        if not ckpt_paths:
            sys.exit(f"ERROR: no checkpoints resolved from spec "
                     f"{args.checkpoints!r} in {run_dir}")
        # Retag any unsuffixed legacy postprocess outputs first — `plot_replay`
        # would otherwise clobber `lyapunov_<tag>_<mode>.npz` with our
        # short-window data before our rename step runs.
        n_migrated = migrate_legacy_outputs(run_dir, args.mode)
        if n_migrated:
            print(f"  [migrate] renamed {n_migrated} legacy "
                  f"unsuffixed `*_{args.mode}.{{npz,png}}` "
                  f"file(s) to `..._{LEGACY_POSTPROCESS_SUFFIX}.*` to "
                  f"preserve them across this run.")
        print(f"\n=== Phase A: per-checkpoint Benettin replays "
              f"({len(ckpt_paths)} checkpoints) ===")
        for i, ckpt in enumerate(ckpt_paths, 1):
            tag = _default_ckpt_tag(ckpt)
            print(f"\n--- [{i}/{len(ckpt_paths)}] ckpt={tag} ---")
            try:
                plot_replay(
                    ckpt_path=ckpt,
                    out_dir=run_dir,
                    mode=args.mode,
                    t_range=t_range,
                    device=args.device,
                    compute_lyapunov=True,
                    lya_M=args.lya_M,
                    lya_d0=args.lya_d0,
                    lya_seed=args.lya_seed,
                    ckpt_tag=tag,
                )
            except Exception as e:
                print(f"  [replay] FAILED for {ckpt.name}: {e}")
                continue
            # Rename the just-written files to embed the window suffix so
            # they don't collide with the postprocess long-window outputs.
            _rename_outputs_for_window(run_dir, tag, args.mode, suffix)
    else:
        print("\n[skip-replay] skipping Phase A — using existing NPZs only")

    print(f"\n=== Phase B: aggregate FTLE-vs-epoch ===")
    variants_filter = (
        [v.strip() for v in args.variants.split(",")] if args.variants else None
    )
    variants, by_variant = collect_lle_table(
        run_dir, args.mode, suffix, variants_filter,
    )
    if not variants:
        sys.exit(
            f"ERROR: no lyapunov_*_{args.mode}_{suffix}.npz files found in "
            f"{run_dir}'s variant subdirs. Did Phase A succeed for at least "
            f"one variant? Did you specify the right --t-warm/--t-end pair?"
        )

    print(f"  found {len(variants)} variant(s) with FTLE data:")
    for v in variants:
        eps = [e for (e, _) in by_variant[v]]
        lles = [l for (_, l) in by_variant[v]]
        rng = f"({min(lles):+.3f} → {max(lles):+.3f})" if lles else ""
        print(f"    {v}: {len(eps)} checkpoints, epochs "
              f"{eps[0] if eps else '?'}…{eps[-1] if eps else '?'}, LLE range {rng}")

    make_aggregate_plot(
        run_dir, args.mode, suffix,
        args.t_warm, args.t_end,
        variants, by_variant,
    )

    print(f"\n=== Done — outputs in {run_dir} ===")


if __name__ == "__main__":
    main()
