"""Post-run analysis: download a run (Modal Volume or GCS), then plots and tables.

    python scripts/postprocess.py <run_name>                     # task cheetah100, seed 1
    python scripts/postprocess.py <gce_run> --source gcs         # runs made on GCE VMs
    python scripts/postprocess.py ring2x5-100e --task cheetah100
    python scripts/postprocess.py myrun --skip-download --variants srnn-skip,srnn-no-adapt-skip
    python scripts/postprocess.py later-run --prepend-runs earlier-run   # concatenate a resumed run

Output layout under $SRNN_CACHE_DIR/<run_name>/ (default $SRNN_HOME/cache):
    init.pt, last.pt, epoch_*.pt, *.csv, *.json, training_log.txt
    curves_{skip,no-skip}.png (+ semilogy_, log_log_, semilogy_direct_ variants)
    lr_schedule.png, weight_evolution.png, report.pdf
    <variant>/tau_evolution.png, W_EI_evolution.png, param_table.txt,
              timeseries_<ckpt>_<mode>.png, lyapunov_<ckpt>_<mode>.npz
"""
from __future__ import annotations

import argparse
import copy
import csv
import os
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from _runs import (cache_dir, download_from_volume, gcloud_storage, read_bucket,  # noqa: E402
                   results_volume, variant_names as _ckpt_names)

DEFAULT_TMP = cache_dir()


# =============================================================================
# Phase 1: download
# =============================================================================

def ensure_local_run(args) -> Path:
    """Download (or reuse) artifacts for <run_name>/srnn/<task>/seed<seed>/."""
    local = Path(args.tmp_dir) / args.run_name
    local.mkdir(parents=True, exist_ok=True)

    if args.skip_download:
        print(f"[download] --skip-download set, using {local} as-is")
        return local

    run_path = f"results-pytorch/{args.run_name}/srnn/{args.task}/seed{args.seed}"
    if args.source == "modal":
        print(f"[download] enumerating srnn-results:/{run_path}/")
        try:
            new, skipped = download_from_volume(results_volume(), run_path, local)
        except FileNotFoundError as exc:
            sys.exit(f"ERROR: nothing at {exc}. Wrong run_name/task/seed, or a GCE run "
                     "(pass --source gcs)?")
        print(f"[download] done: {new} new, {skipped} cached, total {new+skipped} files")
        _require_run_files(local)
        return local

    bucket = args.bucket or read_bucket()
    remote_prefix = f"{bucket}/{run_path}"
    print(f"[download] enumerating {remote_prefix}/*")
    res = gcloud_storage("ls", f"{remote_prefix}/")
    if res.returncode != 0:
        sys.exit(f"ERROR: gcloud storage ls failed:\n{res.stderr}\n"
                 f"Check that {remote_prefix} exists and you're authenticated.")

    remote_files = [line.strip() for line in res.stdout.splitlines() if line.strip().startswith("gs://")]
    if not remote_files:
        sys.exit(f"ERROR: no files at {remote_prefix}/. Wrong run_name/task/seed?")

    new = 0
    skipped = 0
    for rf in remote_files:
        if rf.endswith("/"):  # subdirectory marker
            continue
        name = rf.rsplit("/", 1)[-1]
        dest = local / name
        if dest.exists() and dest.stat().st_size > 0:
            skipped += 1
            continue
        print(f"[download] {name}")
        cp = gcloud_storage("cp", rf, str(dest), capture=False)
        if cp.returncode != 0:
            sys.exit(f"ERROR: download of {rf} failed (exit {cp.returncode})")
        new += 1
    print(f"[download] done: {new} new, {skipped} cached, total {new+skipped} files")

    _require_run_files(local)
    return local


def _require_run_files(local: Path) -> None:
    for fn in ("last.pt", "training_history.csv"):
        if not (local / fn).exists():
            sys.exit(f"ERROR: required file {fn} missing from {local}. Run incomplete?")


# =============================================================================
# Phase 1b (optional): concatenate multiple runs into a synthetic merged dir
# =============================================================================

def _run_epoch_count(run_dir: Path) -> int:
    """How many epochs this run actually completed.

    Read training_history.csv for the first variant; return max(epoch)+1.
    Falls back to last.pt's "epoch" key if the CSV is missing/empty.
    """
    csv_path = run_dir / "training_history.csv"
    if csv_path.exists():
        rows = list(csv.DictReader(open(csv_path)))
        if rows:
            v0 = rows[0]["variant"]
            eps = [int(r["epoch"]) for r in rows if r["variant"] == v0]
            if eps:
                return max(eps) + 1
    last = run_dir / "last.pt"
    if last.exists():
        sd = torch.load(last, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "epoch" in sd:
            return int(sd["epoch"]) + 1
    sys.exit(f"ERROR: cannot determine epoch count for {run_dir}")


def _variant_names_for(run_dir: Path) -> list[str]:
    """Read variant_names from any checkpoint in run_dir."""
    for cand in ("last.pt", "init.pt"):
        p = run_dir / cand
        if p.exists():
            sd = torch.load(p, map_location="cpu", weights_only=False)
            if isinstance(sd, dict) and _ckpt_names(sd):
                return _ckpt_names(sd)
    # CSV fallback
    csv_path = run_dir / "training_history.csv"
    if csv_path.exists():
        rows = list(csv.DictReader(open(csv_path)))
        # Preserve order of first appearance
        seen: list[str] = []
        for r in rows:
            v = r["variant"]
            if v not in seen:
                seen.append(v)
        if seen:
            return seen
    sys.exit(f"ERROR: cannot determine variant_names for {run_dir}")


def _validate_chain_compat(run_dirs: list[Path]) -> list[str]:
    """All runs in a concat chain must share variant_names."""
    canonical = _variant_names_for(run_dirs[0])
    for rd in run_dirs[1:]:
        names = _variant_names_for(rd)
        if names != canonical:
            sys.exit(
                f"ERROR: variant_names mismatch in concat chain.\n"
                f"  {run_dirs[0].name}: {canonical}\n"
                f"  {rd.name}: {names}"
            )
    return canonical


def _concat_training_history(run_dirs: list[Path], offsets: list[int], out_path: Path) -> None:
    """Concat training_history.csv rows; offset each row's epoch column."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = None
    merged_rows: list[dict] = []
    for rd, off in zip(run_dirs, offsets):
        csv_path = rd / "training_history.csv"
        if not csv_path.exists():
            sys.exit(f"ERROR: {csv_path} missing")
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            if fieldnames is None:
                fieldnames = reader.fieldnames
            for r in reader:
                r["epoch"] = str(int(r["epoch"]) + off)
                merged_rows.append(r)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged_rows)


def _concat_test_history(run_dirs: list[Path], offsets: list[int], out_path: Path) -> None:
    """Concat test_history.csv: keep `init` only from earliest, `last` only from latest;
    other tags (e.g. periodic mid-run test eval) are preserved with offsets."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = None
    merged: list[dict] = []
    n_runs = len(run_dirs)
    for i, (rd, off) in enumerate(zip(run_dirs, offsets)):
        csv_path = rd / "test_history.csv"
        if not csv_path.exists():
            continue
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            if fieldnames is None:
                fieldnames = reader.fieldnames
            for r in reader:
                tag = r.get("tag", "")
                if tag == "init" and i != 0:
                    continue
                if tag == "last" and i != n_runs - 1:
                    continue
                r["epoch"] = str(int(r["epoch"]) + off)
                merged.append(r)
    if fieldnames is None:
        return
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged)


def _symlink_force(target: Path, link_path: Path) -> None:
    if link_path.exists() or link_path.is_symlink():
        link_path.unlink()
    # Use absolute target so the symlink works regardless of cwd
    link_path.symlink_to(target.resolve())


def _build_concat_run_dir(prepend_dirs: list[Path], primary_dir: Path, out_dir: Path) -> Path:
    """Materialise a synthetic merged run dir. Symlinks for checkpoints, real
    files for the concatenated CSVs. Idempotent: surgically replaces only the
    files this builder owns (init.pt, last.pt, epoch_*.pt, the two history CSVs)
    so prior plot outputs and replay artefacts in subdirs are preserved across
    reruns (e.g. when iterating with --skip-replay)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in list(out_dir.glob("epoch_*.pt")):
        if stale.is_symlink() or stale.is_file():
            stale.unlink()
    for owned in ("init.pt", "last.pt", "training_history.csv", "test_history.csv"):
        p = out_dir / owned
        if p.exists() or p.is_symlink():
            p.unlink()

    chain = list(prepend_dirs) + [primary_dir]
    counts = [_run_epoch_count(rd) for rd in chain]
    offsets = [sum(counts[:i]) for i in range(len(chain))]

    # init.pt — earliest run's true initial state (epoch -1 / pre-training)
    init_src = chain[0] / "init.pt"
    if init_src.exists():
        _symlink_force(init_src, out_dir / "init.pt")

    # last.pt — final trained state from the primary run. Cannot symlink:
    # the saved `epoch` key reflects the primary run's internal counter
    # (e.g. 119 for resume120), but in the merged timeline it must be
    # offsets[-1] + that. Rewrite the key so load_snapshots places the
    # "last" snapshot at the correct merged epoch.
    last_src = primary_dir / "last.pt"
    if last_src.exists():
        sd = torch.load(last_src, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "epoch" in sd:
            sd["epoch"] = int(sd["epoch"]) + offsets[-1]
        torch.save(sd, out_dir / "last.pt")

    # epoch_*.pt — merge with offsets. Skip the resume runs' "init" copies
    # (which would duplicate the prior run's last state).
    for i, (rd, off) in enumerate(zip(chain, offsets)):
        for ep_file in sorted(rd.glob("epoch_*.pt")):
            ep = int(ep_file.stem.split("_")[1])
            new_ep = ep + off
            dst = out_dir / f"epoch_{new_ep:03d}.pt"
            _symlink_force(ep_file, dst)

    # Concatenated histories
    _concat_training_history(chain, offsets, out_dir / "training_history.csv")
    _concat_test_history(chain, offsets, out_dir / "test_history.csv")

    total_epochs = sum(counts)
    chain_label = " -> ".join(rd.name for rd in chain)
    print(f"[concat] {chain_label}  ({total_epochs} epochs total)")
    return out_dir


# =============================================================================
# Common loaders
# =============================================================================

def load_snapshots(run_dir: Path):
    """Returns (snaps, variant_names) where snaps = [(label, x_epoch, model_state_dict), ...]
    and x_epoch is the true epoch index from the checkpoint (init -> -1)."""
    raw = []
    init = run_dir / "init.pt"
    if init.exists():
        raw.append(("init", -1, torch.load(init, map_location="cpu", weights_only=False)))
    for p in sorted(run_dir.glob("epoch_*.pt")):
        ep = int(p.stem.split("_")[1])
        raw.append((f"ep{ep:03d}", ep, torch.load(p, map_location="cpu", weights_only=False)))
    last = run_dir / "last.pt"
    if last.exists():
        sd = torch.load(last, map_location="cpu", weights_only=False)
        ep = int(sd["epoch"]) if isinstance(sd, dict) and "epoch" in sd else -1
        raw.append(("last", ep, sd))

    out = []
    for label, ep, sd in raw:
        ms = sd.get("model_state_dict", sd) if isinstance(sd, dict) else sd
        out.append((label, ep, ms))
    if not raw:
        return [], []
    variant_names = _ckpt_names(raw[-1][2])
    return out, variant_names


def load_history_by_variant(path: Path) -> dict[str, list[dict]]:
    rows = list(csv.DictReader(open(path)))
    by_variant: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_variant[r["variant"]].append(r)
    return by_variant


def x_axis(snaps) -> list[int]:
    return [ep for (_, ep, _) in snaps]


# =============================================================================
# Phase 2A: loss / metric curves (split skip vs no-skip)
# =============================================================================

def _is_skip(variant: str) -> bool:
    return "-skip" in variant


# Matches the "-seed<int>" suffix the factory appends to variant names when
# `batched_ablation_seeds` is set (train_srnn/models/factory.py:_SEED_SEP).
_SEED_SUFFIX_RE = re.compile(r"^(?P<base>.+)-seed(?P<seed>\d+)$")


def split_variant_seed(variant: str) -> tuple[str, int | None]:
    """('srnn-skip-seed3') -> ('srnn-skip', 3); ('srnn-skip') -> ('srnn-skip', None)."""
    m = _SEED_SUFFIX_RE.match(variant)
    return (m.group("base"), int(m.group("seed"))) if m else (variant, None)


def variant_style_map(variants: list[str], cmap=None):
    """One colour per *base* variant — every seed of a variant shares it.

    Returns ``(colour_by_variant, label_by_variant)``. Only the first variant
    of each base carries a legend label, so a 2-variant x 5-seed run gets a
    2-entry legend rather than 10 near-identical ones. Runs with no seed
    suffix behave exactly as before: one colour and one legend entry each.
    """
    cmap = cmap or plt.get_cmap("tab10")
    base_index: dict[str, int] = {}
    for v in variants:
        base, _ = split_variant_seed(v)
        base_index.setdefault(base, len(base_index))
    n_seeds = Counter(split_variant_seed(v)[0] for v in variants)

    colour, label, labelled = {}, {}, set()
    for v in variants:
        base, _ = split_variant_seed(v)
        colour[v] = cmap(base_index[base] % 10)
        if base in labelled:
            label[v] = "_nolegend_"
        else:
            labelled.add(base)
            label[v] = base if n_seeds[base] == 1 else f"{base} ({n_seeds[base]} seeds)"
    return colour, label


def _draw_curves(axes, th, variant_subset, cmap):
    colour, label = variant_style_map(variant_subset, cmap)
    for v in variant_subset:
        rows = th[v]
        ep = [int(r["epoch"]) + 1 for r in rows]
        c, lb = colour[v], label[v]
        axes[0, 0].plot(ep, [float(r["train_loss"]) for r in rows], color=c, label=lb, lw=1)
        axes[0, 1].plot(ep, [float(r["valid_loss"]) for r in rows], color=c, label=lb, lw=1)
        axes[1, 0].plot(ep, [float(r["train_metric"]) for r in rows], color=c, label=lb, lw=1)
        axes[1, 1].plot(ep, [float(r["valid_metric"]) for r in rows], color=c, label=lb, lw=1)
    axes[0, 0].set_title("train_loss");  axes[0, 1].set_title("valid_loss")
    axes[1, 0].set_title("train_metric"); axes[1, 1].set_title("valid_metric")
    for ax in axes.flat:
        ax.set_xlabel("epoch"); ax.grid(alpha=0.3)
    axes[0, 1].legend(fontsize=7, loc="best")


def _save_curves_group(run_dir: Path, group_label: str, variants_in_group: list[str], th, test):
    """Emit linear / semilogy / log_log curves for a single group of variants."""
    if not variants_in_group:
        return
    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    _draw_curves(axes, th, variants_in_group, cmap)

    final = {r["variant"]: float(r["test_loss"]) for r in test if r["tag"] == "last"}
    init = {r["variant"]: float(r["test_loss"]) for r in test if r["tag"] == "init"}
    txt = f"Final test_loss (init → last) [{group_label}]:\n" + "\n".join(
        f"  {v}: {init.get(v, float('nan')):.5f} → {final.get(v, float('nan')):.5f}"
        for v in variants_in_group
    )
    fig.suptitle(f"{run_dir.name} — {group_label} variants", fontsize=11)
    fig.text(0.01, 0.01, txt, family="monospace", fontsize=8, va="bottom")
    plt.tight_layout(rect=[0, 0.18, 1, 0.97])

    plt.savefig(run_dir / f"curves_{group_label}.png", dpi=120)
    print(f"  wrote curves_{group_label}.png")
    for ax in axes.flat:
        ax.set_yscale("log")
    plt.savefig(run_dir / f"semilogy_curves_{group_label}.png", dpi=120)
    print(f"  wrote semilogy_curves_{group_label}.png")
    for ax in axes.flat:
        ax.set_xscale("log")
        ax.set_xlabel("epoch + 1 (log)")
    plt.savefig(run_dir / f"log_log_curves_{group_label}.png", dpi=120)
    print(f"  wrote log_log_curves_{group_label}.png")
    plt.close(fig)


def _save_semilogy_direct(run_dir: Path, group_label: str, variants_in_group: list[str], th):
    """Direct ax.semilogy(...) version — keeps the 'no NaN dropped' panels for sanity."""
    if not variants_in_group:
        return
    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    colour, label = variant_style_map(variants_in_group, cmap)
    for v in variants_in_group:
        rows = th[v]
        ep = [int(r["epoch"]) + 1 for r in rows]
        c, lb = colour[v], label[v]
        axes[0, 0].semilogy(ep, [float(r["train_loss"]) for r in rows], color=c, label=lb, lw=1)
        axes[0, 1].semilogy(ep, [float(r["valid_loss"]) for r in rows], color=c, label=lb, lw=1)
        axes[1, 0].semilogy(ep, [float(r["train_metric"]) for r in rows], color=c, label=lb, lw=1)
        axes[1, 1].semilogy(ep, [float(r["valid_metric"]) for r in rows], color=c, label=lb, lw=1)
    axes[0, 0].set_title("train_loss");  axes[0, 1].set_title("valid_loss")
    axes[1, 0].set_title("train_metric"); axes[1, 1].set_title("valid_metric")
    for ax in axes.flat:
        ax.set_xlabel("epoch + 1"); ax.grid(alpha=0.3, which="both")
    axes[0, 1].legend(fontsize=7, loc="best")
    fig.suptitle(f"{run_dir.name} — semilogy_direct ({group_label})", fontsize=11)
    plt.tight_layout()
    plt.savefig(run_dir / f"semilogy_direct_curves_{group_label}.png", dpi=120)
    print(f"  wrote semilogy_direct_curves_{group_label}.png")
    plt.close(fig)


def plot_curves_split(run_dir: Path):
    csv_path = run_dir / "training_history.csv"
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        print("  [curves] no training_history.csv — skipping (init-only run?)")
        return
    th = load_history_by_variant(csv_path)
    if not th:
        print("  [curves] training_history.csv empty — skipping")
        return
    test_csv = run_dir / "test_history.csv"
    te = list(csv.DictReader(open(test_csv))) if test_csv.exists() else []
    variants = sorted(th.keys())
    skip = [v for v in variants if _is_skip(v)]
    noskip = [v for v in variants if not _is_skip(v)]
    _save_curves_group(run_dir, "skip", skip, th, te)
    _save_curves_group(run_dir, "no-skip", noskip, th, te)
    _save_semilogy_direct(run_dir, "skip", skip, th)
    _save_semilogy_direct(run_dir, "no-skip", noskip, th)


# =============================================================================
# Phase 2B: LR schedule
# =============================================================================

def plot_lr_schedule(run_dir: Path):
    csv_path = run_dir / "training_history.csv"
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        print("  [lr_schedule] no training_history.csv — skipping")
        return
    rows = list(csv.DictReader(open(csv_path)))
    if not rows:
        print("  [lr_schedule] empty training_history.csv")
        return
    v0 = rows[0]["variant"]
    ep, lr = [], []
    for r in rows:
        if r["variant"] != v0:
            continue
        ep.append(int(r["epoch"]))
        lr.append(float(r["lr"]))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(ep, lr, marker="o", ms=3); axes[0].set_title("LR schedule (linear)"); axes[0].set_xlabel("epoch")
    axes[1].semilogy(ep, lr, marker="o", ms=3); axes[1].set_title("LR schedule (semilogy)"); axes[1].set_xlabel("epoch")
    axes[2].loglog([e + 1 for e in ep], lr, marker="o", ms=3); axes[2].set_title("LR schedule (log-log)"); axes[2].set_xlabel("epoch + 1")
    for ax in axes:
        ax.set_ylabel("lr"); ax.grid(alpha=0.3, which="both")
    plt.tight_layout()
    out = run_dir / "lr_schedule.png"
    plt.savefig(out, dpi=120); plt.close(fig)
    print(f"  wrote lr_schedule.png  (peak={max(lr):.3e}, final={lr[-1]:.3e})")


# =============================================================================
# Phase 2C: weight evolution (top-16 most-changed params)
# =============================================================================

def plot_weight_evolution(run_dir: Path, snaps):
    pts = sorted(run_dir.glob("epoch_*.pt"))
    if not pts:
        print("  [weight_evolution] no epoch_*.pt found"); return
    epochs, snapshots = [], []
    for p in pts:
        epochs.append(int(p.stem.split("_")[1]))
        sd = torch.load(p, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "model_state_dict" in sd:
            sd = sd["model_state_dict"]
        snapshots.append(sd)

    keys = list(snapshots[0].keys())
    norms: dict[str, np.ndarray] = {}
    for k in keys:
        try:
            ts = [s[k].float() for s in snapshots]
        except (KeyError, AttributeError):
            continue
        if ts[0].ndim == 0:
            continue
        n = np.array([t.norm().item() for t in ts])
        if np.allclose(n, n[0]):
            continue
        norms[k] = n

    if not norms:
        print("  [weight_evolution] no varying params"); return

    rel = sorted(norms.items(),
                 key=lambda kv: abs(kv[1][-1] - kv[1][0]) / (abs(kv[1][0]) + 1e-9),
                 reverse=True)[:16]
    fig, ax = plt.subplots(figsize=(13, 8))
    for k, n in rel:
        ax.plot(epochs, n, marker="o", ms=3, lw=1, label=k)
    ax.set_xlabel("epoch"); ax.set_ylabel("|param| (L2)")
    ax.set_title(f"Top-16 most-changed param L2 norms ({run_dir.name}, {len(snapshots)} snapshots)")
    ax.legend(fontsize=7, loc="best", ncol=2); ax.grid(alpha=0.3)
    plt.tight_layout()
    out = run_dir / "weight_evolution.png"
    plt.savefig(out, dpi=120); plt.close(fig)
    print(f"  wrote weight_evolution.png")
    print("  Top-16 most-changed params (final / initial L2 norm):")
    for k, n in rel:
        pct = 100 * (n[-1] - n[0]) / (n[0] + 1e-9)
        print(f"    {k:60s}  {n[0]:.4g} -> {n[-1]:.4g}  ({pct:+.2f}%)")


# =============================================================================
# Phase 3: per-variant tau / W_EI / param table
# =============================================================================

def effective_taus(ms, k):
    out = {}

    def eff(vec_key, gain_key):
        if vec_key not in ms:
            return None
        gain = torch.exp(ms[gain_key])[k].item()
        vec = F.softplus(ms[vec_key])[k].numpy()
        return gain * vec

    out["tau_d"] = eff("cell.isp_tau_d_vec", "cell.log_tau_d_gain")
    out["tau_a_E"] = eff("cell.isp_tau_a_E_vec", "cell.log_tau_a_E_gain")
    out["tau_a_I"] = eff("cell.isp_tau_a_I_vec", "cell.log_tau_a_I_gain")
    out["tau_b_rec_E"] = eff("cell.isp_tau_b_rec_E_vec", "cell.log_tau_b_rec_E_gain")
    out["tau_b_rel_E"] = eff("cell.isp_tau_b_rel_E_vec", "cell.log_tau_b_rel_E_gain")
    out["tau_b_rec_I"] = eff("cell.isp_tau_b_rec_I_vec", "cell.log_tau_b_rec_I_gain")
    out["tau_b_rel_I"] = eff("cell.isp_tau_b_rel_I_vec", "cell.log_tau_b_rel_I_gain")
    return {k_: v for k_, v in out.items() if v is not None}


def effective_W(ms, k):
    """Reproduce SRNNCell._effective_W for variant k."""
    W_raw = ms["cell.W_raw"][k]
    gain = ms["cell.log_W_raw_gain"][k].exp().item()
    dales_mask = ms["cell.dales_mask"][k].item()
    dales_signs = ms["cell.dales_signs"][k]
    sparsity = ms["cell.sparsity_masks"][k]
    if dales_mask >= 0.5:
        W_eff = dales_signs.unsqueeze(0) * F.softplus(W_raw)
    else:
        W_eff = W_raw
    return (gain * W_eff * sparsity).numpy(), dales_signs.numpy(), sparsity.numpy()


def effective_W_in(ms, k):
    W_in = ms["cell.W_in"][k]
    gain = ms["cell.W_in_gain"][k].item()
    mask = ms.get("cell.W_in_mask", torch.ones(1, W_in.shape[0], 1))                     # (1, N, 1) shared across variants
    return (gain * W_in * mask.squeeze(0)).numpy()


def effective_c(ms, k, side):
    vec_key = f"cell.isp_c_{side}_vec"; gain_key = f"cell.log_c_{side}_gain"
    if vec_key not in ms:
        return None
    gain = torch.exp(ms[gain_key])[k].item()
    return (gain * F.softplus(ms[vec_key])[k]).numpy()


def effective_a_0(ms, k):
    return (ms["cell.a_0_vec"][k] + ms["cell.a_0_scalar"][k]).numpy()


def is_per_neuron(ms, k) -> bool:
    """True iff variant k trains its per-neuron parameter tensors."""
    return bool(ms["cell.per_neuron_mask"][k].item() > 0.5)


def effective_W_out(ms, k):
    """readout_weight * W_out_gain (the bias is not gain-scaled)."""
    return (ms["readout_weight"][k] * ms["W_out_gain"][k]).numpy()


def active_js(ms, k, side: str, mechanism: str = "sfa") -> list[int]:
    """Active SFA or STD timescale indices, excluding padded components."""
    m = ms.get(f"cell.{mechanism}_{side}_mask")
    if m is None:
        return []
    v = m[k, 0, :].numpy()
    return [int(j) for j in np.where(v != 0)[0]]


def is_tau_active(ms, k, key):
    """Whether the given effective tau drives the loss for variant k."""
    if key == "tau_d":
        return True
    if key == "tau_a_E":
        m = ms.get("cell.sfa_E_mask"); return bool(m is not None and m[k].any().item())
    if key == "tau_a_I":
        m = ms.get("cell.sfa_I_mask"); return bool(m is not None and m[k].any().item())
    if key in ("tau_b_rec_E", "tau_b_rel_E"):
        m = ms.get("cell.std_E_mask"); return bool(m is not None and m[k].any().item())
    if key in ("tau_b_rec_I", "tau_b_rel_I"):
        m = ms.get("cell.std_I_mask"); return bool(m is not None and m[k].any().item())
    return True


def _grey_overlay(ax):
    """20%-opacity grey covering the whole axes box (data-coords-independent)."""
    ax.add_patch(plt.Rectangle(
        (0, 0), 1, 1, transform=ax.transAxes,
        facecolor="gray", alpha=0.20, zorder=10, edgecolor="none",
    ))


def plot_tau_evolution(out_dir: Path, run_label: str, snaps, k, name):
    xs = x_axis(snaps)
    last_ms = snaps[-1][2]
    per_neuron = is_per_neuron(last_ms, k)

    taus = effective_taus(last_ms, k)
    keys = [key for key in taus if is_tau_active(last_ms, k, key)]
    fig, axes = plt.subplots((len(keys) + 2) // 3, 3, figsize=(12, 3 * ((len(keys) + 2) // 3)), squeeze=False)
    for ax, key in zip(axes.flat, keys):
        mechanism = "sfa" if key.startswith("tau_a_") else "std"
        indices = active_js(last_ms, k, key[-1], mechanism) if taus[key].ndim == 2 else [None]
        for color, j in enumerate(indices):
            samples = [effective_taus(ms, k)[key] for _, _, ms in snaps]
            samples = [arr if j is None else arr[:, j] for arr in samples]
            means = np.array([arr.mean() for arr in samples])
            spreads = np.array([arr.std() for arr in samples])
            label = "neurons" if j is None else f"timescale {j + 1}"
            ax.plot(xs, means, "o-", ms=4, color=f"C{color}", label=label)
            # Fixed initialization heterogeneity remains under shared learning.
            ax.fill_between(xs, means - spreads, means + spreads, alpha=.18, color=f"C{color}")
        ax.set_yscale("log")
        ax.set_title(f"effective {key}")
        ax.set_xlabel("epoch"); ax.set_ylabel("seconds"); ax.grid(alpha=.3)
        ax.legend(fontsize=8)
    for ax in list(axes.flat)[len(keys):]:
        ax.axis("off")
    mode = "per-neuron learning" if per_neuron else "shared learning"
    fig.suptitle(f"{run_label} — time constants ({name}) — {mode}; neuron mean ±std", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_dir / "tau_evolution.png", dpi=120); plt.close(fig)
    print(f"    wrote {name}/tau_evolution.png")


def plot_W_EI_evolution(out_dir: Path, run_label: str, snaps, k, name):
    xs = x_axis(snaps)
    e_means, i_means, e_abs, i_abs = [], [], [], []
    e_sds, i_sds, e_abs_sd, i_abs_sd = [], [], [], []
    for (_, _, ms) in snaps:
        W_eff, signs, sp = effective_W(ms, k)
        e_cols = signs > 0; i_cols = signs < 0; nz = sp != 0
        e_block = W_eff[:, e_cols]; i_block = W_eff[:, i_cols]
        e_nz = nz[:, e_cols]; i_nz = nz[:, i_cols]
        # W_raw is per-element trainable regardless of per_neuron, so the
        # spread across synapses is real and worth a band.
        if e_nz.any():
            v = e_block[e_nz]
            e_means.append(float(v.mean())); e_sds.append(float(v.std()))
            e_abs.append(float(np.abs(v).mean())); e_abs_sd.append(float(np.abs(v).std()))
        else:
            e_means.append(np.nan); e_sds.append(np.nan)
            e_abs.append(np.nan); e_abs_sd.append(np.nan)
        if i_nz.any():
            v = i_block[i_nz]
            i_means.append(float(v.mean())); i_sds.append(float(v.std()))
            i_abs.append(float(np.abs(v).mean())); i_abs_sd.append(float(np.abs(v).std()))
        else:
            i_means.append(np.nan); i_sds.append(np.nan)
            i_abs.append(np.nan); i_abs_sd.append(np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    def _band(ax, m, s, colour, label):
        m = np.asarray(m, dtype=float); s = np.asarray(s, dtype=float)
        ax.plot(xs, m, "o-", color=colour, ms=3, label=label)
        ax.fill_between(xs, m - s, m + s, alpha=0.18, color=colour, lw=0)

    _band(axes[0], e_means, e_sds, "C3", "mean E (signed) ±std")
    _band(axes[0], i_means, i_sds, "C0", "mean I (signed) ±std")
    axes[0].axhline(0, color="k", lw=0.5)
    axes[0].set_title("Mean non-zero W_eff (signed)"); axes[0].set_xlabel("epoch"); axes[0].grid(alpha=0.3); axes[0].legend()
    _band(axes[1], e_abs, e_abs_sd, "C3", "|E| ±std")
    _band(axes[1], i_abs, i_abs_sd, "C0", "|I| ±std")
    axes[1].set_title("Mean |W_eff| non-zero"); axes[1].set_xlabel("epoch"); axes[1].grid(alpha=0.3); axes[1].legend()
    fig.suptitle(f"{run_label} — recurrent E vs I weights ({name})", fontsize=11)
    plt.tight_layout()
    out = out_dir / "W_EI_evolution.png"
    plt.savefig(out, dpi=120); plt.close(fig)
    print(f"    wrote {name}/W_EI_evolution.png")


def _offset_series(snaps, k, getter, js):
    """Collect per-j (or collapsed) mean series plus std, for an offset term."""
    lines, sds = {}, {}
    for (_, _, ms) in snaps:
        arr = getter(ms)
        if arr is None:
            return None, None
        if arr.ndim > 1 and js:
            for j in js:
                lines.setdefault(f"j={j}", []).append(float(arr[:, j].mean()))
                sds.setdefault(f"j={j}", []).append(float(arr[:, j].std()))
        else:
            lines.setdefault("", []).append(float(arr.mean()))
            sds.setdefault("", []).append(float(arr.std()))
    return lines, sds


def plot_offsets_evolution(out_dir: Path, run_label: str, snaps, k, name):
    """Activation threshold, including retained initialization heterogeneity."""
    xs = x_axis(snaps)
    last_ms = snaps[-1][2]
    per_neuron = is_per_neuron(last_ms, k)

    panels = []
    panels.append(("a_0 (threshold)", *_offset_series(
        snaps, k, lambda ms: effective_a_0(ms, k), None)))

    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4),
                             squeeze=False)
    for ax, (title, lines, sds) in zip(axes[0], panels):
        for c, (lbl, ys) in enumerate(sorted(lines.items())):
            ys = np.asarray(ys)
            ax.plot(xs, ys, "o-", color=f"C{c}", ms=3, label=lbl or None)
            s = np.asarray(sds[lbl])
            ax.fill_between(xs, ys - s, ys + s, alpha=0.18,
                            color=f"C{c}", lw=0)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(title); ax.set_xlabel("epoch"); ax.grid(alpha=0.3)
        if len(lines) > 1:
            ax.legend(fontsize=8)
    mode = "mean ±std" if per_neuron else "scalar-driven (vec frozen)"
    fig.suptitle(f"{run_label} — activation threshold ({name}) — {mode}",
                 fontsize=11)
    plt.tight_layout()
    out = out_dir / "offsets_evolution.png"
    plt.savefig(out, dpi=120); plt.close(fig)
    print(f"    wrote {name}/offsets_evolution.png")


def plot_W_io_evolution(out_dir: Path, run_label: str, snaps, k, name):
    """Input / output weights and the output bias, mean ±std.

    Separate from W_EI_evolution.png, which covers the recurrent matrix. All
    three quantities here are per-element trainable irrespective of
    per_neuron -- that flag only gates the adaptation *_vec tensors -- so the
    band is always meaningful. For W_in_eff the std across synapses is 8x its
    mean and for W_out_eff 35x, i.e. the mean alone says very little.
    """
    xs = x_axis(snaps)
    series = {key: ([], []) for key in ("W_in_eff", "W_out_eff", "readout_bias")}

    for (_, _, ms) in snaps:
        # W_in: 75 of 300 rows are mask-zeroed by the neuron partition, so
        # restrict to live entries or the mean is dragged toward zero.
        win = effective_W_in(ms, k)
        win = win[win != 0]
        wout = effective_W_out(ms, k).reshape(-1)
        bias = ms["readout_bias"][k].numpy().reshape(-1)
        for key, v in (("W_in_eff", win), ("W_out_eff", wout),
                       ("readout_bias", bias)):
            series[key][0].append(float(v.mean()) if v.size else np.nan)
            series[key][1].append(float(v.std()) if v.size else np.nan)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    titles = {"W_in_eff": "W_in_eff (live entries)",
              "W_out_eff": "W_out_eff = readout_weight × W_out_gain",
              "readout_bias": "readout_bias (not gain-scaled)"}
    for ax, (key, (m, s)) in zip(axes, series.items()):
        m = np.asarray(m); s = np.asarray(s)
        ax.plot(xs, m, "o-", color="C2", ms=3, label="mean")
        ax.fill_between(xs, m - s, m + s, alpha=0.18, color="C2", lw=0,
                        label="±std")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(titles[key]); ax.set_xlabel("epoch"); ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"{run_label} — input / output weights ({name})", fontsize=11)
    plt.tight_layout()
    out = out_dir / "W_io_evolution.png"
    plt.savefig(out, dpi=120); plt.close(fig)
    print(f"    wrote {name}/W_io_evolution.png")


def stats(arr, mask=None):
    a = np.asarray(arr).astype(np.float64)
    if mask is not None:
        a = a[mask]
    if a.size == 0:
        return float("nan"), float("nan"), 0
    return float(a.mean()), float(a.std()), int(a.size)


def write_param_table(out_dir: Path, run_label: str, snaps, k, name):
    init_ms = snaps[0][2]; last_ms = snaps[-1][2]
    rows = []

    def add(label, a0, a1):
        m0, s0, n0 = stats(a0); m1, s1, _ = stats(a1)
        rows.append((label, f"{m0:+.5g}", f"{m1:+.5g}", f"{s0:.4g}", f"{s1:.4g}", str(n0)))

    t0 = effective_taus(init_ms, k); t1 = effective_taus(last_ms, k)

    # Per-timescale tau_a_E / tau_a_I (rest of taus collapsed)
    for key in ("tau_d", "tau_a_E", "tau_a_I", "tau_b_rec_E", "tau_b_rel_E", "tau_b_rec_I", "tau_b_rel_I"):
        if key not in t0:
            continue
        if key == "tau_a_E":
            for j in active_js(last_ms, k, "E"):
                add(f"tau_a_E[j={j}] (s)", t0[key][:, j], t1[key][:, j])
        elif key == "tau_a_I":
            for j in active_js(last_ms, k, "I"):
                add(f"tau_a_I[j={j}] (s)", t0[key][:, j], t1[key][:, j])
        elif key.startswith("tau_b_"):
            for j in active_js(last_ms, k, key[-1], "std"):
                add(f"{key}[m={j}] (s)", t0[key][:, j], t1[key][:, j])
        else:
            add(f"{key} (s)", t0[key], t1[key])

    # W_eff (split E / I source)
    W0, signs0, sp0 = effective_W(init_ms, k); W1, _, sp1 = effective_W(last_ms, k)
    e_mask0 = (sp0 != 0) & (signs0 > 0)[None, :]
    i_mask0 = (sp0 != 0) & (signs0 < 0)[None, :]
    e_mask1 = (sp1 != 0) & (signs0 > 0)[None, :]
    i_mask1 = (sp1 != 0) & (signs0 < 0)[None, :]
    add("W_eff (E src, signed)", W0[e_mask0], W1[e_mask1])
    add("W_eff (I src, signed)", W0[i_mask0], W1[i_mask1])
    add("|W_eff| (E src)", np.abs(W0[e_mask0]), np.abs(W1[e_mask1]))
    add("|W_eff| (I src)", np.abs(W0[i_mask0]), np.abs(W1[i_mask1]))

    # W_in_eff
    Win0 = effective_W_in(init_ms, k); Win1 = effective_W_in(last_ms, k)
    add("W_in_eff (input neurons)", Win0[Win0 != 0], Win1[Win1 != 0])

    # One total SFA budget per neuron (active populations only)
    for side in ("E", "I"):
        active = active_js(last_ms, k, side)
        if not active:
            continue
        c0 = effective_c(init_ms, k, side); c1 = effective_c(last_ms, k, side)
        if c0 is not None:
            add(f"c_{side} (total SFA budget)", c0, c1)

    # Threshold a_0
    add("a_0 (threshold)", effective_a_0(init_ms, k), effective_a_0(last_ms, k))

    # Readout
    # Effective, i.e. gain-applied — the table header promises post-transform
    # values, and this is what W_io_evolution.png plots. The bias is NOT
    # gain-scaled (sequence_model.py:252), so it stays raw.
    add("W_out_eff (readout × gain)", effective_W_out(init_ms, k), effective_W_out(last_ms, k))
    add("readout_bias", init_ms["readout_bias"][k].numpy(), last_ms["readout_bias"][k].numpy())

    if "ic.ic" in init_ms and init_ms["ic.ic"].shape[0] == last_ms["cell.skip_flags"].shape[0]:
        add("ic.ic (per-variant init)", init_ms["ic.ic"][k].numpy(), last_ms["ic.ic"][k].numpy())

    header = f"{'effective parameter':<32s} {'init mean':>13s} {'final mean':>13s} {'init std':>11s} {'final std':>11s} {'n':>10s}"
    lines = [
        f"# {run_label} — variant {name} (k={k})",
        f"# Effective values (post-transform). Means/stds computed over the indicated dimension.",
        header, "-" * len(header),
    ]
    for r in rows:
        lines.append(f"{r[0]:<32s} {r[1]:>13s} {r[2]:>13s} {r[3]:>11s} {r[4]:>11s} {r[5]:>10s}")
    out = out_dir / "param_table.txt"
    out.write_text("\n".join(lines) + "\n")
    print(f"    wrote {name}/param_table.txt")


# =============================================================================
# Orchestration
# =============================================================================

def build_report(run_dir: Path, variant_names: list[str], variants_filter: list[str] | None):
    """Assemble a single PDF report:
       - log-log loss curves first (skip + no-skip)
       - then per variant: tau_evolution + W_EI_evolution on one page,
         param_table on the next page, with the variant name as an H1 at
         the top-left of every page.
    """
    if not shutil.which("pandoc"):
        print("[report] pandoc not in PATH — skipping PDF assembly")
        return
    if not shutil.which("lualatex"):
        print("[report] lualatex not in PATH — skipping PDF assembly (install MacTeX)")
        return

    selected = variant_names if not variants_filter else [n for n in variant_names if n in variants_filter]

    md = []
    # YAML frontmatter — geometry + small font so the wide param tables fit US-letter
    md += [
        f"# Run: {run_dir.name}",
        "",
        "## Log-log loss curves",
        "",
    ]
    for grp in ("skip", "no-skip"):
        png = run_dir / f"log_log_curves_{grp}.png"
        if png.exists():
            md.append(f"![{grp} variants — log-log loss/metric](log_log_curves_{grp}.png){{ width=95% }}")
            md.append("")

    for name in selected:
        out = run_dir / name
        tau_p = out / "tau_evolution.png"
        w_p = out / "W_EI_evolution.png"
        wio_p = out / "W_io_evolution.png"
        off_p = out / "offsets_evolution.png"
        tbl_p = out / "param_table.txt"
        if not (tau_p.exists() or w_p.exists() or wio_p.exists()
                or off_p.exists() or tbl_p.exists()):
            continue
        md += ["\\newpage", "", f"# {name}", ""]
        if tau_p.exists():
            md.append(f"![Tau evolution]({name}/tau_evolution.png){{ width=95% }}")
            md.append("")
        if w_p.exists():
            md.append(f"![W\\_EI evolution]({name}/W_EI_evolution.png){{ width=95% }}")
            md.append("")
        if wio_p.exists():
            md.append(f"![W\\_io evolution]({name}/W_io_evolution.png){{ width=95% }}")
            md.append("")
        if off_p.exists():
            md.append(f"![Offsets and threshold]({name}/offsets_evolution.png){{ width=95% }}")
            md.append("")
        if tbl_p.exists():
            md += ["\\newpage", "", f"# {name} — parameter table", "", "```text"]
            md.append(tbl_p.read_text().rstrip())
            md += ["```", ""]

    run_dir_abs = run_dir.resolve()
    md_path = run_dir_abs / "report.md"
    md_path.write_text("\n".join(md))
    pdf_path = run_dir_abs / "report.pdf"

    # Header-includes file: small monospace font for the param tables, and
    # fancyhdr running header showing the current section in the upper-left.
    header_tex = run_dir_abs / "_report_header.tex"
    header_tex.write_text(
        "\\usepackage{fvextra}\n"
        "\\DefineVerbatimEnvironment{Highlighting}{Verbatim}"
        "{breaklines,fontsize=\\scriptsize,commandchars=\\\\\\{\\}}\n"
        "\\usepackage{fancyhdr}\n"
        "\\pagestyle{fancy}\n"
        "\\fancyhf{}\n"
        "\\fancyhead[L]{\\leftmark}\n"
        "\\renewcommand{\\sectionmark}[1]{\\markboth{#1}{}}\n"
        "\\renewcommand{\\headrulewidth}{0pt}\n"
    )

    cmd = [
        "pandoc", str(md_path),
        "-f", "markdown+tex_math_single_backslash",
        "--pdf-engine=lualatex",
        "-V", "geometry:top=0.5in,bottom=0.5in,left=0.5in,right=0.5in",
        "-V", "papersize=letter",
        "-V", "fontsize=9pt",
        "-H", str(header_tex),
        "-o", str(pdf_path),
    ]
    # cwd=run_dir_abs so that relative image paths in report.md resolve correctly
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(run_dir_abs))
    if res.returncode != 0:
        print(f"[report] pandoc failed (exit {res.returncode}):")
        print(res.stderr[-2000:])
        return
    print(f"  wrote report.md")
    print(f"  wrote report.pdf")


def run_per_variant(run_dir: Path, snaps, variant_names: list[str], variants_filter: list[str] | None):
    if not variant_names:
        print("[per-variant] no variant_names found in last.pt — skipping per-variant phase")
        return
    selected = variant_names if not variants_filter else [n for n in variant_names if n in variants_filter]
    if variants_filter and not selected:
        print(f"[per-variant] WARNING: --variants {variants_filter} matched none of {variant_names}")
    print(f"[per-variant] {len(selected)} variant(s)")
    for k, name in enumerate(variant_names):
        if name not in selected:
            continue
        out = run_dir / name
        out.mkdir(parents=True, exist_ok=True)
        print(f"  [{k}] {name}")
        plot_tau_evolution(out, run_dir.name, snaps, k, name)
        plot_W_EI_evolution(out, run_dir.name, snaps, k, name)
        plot_W_io_evolution(out, run_dir.name, snaps, k, name)
        plot_offsets_evolution(out, run_dir.name, snaps, k, name)
        write_param_table(out, run_dir.name, snaps, k, name)


def _resolve_replay_checkpoints(run_dir: Path, spec: str) -> list[Path]:
    """Resolve a comma-separated checkpoint specifier to a list of files in run_dir.

    Tokens:
        'init'                   -> run_dir/init.pt
        'last'                   -> run_dir/last.pt
        'epoch_NNN', 'epNNN', 'NNN' (digits) -> run_dir/epoch_<NNN with leading zeros>.pt
        'all'                    -> init + every epoch_*.pt (numeric order) + last
    Tokens that don't resolve to an existing file print a warning and are skipped.
    Returned list deduplicates while preserving first-seen order.
    """
    tokens = [t.strip() for t in spec.split(",") if t.strip()]
    out: list[Path] = []
    seen: set[Path] = set()

    def _add(p: Path):
        rp = p.resolve()
        if rp in seen:
            return
        if not p.exists():
            print(f"  [replay] WARNING: checkpoint not found: {p.name} (skipping)")
            return
        seen.add(rp)
        out.append(p)

    for tok in tokens:
        if tok == "init":
            _add(run_dir / "init.pt")
        elif tok == "last":
            _add(run_dir / "last.pt")
        elif tok == "all":
            _add(run_dir / "init.pt")
            for p in sorted(run_dir.glob("epoch_*.pt"),
                            key=lambda q: int(q.stem.split("_")[1])):
                _add(p)
            _add(run_dir / "last.pt")
        else:
            # Try to parse an epoch index from any of: epoch_NNN, epNNN, NNN
            digits = None
            if tok.startswith("epoch_") and tok[len("epoch_"):].isdigit():
                digits = tok[len("epoch_"):]
            elif tok.startswith("ep") and tok[len("ep"):].isdigit():
                digits = tok[len("ep"):]
            elif tok.isdigit():
                digits = tok
            if digits is None:
                print(f"  [replay] WARNING: unrecognised checkpoint token: {tok!r} (skipping)")
                continue
            _add(run_dir / f"epoch_{int(digits):03d}.pt")
    return out


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_name", help="run name (top-level folder under results-pytorch/)")
    p.add_argument("--task", default="cheetah100", help="task name (default: cheetah100)")
    p.add_argument("--seed", type=int, default=1, help="seed (default: 1)")
    p.add_argument("--source", choices=["modal", "gcs"], default="modal",
                   help="where the run lives: the srnn-results Modal Volume (default) or GCS")
    p.add_argument("--bucket", default=None,
                   help="GCS bucket for --source gcs (default: cloud/config.gpu.env)")
    p.add_argument("--tmp-dir", default=str(DEFAULT_TMP), help=f"local destination root (default: {DEFAULT_TMP})")
    p.add_argument("--skip-download", action="store_true", help="skip the download; use existing local files only")
    p.add_argument("--variants", default=None, help="comma-separated variant names to render per-variant (default: all)")
    p.add_argument("--no-report", action="store_true", help="skip the consolidated PDF report")
    p.add_argument("--skip-replay", action="store_true",
                   help="skip the forward-replay time-series phase")
    p.add_argument("--replay-modes", default="no_input,step,trace",
                   help="comma-separated subset of {no_input,step,trace} (default: all three)")
    p.add_argument("--replay-t-start", type=float, default=-15.0,
                   help="replay start time in seconds (negative = zero-input warm-up; default: -15)")
    p.add_argument("--replay-t-end", type=float, default=30.0,
                   help="replay end time in seconds (default: 30)")
    p.add_argument("--replay-plot-fs", type=float, default=25.0,
                   help="replay plot decimation rate in Hz (default: 25)")
    p.add_argument("--replay-checkpoints", default="init,last",
                   help="comma-separated list of checkpoints to replay. Tokens: "
                        "'init', 'last', 'epoch_NNN' (or 'epNNN' or 'NNN'), or 'all' "
                        "(= init + every epoch_*.pt + last). Default: init,last.")
    p.add_argument("--no-replay-lyapunov", action="store_true",
                   help="skip the Benettin LLE pass during forward-replay")
    p.add_argument("--lya-M", type=int, default=5,
                   help="Benettin rescaling stride in cell-forward steps (default: 5 → lya_dt=5h)")
    p.add_argument("--lya-d0", type=float, default=1e-3,
                   help="Benettin perturbation magnitude (default: 1e-3)")
    p.add_argument("--lya-seed", type=int, default=0,
                   help="RNG seed for the Benettin initial perturbation (default: 0)")
    p.add_argument("--prepend-runs", default=None,
                   help="comma-separated list of earlier run names to concat before "
                        "<run_name> (e.g. --prepend-runs overnight-cl250 to extend "
                        "a resumed run). Builds <cache>/<primary>__concat/ and "
                        "runs all phases on the merged dir.")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"=== Phase 1: download ===")
    run_dir = ensure_local_run(args)

    if args.prepend_runs:
        print(f"\n=== Phase 1b: concat with prepended runs ===")
        prepend_names = [s.strip() for s in args.prepend_runs.split(",") if s.strip()]
        if args.run_name in prepend_names:
            sys.exit(f"ERROR: --prepend-runs cannot include the primary run '{args.run_name}'")
        prepend_dirs = []
        for name in prepend_names:
            sub_args = copy.copy(args)
            sub_args.run_name = name
            prepend_dirs.append(ensure_local_run(sub_args))
        _validate_chain_compat(prepend_dirs + [run_dir])
        merged_dir = Path(args.tmp_dir) / f"{args.run_name}__concat"
        run_dir = _build_concat_run_dir(prepend_dirs, run_dir, merged_dir)
        print(f"[concat] merged dir: {run_dir}")

    print(f"\n=== Phase 2: run-level plots ===")
    plot_curves_split(run_dir)
    plot_lr_schedule(run_dir)

    snaps, variant_names = load_snapshots(run_dir)
    print(f"  {len(snaps)} snapshots: {[s[0] for s in snaps]}")
    plot_weight_evolution(run_dir, snaps)

    print(f"\n=== Phase 3: per-variant outputs ===")
    variants_filter = [s.strip() for s in args.variants.split(",")] if args.variants else None
    run_per_variant(run_dir, snaps, variant_names, variants_filter)

    if not args.skip_replay:
        print(f"\n=== Phase 3b: forward-replay time-series ===")
        ckpts = _resolve_replay_checkpoints(run_dir, args.replay_checkpoints)
        if not ckpts:
            print(f"  no replay checkpoints resolved in {run_dir} — skipping")
        else:
            from plot_srnn_timeseries import plot_replay
            modes = [m.strip() for m in args.replay_modes.split(",") if m.strip()]
            for ckpt_path in ckpts:
                for m in modes:
                    try:
                        plot_replay(
                            ckpt_path=ckpt_path,
                            out_dir=run_dir,
                            mode=m,
                            t_range=(args.replay_t_start, args.replay_t_end),
                            plot_fs=args.replay_plot_fs,
                            compute_lyapunov=not args.no_replay_lyapunov,
                            lya_M=args.lya_M,
                            lya_d0=args.lya_d0,
                            lya_seed=args.lya_seed,
                        )
                    except Exception as e:
                        print(f"  [replay] ckpt={ckpt_path.name} mode={m} failed: {e}")

    if not args.no_report:
        print(f"\n=== Phase 4: consolidated PDF report ===")
        build_report(run_dir, variant_names, variants_filter)

    print(f"\n=== Done — outputs in {run_dir} ===")


if __name__ == "__main__":
    main()
