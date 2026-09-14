#!/usr/bin/env python3
"""Collect and aggregate results from GCS.

For each seed, reads ``training_history.csv`` (per-epoch valid metrics) and
``test_history.csv`` (test metrics at checkpoint epochs). For each variant
independently, finds the checkpoint epoch with best validation metric and
reports the test metric at that epoch.

Works for both single-model runs (no ``variant`` column) and batched ablation
runs (one row per variant per epoch / checkpoint).

Usage:
    python3 cloud/collect_results.py <run_name>
    python3 cloud/collect_results.py <run_name> --seeds 5 --csv out.csv
"""

import argparse
import csv
import datetime
import io
import json
import os
import statistics
import subprocess
import sys
from collections import defaultdict


CLASSIFICATION = {"har", "gesture", "occupancy", "smnist", "ozone_fixed", "person"}
HIGHER_IS_BETTER = CLASSIFICATION  # accuracy & F1


def _default_bucket():
    env = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.gpu.env")
    for line in open(env):
        if line.strip().startswith("GCP_BUCKET="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return "gs://<bucket>"


def gcs_ls(path):
    """List GCS path contents."""
    try:
        result = subprocess.run(
            ["gcloud", "storage", "ls", path],
            capture_output=True, text=True, check=True
        )
        return [l.strip().rstrip("/") for l in result.stdout.strip().split("\n") if l.strip()]
    except subprocess.CalledProcessError:
        return []


def gcs_cat(path):
    """Read GCS file contents."""
    try:
        result = subprocess.run(
            ["gcloud", "storage", "cat", path],
            capture_output=True, text=True, check=True
        )
        return result.stdout
    except subprocess.CalledProcessError:
        return None


def _parse_csv(text):
    """Parse CSV text into list of dicts."""
    reader = csv.DictReader(io.StringIO(text))
    return list(reader)


def _pick_best(train_rows, test_rows, higher_is_better):
    """For each variant, find checkpoint epoch with best valid_metric and
    return the test metric at that epoch.

    Returns dict: {variant: {"best_epoch", "valid_metric", "test_loss", "test_metric"}}.
    If rows lack a ``variant`` column (single-model), uses key "__single__".
    """
    # Index test rows by (variant, epoch) — tag is ignored; epoch is unique per variant
    # except for the edge case where last epoch also lands on a periodic boundary.
    test_by_ve = {}
    for r in test_rows:
        variant = r.get("variant", "__single__")
        epoch = int(r["epoch"])
        # Prefer the 'last' tag if two rows share an epoch (keeps the last-eval weights)
        key = (variant, epoch)
        if key not in test_by_ve or r.get("tag") == "last":
            test_by_ve[key] = r

    # Group training rows by variant, filter to checkpoint epochs that have test data
    by_variant = defaultdict(list)
    for r in train_rows:
        variant = r.get("variant", "__single__")
        epoch = int(r["epoch"])
        if (variant, epoch) in test_by_ve:
            by_variant[variant].append(r)

    # Also consider init / last checkpoints (epoch 0 is covered by "init" + possibly epoch_000)
    # If there's a test row at an epoch with no training row (init), add a synthetic row
    # pointing to the valid metric from the nearest training row (or skip it; init has no valid).
    # Simplest: skip init from the "best" selection (it's random weights — never the best).

    result = {}
    for variant, rows in by_variant.items():
        key = "valid_metric"
        best_row = (max(rows, key=lambda r: float(r[key]))
                    if higher_is_better
                    else min(rows, key=lambda r: float(r[key])))
        epoch = int(best_row["epoch"])
        test_row = test_by_ve[(variant, epoch)]
        result[variant] = {
            "best_epoch": epoch,
            "valid_metric": float(best_row["valid_metric"]),
            "test_loss": float(test_row["test_loss"]),
            "test_metric": float(test_row["test_metric"]),
        }

    return result


def collect(run_name, bucket, seeds=5, models=None, experiments=None):
    """Collect per-seed results for a run.

    Returns (results, timing) where:
      results: list of dicts with keys: model, experiment, seed,
          best_epoch, valid_metric, test_loss, test_metric
      timing: dict of (model, exp, seed) -> run_metadata dict
    """
    base = f"{bucket}/results-pytorch/{run_name}"
    results = []
    timing = {}

    model_dirs = models or [os.path.basename(p) for p in gcs_ls(base)]

    for model in model_dirs:
        exp_dirs = experiments or [os.path.basename(p) for p in gcs_ls(f"{base}/{model}")]
        for exp in exp_dirs:
            higher_is_better = exp in HIGHER_IS_BETTER
            for seed in range(1, seeds + 1):
                seed_path = f"{base}/{model}/{exp}/seed{seed}"

                meta_content = gcs_cat(f"{seed_path}/run_metadata.json")
                if meta_content:
                    try:
                        timing[(model, exp, seed)] = json.loads(meta_content)
                    except json.JSONDecodeError:
                        pass

                train_content = gcs_cat(f"{seed_path}/training_history.csv")
                test_content = gcs_cat(f"{seed_path}/test_history.csv")
                if not train_content or not test_content:
                    continue

                try:
                    train_rows = _parse_csv(train_content)
                    test_rows = _parse_csv(test_content)
                except Exception:
                    continue

                selected = _pick_best(train_rows, test_rows, higher_is_better)
                for variant, d in selected.items():
                    results.append({
                        "model": variant if variant != "__single__" else model,
                        "experiment": exp,
                        "seed": seed,
                        "best_epoch": d["best_epoch"],
                        "valid_metric": d["valid_metric"],
                        "test_loss": d["test_loss"],
                        "test_metric": d["test_metric"],
                    })

    return results, timing


# ── Timing helpers ────────────────────────────────────────────────────

def _parse_utc(s):
    """Parse a UTC ISO timestamp string to datetime."""
    return datetime.datetime.fromisoformat(s.replace("Z", "+00:00"))


def compute_timing_stats(timing):
    """Compute run-level timing stats."""
    starts, ends, durations = [], [], []
    for meta in timing.values():
        completed = meta.get("completed", meta.get("failed_at"))
        dur = meta.get("duration_seconds", 0)
        if completed:
            ct = _parse_utc(completed)
            ends.append(ct)
            if dur:
                starts.append(ct - datetime.timedelta(seconds=dur))
                durations.append(dur)
    if not ends:
        return None
    return {
        "started": min(starts) if starts else None,
        "completed": max(ends),
        "wall_clock": max(ends) - min(starts) if starts else None,
        "cpu_hours": sum(durations) / 3600,
        "n_cells": len(ends),
    }


def _fmt_timedelta(td):
    total_sec = int(td.total_seconds())
    days = total_sec // 86400
    hours = (total_sec % 86400) // 3600
    minutes = (total_sec % 3600) // 60
    parts = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes or not parts:
        parts.append(f"{minutes}m")
    return " ".join(parts)


# ── Table formatting ─────────────────────────────────────────────────

def _fmt_sigfigs(val, n=3):
    from math import log10, floor
    if val == 0:
        return "0"
    digits = n - 1 - floor(log10(abs(val)))
    digits = max(digits, 0)
    return f"{val:.{digits}f}"


def print_table(results, experiments, models):
    """Print mean +/- std test_metric across seeds."""
    grouped = defaultdict(list)
    for r in results:
        grouped[(r["model"], r["experiment"])].append(r["test_metric"])

    cw = 22
    header = f"{'Model':<25}" + "".join(f"{exp:>{cw}}" for exp in experiments)
    print(header)
    print("-" * len(header))

    for model in models:
        parts = [f"{model:<25}"]
        for exp in experiments:
            vals = grouped.get((model, exp), [])
            if not vals:
                parts.append(f"{'--':>{cw}}")
                continue
            n = len(vals)
            mean = statistics.mean(vals)
            if exp in CLASSIFICATION:
                if n > 1:
                    std = statistics.stdev(vals)
                    cell = f"{mean*100:.2f}% +/-{std*100:.2f}"
                else:
                    cell = f"{mean*100:.2f}%"
            else:
                if n > 1:
                    std = statistics.stdev(vals)
                    cell = f"{_fmt_sigfigs(mean)} +/-{_fmt_sigfigs(std)}"
                else:
                    cell = _fmt_sigfigs(mean)
            if n < 5:
                cell += f" (n={n})"
            parts.append(f"{cell:>{cw}}")
        print("".join(parts))


def main():
    parser = argparse.ArgumentParser(description="Collect experiment results from GCS")
    parser.add_argument("run_name", help="Run name")
    parser.add_argument("--bucket", default=_default_bucket(), help="GCS bucket (default: cloud/config.gpu.env)")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--experiments", nargs="+", default=None)
    parser.add_argument("--csv", default=None, help="Output CSV path")
    args = parser.parse_args()

    results, timing = collect(
        args.run_name, args.bucket, args.seeds, args.models, args.experiments,
    )

    if not results:
        print("No results found.")
        return

    experiments = sorted(set(r["experiment"] for r in results))
    models = sorted(set(r["model"] for r in results))

    print(f"Collected {len(results)} results "
          f"({len(models)} models x {len(experiments)} experiments)")

    stats = compute_timing_stats(timing) if timing else None
    if stats:
        if stats["started"]:
            print(f"  Started:    {stats['started'].strftime('%Y-%m-%d %H:%M')} UTC")
        print(f"  Completed:  {stats['completed'].strftime('%Y-%m-%d %H:%M')} UTC")
        if stats["wall_clock"]:
            print(f"  Wall-clock: {_fmt_timedelta(stats['wall_clock'])}")
        print(f"  CPU-hours:  {stats['cpu_hours']:.0f}h ({stats['n_cells']} cells)")

    print()
    print_table(results, experiments, models)

    if args.csv:
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "model", "experiment", "seed", "best_epoch",
                "valid_metric", "test_loss", "test_metric",
            ])
            writer.writeheader()
            writer.writerows(results)
        print(f"\nSaved to {args.csv}")


if __name__ == "__main__":
    main()
