#!/usr/bin/env python3
"""Collect and aggregate results from GCS.

Reads ``test_results.json`` files from each seed directory and aggregates
across seeds. Handles both single-model runs (JSON dict) and batched
ablation runs (JSON list of dicts with ``variant`` key).

Usage:
    python3 cloud/collect_results.py <run_name>
    python3 cloud/collect_results.py <run_name> --seeds 5 --csv out.csv
"""

import argparse
import csv
import datetime
import json
import os
import statistics
import subprocess
import sys


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


def collect(run_name, bucket, seeds=5, models=None, experiments=None):
    """Collect all results for a run from test_results.json files.

    Returns:
        (results, timing) where:
          results: list of dicts with keys: model, experiment, seed,
              best_epoch, test_loss, test_metric, metric_name
          timing: dict of (model, exp, seed) -> metadata dict
    """
    base = f"{bucket}/results-pytorch/{run_name}"
    results = []
    timing = {}

    model_dirs = models or [os.path.basename(p) for p in gcs_ls(base)]

    for model in model_dirs:
        exp_dirs = experiments or [os.path.basename(p) for p in gcs_ls(f"{base}/{model}")]
        for exp in exp_dirs:
            for seed in range(1, seeds + 1):
                seed_path = f"{base}/{model}/{exp}/seed{seed}"

                # Read run metadata if present
                meta_content = gcs_cat(f"{seed_path}/run_metadata.json")
                if meta_content:
                    try:
                        timing[(model, exp, seed)] = json.loads(meta_content)
                    except json.JSONDecodeError:
                        pass

                # Read test_results.json
                content = gcs_cat(f"{seed_path}/test_results.json")
                if not content:
                    continue

                try:
                    data = json.loads(content)
                except json.JSONDecodeError:
                    continue

                # Batched ablation: list of dicts with 'variant'
                if isinstance(data, list):
                    for entry in data:
                        results.append({
                            "model": entry.get("variant", model),
                            "experiment": exp,
                            "seed": seed,
                            "best_epoch": entry.get("best_epoch", 0),
                            "test_loss": entry.get("test_loss", 0),
                            "test_metric": entry.get("test_metric", 0),
                            "metric_name": entry.get("metric_name", ""),
                        })
                # Single model: dict
                else:
                    results.append({
                        "model": model,
                        "experiment": exp,
                        "seed": seed,
                        "best_epoch": data.get("best_epoch", 0),
                        "test_loss": data.get("test_loss", 0),
                        "test_metric": data.get("test_metric", 0),
                        "metric_name": data.get("metric_name", ""),
                    })

    return results, timing


# ── Timing helpers ────────────────────────────────────────────────────

def _parse_utc(s):
    """Parse a UTC ISO timestamp string to datetime."""
    return datetime.datetime.fromisoformat(s.replace("Z", "+00:00"))


def compute_timing_stats(timing):
    """Compute run-level timing stats from per-cell metadata."""
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
    """Format a timedelta as e.g. '2d 11h 50m'."""
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

CLASSIFICATION = {"har", "gesture", "occupancy", "smnist", "ozone_fixed", "person"}
HIGHER_IS_BETTER = CLASSIFICATION  # accuracy & F1


def _fmt_sigfigs(val, n=3):
    """Format a float to n significant figures."""
    from math import log10, floor
    if val == 0:
        return "0"
    digits = n - 1 - floor(log10(abs(val)))
    digits = max(digits, 0)
    return f"{val:.{digits}f}"


def print_table(results, experiments, models):
    """Print results as formatted table with mean +/- std across seeds."""
    # Group by (model, experiment)
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in results:
        grouped[(r["model"], r["experiment"])].append(r)

    cw = 22
    header = f"{'Model':<20}" + "".join(f"{exp:>{cw}}" for exp in experiments)
    print(header)
    print("-" * len(header))

    for model in models:
        parts = [f"{model:<20}"]
        for exp in experiments:
            entries = grouped.get((model, exp), [])
            if not entries:
                parts.append(f"{'--':>{cw}}")
                continue
            vals = [e["test_metric"] for e in entries]
            n = len(vals)
            mean = statistics.mean(vals)
            if exp in CLASSIFICATION:
                # Show as percentage
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
    parser.add_argument("--bucket", default="gs://liquidneuralnets-experiments")
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

    # Determine experiments and models present
    experiments = sorted(set(r["experiment"] for r in results))
    models = sorted(set(r["model"] for r in results))

    print(f"Collected {len(results)} results ({len(models)} models x {len(experiments)} experiments)")

    # Timing summary
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

    # Save CSV if requested
    if args.csv:
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "model", "experiment", "seed", "best_epoch",
                "test_loss", "test_metric", "metric_name",
            ])
            writer.writeheader()
            writer.writerows(results)
        print(f"\nSaved to {args.csv}")


if __name__ == "__main__":
    main()
