#!/usr/bin/env python3
"""Collect and aggregate results from GCS."""

import argparse
import csv
import json
import os
import subprocess
import sys
from collections import defaultdict

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
    """Collect all results for a run."""
    base = f"{bucket}/results-pytorch/{run_name}"
    results = []

    model_dirs = models or [os.path.basename(p) for p in gcs_ls(base)]

    for model in model_dirs:
        exp_dirs = experiments or [os.path.basename(p) for p in gcs_ls(f"{base}/{model}")]
        for exp in exp_dirs:
            for seed in range(1, seeds + 1):
                seed_path = f"{base}/{model}/{exp}/seed{seed}"

                # Try to read CSV
                csv_files = [p for p in gcs_ls(seed_path) if p.endswith(".csv")]
                if not csv_files:
                    continue

                content = gcs_cat(csv_files[0])
                if not content:
                    continue

                lines = content.strip().split("\n")
                if len(lines) < 2:
                    continue

                header = lines[0].split(",")
                values = lines[1].split(",")
                row = dict(zip(header, values))
                row["model"] = model
                row["experiment"] = exp
                row["seed"] = seed
                results.append(row)

    return results

def print_table(results, experiments, models):
    """Print results as formatted table."""
    # Group by model × experiment, average over seeds
    grouped = defaultdict(list)
    for r in results:
        key = (r["model"], r["experiment"])
        # Find the test metric column
        for col in r:
            if col.startswith("test_") and col != "test_loss":
                grouped[key].append(float(r[col]))
                break

    # Print header
    print(f"{'Model':<25}", end="")
    for exp in experiments:
        print(f"{exp:<12}", end="")
    print()
    print("-" * (25 + 12 * len(experiments)))

    for model in models:
        print(f"{model:<25}", end="")
        for exp in experiments:
            values = grouped.get((model, exp), [])
            if values:
                import statistics
                mean = statistics.mean(values)
                std = statistics.stdev(values) if len(values) > 1 else 0
                print(f"{mean:.3f}±{std:.3f} ", end="")
            else:
                print(f"{'—':<12}", end="")
        print()

def main():
    parser = argparse.ArgumentParser(description="Collect experiment results from GCS")
    parser.add_argument("run_name", help="Run name")
    parser.add_argument("--bucket", default="gs://liquidneuralnets-experiments")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--experiments", nargs="+", default=None)
    parser.add_argument("--csv", default=None, help="Output CSV path")
    args = parser.parse_args()

    results = collect(args.run_name, args.bucket, args.seeds, args.models, args.experiments)

    if not results:
        print("No results found.")
        return

    # Determine experiments and models present
    experiments = sorted(set(r["experiment"] for r in results))
    models = sorted(set(r["model"] for r in results))

    print(f"Collected {len(results)} results ({len(models)} models × {len(experiments)} experiments)")
    print()
    print_table(results, experiments, models)

    # Save CSV if requested
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nSaved to {args.csv}")

if __name__ == "__main__":
    main()
