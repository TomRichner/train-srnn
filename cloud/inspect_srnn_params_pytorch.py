#!/usr/bin/env python3
"""Inspect SRNN parameters across Init/Best/Last checkpoints.

Reads PyTorch ``.pt`` checkpoint files saved by ``train_srnn.utils.checkpoint``
and produces comparison tables of SRNN dynamics parameters and weight statistics.

Handles both single SRNNCell models and BatchedSRNNCell models (K variants
stacked as ``(K, ...)`` tensors).

Usage:
    python3 cloud/inspect_srnn_params_pytorch.py --local results/har/srnn_32
    python3 cloud/inspect_srnn_params_pytorch.py --run myrun --experiment har --seed 1
    python3 cloud/inspect_srnn_params_pytorch.py --run myrun --experiment all --seed 1
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np


def softplus(x):
    """Numerically stable softplus."""
    return np.where(x > 20, x, np.log1p(np.exp(x)))


# ── Parameter definitions ────────────────────────────────────────────

# (state_dict_suffix, display_name, transform, applies_to_batched)
# All keys are relative to "cell." prefix in the state_dict.
SCALAR_PARAMS = [
    ("isp_tau_global",   "tau_global",   softplus),
    ("isp_tau_d",        "tau_d",        softplus),
    ("a_0",              "a_0",          None),
    ("isp_tau_a_E",      "tau_a_E",      softplus),
    ("isp_c_E",          "c_E",          softplus),
    ("c_0_E",            "c_0_E",        None),
    ("isp_tau_a_I",      "tau_a_I",      softplus),
    ("isp_c_I",          "c_I",          softplus),
    ("c_0_I",            "c_0_I",        None),
    ("isp_tau_b_rec_E",  "tau_b_rec_E",  softplus),
    ("isp_tau_b_rel_E",  "tau_b_rel_E",  softplus),
    ("isp_tau_b_rec_I",  "tau_b_rec_I",  softplus),
    ("isp_tau_b_rel_I",  "tau_b_rel_I",  softplus),
]

CHECKPOINT_TAGS = {
    "Init": "init",
    "Best": "best",
    "Last": "last",
}


def fmt(v):
    """Format a float for display."""
    if abs(v) < 0.001:
        return f"{v:.6f}"
    elif abs(v) > 100:
        return f"{v:.1f}"
    else:
        return f"{v:.4f}"


def fmt_ms(mean, std):
    """Format mean +/- std."""
    return f"{fmt(mean)}+/-{fmt(std)}"


# ── Single-variant extraction ────────────────────────────────────────

def extract_single_params(state_dict):
    """Extract SRNN parameters from a single-model state_dict.

    Returns dict of {display_name: scalar_value} for scalar/per-neuron params,
    plus weight statistics.
    """
    row = {}

    for key_suffix, display_name, transform in SCALAR_PARAMS:
        full_key = f"cell.{key_suffix}"
        if full_key not in state_dict:
            continue
        val = state_dict[full_key].cpu().numpy().flatten()
        if transform is not None:
            val = transform(val)
        # For scalar params, take mean; for per-neuron, also take mean
        row[display_name] = float(val.mean())
        if val.size > 1:
            row[f"{display_name}_std"] = float(val.std())

    # Weight statistics
    _extract_weight_stats(state_dict, row, prefix="cell.")

    return row


def _extract_weight_stats(state_dict, row, prefix="cell.", k_idx=None):
    """Extract weight matrix statistics into row dict."""
    W_key = f"{prefix}W_raw"
    W_in_key = f"{prefix}W_in"

    if W_key in state_dict:
        W_raw = state_dict[W_key].cpu().numpy()
        if k_idx is not None:
            W_raw = W_raw[k_idx]
        W_sp = softplus(W_raw)
        N = W_raw.shape[0]
        n_E = N // 2
        # After softplus + Dale's law: E columns positive, I columns negated
        W_E = W_sp[:, :n_E]
        W_I = -W_sp[:, n_E:]

        row["W_E_mean"] = float(W_E.mean())
        row["W_E_std"] = float(W_E.std())
        row["W_I_mean"] = float(W_I.mean())
        row["W_I_std"] = float(W_I.std())
        row["W_all_mean"] = float(np.concatenate([W_E, W_I], axis=1).mean())
        row["W_all_std"] = float(np.concatenate([W_E, W_I], axis=1).std())

    if W_in_key in state_dict:
        W_in = state_dict[W_in_key].cpu().numpy()
        if k_idx is not None:
            W_in = W_in[k_idx]
        row["W_in_mean"] = float(W_in.mean())
        row["W_in_std"] = float(W_in.std())


# ── Batched variant extraction ───────────────────────────────────────

def extract_batched_params(state_dict, ablation_names):
    """Extract parameters for each variant in a batched model.

    Returns list of K dicts, one per variant.
    """
    K = len(ablation_names)
    rows = []

    for k in range(K):
        row = {"variant": ablation_names[k]}

        for key_suffix, display_name, transform in SCALAR_PARAMS:
            full_key = f"cell.{key_suffix}"
            if full_key not in state_dict:
                continue
            tensor = state_dict[full_key].cpu().numpy()
            val = tensor[k].flatten()
            if transform is not None:
                val = transform(val)
            row[display_name] = float(val.mean())
            if val.size > 1:
                row[f"{display_name}_std"] = float(val.std())

        _extract_weight_stats(state_dict, row, prefix="cell.", k_idx=k)
        rows.append(row)

    return rows


# ── Checkpoint loading ───────────────────────────────────────────────

def load_ckpt(path):
    """Load a checkpoint, return (state_dict, ablation_names, config)."""
    import torch
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state_dict"]
    ablation_names = ckpt.get("ablation_names")
    config = ckpt.get("config", {})
    return state_dict, ablation_names, config


# ── Table generation ─────────────────────────────────────────────────

def generate_tables(ckpt_dir, out_dir=None):
    """Generate Init/Best/Last parameter comparison tables.

    Args:
        ckpt_dir: Directory containing init.pt, best.pt, last.pt
        out_dir: Optional output directory for markdown file
    """
    md_lines = []
    console_lines = []

    # Load available checkpoints
    stages = {}
    for stage_name, tag in CHECKPOINT_TAGS.items():
        path = os.path.join(ckpt_dir, f"{tag}.pt")
        if os.path.isfile(path):
            stages[stage_name] = load_ckpt(path)

    if not stages:
        print(f"  No checkpoints found in {ckpt_dir}")
        return

    # Detect single vs batched
    first_sd, first_names, first_cfg = next(iter(stages.values()))
    is_batched = first_names is not None and len(first_names) > 0

    if is_batched:
        _generate_batched_tables(stages, md_lines, console_lines)
    else:
        _generate_single_tables(stages, md_lines, console_lines)

    # Print console output
    for line in console_lines:
        print(line)

    # Write markdown if out_dir specified
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        md_path = os.path.join(out_dir, "srnn_params.md")
        with open(md_path, "w") as f:
            f.write("\n".join(md_lines) + "\n")
        print(f"\n  Markdown: {md_path}")


def _generate_single_tables(stages, md_lines, console_lines):
    """Generate tables for a single SRNN model."""
    md_lines.append("# SRNN Parameters: Init vs Best vs Last\n")

    # Collect all param names present across stages
    all_params = []
    for stage_name, (sd, _, _) in stages.items():
        row = extract_single_params(sd)
        for key in row:
            if key not in all_params:
                all_params.append(key)

    # Table: stages as columns, params as rows
    stage_names = list(stages.keys())
    header = "| Parameter | " + " | ".join(stage_names) + " |"
    sep = "|---|" + "|".join(["---:"] * len(stage_names)) + "|"
    md_lines.append(header)
    md_lines.append(sep)

    rows_data = {}
    for stage_name, (sd, _, _) in stages.items():
        rows_data[stage_name] = extract_single_params(sd)

    for param in all_params:
        vals = []
        for sn in stage_names:
            v = rows_data[sn].get(param)
            vals.append(fmt(v) if v is not None else "--")
        md_lines.append(f"| {param} | " + " | ".join(vals) + " |")

    md_lines.append("")

    # Console
    console_lines.append(f"\n{'='*60}")
    console_lines.append("  SRNN Parameters")
    console_lines.append(f"{'='*60}")
    col_w = 12
    hdr = f"{'Parameter':<20}" + "".join(f"{sn:>{col_w}}" for sn in stage_names)
    console_lines.append(hdr)
    console_lines.append("-" * len(hdr))
    for param in all_params:
        line = f"{param:<20}"
        for sn in stage_names:
            v = rows_data[sn].get(param)
            line += f"{fmt(v) if v is not None else '--':>{col_w}}"
        console_lines.append(line)


def _generate_batched_tables(stages, md_lines, console_lines):
    """Generate tables for a batched SRNN model (K variants)."""
    first_names = next(iter(stages.values()))[1]
    K = len(first_names)

    md_lines.append(f"# SRNN Parameters: Init vs Best vs Last (K={K} variants)\n")
    console_lines.append(f"\n{'='*60}")
    console_lines.append(f"  SRNN Parameters ({K} batched variants)")
    console_lines.append(f"{'='*60}")

    for stage_name, (sd, ablation_names, _) in stages.items():
        rows = extract_batched_params(sd, ablation_names)

        # Collect all param names
        all_params = []
        for row in rows:
            for key in row:
                if key != "variant" and key not in all_params:
                    all_params.append(key)

        md_lines.append(f"\n## {stage_name}\n")
        header = "| Variant | " + " | ".join(all_params) + " |"
        sep_line = "|---|" + "|".join(["---:"] * len(all_params)) + "|"
        md_lines.append(header)
        md_lines.append(sep_line)

        console_lines.append(f"\n  {stage_name}:")
        col_w = 12
        # Show first 8 params in console
        show_params = all_params[:8]
        hdr = f"{'Variant':<25}" + "".join(f"{p:>{col_w}}" for p in show_params)
        console_lines.append("  " + hdr)

        for row in rows:
            vals = [fmt(row[p]) if p in row else "--" for p in all_params]
            md_lines.append(f"| {row['variant']} | " + " | ".join(vals) + " |")

            line = f"{row['variant']:<25}"
            for p in show_params:
                v = row.get(p)
                line += f"{fmt(v) if v is not None else '--':>{col_w}}"
            console_lines.append("  " + line)

        md_lines.append("")


# ── Multi-experiment mode ────────────────────────────────────────────

EXPERIMENTS = [
    "har", "gesture", "occupancy", "smnist", "traffic",
    "power", "ozone_fixed", "person", "cheetah",
]


def find_srnn_dirs(base_dir, seed):
    """Find directories containing SRNN checkpoints under a run."""
    found = []
    for model_dir in sorted(os.listdir(base_dir)):
        model_path = os.path.join(base_dir, model_dir)
        if not os.path.isdir(model_path):
            continue
        for exp_dir in sorted(os.listdir(model_path)):
            seed_dir = os.path.join(model_path, exp_dir, f"seed{seed}")
            if not os.path.isdir(seed_dir):
                continue
            # Check for any .pt checkpoint
            if any(f.endswith(".pt") for f in os.listdir(seed_dir)):
                found.append((model_dir, exp_dir, seed_dir))
    return found


def generate_multi_experiment(base_dir, out_dir, run_name, experiment, seed):
    """Generate parameter tables across experiments for a GCS run."""
    if experiment == "all":
        srnn_dirs = find_srnn_dirs(base_dir, seed)
        if not srnn_dirs:
            print("  No SRNN checkpoints found.")
            return
        print(f"  Found checkpoints in {len(srnn_dirs)} model/experiment combinations")
    else:
        # Find all model dirs for this experiment
        srnn_dirs = []
        for model_dir in sorted(os.listdir(base_dir)):
            seed_dir = os.path.join(base_dir, model_dir, experiment, f"seed{seed}")
            if os.path.isdir(seed_dir) and any(f.endswith(".pt") for f in os.listdir(seed_dir)):
                srnn_dirs.append((model_dir, experiment, seed_dir))

    if not srnn_dirs:
        print(f"  No checkpoints found for experiment={experiment}, seed={seed}")
        return

    all_md = [f"# SRNN Parameters: {run_name}\n**Seed:** {seed}\n"]
    all_console = []

    for model_name, exp_name, ckpt_dir in srnn_dirs:
        all_md.append(f"\n---\n## {model_name} / {exp_name}\n")
        all_console.append(f"\n{'='*60}")
        all_console.append(f"  {model_name} / {exp_name}")
        all_console.append(f"{'='*60}")

        stages = {}
        for stage_name, tag in CHECKPOINT_TAGS.items():
            path = os.path.join(ckpt_dir, f"{tag}.pt")
            if os.path.isfile(path):
                stages[stage_name] = load_ckpt(path)

        if not stages:
            all_console.append("  (no checkpoints)")
            continue

        first_names = next(iter(stages.values()))[1]
        is_batched = first_names is not None and len(first_names) > 0

        if is_batched:
            _generate_batched_tables(stages, all_md, all_console)
        else:
            _generate_single_tables(stages, all_md, all_console)

    # Print console
    for line in all_console:
        print(line)

    # Write markdown
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        md_path = os.path.join(out_dir, "srnn_params.md")
        with open(md_path, "w") as f:
            f.write("\n".join(all_md) + "\n")
        print(f"\n  Markdown: {md_path}")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Inspect SRNN parameters across Init/Best/Last checkpoints"
    )
    parser.add_argument("--local", default=None,
                        help="Path to local checkpoint directory (e.g., results/har/srnn_32)")
    parser.add_argument("--run", default=None,
                        help="Run name in GCS (downloads to tmp/collect_results/<run>/)")
    parser.add_argument("--experiment", default="all",
                        help="Experiment name, or 'all' (default: all)")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out_dir", default=None,
                        help="Output directory for markdown (default: results/<run>/)")
    args = parser.parse_args()

    if args.local and not args.run:
        # Single directory mode
        generate_tables(args.local, out_dir=args.out_dir)
    elif args.run:
        # Multi-experiment GCS mode
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(script_dir)
        base_dir = args.local or os.path.join(
            project_dir, "tmp", "collect_results", args.run
        )

        if not os.path.exists(base_dir):
            print(f"  No local data at {base_dir}. Download first with collect_results.py --with-checkpoints")
            sys.exit(1)

        out_dir = args.out_dir or os.path.join(project_dir, "results", args.run)
        generate_multi_experiment(base_dir, out_dir, args.run, args.experiment, args.seed)
    else:
        parser.error("Must specify either --local or --run")


if __name__ == "__main__":
    main()
