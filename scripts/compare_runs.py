"""Compare two local run directories: config, histories and checkpoint tensors.

    uv run python scripts/compare_runs.py <reference_dir> <candidate_dir> [--json out.json]

Used for the GPU regression checks in docs/regression_baseline_20ep.md. Reports:

- config keys that differ, ignoring where a run was written (``output_dir``,
  ``run_name``, ``paths``) and the ``device``/``init_ckpt`` plumbing;
- per numeric column of ``training_history.csv`` and ``test_history.csv``, the
  maximum absolute and relative difference over matching rows (the CSVs round to
  six decimals, so differences below about 1e-6 are not resolved);
- for ``init.pt`` and ``last.pt`` (and every ``epoch_*.pt`` with ``--all-epochs``),
  the maximum absolute difference over all model tensors, and the relative L2
  norm of the difference.

It exits nonzero if row sets, columns or tensor names differ. It does not judge
tolerances; compare against the repeat-run spread of the same setup.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import torch

IGNORED_CONFIG = {"output_dir", "run_name", "paths", "device", "init_ckpt"}
KEYS = {"training_history.csv": ("epoch", "variant"),
        "test_history.csv": ("epoch", "tag", "variant")}


def _flatten(value, prefix=""):
    if isinstance(value, dict):
        out = {}
        for key, sub in value.items():
            out.update(_flatten(sub, f"{prefix}.{key}" if prefix else str(key)))
        return out
    return {prefix: value}


def config_diff(ref: dict, cand: dict) -> dict:
    a = {k: v for k, v in _flatten(ref).items() if k.split(".")[0] not in IGNORED_CONFIG}
    b = {k: v for k, v in _flatten(cand).items() if k.split(".")[0] not in IGNORED_CONFIG}
    return {k: [a.get(k), b.get(k)] for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)}


def _rows(path: Path, key: tuple[str, ...]) -> dict:
    with path.open() as stream:
        return {tuple(row[k] for k in key): row for row in csv.DictReader(stream)}


def history_diff(ref: Path, cand: Path, key: tuple[str, ...]) -> dict:
    a, b = _rows(ref, key), _rows(cand, key)
    if set(a) != set(b):
        return {"error": f"row keys differ: only reference {sorted(set(a) - set(b))[:5]}, "
                         f"only candidate {sorted(set(b) - set(a))[:5]}"}
    columns = [c for c in next(iter(a.values())) if c not in key and c != "timestamp"]
    out = {}
    for col in columns:
        abs_max = rel_max = 0.0
        for k in a:
            x, y = float(a[k][col]), float(b[k][col])
            if not (math.isfinite(x) and math.isfinite(y)):
                return {"error": f"non-finite {col} at {k}"}
            abs_max = max(abs_max, abs(x - y))
            rel_max = max(rel_max, abs(x - y) / max(abs(x), 1e-12))
        out[col] = {"max_abs": abs_max, "max_rel": rel_max}
    out["rows"] = len(a)
    return out


def checkpoint_diff(ref: Path, cand: Path) -> dict:
    a = torch.load(ref, map_location="cpu", weights_only=False)["model_state_dict"]
    b = torch.load(cand, map_location="cpu", weights_only=False)["model_state_dict"]
    if set(a) != set(b):
        return {"error": f"tensor names differ: {sorted(set(a) ^ set(b))[:5]}"}
    max_abs, worst, num, den = 0.0, None, 0.0, 0.0
    for name in sorted(a):
        x, y = a[name].double(), b[name].double()
        if x.shape != y.shape:
            return {"error": f"{name} shape {tuple(x.shape)} vs {tuple(y.shape)}"}
        if x.numel() == 0:
            continue
        d = (x - y).abs().max().item()
        if d > max_abs:
            max_abs, worst = d, name
        num += (x - y).pow(2).sum().item()
        den += x.pow(2).sum().item()
    return {"max_abs": max_abs, "worst_tensor": worst,
            "rel_l2": math.sqrt(num / den) if den else 0.0, "tensors": len(a)}


def compare(ref: Path, cand: Path, all_epochs: bool = False) -> dict:
    ref_cfg = torch.load(ref / "last.pt", map_location="cpu", weights_only=False)["config"]
    cand_cfg = torch.load(cand / "last.pt", map_location="cpu", weights_only=False)["config"]
    report = {"reference": str(ref), "candidate": str(cand),
              "config_diff": config_diff(ref_cfg, cand_cfg)}
    for name, key in KEYS.items():
        report[name] = history_diff(ref / name, cand / name, key)
    names = ["init.pt", "last.pt"]
    if all_epochs:
        names += sorted(p.name for p in ref.glob("epoch_*.pt"))
    report["checkpoints"] = {n: checkpoint_diff(ref / n, cand / n) if (cand / n).exists()
                             else {"error": "missing in candidate"} for n in names}
    return report


def has_errors(report: dict) -> bool:
    parts = [report[n] for n in KEYS] + list(report["checkpoints"].values())
    return any("error" in part for part in parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--all-epochs", action="store_true", help="also compare every epoch_*.pt")
    parser.add_argument("--json", type=Path, help="write the full report here")
    args = parser.parse_args()
    report = compare(args.reference, args.candidate, args.all_epochs)
    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + "\n")
    print(f"config differences: {report['config_diff'] or 'none'}")
    for name in KEYS:
        part = report[name]
        if "error" in part:
            print(f"{name}: ERROR {part['error']}")
            continue
        cols = ", ".join(f"{c} {v['max_abs']:.3g}" for c, v in part.items() if c != "rows")
        print(f"{name} ({part['rows']} rows) max |diff|: {cols}")
    for name, part in report["checkpoints"].items():
        if "error" in part:
            print(f"{name}: ERROR {part['error']}")
        else:
            print(f"{name}: max |diff| {part['max_abs']:.3g} ({part['worst_tensor']}), "
                  f"relative L2 {part['rel_l2']:.3g}")
    sys.exit(1 if has_errors(report) else 0)


if __name__ == "__main__":
    main()
