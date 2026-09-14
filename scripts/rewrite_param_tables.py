#!/usr/bin/env python3
"""Re-emit param_table.txt for one or more cached run dirs without rerunning
the full postprocess pipeline. Uses init.pt + last.pt only.

Usage:
    python scripts/rewrite_param_tables.py $SRNN_HOME/cache/run1 ...
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import postprocess  # noqa: E402


def _load_ms(path: Path) -> dict:
    sd = torch.load(path, map_location="cpu", weights_only=False)
    return sd.get("model_state_dict", sd) if isinstance(sd, dict) else sd


def _epoch_of(path: Path) -> int:
    sd = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict):
        return int(sd.get("epoch", 0))
    return 0


def rewrite(run_dir: Path) -> None:
    init_pt = run_dir / "init.pt"
    last_pt = run_dir / "last.pt"
    if not init_pt.exists() or not last_pt.exists():
        print(f"[skip] {run_dir}: missing init.pt or last.pt")
        return

    variant_names = postprocess._variant_names_for(run_dir)
    init_ms = _load_ms(init_pt)
    last_ms = _load_ms(last_pt)
    e0 = _epoch_of(init_pt)
    e1 = _epoch_of(last_pt)
    snaps = [(e0, e0, init_ms), (e1, e1, last_ms)]

    for k, name in enumerate(variant_names):
        out = run_dir / name
        if not out.exists():
            print(f"  [skip variant] {name}: dir does not exist")
            continue
        postprocess.write_param_table(out, run_dir.name, snaps, k, name)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for arg in sys.argv[1:]:
        run_dir = Path(arg).resolve()
        if not run_dir.is_dir():
            print(f"[skip] {run_dir}: not a directory")
            continue
        print(f"=== {run_dir} ===")
        rewrite(run_dir)


if __name__ == "__main__":
    main()
