#!/usr/bin/env python3
"""Cloud run provenance and one-second whole-device GPU memory samples."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import signal
import subprocess
import time
from pathlib import Path

FIELDS = ["time_unix", "index", "memory_used_mib", "memory_total_mib", "utilization_pct"]


def query_gpu(fields: str) -> str:
    return subprocess.check_output(
        ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
        text=True, timeout=10,
    )


def summarize(output: Path) -> dict:
    devices = {}
    with (output / "gpu_memory_samples.csv").open() as stream:
        for row in csv.DictReader(stream):
            device = devices.setdefault(row["index"], {
                "peak_used_mib": 0.0, "total_mib": float(row["memory_total_mib"]),
                "samples": 0,
            })
            device["peak_used_mib"] = max(device["peak_used_mib"], float(row["memory_used_mib"]))
            device["samples"] += 1
    for device in devices.values():
        device["peak_fraction"] = device["peak_used_mib"] / device["total_mib"]
    report = {"sample_interval_seconds": 1, "measurement": "whole-device nvidia-smi",
              "devices": devices}
    (output / "gpu_memory_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def sample(output: Path) -> None:
    running = True

    def stop(*_args):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    with (output / "gpu_memory_samples.csv").open("w", buffering=1) as stream:
        writer = csv.writer(stream)
        writer.writerow(FIELDS)
        while running:
            started = time.monotonic()
            try:
                rows = query_gpu("index,memory.used,memory.total,utilization.gpu")
                for row in csv.reader(rows.splitlines()):
                    writer.writerow([time.time(), *[value.strip() for value in row]])
            except (subprocess.SubprocessError, OSError) as error:
                print(f"GPU sampling failed: {error}", flush=True)
            time.sleep(max(0, 1 - (time.monotonic() - started)))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def provenance(output: Path, data_dir: Path, commit: str) -> None:
    import torch

    hashes = {}
    for path in sorted(data_dir.rglob("*")):
        if path.is_file() and (path.suffix == ".npz" or path.name == "manifest.json"):
            hashes[str(path.relative_to(data_dir))] = file_sha256(path)
    report = {
        "commit": commit, "python": platform.python_version(), "platform": platform.platform(),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(), "dataset_sha256": hashes,
        "gpus": [dict(name=torch.cuda.get_device_name(index),
                      total_memory_bytes=torch.cuda.get_device_properties(index).total_memory)
                 for index in range(torch.cuda.device_count())],
        "nvidia_smi": query_gpu("index,uuid,name,driver_version,memory.total").strip(),
    }
    (output / "runtime_provenance.json").write_text(json.dumps(report, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["provenance", "sample", "summarize"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--commit")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.action == "provenance":
        if args.data_dir is None or args.commit is None:
            parser.error("provenance requires --data-dir and --commit")
        provenance(args.output, args.data_dir, args.commit)
    elif args.action == "sample":
        sample(args.output)
    else:
        print(json.dumps(summarize(args.output)))


if __name__ == "__main__":
    main()
