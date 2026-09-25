"""Next-step error of every network as a function of the local playback rate.

    uv run python scripts/eval_speed.py <run_dir> --data-root "$SRNN_HOME/data" \
        --tests cheetah100_warp_multi/test.npz cheetah100_fixed_speeds/test_r*.npz \
        [--checkpoint last.pt] [--device cuda] [--out <run_dir>/speed_eval]

For each test trace (a ``timewarp.py`` npz holding ``obs`` and a per-sample ``rate``;
a plain trace counts as rate 1), the checkpoint's networks predict every sample one
step ahead in teacher-forcing mode, and each squared error is recorded. The trace is
cut into segments of ``--segment-len`` scored samples, run in parallel; each starts
from the frozen initial condition ``--warmup-s`` seconds earlier, and that lead-in is
not scored. Traces are z-scored with the
statistics of the run's own training trace, exactly as ``TraceTask.load`` does, read
from ``<data-root>/<task.dataset>``. 
Errors are summarized overall and in bins of ``log2 rate``, next to the persistence
baseline (predict the current sample) in the same bins, because faster segments have
larger sample-to-sample changes and would otherwise dominate a plain MSE. Segments and
lead-in differ from the trainer's windows (1,500 samples, readout after its BPTT
lead-in), so the numbers differ slightly from the trainer's test loss.

Writes ``speed_eval.json`` (summaries) and ``speed_eval_<test>.npz`` (per-sample
errors, ``(K, T)`` float32) to ``--out``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from _runs import rebuild_model  # noqa: E402

BIN_EDGES = np.arange(-1.5, 1.5001, 0.25)   # log2 rate


def train_stats(cfg, data_root: Path) -> tuple[np.ndarray, np.ndarray]:
    """Mean and SD of the run's training trace, as ``TraceTask.load`` computes them."""
    task = cfg.task
    with np.load(data_root / task.dataset / "train.npz") as z:
        train = z["obs"].astype(np.float32)
    train = train[int(round(task.skip_transient_s * task.sample_rate_hz)):]
    if task.train_trace_max_len is not None:
        train = train[:task.train_trace_max_len]
    mu, sd = train.mean(axis=0), train.std(axis=0)
    sd[sd < 1e-8] = 1.0
    return mu, sd


@torch.no_grad()
def sample_errors(model, x: torch.Tensor, warmup: int, seg_len: int = 1000,
                  chunk: int = 500) -> torch.Tensor:
    """Squared error per network for predictions ``warmup .. T-2``, ``(K, T-1-warmup)``.

    The trace is cut into segments of ``seg_len`` scored predictions, evaluated in
    parallel as one batch (a single long pass is launch-bound and far too slow). Each
    segment starts from the initial condition ``warmup`` samples before its first
    scored prediction, like the trainer's windowed evaluation.
    """
    n_pred = x.shape[0] - 1
    starts = list(range(warmup, n_pred, seg_len))
    length = warmup + seg_len
    inputs = torch.zeros(len(starts), length, x.shape[1], device=x.device, dtype=x.dtype)
    targets = torch.zeros_like(inputs)
    valid = torch.zeros(len(starts), length, dtype=torch.bool, device=x.device)
    for b, a in enumerate(starts):
        lo, hi = a - warmup, min(a + seg_len, n_pred)
        inputs[b, :hi - lo] = x[lo:hi]
        targets[b, :hi - lo] = x[lo + 1:hi + 1]
        valid[b, warmup:hi - lo] = True
    state = model.initial_state(len(starts))
    hoisted = model.cell.hoist()
    errors = []
    for t0 in range(0, length, chunk):
        seg = inputs[:, t0:t0 + chunk]
        res = model.unroll(seg, state, hoisted=hoisted)
        state = res.state
        pred = model.apply_readout(res.hidden, seg)
        pred = pred if pred.dim() == 4 else pred[None]          # (K, B, T, O)
        errors.append(((pred - targets[None, :, t0:t0 + chunk]) ** 2).mean(-1))
    err = torch.cat(errors, dim=2)                                # (K, B, length)
    return err[:, valid]                                          # segments in order


def summarize(err: np.ndarray, persistence: np.ndarray, rate: np.ndarray, warmup: int) -> dict:
    """``err`` already starts at prediction ``warmup``; trim the others to match."""
    persistence, lr = persistence[warmup:], np.log2(rate[1:][warmup:])
    bins = np.digitize(lr, BIN_EDGES) - 1
    per_bin = []
    for b in range(len(BIN_EDGES) - 1):
        idx = bins == b
        if idx.sum() == 0:
            continue
        per_bin.append({"log2_rate_lo": float(BIN_EDGES[b]), "log2_rate_hi": float(BIN_EDGES[b + 1]),
                        "samples": int(idx.sum()), "mse": err[:, idx].mean(axis=1).tolist(),
                        "persistence_mse": float(persistence[idx].mean())})
    return {"samples": int(err.shape[1]), "mse": err.mean(axis=1).tolist(),
            "persistence_mse": float(persistence.mean()),
            "rate_mean": float(rate.mean()), "bins": per_bin}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--tests", nargs="+", required=True, help="npz paths relative to --data-root")
    p.add_argument("--checkpoint", default="last.pt")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--warmup-s", type=float, default=5.0,
                   help="unscored lead-in from the initial condition before each segment")
    p.add_argument("--segment-len", type=int, default=1000, help="scored samples per segment")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    out = args.out or args.run_dir / "speed_eval"
    out.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.run_dir / args.checkpoint, map_location="cpu", weights_only=False)
    model, cfg, names = rebuild_model(ckpt, device=args.device)
    mu, sd = train_stats(cfg, args.data_root)
    report = {"run_dir": str(args.run_dir), "checkpoint": args.checkpoint, "epoch": ckpt.get("epoch"),
              "variants": names, "train_dataset": cfg.task.dataset, "warmup_s": args.warmup_s,
              "device": args.device, "tests": {}}
    tests = [str(q.relative_to(args.data_root)) for pat in args.tests
             for q in sorted(args.data_root.glob(pat))] if any("*" in t for t in args.tests) else args.tests
    for rel in tests:
        path = args.data_root / rel
        with np.load(path) as z:
            obs = z["obs"].astype(np.float32)
            rate = z["rate"].astype(np.float64) if "rate" in z.files else np.ones(len(obs))
        name = f"{path.parent.name}/{path.stem}"
        x = torch.tensor((obs - mu) / sd, device=args.device)
        t0 = time.time()
        warmup = int(round(args.warmup_s * cfg.task.sample_rate_hz))
        err = sample_errors(model, x, warmup, args.segment_len).cpu().numpy().astype(np.float32)
        xn = x.cpu().numpy()
        persistence = ((xn[1:] - xn[:-1]) ** 2).mean(axis=1)
        report["tests"][name] = summarize(err, persistence, rate, warmup)
        np.savez_compressed(out / f"speed_eval_{path.parent.name}__{path.stem}.npz",
                            err=err, rate=rate.astype(np.float32), persistence=persistence)
        print(f"{name}: {len(obs)} samples in {time.time() - t0:.1f} s; "
              f"median network MSE {np.median(report['tests'][name]['mse']):.4f}, "
              f"persistence {report['tests'][name]['persistence_mse']:.4f}", flush=True)
    (out / "speed_eval.json").write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
