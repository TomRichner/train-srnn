"""test_cheetah100_loader.py — assertions for load_cheetah100.

Checks the pieces that would otherwise fail silently at loss time or, worse,
train on subtly wrong data: channel counts for both include_actions settings,
train-only z-score statistics, the autoregressive offset, transient removal,
and that train_trace is the same array the windows were cut from.

Usage:
    PYTHONPATH=. python scripts/test_cheetah100_loader.py
"""

import json
import pathlib

import numpy as np

from train_srnn.data.datasets import load_dataset

DATA_DIR = "train_srnn/data/cheetah100"
HZ = 100.0
PASS, FAIL = 0, 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


def main():
    if not pathlib.Path(DATA_DIR, "train.npz").exists():
        raise SystemExit(
            f"{DATA_DIR}/train.npz missing — stage the dataset first "
            "(see conf/task/cheetah100.yaml)")

    # ---- shapes, both variants ------------------------------------------
    print("\n[shapes]")
    for include, want_c in ((False, 17), (True, 23)):
        d = load_dataset("cheetah100", DATA_DIR, include_actions=include,
                         seq_len=1500, stride=500, train_trace_max_len=118973)
        tr_x, tr_y = d["train"]
        tag = "obs+act" if include else "obs"
        check(f"{tag}: meta input_size == {want_c}",
              d["meta"]["input_size"] == want_c, d["meta"]["input_size"])
        check(f"{tag}: meta input_size == output_size (ring requires it)",
              d["meta"]["input_size"] == d["meta"]["output_size"])
        check(f"{tag}: train windows are (N, 1500, {want_c})",
              tr_x.shape[1:] == (1500, want_c), tr_x.shape)
        check(f"{tag}: x and y windows same shape",
              tr_x.shape == tr_y.shape)
        check(f"{tag}: train_trace present and 2-D",
              d["train_trace"].ndim == 2 and d["train_trace"].shape[1] == want_c)

    # ---- the obs-only case in depth -------------------------------------
    d = load_dataset("cheetah100", DATA_DIR, seq_len=1500, stride=500,
                     train_trace_max_len=118973, skip_transient_s=10.0)
    trace = d["train_trace"]
    tr_x, tr_y = d["train"]

    print("\n[truncation + ring sizing]")
    check("train_trace truncated to train_trace_max_len",
          trace.shape[0] == 118973, trace.shape[0])
    steps = -(-trace.shape[0] // (24 * 250))
    drift = steps * 24 * 250 - trace.shape[0]
    # Nonzero drift moves the state.detach() boundaries between epochs;
    # gcd(drift, chunk)==1 makes them sweep every offset rather than a subset.
    check("phase drift is nonzero (B=24, chunk=250)",
          drift > 0, f"steps={steps} drift={drift}")
    check("drift coprime to chunk_len -> boundaries sweep all offsets",
          np.gcd(drift, 250) == 1, f"gcd={np.gcd(drift, 250)}")

    print("\n[normalization: train-only statistics]")
    check("train mean ~ 0", np.abs(trace.mean(0)).max() < 1e-4,
          np.abs(trace.mean(0)).max())
    check("train std ~ 1", np.abs(trace.std(0) - 1).max() < 1e-4,
          np.abs(trace.std(0) - 1).max())
    va_x, _ = d["valid"]
    va_mean = np.abs(va_x.reshape(-1, va_x.shape[-1]).mean(0)).max()
    check("valid NOT separately centered (train stats applied)",
          va_mean > 1e-3, f"max |mean| = {va_mean:.2e}")

    print("\n[normalization: actions are z-scored too]")
    da = load_dataset("cheetah100", DATA_DIR, include_actions=True,
                      seq_len=1500, stride=500, train_trace_max_len=118973)
    at = da["train_trace"][:, 17:]
    check("action channels have unit std after z-score",
          np.abs(at.std(0) - 1).max() < 1e-4, np.abs(at.std(0) - 1).max())

    print("\n[autoregressive offset]")
    check("y[i, t] == x[i, t+1] within a window",
          np.array_equal(tr_x[0, 1:], tr_y[0, :-1]))
    check("windows are cut from train_trace",
          np.array_equal(tr_x[0], trace[:1500]))

    print("\n[transient removal]")
    raw = np.load(pathlib.Path(DATA_DIR, "train.npz"))["obs"]
    check("raw trace starts at a standstill", abs(raw[0, 8]) < 1.0, raw[0, 8])
    kept = load_dataset("cheetah100", DATA_DIR, seq_len=1500, stride=500,
                        skip_transient_s=0.0, normalize=False)["train_trace"]
    dropped = load_dataset("cheetah100", DATA_DIR, seq_len=1500, stride=500,
                           skip_transient_s=10.0, normalize=False)["train_trace"]
    check("skip_transient_s=0 keeps the standstill",
          abs(kept[0, 8]) < 1.0, kept[0, 8])
    check("skip_transient_s=10 starts at cruise",
          dropped[0, 8] > 5.0, dropped[0, 8])
    check("exactly 10 s dropped",
          kept.shape[0] - dropped.shape[0] == int(10.0 * HZ),
          kept.shape[0] - dropped.shape[0])

    print("\n[manifest cross-check]")
    man = json.loads(pathlib.Path(DATA_DIR, "manifest.json").read_text())
    ref = np.array(man["zscore_train_stats"]["obs_std"])
    check("recomputed raw std matches manifest reference",
          np.allclose(kept.std(0), ref, rtol=1e-3),
          f"max rel diff {np.abs(kept.std(0) / ref - 1).max():.2e}")

    print(f"\n{PASS} passed, {FAIL} failed")
    raise SystemExit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
