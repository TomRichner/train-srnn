# Time-warped cheetah100: the two-STD penalty, not the SFA ladder

Runs `tw30-cheetah100-20260925`, `tw30-warpslow-20260925` and `tw30-warpmulti-20260925`
(Modal L4, commit `a4a6eb9`, 30 epochs). The full report with figures, the per-run
parameter-evolution reports and the interpretation is generated outside the repository
by `scripts/report_timewarp.py`, at
`$SRNN_HOME/results/timewarp-tw30-20260925/report.md`.

## Question and design

In the manuscript run `m5-15s-100e-0916`, SFA1/STD1 networks predicted the 100 Hz
HalfCheetah better than SFA3/STD2 networks. One explanation is that the steady gait
carries essentially one timescale. To test it, `gen-half-cheetah-dataset/timewarp.py`
replays the same traces at a hidden playback rate that drifts smoothly and randomly
(log2 r a bounded sum of Gaussian-smoothed noise; velocities scaled by r; anti-aliased
speed-ups; one seed per split):

| Dataset | Rate process | Training rates |
|---|---|---|
| `cheetah100` (control) | none | 1x |
| `cheetah100_warp_slow` | one component, kernel SD 5 s | 0.54-1.78x |
| `cheetah100_warp_multi` | components at 1, 5 and 25 s | 0.54-1.74x |

`cheetah100_fixed_speeds` holds the test split at constant 0.4-2.5x for interpolation
and extrapolation. Each dataset trained 75 networks in one batch: no adaptation,
SFA1/STD1, SFA3/STD2 and the factorial controls SFA3/STD1 and SFA1/STD2 (new
`variants.py` tokens) x 15 paired seeds, otherwise with the manuscript settings
(500 units, SRA1 with four substeps, 24 ring readers, 250-sample chunks, Dale, no
skip). `scripts/eval_speed.py` measured next-step error by local playback rate.

Reproduce (datasets staged on `srnn-data` first, see `docs/modal.md`):

```bash
ARGS="model.variants=[srnn-no-adapt,srnn-sfa1-std1,srnn-sfa3-std2,srnn-sfa3-std1,srnn-sfa1-std2] \
model.variant_seeds=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15] model.num_units=500 model.solver=sra1 \
model.ode_unfolds=4 task.batch_size=24 task.bptt_chunk_len=250 burn_in=10 lr=0.0005 warmup_epochs=3 \
cosine_decay=false grad_clip=1 amp=fp32 grad_checkpoint=true grad_checkpoint_segment_len=5 \
checkpoint_interval=5 log_interval=5 compile.enabled=true device=cuda epochs=30"
uv run modal run --detach cloud/modal_app.py --run-name <run> --task cheetah100 --args "$ARGS"
uv run modal run --detach cloud/modal_app.py --run-name <run> --task cheetah100 \
    --dataset cheetah100_warp_multi --args "$ARGS task.skip_transient_s=0"
```

## Results (final test MSE, geometric mean over 15 seeds)

| Condition | control | slow | multi |
|---|---:|---:|---:|
| No adaptation | 0.815 | 0.844 | 0.809 |
| SFA1 / STD1 | 0.118 | 0.162 | 0.156 |
| SFA3 / STD1 | 0.119 | 0.166 | 0.161 |
| SFA1 / STD2 | 0.340 | 0.376 | 0.340 |
| SFA3 / STD2 | 0.351 | 0.374 | 0.349 |

| Factorial effect on MSE (p, exact paired sign-flip) | control | slow | multi |
|---|---:|---:|---:|
| Two vs one STD timescale | 2.92x (6e-5) | 2.28x (6e-5) | 2.17x (6e-5) |
| Three vs one SFA timescale | 1.02x (0.16) | 1.01x (0.54) | 1.03x (0.055) |
| Interaction | 1.02x (0.45) | 0.97x (0.38) | 0.99x (0.79) |

- The whole SFA1/STD1 versus SFA3/STD2 gap comes from the second STD timescale; the SFA
  ladder has no measurable effect, with or without speed variation.
- Speed variation shrinks the two-STD penalty (2.92x to 2.2-2.3x) but does not reverse
  it: SFA3/STD2 is worse than SFA1/STD1 in 15 of 15 seeds on every dataset.
- Training on warped data improves generalization across speeds (at 1.41x, error
  relative to persistence 2.3 versus 5.1), at a small cost at 1x. No condition
  extrapolates, least of all to slower speeds.
- Training weakens depression in every adapting condition (recovery times about 20%
  shorter, release times about 25% longer); the two-STD networks also raise their
  rate setpoint a_0 from 0.20 to 0.31.

Caveats: the one- and two-STD conditions are not strength-matched, so the penalty may
be a depression-strength effect rather than a timescale effect; 30 epochs is early
training (the best condition only reaches the persistence baseline at 1x, 0.118 versus
0.117); one warp realization per split.
