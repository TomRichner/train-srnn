# MATLAB alignment validation — 2026-09-16

The version-2 SRNN implements the manuscript's no-adaptation, SFA1/STD1,
and SFA3/STD2 conditions. The model has a total SFA budget, independent STD
recovery/release timescales entering a product, raw-rate adaptation drive,
and deterministic SRA1 integration. Batched ablations, skip connections,
optional Dale enforcement during training, and per-neuron training remain.
Historical checkpoints require their original source revision; version 2
rejects them rather than silently interpreting incompatible tensors.

## Completed checks

- CPU suite with MATLAB fixtures: 174 passed, 6 skipped, 1 deselected.
  The deselected slow 40-second MATLAB trajectory check also passed in a
  separate parity run. The six skips are existing optional baseline checks.
- Production MATLAB versus PyTorch float64: maximum 40-second state error
  below 8e-14 for an unforced and a step-driven 500-neuron SFA3/STD2 model.
  Derivative and single-step probes cover all three conditions. See
  [matlab_parity.md](matlab_parity.md) for convergence and contraction limits.
- GPU compiled/eager comparison on PyTorch 2.9.1+cu129: output difference
  3.73e-8, state difference 5.96e-8; every parameter gradient passed
  atol=2e-6, rtol=2e-4 (maximum absolute difference 3.82e-6 on the setpoint
  scalar, within its relative tolerance).
- Continuous-ring gradient checkpointing was previously ignored. It now
  recomputes five-sample segments while retaining the 250-sample BPTT
  boundary. Tests compare logits, gradients, optimizer updates, carried
  states, previous outputs, and reader positions in teacher-forced and
  closed-loop operation, including unequal final segments.
- Golden SRNN snapshots were deliberately regenerated for changed model
  equations and paired initialization. Baseline-cell snapshots are unchanged.

## GPU execution provenance

The existing L4 VM `smoke-dales-skip-srnn-cheetah-seed1` was reused. Its
loaded/installed NVIDIA driver mismatch was resolved with a controlled reboot,
with startup training replay disabled during the reboot. Startup service
configuration was restored without executing the old job. Driver 580.178.04
now supports successful CUDA forward/backward execution. No packages were
installed. The historical `/opt/train-srnn` checkout and regression results
remain untouched; the VM is retained running.

Source was deployed as explicit hashed working-tree snapshots rather than
claiming that the uncommitted changes were present in the historical checkout:

- `/opt/train-srnn-matlab-v2-20260916`: first, uncheckpointed profile.
- `/opt/train-srnn-matlab-v2-checkpoint-20260916`: corrected checkpointed trainer.

Each contains `SOURCE_REVISION` and `SOURCE_MANIFEST.json`. Run metadata
additionally hashes Python sources and dataset files. Runs are isolated under
`/opt/srnn-work/results-matlab-v2-20260916/cheetah100`.

The first three-seed profile completed training, evaluation, and checkpoint
reload, but used 19,018 of 23,034 MiB (82.57%), above the intended reserve.
Although its saved config says `grad_checkpoint=true`, the pre-fix ring trainer
ignored that flag. It must not be reused as a checkpointed capacity measurement.
Subsequent runs use the corrected implementation and fresh output directories.

The corrected three-seed profile completed in 304 seconds. Sampled whole-device
peak was 2,844 MiB; PyTorch peak allocated/reserved memory was 1,838.55/2,598
MiB. The conservative reserved-plus-1-GiB measure is 3,622 MiB, or 15.72% of
device capacity. Its initial validation, training, final validation, and test
loss arrays were exactly equal to the uncheckpointed profile's saved arrays.
These timings include compilation/cache differences and are not a speed test.
Dataset hashes match the existing local cheetah100 dataset. Both profiles and
the CUDA comparison log are mirrored under
`/Users/tom/Desktop/local_data/srnn/cache/matlab-alignment-20260916`.

## Cancelled pilot and replacement experiment

The first durable pipeline was cancelled at the user's request after six
completed epochs (120 optimizer steps). The saved `epoch_004.pt` contains
five completed epochs (100 steps). No capacity search or final training was
started. The pipeline status and pilot `CANCELLED.json` record cancellation.
The pilot directory `matlab-v2-20260916-052434-bce3a7-pilot-s3` and pipeline
records are mirrored under the local cache alongside the parity artifacts.
The old VM was stopped after preserving these artifacts.

The user approved a fresh VM through the repository cloud scripts. The
initial eight-seed/three-condition one-epoch preflight completed successfully:
24 finite model states and training/validation losses after 20 optimizer steps,
a loadable saved checkpoint, finite test losses, and GCS `run_metadata.json`
with exit code zero. Whole-device sampled peak was 6,268 / 23,034 MiB (27.21%).
Its source is pinned to `8e2ce4cc7cbb5534447375a4f49ede4f18880308`; dataset
hashes match the validated dataset. Python 3.10.12, PyTorch 2.9.1+cu129,
CUDA 12.9 and NVIDIA driver 580.178.04 were recorded. This one-epoch preflight
uses the standard trainer's short-run warmup cap (four steps); the manuscript
run uses the approved 60-step warmup. The preflight is not a scientific result.

The user subsequently selected **15 paired seeds per condition, 100 epochs**
(45 networks, 2,000 optimizer steps). This fixed count supersedes the earlier
eight-seed plan and maximum-capacity search. The manuscript run is
`m5-15s-100e-0916`, submitted through `cloud/submit.sh` on the fresh L4 VM
`m5-8s-preflight-0916-srnn-cheetah100-seed1` in `us-central1-b`. The VM name
reflects its initial preflight; the manuscript run actually uses seeds 1–15.
It starts from fresh initialization (`init_ckpt=null`), with no skip connection,
Dale enforcement, FP32, 500 neurons, SRA1 with four internal substeps, 24 ring
readers, 250-sample chunks and five-sample gradient-checkpoint segments.
The deployed commit remains pinned to `8e2ce4c`.

Cloud artifacts:

- Preflight: `gs://liquidneuralnets-experiments/results-pytorch/m5-8s-preflight-0916/srnn/cheetah100/seed1/`.
- Manuscript: `gs://liquidneuralnets-experiments/results-pytorch/m5-15s-100e-0916/srnn/cheetah100/seed1/`.

The outer `seed1` directory is the cloud dispatch seed; all 45 named variants
are saved within it. The VM is configured to upload results periodically and
stop after completion. The first epoch completed at 2026-09-16 06:04 UTC with finite training losses
for all 45 networks. Sampled peak through that epoch was 9,376 MiB (40.70%
of device memory). The manuscript and its figure links have not been edited
by this implementation.

## Completed manuscript run

All 45 networks completed 100 epochs / 2,000 optimizer steps. Cloud metadata
records exit code zero and completion at 2026-09-16 10:04:55 UTC. Runtime was
14,890 seconds (about 4 h 8 min), including startup/compilation and evaluation.
The VM is verified TERMINATED (stopped). `last.pt` exists in GCS. No traceback
or ERROR entries were found in the training log. The summary verified every
condition/seed has the complete validation grid from step 0 through 2,000
at 100-step intervals and a finite final test result.

Peak sampled whole-device memory was 10,414 / 23,034 MiB (45.21%), across
14,861 samples. Median training-epoch duration excluding the first was
117.3 seconds; this is not a controlled checkpointing speed comparison.

| Condition | Final test MSE, mean ± sample SD | Final test MAE, mean |
|---|---:|---:|
| No adaptation | 0.489535 ± 0.423500 | 0.504246 |
| SFA1 / STD1 | 0.041746 ± 0.003331 | 0.140524 |
| SFA3 / STD2 | 0.088803 ± 0.007974 | 0.216172 |

Each condition has 15 paired seeds. Single-timescale adaptation gave the
lowest final loss and best validation learning-curve score. Both adaptation
conditions improved average performance relative to no adaptation. These
results do not support an MTS advantage on this task and protocol. The
unadapted final MSE varied widely across seeds (0.093601 to 0.980178).

For the prespecified average integral of log validation loss, lower is better:

| Condition | Mean score | Bootstrap 95% interval |
|---|---:|---:|
| No adaptation | -0.488129 | [-0.781289, -0.210682] |
| SFA1 / STD1 | -2.305354 | [-2.334357, -2.274226] |
| SFA3 / STD2 | -1.441099 | [-1.505262, -1.381057] |

Exact paired sign-flip P values for learning-curve scores are 0.0000610
(no adaptation vs SFA1/STD1), 0.0001221 (no adaptation vs SFA3/STD2), and
0.0000610 (SFA1/STD1 vs SFA3/STD2), unadjusted for three comparisons.
Bootstrap intervals use 2,000 resamples. Test data were evaluated periodically,
so these are not untouched holdout-test results. Single- and multiple-timescale
STD are not strength-matched, as recorded in the protocol.

Downloaded histories, execution metadata, full log, SVG/PNG learning curves,
and `matlab_aligned_summary.json` are stored in
`/Users/tom/Desktop/local_data/srnn/cache/m5-15s-100e-0916/`.
Reproduce the analysis with `scripts/summarize_matlab_aligned.py <run-directory>`.

