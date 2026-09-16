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

## Running experiment

The durable pipeline was launched at 2026-09-16 05:24 UTC. Its atomic status
file is `/opt/srnn-work/results-matlab-v2-20260916/pipeline-20260916/pipeline_status.json`
on the retained VM. The current fresh pilot run is
`matlab-v2-20260916-052434-bce3a7-pilot-s3` under the `cheetah100` results directory.
It executes 2,000 optimizer steps for all nine condition/seed combinations.
The pipeline then summarizes the pilot, measures and confirms the largest
whole-seed batch satisfying the conservative 80% memory bound, and starts a
fresh final 2,000-step experiment at that count. Any failed stage halts it.

Pilot and final scientific results are pending; model parity does not establish
an advantage for adaptation. The manuscript and its figure links have not been
edited by this implementation.
