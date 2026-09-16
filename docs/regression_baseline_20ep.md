# SRNN 20-epoch regression baseline

This run is the reference for checking future refactors of the default SRNN
and continuous trainer on `cheetah100`. It starts from seed 1, not from the
three-epoch smoke-test checkpoint.

## Identity and artifacts

- Run: `regression-baseline-20ep-20260915` (seed 1).
- Code: `5e92be7510f95471128873ef63843294360ae05f`; tracked VM checkout clean.
- VM: `smoke-dales-skip-srnn-cheetah-seed1`, `us-central1-b`, NVIDIA L4.
- VM output: `/opt/srnn-work/results/cheetah100/regression-baseline-20ep-20260915_seed1/`.
- GCS: `gs://liquidneuralnets-experiments/results-pytorch/regression-baseline-20ep-20260915/srnn/cheetah100/seed1/`.
- Local mirror: `/Users/tom/Desktop/local_data/srnn/cache/regression-baseline-20ep-20260915/srnn/cheetah100/seed1/`.

The result directory preserves `init.pt`, every `epoch_*.pt`, `last.pt`,
training/test histories, logs, and completion metadata. Additional baseline
artifacts are `resolved_config.yaml`, `baseline_provenance.json` (including
dataset SHA-256 hashes), `python_packages.txt`, `hydra_config.tar.gz`,
`baseline_verification.json`, and `artifact_sha256.txt`. All four dataset
hashes match `/Users/tom/Desktop/local_data/srnn/data/cheetah100/`.

## Reproduction

Use the saved resolved configuration as the authority for defaults: a
future refactor may change the defaults behind the same CLI arguments.
The original invocation, with `SRNN_HOME=/opt/srnn-work`, is:

```bash
python3 train.py model=srnn task=cheetah100 seed=1 \
  epochs=20 device=cuda checkpoint_interval=1 warmup_epochs=1 \
  run_name=regression-baseline-20ep-20260915_seed1
```

Choose a new run name for comparisons so the reference is never overwritten.
For a configured, idle VM with matching code, dataset, and dependencies:

```bash
bash cloud/submit.sh <vm> <new_comparison_run> cheetah100 srnn 1 \
  --cleanup=keep --skip-refresh \
  "epochs=20 device=cuda checkpoint_interval=1 warmup_epochs=1"
```

The reference uses the default 300-unit `srnn` variant, FP32, compiled cell,
24 ring readers, 250-step BPTT chunks, and 20 optimizer steps per epoch.
Python is 3.10.12; PyTorch is 2.9.1+cu129; CUDA is 12.9; the NVIDIA driver is
580.173.02. The full package inventory is saved with the run. Do not replace
this VM environment with the local Python 3.12 environment for a controlled
comparison.

Warmup is explicitly one epoch. The scheduler caps warmup at 20% of total
steps, so this run uses 20 warmup steps, whereas the preceding three-epoch
smoke test used 12. Their first three epochs are not an equality check.

## Comparison criteria

Verify matching dataset hashes and configuration, then compare the entire
training, validation, and test trajectories, as well as initial and final
model states. Epoch indices in CSVs/checkpoints are zero-based: epoch 19
is the twentieth epoch. Check for finite metrics, all 20 completed epochs,
and successful completion metadata before comparing performance.

Strict deterministic algorithms were not enabled. A fixed seed alone does
not establish bitwise GPU reproducibility; measure same-code repeat-run
variation before setting a numerical acceptance tolerance. Keep the existing
CPU golden tests as the separate check for pinned cell/trainer numerics.

## Outcome

Completed successfully on 2026-09-16 at 03:36:35 UTC (September 15 at
10:36:35 p.m. America/Chicago), with exit code 0. Recorded runtime was
839 seconds (13 minutes 59 seconds). The VM was left running.

| Final metric (epoch 19, twentieth epoch) | Value |
|---|---:|
| Training loss | 0.233011 |
| Validation loss | 0.216722 |
| Test loss | 0.216856 |
| Training MAE | 0.370861 |
| Validation MAE | 0.358497 |
| Test MAE | 0.357196 |

The final epoch also had the lowest validation loss. Verification confirmed
20 consecutive training epochs, 22 test records (initial, each epoch, and
final), 22 checkpoints (initial, each epoch, and final), finite history
metrics, and finite floating-point tensors in the final model state.
