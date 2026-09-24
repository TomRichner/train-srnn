# Modal dispatch (`cloud/modal_app.py`)

Reference for running training on a [Modal](https://modal.com/docs) GPU. It
replaces the GCE VM pipeline in [cloud.md](cloud.md) without removing it: there
is no VM to create, reset, stop or delete, no deploy key, and no quota of one
GPU. Each run is one call of a Modal function that exists only while it runs and
is billed per second. The GCE scripts remain for reproducing older runs.

## Prerequisites

- A Modal account, and `uv sync` (the `modal` package is in the `dev` group).
- `uv run modal setup` once per machine. It opens a browser and stores a token
  in `~/.modal.toml`, outside the repository. Every project on the machine
  shares it.
- The task's dataset on the `srnn-data` Volume (next section). Data-free tasks
  such as `synthetic` need nothing.

Modal Volumes need no enabling; `cloud/modal_app.py` creates `srnn-data` and
`srnn-results` on first use.

## Staging a dataset (once per task)

```bash
uv run modal volume put srnn-data "$SRNN_HOME/data/cheetah100" /cheetah100
uv run modal volume ls srnn-data /cheetah100
```

The container reads `srnn-data:/<task>/` through `SRNN_DATA_DIR`, and the
Volume is mounted read-only. `runtime_provenance.json` records the SHA-256 of
every `*.npz` and `manifest.json` it saw, so a stale upload shows up in the run.

## Launching a run

```bash
uv run modal run --detach cloud/modal_app.py --run-name myrun --task cheetah100 \
    --args "epochs=3 device=cuda checkpoint_interval=1 warmup_epochs=1"
```

| Option | Default | Meaning |
|---|---|---|
| `--run-name` | required | Top-level result folder; must not already exist. |
| `--task` | required | Task name, also the dataset folder. |
| `--model` | `srnn` | Model config. |
| `--seed` | `1` | Dispatch seed, as on GCE. |
| `--args` | empty | Hydra overrides as one string. |
| `--gpu` | `L4` | Any Modal GPU string, e.g. `A10`, `L40S`, `A100-40GB`. |
| `--image` | `default` | `default` (`uv.lock`) or `reference` (see below). |
| `--commit` | worktree | Ship `git archive` of this full 40-character SHA. |
| `--allow-dirty` | off | Ship a working tree with uncommitted tracked changes. |
| `--overwrite` | off | Replace an existing result directory. |

Always pass `--detach`. The launcher waits for the run and streams its log, but
with `--detach` a closed laptop, lost connection or Ctrl-C leaves the run going.
Without it, the run stops when the local process does.

Arguments are merged exactly as `submit.sh` merges them: `ARGS` from
`cloud/experiments/<task>.env` in the shipped code comes first, then `--args`
with single quotes removed. Words are split on whitespace, so
`model.variants=[a,b]` needs no quoting. Hydra is last-wins, so a later
`epochs=20` overrides the `epochs=60` in `cheetah100.env`.

## What code runs

The launcher ships the code as a tarball argument to the function, and the
container extracts it to `/root/train-srnn`:

- **Default: the working tree.** The launcher refuses if any tracked file has
  uncommitted changes, unless `--allow-dirty` is passed. Only tracked files
  ship (`git ls-files`); untracked files never do, so a stray `.env` or scratch
  script cannot leak into a run. The launcher warns about untracked `*.py`
  files, since they will be missing in the container.
- **`--commit=<sha>`:** ship `git archive <sha>`, regardless of the working tree.
  Use this to rerun old code, such as the regression baseline at `5e92be7`.

The run directory keeps `code.tar.gz` and `code.sha256`: the exact code that
ran, including any uncommitted edits. The working-tree tarball is
deterministic, so identical content gives an identical hash.

`cloud/modal_run.py` and `cloud/run_telemetry.py` always come from the
current checkout, whatever `--commit` ships, so an old commit runs under the
current launcher.

## Images

- **`default`:** Debian with Python 3.12 and exactly the packages in `uv.lock`
  (`uv sync --frozen --no-dev`), currently torch 2.14.0+cu130. This differs from
  GCE, where the Deep Learning VM's system Python 3.10 and torch 2.9.1+cu129
  were used rather than the lock.
- **`reference`:** Python 3.10, torch 2.9.1+cu129 and the numpy, scipy,
  pandas, hydra-core, omegaconf and tqdm versions of the GCE regression baseline
  (`python_packages.txt`). It exists to separate "Modal vs GCE" from
  "torch 2.14 vs 2.9.1" in the regression comparison; use `default` otherwise.

Images build once and are cached, so later runs start in seconds. A change to
`uv.lock` rebuilds `default`. Code changes never rebuild an image, since the
code arrives as an argument.

## Results

A run writes directly to the `srnn-results` Volume, in the same layout as the
GCS bucket:

```
srnn-results:/results-pytorch/<run>/<model>/<task>/seed<seed>/
    init.pt, epoch_*.pt, last.pt, training_history.csv, test_history.csv,
    progress.json, train.log, .hydra/                    (from train.py)
    training_log.txt                                      (full stdout/stderr)
    runtime_provenance.json, gpu_memory_samples.csv,
    gpu_memory_summary.json, run_metadata.json            (as on GCE)
    code.tar.gz, code.sha256                              (Modal only)
```

`train.py` gets `output_dir=<that path>`, so the run's resolved config records
it instead of `$SRNN_HOME/results/...`. Volumes save in the background every
few seconds and once more at shutdown, so a crashed run keeps what it wrote.
There is no upload loop.

During a run, the Volume shows each file as it was when last closed.
Checkpoints, `progress.json` and the history CSVs are rewritten or reopened for
each write, so they stay current. `training_log.txt`, `train.log` and
`gpu_memory_samples.csv` stay open for appending, so on the Volume they look
frozen near their start until the run ends. Follow a live run with
`modal app logs` instead.

- **`runtime_provenance.json`:** the GCE fields, plus `launcher: "modal"`,
  image, GPU, code source and hash, the Modal function-call, container and
  image IDs, the full `train.py` argv, and every installed package version.
- **`run_metadata.json`:** the GCE fields. `vm_name` holds the Modal
  function-call ID and `hardware` is `modal-<gpu>`. A nonzero exit adds `error`,
  `failed_at` and, for exceptions, `error_message`. It is written on every
  exit, including `modal app stop`.

Pull runs back with the usual tools. Modal is their default source; add
`--source gcs` for GCE runs:

```bash
uv run python scripts/postprocess.py myrun --task cheetah100       # into $SRNN_CACHE_DIR/myrun/
uv run python cloud/collect_results.py myrun --seeds 1
uv run modal volume get srnn-results results-pytorch/myrun/srnn/cheetah100/seed1 ./myrun
```

## Monitoring and stopping

```bash
uv run modal app list                          # running and recent apps
uv run modal app logs <app-id>                 # printed by the launcher
uv run modal app stop <app-id>                 # SIGTERM: metadata is still written
uv run modal volume get srnn-results results-pytorch/<run>/.../run_metadata.json -
```

The dashboard (`uv run modal dashboard`) shows the same logs, plus GPU and
memory use.

## Guards and limits

- **Collision:** the launcher refuses an existing result directory unless
  `--overwrite` is passed, and `--overwrite` deletes that directory before
  training. GCE silently overwrote in place.
- **Preemption:** GPU functions can be preempted, and Modal cannot exempt them.
  The docs call it rare. A preempted input may be delivered again even with
  `retries=0`. The runner then finds a populated result directory, writes
  `redelivery_<time>.json` beside it, and stops without touching the existing
  files. The run has to be relaunched under a new name. Runs are not yet
  resumable (see [known_issues.md](known_issues.md)); making them resumable
  would let Modal retry and continue instead.
- **Timeout:** 24 hours, Modal's maximum. The 15-seed manuscript run took about
  4 hours. A run killed at the timeout may skip `run_metadata.json`.
- **Concurrency:** the Starter plan allows 10 GPUs at once. Separate launches
  run in parallel.

## Debugging on a GPU

```bash
uv run modal shell cloud/modal_app.py::train          # same image, GPU and Volumes
uv run modal container list
uv run modal container exec <container-id> bash       # into a live run
```

A `modal shell` session starts without the code. Extract a run's code with
`mkdir -p /root/train-srnn && tar -xzf /vol/results/results-pytorch/<run>/.../code.tar.gz -C /root/train-srnn`,
or clone the public repository. Anything written outside a Volume is lost when
the container exits. torch.compile caches are not persisted; compiling the cell
takes about 20 seconds.

## Cost

An L4 costs $0.000222 per second (about $0.80 per hour), plus small CPU and
memory charges. The Starter plan includes $30 of compute per month. The
20-epoch regression run costs about $0.20. See
[modal.com/pricing](https://modal.com/pricing) for current rates.

## Files

- `cloud/modal_app.py`: Modal definitions only. Two images, the Volumes,
  `train` and `train_reference` (one body, one per image; `gpu="L4"`,
  `timeout=24h`, `retries=0`), and the local entrypoint.
- `cloud/modal_run.py`: plain Python, unit-tested in `tests/test_modal_run.py`.
  Launch side: git checks, the code tarball, argument merging. Container side:
  `run_training()`, the counterpart of `startup_gpu.sh`.
- `cloud/run_telemetry.py`: shared with GCE. Provenance, GPU sampling and
  summary, and `run_metadata.json`.
