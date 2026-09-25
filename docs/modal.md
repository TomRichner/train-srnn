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
| `--resume` | off | Continue an existing result directory (see Resuming). |
| `--dataset` | the task name | Train on `srnn-data:/<dataset>` (adds `task.dataset=`). |
| `--wait` | off | Wait for the run and print its exit status. |

Always pass `--detach`. The launcher *spawns* the training call and returns at
once, printing the call ID; the run then continues on Modal whatever happens to
the laptop. Follow it with `modal app logs` or the Volume. `--wait` also waits for
the result; killing a waiting launcher does not cancel the run. (An awaited
`.remote()` call would be cancelled when the launcher process dies, even under
`--detach`, which is why the launcher spawns.) Without `--detach`, the app stops
when the launcher returns.

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
- **Preemption and crashes:** GPU functions can be preempted, and Modal cannot
  exempt them; the docs call it rare. A preempted input is delivered again, and
  a crashed container is retried up to twice (`modal.Retries`, 30 s apart). The
  runner always passes `resume=true` when the shipped code supports it, so a
  second attempt continues from the newest checkpoint (see Resuming). Each
  attempt is appended to `attempts.jsonl`; later attempts write
  `runtime_provenance_attempt<N>.json` and append to the GPU sample CSV. Code
  older than `resume=true` (for example `--commit=5e92be7…`) cannot resume: a
  second delivery then writes `redelivery_<time>.json` and stops without touching
  the existing files.
- **Timeout:** 24 hours, Modal's maximum. The 15-seed manuscript run took about
  4 hours. A run killed at the timeout may skip `run_metadata.json`; continue it
  with `--resume`.
- **Concurrency:** the Starter plan allows 10 GPUs at once. Separate launches
  run in parallel.

## Resuming

`resume=true` (see [architecture.md](architecture.md), "Run directory") continues
a run directory from its newest checkpoint with the same epoch counter, carried
state and RNG streams; on CPU the result is bitwise identical to an uninterrupted
run. To continue a run that was stopped, timed out or failed:

```bash
uv run modal run --detach cloud/modal_app.py --run-name myrun --task cheetah100 \
    --args "<the same args>" --resume
```

Up to `checkpoint_interval` epochs are recomputed. Pass the same arguments as the
original launch; the run keeps its directory, histories and `run_metadata.json`
is rewritten at the end.

Checked on an L4 (2026-09-25): a 6-epoch `cheetah100` run stopped with
`modal app stop` after its epoch-2 checkpoint and relaunched with `--resume`
continued at epoch 3 in a new container. Against an uninterrupted twin, its
histories were identical at the CSVs' six decimals, each epoch appeared once, and
its weights differed by at most 1.2e-7 (relative L2 4.5e-10), the float32 noise
already present between the two runs at epoch 0. The twin's launcher was killed
right after spawning, and the run still completed.

## Running analysis scripts on a GPU

`cloud/modal_analyze.py` runs any repository script next to the Volumes on an L4,
with the `default` image. `{results}` and `{data}` in `--args` become the mount
points, and the log goes to `srnn-results:/analysis_logs/<time>_<script>.log`:

```bash
uv run modal run --detach cloud/modal_analyze.py --script scripts/eval_speed.py \
    --args "{results}/results-pytorch/myrun/srnn/cheetah100/seed1 --data-root {data} \
            --tests cheetah100_fixed_speeds/test_r*.npz"
```

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
  `timeout=24h`, two retries), and the local entrypoint.
- `cloud/modal_run.py`: plain Python, unit-tested in `tests/test_modal_run.py`.
  Launch side: git checks, the code tarball, argument merging. Container side:
  `run_training()`, the counterpart of `startup_gpu.sh`, and `run_script()`.
- `cloud/modal_analyze.py`: the analysis app (`analyze` function and launcher).
- `cloud/run_telemetry.py`: shared with GCE. Provenance, GPU sampling and
  summary, and `run_metadata.json`.
