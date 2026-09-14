# Cloud dispatch layer (`cloud/`)

Reference for how a training run gets onto a GCP GPU VM, what happens on the
VM, where results land in GCS, and how they come back to the laptop. One
pipeline: a VM-native startup script triggered by `gcloud compute instances
create` (first launch) or by `add-metadata` + `start`/`reset` (re-dispatch).
The laptop is never tethered to the run; it can sleep.

Everything in `config.gpu.env` (project, bucket, zone, deploy-key repo URL)
is the author's own GCP setup. Anyone else must point these at their own
project and bucket before any script here will work.

## Prerequisites

- `gcloud` installed and authenticated (`gcloud auth login`), with the
  project in `config.gpu.env` selected or passed via `--project`.
- Compute Engine, Cloud Storage and (for private repos) Secret Manager APIs
  enabled. VMs are created with `--scopes=cloud-platform`, so the default
  compute service account needs `secretmanager.secretAccessor` on the
  deploy-key secret and write access to the bucket.
- A regional GPU quota for the chosen accelerator (default `nvidia-l4`,
  which starts at 1 per region). `MAX_CONCURRENT_VMS=1` reflects that.
- If `REPO_URL` is an ssh URL (`git@github.com:...`), a Secret Manager
  secret named `train-srnn-deploy-key` holding a read-only GitHub deploy
  key for the repo. An https URL to a public repo needs no secret.
- The dataset for the task already uploaded to
  `gs://<bucket>/datasets/<task>/` (see GCS layout below).

## Files

### `config.gpu.env`

Sourced by every local script. Sets `GCP_PROJECT`, `GCP_ZONE`,
`GCP_BUCKET`, `GCP_USE_SPOT`, `REPO_URL`, the Deep Learning VM image
(`GCP_IMAGE_FAMILY` / `GCP_IMAGE_PROJECT`, a `pytorch-*-cu*` family with
torch, CUDA and the NVIDIA driver preinstalled), and the machine shape
(`g2` + `standard-8`, `nvidia-l4` x1, 100 GB `pd-balanced`). Zone and
machine values are `${VAR:-default}` so they can be overridden per call
without editing the file, e.g. `GCP_ZONE=us-central1-a` when a zone is
stocked out, or `GPU_TYPE=nvidia-tesla-t4 DEFAULT_MACHINE_TYPE_FAMILY=n1`
for a T4 fallback. `ALL_MODELS` / `ALL_EXPERIMENTS` are informational; no
script iterates them. `scripts/_runs.py:read_bucket()` parses `GCP_BUCKET`
out of this file so the analysis scripts share the bucket.

### `experiments/<task>.env`

One per task, each defining a single `ARGS` string of Hydra overrides that
is prepended to whatever the caller passes (Hydra is last-wins, so caller
args override). Kept minimal so the task config in `train_srnn/config.py`
stays authoritative: most are `epochs=200 model.num_units=32`;
`cheetah100.env` is `epochs=60` plus comments recording the exact launch
lines for the `ring6-400e` and `ring2x5-100e` runs. A task with no `.env`
launches with the caller's args only.

### `launch_run_gpu.sh` (local, first launch)

```
cloud/launch_run_gpu.sh [--cleanup=delete|stop|keep] [--branch=<name>] \
    <run_name> <task> <model> <seed> [hydra args...]
```

One VM per run. Steps:

1. Parse `--cleanup` (default `delete`) and `--branch` (default `main`).
2. Strip single quotes from the trailing args so `'[a,b]'` typed for zsh is
   stored as `[a,b]`.
3. VM name is `<run_name>-<model>-<task>-seed<seed>` with underscores
   replaced by hyphens. If a VM of that name exists, exit 0 without doing
   anything. If the count of VMs whose name starts with `<run_name>-` is at
   `MAX_CONCURRENT_VMS`, exit 1.
4. Source `experiments/<task>.env` and prepend its `ARGS`.
5. Write the final train args to a temp file and pass it with
   `--metadata-from-file=train-args=...`. This dodges gcloud's
   comma-delimited parsing of `--metadata=`, which would otherwise split
   `model.variants=[a,b]`.
6. `gcloud compute instances create` with the accelerator,
   `--maintenance-policy=TERMINATE` (required for GPU VMs; spot adds
   `--provisioning-model=SPOT --instance-termination-action=STOP`),
   `--scopes=cloud-platform`, and metadata keys `run-name`, `experiment`,
   `model`, `seed`, `bucket`, `cleanup`, `skip-refresh=0`, `branch`,
   `repo-url`, `install-nvidia-driver=True`, plus
   `startup-script=cloud/startup_gpu.sh` and `train-args` from file.
7. Return immediately. For `stop`/`keep` it prints the matching
   `submit.sh` and `stop_vm.sh`/`start_vm.sh` lines.

### `startup_gpu.sh` (runs on the VM at every boot)

Executed by `google-startup-scripts.service` on first boot and again on
every `start` or `reset`, so it is the single entry point for both first
launches and re-dispatches. Reads all metadata keys from the instance
metadata server (`cleanup` defaults to `delete`, `skip-refresh` to `0`,
`branch` to `main`, `repo-url` to the ssh URL in `config.gpu.env`).

- Log: `/var/log/training-<run_name>-<seed>.log`, per run so re-dispatches
  on one VM do not accumulate in a single file. Uploaded to GCS as
  `training_log.txt`.
- Working dirs depend on the lifecycle mode. `delete`: repo at
  `/tmp/workdir`, `SRNN_HOME=/tmp/srnn-work`. `stop`/`keep`: repo at
  `/opt/train-srnn`, `SRNN_HOME=/opt/srnn-work`, both persistent across
  reboots so a re-dispatch reuses the checkout, the staged dataset and the
  installed Python packages. `SRNN_HOME` is exported so `train_srnn/paths.py`
  resolves `data_dir`, `results_dir` and `cache_dir` beneath it.
- Repo (step 1, skipped when `skip-refresh=1` and a checkout exists). For a
  `git@` URL, fetch `train-srnn-deploy-key` from Secret Manager into
  `/root/.ssh/deploy_key`, write an ssh config pinned to that key, and scrub
  both after the clone. For an https URL none of that happens. A fresh VM
  does `git clone --depth 1 --branch <branch>` with three retries; an
  existing checkout does `git fetch --depth 1 origin <branch>` and
  `git reset --hard FETCH_HEAD` (FETCH_HEAD rather than `origin/<branch>`
  because a shallow clone only tracks the branch it was cloned with). The
  full commit sha is recorded for `run_metadata.json`.
- Dataset (step 2, skipped on `skip-refresh=1` if present):
  `gs://<bucket>/datasets/<task>/*` copied to `$SRNN_HOME/data/<task>/`.
  This must match `task.dataset` in `train_srnn/config.py`; every task's
  dataset folder is named after the task.
- Python (step 3, skipped on `skip-refresh=1`): `sudo pip3 install
  hydra-core omegaconf scipy pandas` into the system Python. No venv,
  because the DLVM ships its CUDA-matched torch in system Python and a venv
  would shadow it.
- Waits up to 10 minutes for `nvidia-smi` (the DLVM installs the driver in
  the background on first boot), then asserts `torch.cuda.is_available()`.
- Training (step 4). Output dir is
  `RUN_OUTPUT=$SRNN_HOME/results/<task>/<run_name>_seed<seed>`, matched by
  passing `run_name=<run_name>_seed<seed>` to `train.py` (whose
  `output_dir` is `${paths.results_dir}/${task.name}/${run_name}`). The
  per-run dir keeps re-dispatches on one VM isolated so uploads never ship
  a previous run's files. Invoked as `python3 train.py model=<model>
  task=<task> seed=<seed> <train-args> run_name=...` under `set -f` so the
  shell does not glob `[a,b]`. `PYTHONUNBUFFERED=1` keeps the tee'd log
  live.
- Upload watcher: a background loop that every 30 s reads
  `progress.json` under `RUN_OUTPUT` and, every `epochs/10` epochs (5 if
  `epochs<=50`; `epochs` is parsed from the train args, default 200),
  copies the whole run dir plus the log to the GCS results prefix. A
  preempted or crashed VM therefore leaves partial checkpoints and CSVs
  behind.
- EXIT trap (`cleanup`), fires on success and failure: kills the watcher,
  uploads `RUN_OUTPUT/*` and the log, writes `run_metadata.json`
  (`run_name`, `experiment`, `model`, `seed`, `exit_code`, `vm_name`,
  `hardware`, `start_time`, `completed`, `duration_seconds`, `commit`,
  `train_args`, `cleanup_mode`, `skip_refresh`, and `error`/`failed_at` on
  nonzero exit), then applies the lifecycle mode:
  - `delete`: `gcloud compute instances delete` itself. One-shot, nothing
    persists but the GCS upload.
  - `stop`: `gcloud compute instances stop` itself. Disk persists at
    disk-only cost; restart with `start_vm.sh` or `submit.sh`.
  - `keep`: stay RUNNING for the next `submit.sh`.

### `submit.sh` (local, re-dispatch to an existing VM)

```
cloud/submit.sh <vm_name> <run_name> <task> <model> <seed> \
    [--cleanup=delete|stop|keep] [--skip-refresh] [--branch=<name>] [hydra args...]
```

Default `--cleanup=keep`. Finds the VM's zone by name (it may not be in the
config default zone), reads its status, sources `experiments/<task>.env`,
then `gcloud compute instances add-metadata` rewrites every per-run key
(`run-name`, `experiment`, `model`, `seed`, `bucket`, `cleanup`,
`skip-refresh`, `branch`, `repo-url`) plus `train-args` and the current
`startup_gpu.sh` from file. Then: `start` if the VM is TERMINATED, `reset`
if RUNNING (this kills any in-flight job), fatal otherwise. Both re-run the
startup script with the new metadata. `--skip-refresh` skips git fetch,
dataset copy and pip install as one unit, so use it only when none of the
three changed. `--branch` dispatches a feature branch without merging.
Prints the results prefix, a `gcloud compute ssh ... tail -f` line for the
log, and the `run_metadata.json` path to poll for completion.

### `start_vm.sh`, `stop_vm.sh`

Manual lifecycle for a parked VM. `start_vm.sh <vm>` locates the zone,
starts the instance and polls ssh until sshd answers (its closing hint
still names the removed `run_on_vm.sh`; use `submit.sh`). A plain `start`
re-runs `startup_gpu.sh` with the metadata already on the VM, i.e. the last
dispatched job. `stop_vm.sh <vm>` stops it.

### `collect_results.py`

`python3 cloud/collect_results.py <run_name> [--bucket ...] [--seeds N]
[--models ...] [--experiments ...] [--csv out.csv]`. Read-only against GCS.
Walks `results-pytorch/<run>/<model>/<task>/seed<n>/` for seeds `1..N`,
`gcloud storage cat`s `training_history.csv`, `test_history.csv` and
`run_metadata.json`, and for each variant (batched runs have one row per
variant; single-model runs are keyed `__single__`) picks the checkpoint
epoch with the best validation metric and reports the test metric there.
Prints a mean and std table per model x task and a wall-clock summary from
`duration_seconds`. The `--bucket` default is hardcoded to the author's
bucket rather than read from `config.gpu.env`.

### `inspect_srnn_params.py`

Tabulates SRNN dynamics parameters (softplus-transformed `isp_*` taus,
`a_0`, `c_0_*`) and recurrent weight statistics across `init.pt`,
`best.pt` and `last.pt`, for single-cell and K-variant checkpoints.
`--local <dir>` reads one checkpoint directory. `--run <run>` expects a
mirror of the GCS tree at `$SRNN_CACHE_DIR/collect_results/<run>/<model>/
<task>/seed<n>/` and writes `srnn_params.md` to `--out_dir` (default
`results/<run>/` in the repo). Nothing populates that mirror
automatically; the script's hint about `collect_results.py
--with-checkpoints` refers to a flag that does not exist. Copy with
`gcloud storage cp -r` first.

## GCS layout

```
gs://<bucket>/
  datasets/<task>/                       staged to $SRNN_HOME/data/<task>/ on the VM
  results-pytorch/<run>/<model>/<task>/seed<seed>/
    init.pt, epoch_*.pt, last.pt (best.pt if written)
    training_history.csv, test_history.csv, progress.json
    training_log.txt, run_metadata.json
```

`<run>` is the bare run name (`ring2x5-100e`); the `_seed<n>` suffix is only
on the VM's local output dir. Re-dispatching the same run name and seed
overwrites files in place.

## Local paths (`train_srnn/paths.py`)

All locations derive from `SRNN_HOME` (default `~/srnn`): `SRNN_DATA_DIR`
(`$SRNN_HOME/data`), `SRNN_RESULTS_DIR` (`$SRNN_HOME/results`) and
`SRNN_CACHE_DIR` (`$SRNN_HOME/cache`), each individually overridable and
exposed to Hydra as `${srnn_path:data|results|cache}`. The VM sets
`SRNN_HOME` itself; locally export it in your shell.

## Pulling a run back: `scripts/postprocess.py`

`python scripts/postprocess.py <run> [--task cheetah100] [--seed 1]
[--bucket ...] [--tmp-dir ...] [--skip-download]`. `ensure_local_run` lists
`gs://<bucket>/results-pytorch/<run>/srnn/<task>/seed<seed>/` (model is
fixed to `srnn`), copies every file not already present and non-empty into
`$SRNN_CACHE_DIR/<run>/`, and requires `last.pt` and
`training_history.csv` to exist before continuing to the plots and report.
The bucket default comes from `config.gpu.env` via `scripts/_runs.py`.

## Flow

```
laptop: launch_run_gpu.sh -> instances create      submit.sh -> add-metadata -> start/reset
VM:     startup_gpu.sh (every boot)
          read metadata -> clone or fetch+reset (deploy key if ssh URL)
          stage datasets/<task> -> pip install -> wait for driver, CUDA check
          train.py -> $SRNN_HOME/results/<task>/<run>_seed<s>   (watcher uploads every epochs/10)
          EXIT: upload + run_metadata.json -> delete | stop | keep
laptop: postprocess.py / collect_results.py <- results-pytorch/<run>/<model>/<task>/seed<s>
```

## Recipes

First launch, VM parks itself after the run:

```bash
bash cloud/launch_run_gpu.sh --cleanup=stop ring2x5-100e cheetah100 srnn 1 \
    "model.variants=[srnn-no-dales-skip,srnn-no-adapt-no-dales-skip] \
     model.variant_seeds=[1,2,3,4,5] epochs=100 checkpoint_interval=5 warmup_epochs=3"
# VM name: ring2x5-100e-srnn-cheetah100-seed1
```

Re-dispatch a new run to that VM (stopped or running), on a feature branch,
and delete the VM when done:

```bash
bash cloud/submit.sh ring2x5-100e-srnn-cheetah100-seed1 ring2x5-200e cheetah100 srnn 1 \
    --cleanup=delete --branch=cleanUp \
    "model.variants=[srnn-no-dales-skip,srnn-no-adapt-no-dales-skip] epochs=200"
```

Same code and data already on the VM, skip the refresh:

```bash
bash cloud/submit.sh <vm> <run> cheetah100 srnn 2 --skip-refresh "epochs=50"
```

Watch and download:

```bash
gcloud compute ssh <vm> --zone=<zone> -- sudo tail -f /var/log/training-<run>-<seed>.log
gcloud storage cat gs://<bucket>/results-pytorch/<run>/srnn/cheetah100/seed1/run_metadata.json
python scripts/postprocess.py <run> --task cheetah100        # into $SRNN_CACHE_DIR/<run>/
python3 cloud/collect_results.py <run> --seeds 1
```

Park or wake a kept VM by hand:

```bash
bash cloud/stop_vm.sh <vm>
bash cloud/start_vm.sh <vm>    # note: re-runs the last dispatched job
```
