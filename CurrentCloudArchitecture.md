# Current Cloud Architecture (`cloud/`)

A descriptive snapshot of every script under `cloud/` as of this writing.
This document is intentionally non-prescriptive — it records what each
piece does and how they connect, so we can decide what to keep, what to
change, and what to delete.

## Mental model

There are **two pipelines** that share a common pattern (VM-native
startup script triggered by `gcloud compute instances create`):

- **CPU pipeline** — `launch_run.sh` + `startup.sh`, configured via
  `config.env`. Used for sweeps across `n4d` family + custom-built
  PyTorch CPU image.
- **GPU pipeline** — `launch_run_gpu.sh` + `startup_gpu.sh`, configured
  via `config.gpu.env`. Uses Google's Deep Learning VM (DLVM) base
  image with CUDA-matched torch pre-installed, on `g2-standard-8` (L4)
  by default.

Both pipelines follow the same lifecycle on the VM: boot →
google-startup-scripts.service runs the metadata-loaded script → fetch
GitHub deploy key from Secret Manager → clone repo → download dataset
→ run training → upload results to GCS → self-delete VM.

A third, **bolted-on dev-mode pipeline** sits on top of the GPU
pipeline:

- `launch_run_gpu.sh --keep-alive ...` — first-run path; same as a
  one-shot launch but the VM doesn't self-delete and the workdir lives
  at `/opt/train-srnn` instead of `/tmp/workdir`.
- `run_on_vm.sh` + `run_on_vm_remote.sh` — re-run path; SSHes into the
  existing keep-alive VM, refreshes the repo, runs another training
  session, uploads results.
- `start_vm.sh` / `stop_vm.sh` — lifecycle helpers for parking the dev
  VM overnight.

This dev-mode is the part that's currently uncomfortable: it tethers a
long-running training process to a synchronous SSH session from the
local laptop.

## Files

### Entry-point scripts (local → cloud)

#### `launch_run.sh` (75 lines, CPU)
- One VM = one training run.
- Reads `config.env` (project, zone, image family, machine type, spot
  flag, concurrency cap, experiment matrix).
- Reads `experiments/<task>.env` if present to override `MACHINE_TIER`
  and prepend a per-task `ARGS` string.
- Names the VM `<run_name>-<model>-<experiment>-seed<seed>` (with
  underscores → hyphens, since GCP doesn't allow underscores).
- Skips creation if the VM already exists; refuses to launch if the
  global concurrency cap (`MAX_CONCURRENT_VMS`) is reached.
- Writes `train-args` to a temp file and passes it via
  `--metadata-from-file=` so commas in Hydra list values
  (`batched_ablations=[a,b,c]`) aren't mis-parsed as gcloud flag
  delimiters.
- Creates the VM with `--metadata-from-file=startup-script=startup.sh,
  train-args=...` and `--scopes=cloud-platform` (needed for the VM to
  read Secret Manager + write GCS).
- Returns immediately after the create call; never SSHes into the VM.

#### `launch_run_gpu.sh` (91 lines, GPU)
- Same shape as `launch_run.sh` but for GPU.
- Reads `config.gpu.env` (env-var overridable for GPU type, machine
  family, disk size).
- Adds `--accelerator=type=...,count=...` and the
  `install-nvidia-driver=True` metadata flag (DLVM honors this on first
  boot to install the matched proprietary driver).
- New `--keep-alive` first-positional flag (added recently): when set,
  passes `keep-alive=1` in metadata; the startup script reads this and
  skips the self-delete. Also prints helper text suggesting follow-up
  `run_on_vm.sh` / `stop_vm.sh` invocations.
- Forces `--maintenance-policy=TERMINATE` (required for GPU VMs).

#### `launch_all.sh` (78 lines, CPU sweep)
- Wraps `launch_run.sh` to fan out a full
  models × experiments × seeds matrix.
- Phase 1: bursts launches in batches of 8 with a 15 s pause; respects
  the global concurrency cap.
- Phase 2: when the cap fills, switches to a sequential mode that polls
  for free slots before each launch.
- No GPU equivalent currently exists.

### VM-side scripts (run on the VM, not locally)

#### `startup.sh` (197 lines, CPU pipeline)
Triggered by `google-startup-scripts.service` on first boot.
1. `tee` everything to `/var/log/training.log`.
2. Read `run-name`, `experiment`, `model`, `seed`, `train-args`,
   `bucket` from the instance metadata server.
3. Install an `EXIT trap` (`cleanup`) that always uploads results,
   uploads the log, writes `run_metadata.json` with exit code +
   timing + commit, **then deletes the VM**.
4. Fetch deploy key from Secret Manager
   (`gcloud secrets versions access latest --secret=train-srnn-deploy-key`).
5. `git clone --depth 1` the private repo into `/tmp/workdir`. Scrub
   the deploy key from disk after clone.
6. Copy the dataset from `gs://.../datasets/<experiment>/`.
7. Activate `/opt/python-venv` if the custom image is in use, else
   build a fresh `/tmp/venv` and `pip install -r requirements.txt`.
8. Spawn a background "upload watcher" that every 30 s reads
   `progress.json` and uploads partial results to GCS every
   `EPOCHS/10` epochs (so a long sweep keeps incremental data even if
   the VM gets preempted).
9. Run `python3 train.py` with `set -f` (so Hydra's `[a,b,c]` list
   syntax doesn't get globbed by the shell).
10. EXIT trap fires whether training succeeded or failed → final
    upload + self-delete.

#### `startup_gpu.sh` (233 lines, GPU pipeline)
Almost identical to `startup.sh`, with these differences:
- Reads an extra `keep-alive` metadata key.
- If `keep-alive=1`: `WORKDIR=/opt/train-srnn` (persistent across
  reboots); `cleanup` skips the self-delete; clone path includes a
  "if .git already exists, just `git fetch + reset --hard origin/main`"
  branch so a stop/start cycle reuses the on-disk repo.
- DLVM-specific Python setup: installs the Hydra stack into the
  *system* Python (no venv) because the DLVM ships a CUDA-matched
  torch in system Python that a venv would shadow.
- "Wait for NVIDIA driver" loop (up to 10 min) before the CUDA sanity
  check. The DLVM installs the driver on first boot in the background;
  jumping straight to `python -c "import torch; assert
  torch.cuda.is_available()"` would race and fail.
- Sets `PYTHONUNBUFFERED=1` so log lines flush in real time through
  the `tee` pipe.

#### `run_on_vm_remote.sh` (120 lines, dev-mode)
A trimmed sibling of `startup_gpu.sh` that runs on an *existing*
keep-alive VM. Invoked over SSH by `run_on_vm.sh`.
- Re-fetches the deploy key per run, scrubs after — uniform "no
  long-lived key on disk" pattern.
- `cd /opt/train-srnn && git fetch + reset --hard origin/main` to
  refresh the checkout to current main.
- Re-copies the dataset (idempotent overwrite).
- Re-runs the system pip install (no-op if already satisfied).
- CUDA sanity check (no driver wait — assumes driver is already up,
  which is true if the VM has been booted at least once since image
  install).
- `python3 train.py ...` writes results into a unique
  `results/<exp>/<run_name>_seed<seed>/` directory so concurrent /
  successive runs don't clobber each other.
- Uploads results, log, and `run_metadata.json` to
  `gs://.../results-pytorch/<run_name>/<model>/<exp>/seed<seed>/`.
- Exits with the python exit code.
- **Does not self-delete** — the dev VM is persistent.

### Local-side dev-mode wrappers

#### `run_on_vm.sh` (65 lines)
- Looks up the dev VM's zone via `gcloud compute instances list
  --filter=name=$VM_NAME` (so the VM can live in any zone, not just
  the config default).
- Refuses to run if the VM isn't `RUNNING`.
- `gcloud compute scp` the runner script + a temp file containing the
  Hydra train-args.
- `gcloud compute ssh ... --command="sudo bash run_on_vm_remote.sh
  ..."` — **synchronous foreground SSH**.
- Adds `ServerAliveInterval=60`, `ServerAliveCountMax=10`, and
  `TCPKeepAlive=yes` in the SSH command to fight idle-timeout NAT
  drops during long quiet training stretches.
- **Pain point:** the remote training is tethered to this SSH
  session. If the local laptop sleeps, ServerAlive can't help (no
  packets from our side), the channel eventually dies, the remote
  bash gets `SIGHUP`, and the python process is killed. This is what
  killed last night's overnight 10-epoch run mid-training.

#### `start_vm.sh` (41 lines)
- Looks up VM zone from list filter.
- `gcloud compute instances start`.
- Polls `gcloud compute ssh --command=true` (with 5 s connect
  timeout, 30 attempts × 10 s) so it doesn't return until sshd is
  actually answering. (Earlier version returned as soon as the GCE
  API said RUNNING — premature, since the OS may not have booted.)

#### `stop_vm.sh` (23 lines)
- Looks up zone, calls `gcloud compute instances stop`.
- Stopped VM costs only ~$0.02/hr (disk only); RUNNING L4 is
  ~$0.70/hr. No work happens while stopped, but a `start_vm.sh` later
  has the same capacity-stockout risk as a fresh launch.

### Operator tooling

#### `monitor.sh` (58 lines, CPU sweep)
- For a given `<run_name>`, prints two tables:
  1. Currently running VMs whose names start with `<run_name>-`.
  2. A models × experiments completion grid — counts how many seed
     directories exist in `gs://.../results-pytorch/<run_name>/...`
     and shows progress like `3/5`, `done`, `2+run`, `.` (not
     started).
- Pulls model/experiment lists from `config.env`'s `ALL_MODELS` /
  `ALL_EXPERIMENTS`.
- No GPU-specific awareness; the same script works for any sweep.

#### `collect_results.py` (309 lines)
- Aggregates final per-seed metrics from GCS for a given run name.
- For each `(model, experiment, seed)`, reads
  `training_history.csv` + `test_history.csv` from GCS and picks the
  checkpoint epoch with best validation metric, reporting the test
  metric at that epoch.
- Handles both single-model runs and batched-ablation runs (one row
  per variant).
- Emits a wall-clock + CPU-hours summary using
  `run_metadata.json.duration_seconds`, plus a mean ± std table.
- Optional CSV export.

#### `inspect_srnn_params_pytorch.py` (437 lines)
- Loads `init.pt` / `best.pt` / `last.pt` checkpoints (local or
  downloaded from GCS) and prints/exports a markdown table of
  SRNN-specific dynamics parameters and weight statistics across the
  three stages.
- Handles both single-cell and `BatchedSRNNCell` (K-variant)
  checkpoints.
- Independent of the launch/runtime layer — a pure analysis tool.

### Image build

#### `build_image.sh` (57 lines, CPU only)
- One-time helper to build a custom Debian + PyTorch CPU image:
  spawns a temp e2 VM, installs python+torch+hydra into
  `/opt/python-venv`, stops the VM, snapshots the disk into a
  `srnn-pytorch` image family, deletes the temp VM.
- Uses the `--scopes=storage-full` for the build VM only.
- Not used by the GPU pipeline (DLVM has torch pre-installed).
- Has not been re-run in a while; the family `srnn-pytorch` is what
  `config.env` points at.

### Config files

#### `config.env`
- Project, zone, bucket, custom CPU image family, spot flag.
- `n4d-standard-8` default with `hyperdisk-balanced` (n4d requires
  hyperdisk).
- `MAX_CONCURRENT_VMS=32` (global concurrency cap, used by both
  `launch_run.sh` and `launch_all.sh`).
- `ALL_MODELS` + `ALL_EXPERIMENTS` strings consumed by
  `launch_all.sh` and `monitor.sh`.

#### `config.gpu.env`
- GPU equivalent: L4 + g2-standard-8 + pd-balanced (g2 doesn't
  support hyperdisk).
- All values are env-var overridable
  (`GCP_ZONE=... GPU_TYPE=... ./cloud/launch_run_gpu.sh ...`) so a
  caller can swap GPU type / zone for stockout cycling without
  editing the file.
- Pinned to DLVM family `pytorch-2-9-cu129-ubuntu-2204-nvidia-580`
  (Turing+ only, sm_75+); a comment documents the deprecated
  `nvidia-570` image as a backup for Pascal cards (P4/P100).
- `MAX_CONCURRENT_VMS=1` (regional L4 quota usually 1).
- `ALL_MODELS` is a *narrower* list than `config.env`'s — excludes
  models that don't benefit from GPU (LSTM, CTGRU, LTC variants).

#### `experiments/<task>.env`
- One per task. Sets `EXPERIMENT_NAME`, an `ARGS` string of Hydra
  overrides specific to that task (e.g. epochs, bptt config, batch
  size), an optional `MACHINE_TIER` override, and `N_SEEDS`.
- Sourced by `launch_run.sh` and `launch_run_gpu.sh`. The `ARGS`
  string is *prepended* to user-supplied trailing CLI args, so user
  overrides win (Hydra last-wins).
- `seeg.env` is the only one carrying notes about model.h /
  model.ode_unfolds being CLI-only because they'd error on non-SRNN
  models if put in `ARGS`.

## Coupling map

```
local                              VM
─────────────                      ──────────────────────
launch_run.sh ──────► gcloud create ────► startup.sh
launch_all.sh ──┐
                └──► launch_run.sh

launch_run_gpu.sh ──► gcloud create ────► startup_gpu.sh
                                            │
                                            ├ if keep-alive=0: self-delete (one-shot)
                                            └ if keep-alive=1: stays up

run_on_vm.sh ──► scp + ssh --command  ──► run_on_vm_remote.sh
                 (synchronous, tethered)    (runs in foreground bash)

start_vm.sh ──► gcloud start + ssh-ready poll
stop_vm.sh  ──► gcloud stop

monitor.sh           ──► gcloud list + GCS ls   (read-only)
collect_results.py   ──► GCS cat + parse        (read-only)
inspect_srnn_*.py    ──► local/GCS .pt loader   (read-only)
```

## Observations / friction points (descriptive, not prescriptive)

1. **Two parallel pipelines (CPU + GPU) with ~80% overlapping logic.**
   `startup.sh` and `startup_gpu.sh` differ mainly in Python setup +
   driver wait + keep-alive support. CPU pipeline doesn't have a
   keep-alive equivalent.

2. **One-shot launch pattern is robust.** `launch_run_gpu.sh` (no
   `--keep-alive`) is bulletproof: VM-native startup script, fully
   detached from local. The CPU pipeline shares this property.

3. **Dev-mode dispatch is fragile.** `run_on_vm.sh` runs the long
   training session via a synchronous foreground SSH command. Local
   laptop sleep / network drop kills the remote work. `launch_run_gpu.sh`
   does not have this problem because it never opens an SSH session
   for the work — the startup script runs as a systemd unit on the VM.

4. **Three lifecycle modes are not symmetric.** Today:
   - **Self-delete** (one-shot): supported by both pipelines via the
     EXIT trap in `startup_gpu.sh` / `startup.sh`. No flag needed; the
     trap deletes by default.
   - **Stay running** ("keep-alive"): supported only by GPU pipeline,
     only at first launch (`launch_run_gpu.sh --keep-alive`). There's
     no way to dispatch a *new* job to a running VM without going
     through SSH.
   - **Stop after run**: not supported as a built-in mode. The user
     can chain `... ; bash cloud/stop_vm.sh ...` after a `run_on_vm.sh`
     invocation, but that piggybacks on `run_on_vm.sh` blocking until
     the SSH-tethered work completes — and inherits its fragility.

5. **Dispatch mechanisms in use:**
   - `gcloud compute instances create --metadata-from-file=startup-script=...`
     → robust, VM-native, used by both `launch_run*.sh`.
   - `gcloud compute ssh --command="bash ..."` → fragile (tethered),
     used by `run_on_vm.sh`.
   - `gcloud compute instances reset` → not used anywhere, but would
     re-trigger the existing startup script; would let us re-use the
     VM-native dispatch path for re-runs.
   - `gcloud compute instances add-metadata` → not used, but supports
     updating `train-args` on a running VM in place.

6. **GCS layout is consistent across all run types**:
   `gs://liquidneuralnets-experiments/results-pytorch/<run_name>/<model>/<exp>/seed<seed>/`
   contains `init.pt`, `last.pt`, `best.pt` (if present),
   `training_history.csv`, `test_history.csv`, `progress.json`,
   `training_log.txt`, `run_metadata.json`. `collect_results.py` and
   `monitor.sh` both rely on this layout.

7. **Local laptop is in the loop in three places today**:
   - `launch_*.sh` (only at create-time — fire-and-forget afterwards). OK.
   - `monitor.sh`, `collect_results.py`, `inspect_srnn_params_pytorch.py`
     (interactive read-only tools — running them is always a manual
     "check on things" action). OK.
   - `run_on_vm.sh` (long-running tether for the duration of training).
     Not OK.

## What's not in `cloud/`

- The training pipeline itself (`train.py`, `train_srnn/`).
- Hydra configs (`conf/`).
- Forecasting / analysis scripts (`scripts/`).

These are independent of the launch infrastructure and can be edited
without touching anything in `cloud/`.
