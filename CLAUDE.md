# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

PyTorch 2.2+ / Hydra reimplementation of liquid time-constant and spiking RNN experiments (originally TF 1.x). Trains RNN cells (LSTM, LTC, CTRNN, NODE, CTGRU, SRNN + ablations) on 11 task configs in `conf/task/`: the original classification/regression set (HAR, sMNIST, gesture, occupancy, ozone, person, power, traffic, cheetah) plus the autoregressive forecasting tasks **seeg**, **cheetah100** and **cheetah100_act**.

**Active focus.** Day-to-day work is **SRNN run as K-batched ablations** (`BatchedSRNNCell`) on the **autoregressive tasks under the continuous (ring) trainer** (`train_srnn/training/continuous.py`). The most recent work (late Aug 2026) is on **cheetah100**: adaptation vs. no-adaptation comparisons (`ring6-400e`, `ring2x5-100e` — 5 recurrent-matrix seeds per variant) and the resulting report in `docs/report_to_Brian/`. seeg uses the same trainer and remains a primary task. The windowed-trainer 9-task / 6-cell matrix is kept passing but is not the development frontier — don't optimize for it, and don't run the full 90-combo `smoke_test.sh` as a default integration check (it costs 30–60 min on local CPU). See **Test strategy** below for what to run instead.

**Repo hygiene is being overhauled (Sept 2026).** The user plans a cleanup/restructure (branch `cleanUp`, possibly on a fresh clone on another Mac). See **Repo state & housekeeping** at the end before moving, deleting, or reorganizing files.

Companion docs in the repo are authoritative deep references — read them when working on non-trivial changes:
- `README.md` — model/task tables, CLI usage
- `pytorch_hydra_code_data_structure.md` — exhaustive structure, tensor shapes, SRNN state layout
- `CurrentCloudArchitecture.md` — canonical, file-by-file description of `cloud/`. Read this before editing launch/startup scripts
- `FullModel.md` — LaTeX-form spec of every parameter, buffer, ODE RHS, solver, and ablation knob, with code-symbol → math-symbol cheat sheet
- `KnownIssues.md` — tracked limitations and bugs (§1–§15, several marked **FIXED**); check before making assumptions about SRNN/batched behavior

Other top-level `.md` files (`Pytorch_Refactor.md`, `code_audit.md`, `effectiveWPlan.md`, `refactor_split_GPU_notes.md`, `data_export_for_srnn.md`, `time_stretch_note.md`) and `docs/{equations,full_equations,srnn_variants}.md` are historical plans/notes — useful context, not guaranteed current.

## Environment & worktrees

The project uses a **shared `uv` venv outside the repo** at `/Users/richner.thomas/Desktop/local_venv/srnn-train/.venv` (Python 3.12.6). Activate it from any worktree with:

```bash
source /Users/richner.thomas/Desktop/local_venv/srnn-train/.venv/bin/activate
```

Don't create per-worktree `.venv` directories. The legacy in-repo `.venv` is gitignored and being phased out. On a fresh machine: `uv venv --python 3.12 <path>` then `uv pip install -r requirements.txt`. Note `requirements.txt` is **unpinned** (no lockfile, no `pyproject.toml`), so a fresh install pulls newer torch etc. than the existing venv.

Development uses **`git worktree`** rather than separate clones, so multiple Claude Code agents can work on different branches in parallel without colliding on `HEAD`. Typical layout:

```
~/Desktop/MayoVertex/
  train-srnn/             # primary worktree
  train-srnn-<feature>/   # sibling worktrees on feature branches
```

Create with `git worktree add ../train-srnn-<feature> -b <feature>`; remove with `git worktree remove <path>`. Note that `tmp/`, `results/`, `outputs/` and `train_srnn/data/` are per-worktree (not shared). Shared external resources (GCS paths, GCP VM names, the `train-srnn-deploy-key` secret) still need manual coordination across worktrees.

The git remote is a **private GitHub repo** (`git@github.com:TomRichner/train-srnn.git`). The user pushes; agents commit but don't push.

## Common commands

```bash
# one-time, shared across all worktrees
uv pip install -r requirements.txt   # into /Users/richner.thomas/Desktop/local_venv/srnn-train/.venv

# Primary path: SRNN K-batched ablations on a continuous (ring) task
python train.py task=cheetah100 model=srnn seed=1 \
    batched_ablations='[srnn-no-dales-skip,srnn-no-adapt-no-dales-skip]' \
    batched_ablation_seeds='[1,2,3]'
python train.py task=seeg model=srnn seed=1 \
    batched_ablations='[srnn-e-only-per-neuron,srnn-e-only-skip-per-neuron]'

# Other tasks/cells still work — kept passing, not the active frontier
python train.py model=lstm task=smnist epochs=100 size=64
python train.py model=ltc task=gesture device=cuda
```

## Test strategy

There is no pytest runner. The primary integration checks are **targeted scripts under `scripts/test_*.py`**, not the full model×task smoke matrix. They run on local CPU in seconds-to-minutes and exercise the SRNN/SequenceModel/continuous-trainer codepaths that all active work touches.

Run these before pushing changes that touch SRNN cells, `SequenceModel`, or `train_srnn/training/continuous.py`:

```bash
PYTHONPATH=. python scripts/test_closed_loop_grad_checkpoint.py   # 9 tests, ~30s
PYTHONPATH=. python scripts/test_effective_w_hoist.py             # 4 tests, ~10s
PYTHONPATH=. python scripts/test_closed_loop_forward.py
PYTHONPATH=. python scripts/test_srnn_defaults.py
```

Other targeted suites, run when touching the relevant code:
- `test_cell_loop.py` — slice-assign output buffer is byte-identical to append+stack (forward + grads)
- `test_w_out_gain.py` — `W_out_gain` readout gain: identity at 1.0, gradient flow, doubling (single + K-batched)
- `test_std_zero_floor.py` — `std_zero_floor` flag (rescales STD `b` to [0,1] for the readout); bit-identity when off
- `test_rmt_matrix.py` — `RMTMatrix` spectral radius, sparsity, export round-trip
- `test_cheetah100_loader.py` — `load_cheetah100` channel counts, train-only z-score, AR offset, transient removal
- `test_closed_loop_schedule.py` — alpha ramp schedule per epoch
- `test_isp_rename_regression.py` — golden-file check for the `log_*` → `isp_*` parameter rename
- `test_srnn_defaults_pyOnly.py`, `test_tau_global.py` — pure-Python forward sims / visual checks (produce PNGs). Note `test_srnn_defaults.py` hardcodes a MATLAB `.mat` path from another machine.

Each script is a self-contained suite that runs paired models / paired forward paths and asserts byte-identical (or tight-tolerance) forward outputs and per-parameter gradients. Add new tests in this style when adding a code path.

`smoke_test.sh` (90 model×task combos × 2 epochs at size=8) still exists and works, but **don't run it as a default integration check** — it costs 30–60 min on local CPU and reproves things the targeted scripts already cover. Use it only when intentionally validating the broad cell×task matrix.

The real production signal — does this change move the loss curve or epoch wall-clock? — comes from a **GPU dispatch on a cloud VM**, not from local CPU runs. See `cloud/submit.sh`, `cloud/launch_run_gpu.sh`, and the launch recipes documented in `cloud/experiments/cheetah100.env`.

MATLAB MCP tools are available but not part of this Python project.

## Architecture

**Hydra config composition.** `conf/config.yaml` selects one `conf/model/*.yaml` and one `conf/task/*.yaml` via defaults. The root `size` parameter is interpolated into `model.num_units` via `${size}`. Many root keys (`batch_size`, `window_len`, `bptt_len`, `bptt_chunk_len`, `loss_over_bptt`, `no_augment`, `continuous_train`, …) pull per-task values via `${oc.select:task.<key>, <default>}`, so the task YAML is the single source of truth for them. Model YAMLs interpolate ODE step size via `${task.h}`. Any value is CLI-overridable. `output_dir` defaults to `results/${task.name}/${model.name}_${size}`; Hydra's own run logs go to `outputs/<date>/<time>/`.

**Cell interface.** Every RNN cell (`train_srnn/models/{ltc_cell,srnn_cell,ctrnn_cell}.py` + `LSTMCellWrapper` in `sequence_model.py`) implements `cell(input, state) -> (output, new_state)` with a `state_size: int` attribute and optional `constrain_parameters()`. State is always packed into a single flat `(batch, state_size)` tensor — notably for SRNN where state is `[a_E | a_I | b_E | b_I | x]` (see structure doc §2.2 for unpacking).

**Factory dispatch.** `train_srnn/models/factory.py:build_model(cfg)` generates a neuron partition (input/inter/output ~25/50/25%), builds a `W_in_mask`, dispatches by `cfg.model.type` to the right cell + dataclass config, and wraps in `SequenceModel`. `_cfg_to_dataclass` filters DictConfig to dataclass fields. SRNN recurrent weights come from `train_srnn/models/rmt_matrix.py:RMTMatrix` (Python port of the MATLAB RMTMatrix/RMTConnectivity, Harris et al. 2023: E/I structure, sparsity, controlled spectral radius). `build_batched_model` stacks K `SRNN_PRESETS` configs into a `BatchedSRNNCell` for parallel ablation runs (all variants must share `solver`/`h`/`ode_unfolds`). With `batched_ablation_seeds=[...]`, variants are crossed with seeds (variant-major, K = n_variants × n_seeds) and renamed `<variant>-seed<n>`; variants sharing a seed share one `RMTMatrix`, so comparisons are paired on W.

**SequenceModel wrapper** (`train_srnn/models/sequence_model.py`) handles unrolling, optional I/O neuron masking (readout sees only ~25% of units), optional `TrainableIC` (with burn-in initialization via `train_srnn/utils/trainable_ic.py:compute_burn_in`), `bptt_start_idx` gradient detaching for truncated BPTT, gradient-checkpointed segments (`_run_segment` / `_cl_run_segment`), readout selection at `readout_idx`, the `skip` residual (`y = readout(state) + x`, autoregressive tasks only), and a final `nn.Linear` head with a learnable `W_out_gain`.

**Data + augmentation pipeline.** `train_srnn/data/datasets.py:load_dataset` returns numpy `(N, T, F)` arrays (batch-first). Autoregressive loaders (`load_seeg`, `load_cheetah100`) additionally return a `train_trace` (one long z-scored trace, train-split stats) for the continuous trainer. `train_srnn/data/transforms.py` then does, **per training batch** (windowed trainer only): PCHIP time-stretch (random factor in `[stretch_lo, stretch_hi]`) → palindrome loop → random contiguous window, returning `(aug_x, aug_y, readout_idx, bptt_start_idx)`. The autoregressive tasks set `no_augment: true` (real physical time) and `loss_over_bptt: true` (loss at every step in the grad region). For per-timestep-label tasks, `train.py` extracts `batch_y[:, readout_idx]` before loss.

**Training loop** (`train.py`). `@hydra.main def main(cfg)`: seed → device autodetect (cuda > mps > cpu) → load data → build model (`build_batched_model` if `batched_ablations` set) → optional `freeze_params` (pin named SRNN params at init) → optional `torch.compile` (CUDA only; `compile_cell=true` compiles `model.cell`, deferred to the continuous trainer) → Adam + `WarmupHoldCosineSchedule` → optional `init_ckpt` resume → optional burn-in → then one of two trainers:
- **Windowed** (`continuous_train=false`, the classic tasks): epoch loop calling `run_epoch` (in `train.py`) for train/valid, `model.constrain_parameters()` after each optimizer step.
- **Continuous / ring** (`continuous_train=true`, set in `seeg.yaml` and `cheetah100.yaml`): `train_srnn/training/continuous.py:run_continuous_training`. B parallel readers sit at fixed phase offsets around a circular `train_trace` of length `train_trace_max_len` (chosen prime so detach boundaries drift across epochs); every step advances `bptt_chunk_len` samples, target is the trace shifted by one sample, state is never reset. A "logical epoch" is one sweep of T. Valid/test still use windowed `run_epoch`. Scheduler steps per chunk; `warmup_epochs=null` auto-picks.

Final test → CSVs (`training_history.csv`, `test_history.csv`), checkpoints (`init.pt`, `epoch_*.pt`, `last.pt`) and `progress.json` in `output_dir`.

**Gradient clipping.** `grad_clip` (default 1.0). In batched-ablation mode it is applied **per variant** via `train_srnn/utils/grad_clip.py:clip_grad_norm_per_variant`, so one variant's large gradient doesn't throttle the others (KnownIssues §14).

**SRNN specifics.** Dale's law applied via `_effective_W` (softplus + sparsity_mask, inhibitory columns negated), hoisted out of the per-step cell call (KnownIssues §10). Positive params are stored inverse-softplus as `isp_*`. Multi-timescale SFA: when `n_a_E >= 2`, learnable `isp_tau_a_E_lo`/`hi` endpoints are interpolated across `n_a_E` timescales at runtime (default `tau_a_hi` 4 s). STD state `b` is rescaled to [0,1] for the readout when `std_zero_floor=true` (default in `conf/model/srnn.yaml`). `per_neuron=True` switches adaptation params to per-neuron shape; in `BatchedSRNNCell` this is implemented as scalar + per-neuron `_vec` params with a per-variant gradient mask (KnownIssues §1, FIXED). `echo=True` freezes recurrent W (reservoir mode). Activation is `piecewise_sigmoid` with 5 regions. `a_0` and `c_0_E`/`c_0_I` share an exact flat direction (KnownIssues §15).

Current variant families in `SRNN_PRESETS` (`train_srnn/models/srnn_cell.py`): `srnn`, `srnn-skip`, `srnn-no-adapt`, `srnn-no-dales`, `srnn-no-dales-skip`, `srnn-no-adapt-no-dales[-skip]`, `srnn-{sfa,std}-only`, `srnn-E-only` / `srnn-e-only-*` and `srnn-{sfa,std}-e-only-*` (with `-skip`, `-per-neuron`, `-echo` suffixes), `srnn-multi-sfa[-E]`, `srnn-explicit`, `srnn-rk4`, `srnn-echo`, `srnn-per-neuron`. `batched_ablations` names resolve against `SRNN_PRESETS`.

**Closed-loop teacher forcing.** Configured by the `closed_loop:` block in `conf/config.yaml` (`closed_loop.enabled=false` by default). When enabled, inputs are a per-channel blend `x_in = (1-α)·x_real + α·y_pred[t-1]`, with α sampled per batch from an envelope × (baseline + sparse per-channel jitter), and an optional per-epoch `alpha_baseline_start → alpha_baseline` ramp. `train_srnn/training/closed_loop.py` holds only `ClosedLoopConfig` and the α samplers (`sample_alpha_schedule` for windowed, `sample_continuous_alpha` for continuous); the unroll itself lives in `run_epoch` (`train.py`), the continuous trainer, and `SequenceModel._cl_run_segment` (checkpointed). Requires `input_size == output_size`. Eval is unaffected.

**Cloud (`cloud/`).** Two pipelines (CPU and GPU) sharing a VM-native pattern: `gcloud compute instances create --metadata-from-file=startup-script=...` boots a VM that pulls a deploy key from Secret Manager (`train-srnn-deploy-key` in project `liquidneuralnets`), clones the repo, downloads the dataset from `gs://liquidneuralnets-experiments/datasets/<task>/`, runs `train.py`, uploads to `gs://liquidneuralnets-experiments/results-pytorch/<run>/<model>/<task>/seed<seed>/`, and self-deletes (or stays per `cleanup` metadata). **Read `CurrentCloudArchitecture.md` for the file-by-file breakdown** — the rest of this section only documents what isn't there.

- **Batched ablations from CLI**: `bash cloud/launch_run.sh my-run smnist srnn 1 "batched_ablations='[srnn-E-only,srnn-e-only-echo]' epochs=15"`. `launch_run.sh` strips quotes and passes `train-args` via `--metadata-from-file` to dodge gcloud comma-delimiter parsing; `startup.sh` uses `set -f` to keep the shell from globbing `[...]`.
- **GPU lifecycle modes** via the `cleanup` metadata key: `delete` (one-shot, default), `stop` (self-stop after upload, ~$0.02/hr disk-only), `keep` (stay RUNNING for active dev). First launch: `bash cloud/launch_run_gpu.sh [--cleanup=delete|stop|keep] <run> <task> <model> <seed> [args]` — defaults to `g2-standard-8` + 1× L4 per `cloud/config.gpu.env`. Re-dispatch to an existing VM: `bash cloud/submit.sh <vm> <run> <task> <model> <seed> [--cleanup=...] [--skip-refresh] [--branch=<name>] [args]` writes per-run knobs via `gcloud compute instances add-metadata` then `start` (stopped) or `reset` (running) to re-trigger `startup_gpu.sh`. No SSH tether — the laptop can sleep mid-run. `--skip-refresh` bypasses git fetch + dataset re-copy + pip check (all-or-nothing, KnownIssues §12). `--branch` (default `main`) dispatches a feature branch.
- Per-task defaults live in `cloud/experiments/<task>.env`. Keep `ARGS` there minimal so the task YAML stays authoritative; `cheetah100.env` documents the exact `ring6-400e` and `ring2x5-100e` launch commands.
- Per-run output dir is `results/<task>/<run>_seed<seed>` so re-dispatches don't cross-contaminate. The full commit sha lands in `run_metadata.json.commit`.

## Analysis & plotting

Standalone analysis scripts live in `scripts/`. Plotting helpers grouped under `scripts/plots/`.

**Post-run analysis** (one command — download + all plots + tables):
```bash
python scripts/postprocess.py <run_name>                  # default task=seeg, seed=1
python scripts/postprocess.py ring2x5-100e --task cheetah100
python scripts/postprocess.py <run> --skip-download --variants srnn-skip,srnn-no-adapt-skip
```
Downloads `gs://<bucket>/results-pytorch/<run>/srnn/<task>/seed<seed>/` into `tmp/<run>/` (idempotent — skips already-cached files), then emits at the run-dir top level: `curves_{skip,no-skip}.png` plus `log_log_`, `semilogy_` and `semilogy_direct_` variants, `lr_schedule.png`, `weight_evolution.png` (with std bands), `W_io_evolution.png`, `offsets_evolution.png` (`c_0_E`, `c_0_I`, `a_0`), and `report.pdf`. Multi-seed runs get one colour per base variant across all its seeds. Per-variant outputs go into `tmp/<run>/<variant>/{tau_evolution.png, W_EI_evolution.png, param_table.txt}`; with `per_neuron=False` tau panels show one line per timescale. Tau panels for inactive sides get a 20%-grey overlay ("masked — tau_global only"). Tables report **effective (post-transform) values**. `scripts/rewrite_param_tables.py tmp/<run> …` re-emits just the param tables.

**Learning-curve and comparison figures** (batched-ablation runs, read from a `tmp/<run>` dir):
- `plot_learning_curves.py` — per-variant loss vs epoch, skip / non-skip in separate panels (`--loglog`, `--both`)
- `plot_adaptation_comparison.py` — `srnn` vs `srnn-no-adapt` on log-log (curves cross past ~100 epochs)
- `plot_for_Brian.py`, `plot_for_Brian_seeds.py` — presentation figures for `docs/report_to_Brian/` (single-seed and 5-seed adaptation vs none)
- `compare_ablations.py`, `compare_all_ablations.py`, `compare_solvers.py` — older bar/line comparisons across variants / solvers
- `check_weights.py` — `init.pt` / `last.pt` weight histograms

**Dynamics / data inspection:**
- `plots/plot_srnn_timeseries.py` — forward-replay a batched checkpoint under no-input / step / seeg input and plot per-variant state time series
- `lyapunov_evolution.py` — Benettin FTLE on every saved checkpoint, FTLE-vs-epoch plot
- `plot_cheetah_traces.py` — raw cheetah rollouts, one figure per episode
- `plot_seeg_forecast.py`, `plots/plot_kstep_forecast.py` — autoregressive / k-step forecast quality
- `diag_std_b_e_step.py` — diagnostic for how far STD `b_E` depresses
- `plots/plot_all_taus.py`, `plot_tau_global.py`, `plot_effective_taus_overlay.py`, `plot_cmp_val_acc.py`, `plot_lstm_curves.py` — older single-run helpers with hardcoded `tmp/cmp-20ep/...` paths

**Data preprocessing:** `preprocess_seeg_filter.py` — offline zero-phase Butterworth bandpass producing the `*_bp0p1-50.mat` files that `seeg.yaml`'s `filter_tag` selects.

**Gradient diagnostics** (one-shot probes, outputs in `tmp/grad_probe/`):
- `grad_norm_probe.py` — sweeps `bptt_chunk_len ∈ {16…512}` on one real seeg batch, captures per-parameter `|∇θ|₂`. Variant list in an `ABLATIONS` constant near the top.
- `plot_grad_groups.py` — re-plots `grad_norms.csv` grouped by physical role
- `grad_cosine_consistency.py` — mean pairwise cosine of each param's gradient across N batches

**Cosine cooldown caveat.** `cosine_decay=false` is the default. With it off, lr ramps over `warmup_epochs` then stays flat at `lr` for the rest of the run (no decay tail) — useful when you intend to stop early or chain re-dispatches.

## Conventions worth knowing

- Tensors are batch-first `(B, T, F)` everywhere (TF 1.x version was time-major).
- Don't add cells without also adding a model YAML in `conf/model/` and a dispatch branch in `train_srnn/models/factory.py:build_cell`.
- SRNN ablation variants are defined in `SRNN_PRESETS` in `train_srnn/models/srnn_cell.py` (what `batched_ablations` uses) and many also as `conf/model/srnn_*.yaml` (for single-variant runs). Keep the two in sync when editing a variant that has both; not every preset has a YAML.
- Per-timestep vs sequence-level labels is set by `task.per_timestep_labels` in the task YAML; getting this wrong silently produces shape errors at loss time.
- For continuous tasks, pick `train_trace_max_len` so the per-epoch phase drift is coprime to `bptt_chunk_len` (see the comment in `conf/task/cheetah100.yaml`).

## Repo state & housekeeping (as of 2026-09-13)

Facts for the overhaul — verify before acting, they will go stale:

- **Datasets** live in gitignored `train_srnn/data/<task>/` (seeg: 4 `.mat` files, 1.3 GB; cheetah100: `{train,valid,test}.npz` + `manifest.json`; cheetah: `trace_*.npy`; smnist: IDX gz). All are mirrored in `gs://liquidneuralnets-experiments/datasets/<task>/` — restore with `gcloud storage cp -r gs://liquidneuralnets-experiments/datasets/<task> train_srnn/data/`. cheetah100 was generated by the separate repo `~/Desktop/local_code/gen-half-cheetah-dataset`.
- **`outputs/`** (gitignored): Hydra run logs from local `train.py` runs.
- **`results/`** (gitignored): `output_dir` of local runs (checkpoints, CSVs). Cloud runs write here on the VM, then upload to GCS.
- **`tmp/`**: mostly `postprocess.py` download caches plus ad-hoc probe output (~5 GB locally). It is **partly tracked**: `*.png`/`*.pt`/`*.npz` are gitignored, but CSVs, `param_table.txt`, logs and `run_metadata.json` of archived runs are committed (~330 files). Everything in it is regenerable from GCS via `postprocess.py`.
- **Committed binaries in history**: several 3–11 MB PNGs under `scripts/` (now gitignored but still in git history).
- **Hardcoded machine paths**: `scripts/overnight_run.sh` (repo + venv paths), `scripts/test_srnn_defaults.py:84` (MATLAB `.mat` on another machine).
- **Branches/worktrees**: `compile-chunk` (worktree `../train-srnn-compile-chunk`) is an abandoned chunk-compile spike, pushed and tagged `spike/chunk-compile`; its postmortem recommends not shipping. `vertex` (worktree `../train-srnn-vertex`) is merged into `main`; the worktree holds uncommitted notes on a possible port to Mayo AIF / Vertex AI. `cloud-branch-knob` is merged.
- Proposed direction (not yet implemented): move data/results/run caches outside the repo behind an env var (e.g. `SRNN_WORK_ROOT`, defaulting to the current in-repo paths so cloud VMs keep working), and give committed analysis artifacts a deliberate home instead of `tmp/`.

## IDE integration

The user runs Antigravity (a VS Code fork). Both `code` and `agy` resolve to the Antigravity CLI with identical VS Code flags (they may not be on the Bash tool's PATH). Before editing a file the user may not have open, pre-open it in their active window so the inline diff renders where they can see it:

```bash
code -r <path>           # open in current window (--reuse-window)
code -g <path>:<line>    # jump to a specific line
code -d <fileA> <fileB>  # side-by-side diff view
```

Call `code -r` just before an `Edit`/`Write` on a file that isn't already open. No-op if the file is already open. Not needed when the user already has the file visible (system-reminders flag open files).
