# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

PyTorch 2.2+ / Hydra reimplementation of liquid time-constant and spiking RNN experiments (originally TF 1.x). Trains RNN cells (LSTM, LTC, CTRNN, NODE, CTGRU, SRNN + ablations) on 9 sequence tasks (HAR, sMNIST, gesture, occupancy, ozone, person, power, traffic, cheetah).

Companion docs in the repo are authoritative deep references — read them when working on non-trivial changes:
- `README.md` — model/task tables, CLI usage
- `pytorch_hydra_code_data_structure.md` — exhaustive structure, tensor shapes, SRNN state layout, cloud infra
- `KnownIssues.md` — tracked limitations and bugs; check before making assumptions about SRNN/batched behavior

## Environment & worktrees

The project uses a **shared `uv` venv outside the repo** at `/Users/richner.thomas/Desktop/local_venv/srnn-train/.venv` (Python 3.12.6). Activate it from any worktree with:

```bash
source /Users/richner.thomas/Desktop/local_venv/srnn-train/.venv/bin/activate
```

Don't create per-worktree `.venv` directories. The legacy in-repo `.venv` is gitignored and being phased out.

Development going forward uses **`git worktree`** rather than separate clones, so multiple Claude Code agents can work on different branches in parallel without colliding on `HEAD`. Typical layout:

```
~/Desktop/MayoVertex/
  train-srnn/             # primary worktree, usually on main
  train-srnn-<feature>/   # sibling worktrees on feature branches
```

Create with `git worktree add ../train-srnn-<feature> -b <feature>`; remove with `git worktree remove <path>`. Note that `tmp/`, `results/`, and other gitignored scratch dirs are per-worktree (not shared). Shared external resources (GCS paths, GCP VM names, the `train-srnn-deploy-key` secret) still need manual coordination across worktrees.

## Common commands

```bash
# one-time, shared across all worktrees
uv pip install -r requirements.txt   # into /Users/richner.thomas/Desktop/local_venv/srnn-train/.venv

# Train (Hydra overrides on the command line)
python train.py model=srnn task=har seed=1
python train.py model=lstm task=smnist epochs=100 size=64
python train.py model=ltc task=gesture device=cuda

# Run K SRNN ablations in parallel via BatchedSRNNCell (torch.bmm)
python train.py task=har batched_ablations='[srnn,srnn-no-adapt,srnn-E-only]'

# Smoke test all model x task combos (2 epochs each)
bash smoke_test.sh
```

There is no separate lint/test runner — `smoke_test.sh` is the integration check. MATLAB MCP tools are available but not part of this Python project.

## Architecture

**Hydra config composition.** `conf/config.yaml` selects one `conf/model/*.yaml` and one `conf/task/*.yaml` via defaults. The root `size` parameter is interpolated into `model.num_units` via `${size}`. Any value is CLI-overridable. `output_dir` defaults to `results/${task.name}/${model.name}_${size}`.

**Cell interface.** Every RNN cell (`train_srnn/models/{ltc_cell,srnn_cell,ctrnn_cell}.py` + `LSTMCellWrapper` in `sequence_model.py`) implements `cell(input, state) -> (output, new_state)` with a `state_size: int` attribute and optional `constrain_parameters()`. State is always packed into a single flat `(batch, state_size)` tensor — notably for SRNN where state is `[a_E | a_I | b_E | b_I | x]` (see structure doc §2.2 for unpacking).

**Factory dispatch.** `train_srnn/models/factory.py:build_model(cfg)` generates a neuron partition (input/inter/output ~25/50/25%), builds a `W_in_mask`, dispatches by `cfg.model.type` to the right cell + dataclass config, and wraps in `SequenceModel`. `_cfg_to_dataclass` filters DictConfig to dataclass fields. `build_batched_model` stacks K SRNN preset configs into a `BatchedSRNNCell` for parallel ablation runs (all variants must share `solver`/`h`/`ode_unfolds`).

**SequenceModel wrapper** (`train_srnn/models/sequence_model.py`) handles unrolling, optional I/O neuron masking (readout sees only ~25% of units), optional `TrainableIC` (with burn-in initialization via `train_srnn/utils/trainable_ic.py:compute_burn_in`), `bptt_start_idx` gradient detaching for truncated BPTT, readout selection at `readout_idx`, and a final `nn.Linear` head.

**Data + augmentation pipeline.** `train_srnn/data/datasets.py:load_dataset` returns numpy `(N, T, F)` arrays (batch-first). `train_srnn/data/transforms.py` then does, **per training batch**: PCHIP time-stretch (random factor in `[stretch_lo, stretch_hi]`) → palindrome loop (`[fwd, bwd, fwd, …]`, count from `min_loops`/`min_loop_len`) → random contiguous window, returning `(aug_x, aug_y, readout_idx, bptt_start_idx)`. Eval skips stretching and reads out at the last timestep. For per-timestep-label tasks, `train.py` extracts `batch_y[:, readout_idx]` before loss.

**Training loop** (`train.py`) is a single `@hydra.main`-decorated function: seed → device autodetect (cuda > mps > cpu) → load data → build model → optional `torch.compile` (CUDA only) → Adam + `WarmupHoldCosineSchedule` (per-step, 20% warmup / 70% hold / cosine decay) → optional burn-in → epoch loop calling `run_epoch` for train/valid, then `model.constrain_parameters()` after each optimizer step → final test → single-row CSV in `output_dir`.

**SRNN specifics.** Dale's law applied via `_effective_W` (softplus + sparsity_mask, inhibitory columns negated). Multi-timescale SFA: when `n_a_E >= 2`, learnable `log_tau_a_E_lo`/`hi` endpoints are interpolated linearly across `n_a_E` timescales at runtime. `per_neuron=True` switches adaptation params from shape `(1,)` to `(n_E,)`/`(n_I,)` in `SRNNCell` — but note that `BatchedSRNNCell` always stores per-neuron regardless of the flag (see `KnownIssues.md §1`). `echo=True` freezes recurrent W (reservoir mode). Activation is `piecewise_sigmoid` with 5 regions.

**Cloud (`cloud/`).** GCP VM-per-run model: `launch_run.sh` / `launch_all.sh` create VMs (scoped `cloud-platform`) whose `startup.sh` fetches an SSH deploy key from GCP Secret Manager (`train-srnn-deploy-key` in project `liquidneuralnets`), clones the private `train-srnn` repo via SSH, downloads the dataset from GCS, runs `train.py`, uploads results, and self-deletes. The deploy key is scrubbed from disk after clone. `monitor.sh` shows a model×task completion matrix; `collect_results.py` aggregates seed CSVs from GCS (handles both single-row and multi-row batched ablation CSVs). Per-task overrides live in `cloud/experiments/<task>.env`. Defaults in `cloud/config.env` (project, bucket, machine type — n4d requires hyperdisk-balanced). Cloud batched ablations: `bash cloud/launch_run.sh my-run smnist srnn 1 "batched_ablations='[srnn-E-only,srnn-e-only-echo]' epochs=15"` — `launch_run.sh` strips quotes and passes `train-args` via `--metadata-from-file` to avoid gcloud comma-delimiter issues; `startup.sh` uses `set -f` to prevent shell globbing of `[...]`.

**GPU dispatch (`cloud/launch_run_gpu.sh` + `cloud/submit.sh`).** Same VM-native model with three lifecycle modes set per-run via the `cleanup` metadata key: `delete` (one-shot, default), `stop` (self-stop after upload, ~$0.02/hr disk-only), `keep` (stay RUNNING for active dev). First launch: `bash cloud/launch_run_gpu.sh [--cleanup=delete|stop|keep] <run> <task> <model> <seed> [args]`. Subsequent dispatches to a stopped/running VM: `bash cloud/submit.sh <vm> <run> <task> <model> <seed> [--cleanup=...] [--skip-refresh] [args]` — writes per-run knobs via `gcloud compute instances add-metadata`, then `start` (if stopped) or `reset` (if running) to re-trigger `startup_gpu.sh`. No SSH tether to the local laptop; the laptop can sleep mid-run. `--skip-refresh` bypasses git fetch + dataset re-copy + pip check when iterating different ablations on the same code. Per-run output dir is `results/<task>/<run>_seed<seed>` so re-dispatches don't cross-contaminate. The exact full commit sha lands in `run_metadata.json.commit`.

## Analysis & plotting

Standalone analysis scripts live in `scripts/`. Plotting helpers grouped under `scripts/plots/`.

**Post-run analysis** (one command — download + all plots + tables):
```bash
python scripts/postprocess.py <run_name>           # default task=seeg, seed=1
python scripts/postprocess.py <run> --skip-download --variants srnn-skip,srnn-no-adapt-skip
```
Downloads `gs://<bucket>/results-pytorch/<run>/srnn/<task>/seed<seed>/` into `tmp/<run>/` (idempotent — skips already-cached files), then emits at the run-dir top level: `curves_{skip,no-skip}.png` (linear + 3 log variants), `lr_schedule.png`, `weight_evolution.png`. Per-variant outputs go into `tmp/<run>/<variant>/{tau_evolution.png, W_EI_evolution.png, param_table.txt}`. Tau panels for inactive sides (e.g. tau_b_* in an SFA-only variant) get a 20%-grey overlay with the title note "(masked — tau_global only)" since their motion is purely tau_global rescaling. Tables report **effective (post-transform) values** — what multiplies things on the RHS of the ODE.

**Other plot helpers** (operate on a fixed `output_dir`, single-variant runs only):
- `plot_all_taus.py`, `plot_tau_global.py`, `plot_effective_taus_overlay.py` — SRNN time-constant inspection
- `plot_cmp_val_acc.py` — multi-run validation accuracy comparison
- `plot_lstm_curves.py` — LSTM training-curve plot
- `plot_kstep_forecast.py` — multi-step regression forecast quality

**Gradient diagnostics** (one-shot probes for chunk-len / direction-consistency questions, outputs in `tmp/grad_probe/`):
- `scripts/grad_norm_probe.py` — sweeps `bptt_chunk_len ∈ {16…512}` on one real seeg batch, captures per-parameter `|∇θ|₂`. Variant list controlled by an `ABLATIONS` constant near the top.
- `scripts/plot_grad_groups.py` — re-plots `grad_norms.csv` grouped by physical role (output / recurrent / dendritic / SFA / STD)
- `scripts/grad_cosine_consistency.py` — runs N batches at fixed chunk_len, computes mean pairwise cosine of each param's gradient (≈1 = consistent, ≈0 = direction noise → Adam can't accumulate)

**Authoritative model spec.** `FullModel.md` (and `FullModel.pdf` rendered via the `md2pdf` skill) gives the complete LaTeX-form description of every parameter, buffer, ODE RHS, solver, and ablation knob, with code-symbol → math-symbol cheat sheet. Read it before reasoning about effective values or what gates which path.

Generated artifacts (`*.png`, `*.pt`) are gitignored. `tmp/` is NOT gitignored — it's the working scratch dir for downloaded GCS run bundles and ad-hoc analysis output. Keep heavy binaries out of commits; small CSVs are fine.

**Cosine cooldown caveat.** `cosine_decay=false` is now the default. With it off, lr stays flat at the configured `lr` value through warmup→hold for the entire run (no decay tail) — useful when you intend to stop early or chain re-dispatches.

## Conventions worth knowing

- Tensors are batch-first `(B, T, F)` everywhere (TF 1.x version was time-major).
- Don't add cells without also adding a model YAML in `conf/model/` and a dispatch branch in `train_srnn/models/factory.py:build_cell`.
- SRNN ablation variants are defined both as YAML configs (`conf/model/srnn_*.yaml`) and as entries in `SRNN_PRESETS` in `train_srnn/models/srnn_cell.py` — keep the two in sync when editing.
- Per-timestep vs sequence-level labels is set by `task.per_timestep_labels` in the task YAML; getting this wrong silently produces shape errors at loss time (recent fix: 47795bf).

## IDE integration

The user runs Antigravity (a VS Code fork). Both `code` and `agy` resolve to the Antigravity CLI with identical VS Code flags. Before editing a file the user may not have open, pre-open it in their active window so the inline diff renders where they can see it:

```bash
code -r <path>           # open in current window (--reuse-window)
code -g <path>:<line>    # jump to a specific line
code -d <fileA> <fileB>  # side-by-side diff view
```

Call `code -r` just before an `Edit`/`Write` on a file that isn't already open. No-op if the file is already open. Not needed when the user already has the file visible (system-reminders flag open files).
