# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

PyTorch 2.2+ / Hydra reimplementation of liquid time-constant and spiking RNN experiments (originally TF 1.x). Trains RNN cells (LSTM, LTC, CTRNN, NODE, CTGRU, SRNN + ablations) on 9 sequence tasks (HAR, sMNIST, gesture, occupancy, ozone, person, power, traffic, cheetah).

Companion docs in the repo are authoritative deep references — read them when working on non-trivial changes:
- `README.md` — model/task tables, CLI usage
- `pytorch_hydra_code_data_structure.md` — exhaustive structure, tensor shapes, SRNN state layout, cloud infra
- `KnownIssues.md` — tracked limitations and bugs; check before making assumptions about SRNN/batched behavior

## Common commands

```bash
pip install -r requirements.txt

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
