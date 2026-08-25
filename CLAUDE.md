# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

PyTorch 2.2+ / Hydra reimplementation of liquid time-constant and spiking RNN experiments (originally TF 1.x). Trains RNN cells (LSTM, LTC, CTRNN, NODE, CTGRU, SRNN + ablations) on 9 sequence tasks (HAR, sMNIST, gesture, occupancy, ozone, person, power, traffic, cheetah).

**Active focus.** Day-to-day work targets the **seeg** task with the **SRNN cell run as K-batched ablations** via `BatchedSRNNCell` (continuous trainer in `train_srnn/training/continuous.py`). The other 9-task / 6-cell matrix is kept passing but is not the development frontier — don't optimize for it, and don't run the full 90-combo `smoke_test.sh` as a default integration check (it costs 30–60 min on local CPU). See **Test strategy** below for what to run instead.

Companion docs in the repo are authoritative deep references — read them when working on non-trivial changes:
- `README.md` — model/task tables, CLI usage
- `pytorch_hydra_code_data_structure.md` — exhaustive structure, tensor shapes, SRNN state layout
- `CurrentCloudArchitecture.md` — canonical, file-by-file description of `cloud/`. Read this before editing launch/startup scripts
- `FullModel.md` — LaTeX-form spec of every parameter, buffer, ODE RHS, solver, and ablation knob, with code-symbol → math-symbol cheat sheet
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

# Primary path: seeg + SRNN K-batched ablations (continuous trainer)
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

Each script is a self-contained suite that runs paired models / paired forward paths and asserts byte-identical (or tight-tolerance) forward outputs and per-parameter gradients. Add new tests in this style when adding a code path.

`smoke_test.sh` (90 model×task combos × 2 epochs at size=8) still exists and works, but **don't run it as a default integration check** — it costs 30–60 min on local CPU and reproves things the targeted scripts already cover. Use it only when intentionally validating the broad cell×task matrix (e.g. before a release of the non-SRNN cells).

The real production signal — does this change move the seeg loss curve or epoch wall-clock? — comes from a **GPU dispatch on a cloud VM**, not from local CPU runs. See `cloud/submit.sh` and the dispatch examples in `effectiveWPlan.md` / `CompileChunkOutline.md` for the canonical 4-epoch K=2 size=30 non-regression run.

MATLAB MCP tools are available but not part of this Python project.

## Architecture

**Hydra config composition.** `conf/config.yaml` selects one `conf/model/*.yaml` and one `conf/task/*.yaml` via defaults. The root `size` parameter is interpolated into `model.num_units` via `${size}`. Any value is CLI-overridable. `output_dir` defaults to `results/${task.name}/${model.name}_${size}`.

**Cell interface.** Every RNN cell (`train_srnn/models/{ltc_cell,srnn_cell,ctrnn_cell}.py` + `LSTMCellWrapper` in `sequence_model.py`) implements `cell(input, state) -> (output, new_state)` with a `state_size: int` attribute and optional `constrain_parameters()`. State is always packed into a single flat `(batch, state_size)` tensor — notably for SRNN where state is `[a_E | a_I | b_E | b_I | x]` (see structure doc §2.2 for unpacking).

**Factory dispatch.** `train_srnn/models/factory.py:build_model(cfg)` generates a neuron partition (input/inter/output ~25/50/25%), builds a `W_in_mask`, dispatches by `cfg.model.type` to the right cell + dataclass config, and wraps in `SequenceModel`. `_cfg_to_dataclass` filters DictConfig to dataclass fields. `build_batched_model` stacks K SRNN preset configs into a `BatchedSRNNCell` for parallel ablation runs (all variants must share `solver`/`h`/`ode_unfolds`).

**SequenceModel wrapper** (`train_srnn/models/sequence_model.py`) handles unrolling, optional I/O neuron masking (readout sees only ~25% of units), optional `TrainableIC` (with burn-in initialization via `train_srnn/utils/trainable_ic.py:compute_burn_in`), `bptt_start_idx` gradient detaching for truncated BPTT, readout selection at `readout_idx`, and a final `nn.Linear` head.

**Data + augmentation pipeline.** `train_srnn/data/datasets.py:load_dataset` returns numpy `(N, T, F)` arrays (batch-first). `train_srnn/data/transforms.py` then does, **per training batch**: PCHIP time-stretch (random factor in `[stretch_lo, stretch_hi]`) → palindrome loop (`[fwd, bwd, fwd, …]`, count from `min_loops`/`min_loop_len`) → random contiguous window, returning `(aug_x, aug_y, readout_idx, bptt_start_idx)`. Eval skips stretching and reads out at the last timestep. For per-timestep-label tasks, `train.py` extracts `batch_y[:, readout_idx]` before loss.

**Training loop** (`train.py`) is a single `@hydra.main`-decorated function: seed → device autodetect (cuda > mps > cpu) → load data → build model → optional `torch.compile` (CUDA only) → Adam + `WarmupHoldCosineSchedule` (per-step, 20% warmup / 70% hold / cosine decay) → optional burn-in → epoch loop calling `run_epoch` for train/valid, then `model.constrain_parameters()` after each optimizer step → final test → single-row CSV in `output_dir`.

**SRNN specifics.** Dale's law applied via `_effective_W` (softplus + sparsity_mask, inhibitory columns negated). Multi-timescale SFA: when `n_a_E >= 2`, learnable `isp_tau_a_E_lo`/`hi` endpoints are interpolated linearly across `n_a_E` timescales at runtime. `per_neuron=True` switches adaptation params from shape `(1,)` to `(n_E,)`/`(n_I,)` in `SRNNCell` — but note that `BatchedSRNNCell` always stores per-neuron regardless of the flag (see `KnownIssues.md §1`). `echo=True` freezes recurrent W (reservoir mode). Activation is `piecewise_sigmoid` with 5 regions.

**Closed-loop training** (`train_srnn/training/closed_loop.py`). For autoregressive seeg-style forecasting tasks, `run_epoch_closed_loop` runs the model with **per-channel teacher forcing** (mixing prediction and ground truth via a learnable/scheduled `alpha` per channel) and a per-epoch `alpha_baseline` ramp. Supports gradient checkpointing through the unrolled segments via the `_cl_run_segment` helper (needed because the closed-loop unroll is too long for a single backward graph at typical chunk lengths). Activated by `closed_loop=true` in the task config; orthogonal to the standard `run_epoch` path used for classification tasks.

**Cloud (`cloud/`).** Two pipelines (CPU and GPU) sharing a VM-native pattern: `gcloud compute instances create --metadata-from-file=startup-script=...` boots a VM that pulls a deploy key from Secret Manager (`train-srnn-deploy-key` in project `liquidneuralnets`), clones the repo, downloads the dataset, runs `train.py`, uploads to `gs://liquidneuralnets-experiments/results-pytorch/<run>/<model>/<task>/seed<seed>/`, and self-deletes (or stays per `cleanup` metadata). **Read `CurrentCloudArchitecture.md` for the file-by-file breakdown** — the rest of this section only documents what isn't there.

- **Batched ablations from CLI**: `bash cloud/launch_run.sh my-run smnist srnn 1 "batched_ablations='[srnn-E-only,srnn-e-only-echo]' epochs=15"`. `launch_run.sh` strips quotes and passes `train-args` via `--metadata-from-file` to dodge gcloud comma-delimiter parsing; `startup.sh` uses `set -f` to keep the shell from globbing `[...]`.
- **GPU lifecycle modes** via the `cleanup` metadata key: `delete` (one-shot, default), `stop` (self-stop after upload, ~$0.02/hr disk-only), `keep` (stay RUNNING for active dev). First launch: `bash cloud/launch_run_gpu.sh [--cleanup=delete|stop|keep] <run> <task> <model> <seed> [args]` — defaults to `g2-standard-8` + 1× L4 per `cloud/config.gpu.env`. Re-dispatch to an existing VM: `bash cloud/submit.sh <vm> <run> <task> <model> <seed> [--cleanup=...] [--skip-refresh] [args]` writes per-run knobs via `gcloud compute instances add-metadata` then `start` (stopped) or `reset` (running) to re-trigger `startup_gpu.sh`. No SSH tether — the laptop can sleep mid-run. `--skip-refresh` bypasses git fetch + dataset re-copy + pip check when iterating different ablations on the same code.
- Per-run output dir is `results/<task>/<run>_seed<seed>` so re-dispatches don't cross-contaminate. The full commit sha lands in `run_metadata.json.commit`.

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
- `plot_kstep_forecast.py`, `plot_seeg_forecast.py` — multi-step regression / seeg autoregressive forecast quality

**Ablation comparison.**
- `compare_ablations.py`, `compare_all_ablations.py` — bar/line plots across model variants at a fixed task. The `compare_all_ablations_N{32,64,128,256,300}.png` snapshots in the dir are saved outputs from prior sweeps at different `size=` values; regenerate by re-running with the same args.
- `compare_solvers.py` — fixed-step solver comparison (Euler vs. Heun vs. RK4).
- `check_weights.py` — quick `init.pt` / `last.pt` weight-distribution sanity check (histograms saved to `weight_histograms.png`).

**Closed-loop tests** (numerical regression / behavioral sanity for the closed-loop pathway, run as plain Python):
- `test_closed_loop_forward.py` — single-step prediction agrees with `run_epoch` when alpha=1
- `test_closed_loop_grad_checkpoint.py` — gradients match between `_cl_run_segment` checkpointed and non-checkpointed paths
- `test_closed_loop_schedule.py` — alpha ramp schedule produces the expected per-epoch values

**Gradient diagnostics** (one-shot probes for chunk-len / direction-consistency questions, outputs in `tmp/grad_probe/`):
- `scripts/grad_norm_probe.py` — sweeps `bptt_chunk_len ∈ {16…512}` on one real seeg batch, captures per-parameter `|∇θ|₂`. Variant list controlled by an `ABLATIONS` constant near the top.
- `scripts/plot_grad_groups.py` — re-plots `grad_norms.csv` grouped by physical role (output / recurrent / dendritic / SFA / STD)
- `scripts/grad_cosine_consistency.py` — runs N batches at fixed chunk_len, computes mean pairwise cosine of each param's gradient (≈1 = consistent, ≈0 = direction noise → Adam can't accumulate)

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
