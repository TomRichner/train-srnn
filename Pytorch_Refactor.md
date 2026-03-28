# PyTorch + Hydra Refactor

**Started:** 2026-03-26
**Source:** `experiments_with_ltcs/` (TF 1.x) + `cloud/` (GCP scripts)
**Target:** `pytorch_refactor/` (PyTorch 2.2+, Hydra, torch.compile, bmm ablation tiling)

---

## Status

### Models (✅ complete)
- **`models/srnn_cell.py`** — `SRNNCell` + `BatchedSRNNCell`. 15 presets in `SRNN_PRESETS`. 4 ODE solvers (semi_implicit, explicit, rk4, exponential). Piecewise sigmoid, Dale's law, SFA (multi-timescale), STD. torch.compile-friendly: ablation differences are float buffer masks, no Python branching on tensor values.
- **`models/ltc_cell.py`** — `LTCCell` with sensory/recurrent sigmoid gating, trainable reversal potentials, 3 ODE solvers, parameter constraints.
- **`models/ctrnn_cell.py`** — `CTRNNCell` (Euler), `NODECell` (RK4), `CTGRUCell` (M=8 multi-timescale GRU).
- **`models/sequence_model.py`** — `SequenceModel` wrapper: unrolling, I/O masks, truncated BPTT, readout head, trainable IC. `LSTMCellWrapper` baseline.
- **`models/factory.py`** — `build_model(cfg)` / `build_batched_model(cfg, names)` from Hydra config.

### Data (✅ complete)
- **`data/datasets.py`** — 9 loaders (SMNIST, HAR, gesture, occupancy, traffic, power, ozone, person, cheetah). Batch-first numpy `(N, T, F)`.
- **`data/transforms.py`** — PCHIP time stretch, palindrome looping, random windowing, `wrap_train_batch()` / `wrap_eval_batch()`.

### Utils (✅ complete)
- **`utils/io_masks.py`** — Neuron partitioning (1/4 input, 1/2 inter, 1/4 output).
- **`utils/lr_schedule.py`** — `WarmupHoldCosineSchedule` (warmup → hold → cosine decay).
- **`utils/trainable_ic.py`** — `TrainableIC` module + `compute_burn_in()`. Fully integrated (was deferred in TF version).

### Hydra Configs (✅ complete)
- `conf/config.yaml` — main config with CLI overrides
- `conf/model/` — 17 configs: lstm, ltc, ltc_rk, ltc_ex, ctrnn, node, ctgru, srnn + 9 SRNN ablation variants (defaults inheritance)
- `conf/task/` — 9 task configs
- Usage: `python train.py model=srnn_e_only task=har epochs=200 seed=1`

### Training (✅ complete)
- **`train.py`** — Hydra entry point. Auto device, optional `torch.compile` on CUDA, Adam + warmup-hold-cosine, burn-in IC, checkpointing, results CSV.

### BMM Ablation Tiling (✅ complete)
- `BatchedSRNNCell` tiles K ablations into parallel GPU compute via `torch.bmm`
- Weights `(K, N, N)`, SFA/STD/Dale's masks `(K, 1, ...)` as buffers
- Usage: `python train.py batched_ablations='[srnn, srnn-no-adapt, srnn-E-only]'`

### Cloud (✅ complete)
- `cloud/config.env` — GCP config for `srnn-pytorch` image family
- `cloud/startup.sh` — VM boot → clone → train via Hydra → upload to GCS → self-delete
- `cloud/launch_run.sh`, `launch_all.sh` — single + matrix launchers with concurrency
- `cloud/monitor.sh` — completion status grid
- `cloud/collect_results.py` — GCS aggregation with mean±std tables
- `cloud/build_image.sh` — PyTorch VM image builder
- `cloud/experiments/*.env` — 9 per-experiment configs

### Integration Tests (✅ passing)
- All 9 cell types build from Hydra config, correct shapes, gradients flow
- `BatchedSRNNCell(K=3)` bmm forward + backward verified
- All 17 model × 9 task config compositions load correctly
- Data transforms produce correct shapes

---

## Code Review (2026-03-27)

### Critical Bugs — ✅ ALL FIXED (2026-03-28)

**1. ~~Piecewise sigmoid was mathematically wrong~~** — `models/srnn_cell.py` — **FIXED**
- Critical points now use `a = S_a / 2.0`, matching TF exactly. Linear region is 90% of output range.

**2. ~~Multi-timescale SFA had no interpolation~~** — `models/srnn_cell.py` — **FIXED**
- Added `_get_tau_a_E()` / `_get_tau_a_I()` helpers that interpolate N evenly-spaced timescales between trainable lo/hi endpoints at runtime, matching TF `_make_tau_range()`. BatchedSRNNCell initializes intermediate values via interpolation.

**3. ~~CTGRUCell was simplified/incorrect~~** — `models/ctrnn_cell.py` — **FIXED**
- Complete rewrite with learned `tau_r_dense` and `tau_s_dense` Dense layers, data-dependent softmax weighting over timescales, exponential decay per timescale — matches TF architecture.

**4. ~~LTC initialization values were wrong~~** — `models/ltc_cell.py` — **FIXED**
- `w_init_min=0.01`, `cm_init=0.5` (constant), `gleak_init=1.0` (constant) — matches TF.

**5. ~~LTC method name mismatch~~** — `models/ltc_cell.py` — **FIXED**
- Renamed `apply_weight_constraints()` → `constrain_parameters()` to match `SequenceModel` call.

### High Priority Issues — ✅ ALL FIXED (2026-03-28)

**6. ~~Default SRNN config had wrong adaptation timescales~~** — **FIXED**
- `conf/model/srnn.yaml` and `SRNNConfig` default: `n_a_E: 3, n_a_I: 3`.

**7. ~~W_in_mask never passed through factory~~** — **FIXED**
- `build_model()` now creates W_in_mask from neuron partition and passes to `build_cell()`.
- All cell types (LTC, SRNN, CTRNN, NODE, CTGRU) and `BatchedSRNNCell` accept and apply W_in_mask.
- SRNNCell applies mask to `W_in` rows; other cells use existing mask mechanisms.

**8. `wrap_train_batch` RNG state management was fragile** — `data/transforms.py` — **FIXED** (prior vectorization)
- Replaced per-sample loops and fragile RNG save/restore with clean vectorized batch operations.

### What's Correct

- **ODE solvers** — semi-implicit, explicit, RK4, exponential Euler all match TF for both SRNN and LTC
- **State packing/unpacking** — order matches exactly: `[a_E_flat, a_I_flat, b_E, b_I, x]`
- **Dale's law** — softplus + sign flip on I columns is correct
- **STD dynamics** — recovery/depression semi-implicit form matches
- **BatchedSRNNCell bmm** — reshaping math works, ablation masks as buffers is compile-friendly, gradients flow
- **Hydra config architecture** — ablation variants inherit from base srnn.yaml, CLI overrides work
- **SequenceModel** — unrolling, truncated BPTT, output mask indexing all correct
- **LR schedule** — three-phase warmup/hold/cosine with identical boundaries
- **torch.compile compatibility** — no tensor-valued control flow in forward passes
- **Cloud infrastructure** — clean port of launch/monitor/collect scripts

### Unfinished

- **Lyapunov analysis** — not ported (numpy-based, needs PyTorch cell adapter)
- **Smoke test** — never actually run (needs dataset files in `data/`)
- **No numerical equivalence test** — should compare TF and PyTorch outputs on identical inputs
- **Sparsity/L0 regularization** — TF experiments (ozone, occupancy, person, cheetah) support `--sparsity` for L0 masking. Not ported.

### Future Upgrades

- **DataLoader** — currently all numpy, no `torch.utils.data.Dataset`, no pinned memory, no async prefetch. Fine for CPU but leaves GPU throughput on the table.
- **Gradient clipping** — not in TF version either, but standard for RNN training stability.
- **Mixed precision** — `torch.amp` autocast would speed up GPU runs alongside `torch.compile`.
- **Logging** — just CSV and stdout. Could add wandb/tensorboard.
- ~~**Vectorized augmentation**~~ — Done. `wrap_train_batch` now uses vectorized batch operations.
- **Checkpoint format** — uses `torch.save`; could use `safetensors` for faster/safer serialization.

---

## Remaining Work

- **Fix critical bugs 1–5 above** (blocking for correctness)
- **Fix high priority issues 6–8** (blocking for equivalence with TF)
- **Run smoke test** after bug fixes — needs dataset files in `data/`
- **Numerical equivalence test** — run TF and PyTorch on same inputs, compare outputs
- **Cloud deploy** — build `srnn-pytorch` image, upload datasets, launch full run
- **Port Lyapunov analysis** to PyTorch

---

## Key Differences from TF Version

| | TF 1.x (`experiments_with_ltcs/`) | PyTorch (`pytorch_refactor/`) |
|---|---|---|
| Config | argparse per script | Hydra YAML with overrides |
| Models | `tf.nn.rnn_cell.RNNCell` | `nn.Module` with `(input, state) → (output, state)` |
| Ablations | One model per process | `BatchedSRNNCell` tiles K via bmm |
| Compilation | N/A | `torch.compile` on CUDA |
| Trainable IC | Deferred / partial | Fully integrated |
| Data format | Time-major `(T, B, F)` | Batch-first `(B, T, F)` |
| Training loop | Per-script with copy-paste | Shared `train.py` + Hydra |
| Cloud args | Shell flags → `--model lstm` | Hydra overrides → `model=lstm` |
