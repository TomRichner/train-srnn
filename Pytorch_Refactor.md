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

### Critical Bugs (will produce wrong results)

**1. Piecewise sigmoid is mathematically wrong** — `models/srnn_cell.py`
- TF uses `S_a=0.9` as the **linear** region width (90% of output range is linear, quadratic rounding only at edges).
- PyTorch treats `S_b = 1 - S_a = 0.1` as the linear width — making the activation almost entirely quadratic.
- The critical points `x1, x2, x3, x4` are computed with swapped roles of `S_a` and `S_b`.
- **Impact:** Fundamentally different activation shape. SRNN dynamics will not match TF.

**2. Multi-timescale SFA has no interpolation** — `models/srnn_cell.py`
- When `n_a_E >= 2`, TF interpolates N evenly-spaced time constants between `tau_lo` and `tau_hi` via `_make_tau_range()`.
- PyTorch just concatenates `[lo, hi]` as a 2-element parameter — no intermediate timescales generated.
- **Impact:** Multi-timescale SFA is broken. Only boundary values are used.

**3. CTGRUCell is a simplified/incorrect implementation** — `models/ctrnn_cell.py`
- TF CTGRU uses **learned Dense layers** (`tau_r`, `tau_s`) with data-dependent softmax weighting over timescales.
- PyTorch uses **static pre-computed** `alpha = softmax(-ln_tau)` with no learned adaptation.
- **Impact:** Completely different model. PyTorch CTGRU has no input-dependent timescale selection.

**4. LTC initialization values are wrong** — `models/ltc_cell.py`

| Parameter | TF (correct) | PyTorch (wrong) |
|-----------|-------------|----------------|
| `w_init_min` | 0.01 | 0.001 (10x too small) |
| `cm_init` | constant 0.5 | U(0.4, 0.6) |
| `gleak_init` | constant 1.0 | U(0.001, 1.0) |

**5. LTC `constrain_parameters()` method name mismatch** — `models/ltc_cell.py`, `models/sequence_model.py`
- `SequenceModel.constrain_parameters()` calls `self.cell.constrain_parameters()`.
- `LTCCell` names its method `apply_weight_constraints()`.
- **Impact:** Parameter clamping (cm_t, gleak, W, sensory_W) silently never runs for LTC.

### High Priority Issues

**6. Default SRNN config has wrong adaptation timescales** — `conf/model/srnn.yaml`
- Config sets `n_a_E: 1, n_a_I: 1`.
- TF experiments all use `n_a_E: 3, n_a_I: 3`.
- Default config produces a simpler model than what was actually trained.

**7. W_in_mask never passed through factory** — `models/factory.py`, `models/sequence_model.py`
- All cells accept `W_in_mask` in their constructors, but `factory.py` never creates or passes it.
- `SequenceModel` generates `input_mask` as a buffer but never feeds it to the cell.
- **Impact:** Input neuron partitioning is effectively disabled. All neurons receive input.

**8. `wrap_train_batch` RNG state management is fragile** — `data/transforms.py`
- Applies identical stretch/loop/window across a batch by saving and restoring numpy RNG state.
- State restoration after `random_window()` consumes RNG calls is error-prone.
- Could produce misaligned augmentations between samples in a batch.

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
- **Vectorized augmentation** — `wrap_train_batch` loops over batch elements in Python for stretch/loop. Could be much faster.
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
