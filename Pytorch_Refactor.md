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

## Remaining

- **Smoke test:** Run `smoke_test.sh` (2 epochs, all combos) — needs dataset files in `data/`
- **Cloud deploy:** Build `srnn-pytorch` image, upload datasets, launch full run
- **Lyapunov:** Port to PyTorch (currently numpy, needs cell interface adapter)

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
