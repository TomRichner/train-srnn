# PyTorch + Hydra Refactor: Liquid Time-Constant Networks

PyTorch 2.2+ reimplementation of the liquid time-constant neural network experiments originally written in TensorFlow 1.x (`experiments_with_ltcs/`). Uses Hydra for configuration, `torch.compile` for GPU acceleration, and batched `torch.bmm` for running multiple SRNN ablation variants in parallel.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train SRNN on Human Activity Recognition
python train.py model=srnn task=har seed=1

# Train LSTM baseline on Sequential MNIST
python train.py model=lstm task=smnist epochs=100 size=64

# Run 3 SRNN ablations in parallel (GPU recommended)
python train.py task=har batched_ablations='[srnn,srnn-no-adapt,srnn-E-only]'

# Use a specific device
python train.py model=ltc task=gesture device=cuda
```

All configuration is managed by [Hydra](https://hydra.cc). Any config value can be overridden from the command line.

## Models

| Model | Config | Cell | Description |
|-------|--------|------|-------------|
| **LSTM** | `model=lstm` | `LSTMCellWrapper` | Standard LSTM baseline |
| **LTC** | `model=ltc` | `LTCCell` | Liquid Time-Constant cell, semi-implicit solver |
| **LTC-RK** | `model=ltc_rk` | `LTCCell` | LTC with 4th-order Runge-Kutta solver |
| **LTC-EX** | `model=ltc_ex` | `LTCCell` | LTC with explicit Euler solver |
| **CTRNN** | `model=ctrnn` | `CTRNNCell` | Continuous-time RNN, Euler integration |
| **NODE** | `model=node` | `NODECell` | Neural ODE, RK4 integration |
| **CTGRU** | `model=ctgru` | `CTGRUCell` | Multi-timescale continuous-time GRU (M=8) |
| **SRNN** | `model=srnn` | `SRNNCell` | Full SRNN: Dale's law, SFA, STD |

### SRNN Ablation Variants

| Config | Dale's | SFA | STD | Notes |
|--------|--------|-----|-----|-------|
| `model=srnn` | yes | E+I | E+I | Full model |
| `model=srnn_no_adapt` | yes | off | off | No adaptation |
| `model=srnn_no_adapt_no_dales` | no | off | off | No adapt, no Dale's law |
| `model=srnn_e_only` | yes | E only | E only | Adaptation on excitatory neurons only |
| `model=srnn_sfa_only` | yes | E+I | off | SFA only |
| `model=srnn_std_only` | yes | off | E+I | STD only |
| `model=srnn_echo` | yes | E+I | E+I | Reservoir mode (frozen recurrent W) |
| `model=srnn_per_neuron` | yes | E+I | E+I | Per-neuron adaptation parameters |
| `model=srnn_e_only_echo` | yes | E only | E only | E-only + frozen W |
| `model=srnn_e_only_per_neuron` | yes | E only | E only | E-only + per-neuron params |

## Tasks

| Task | Config | Type | Input | Output | Seq Len |
|------|--------|------|-------|--------|---------|
| Human Activity Recognition | `task=har` | classification (6) | 561 | 6 | 16 |
| Sequential MNIST | `task=smnist` | classification (10) | 28 | 10 | 28 |
| Gesture | `task=gesture` | classification (5) | 32 | 5 | 32 |
| Occupancy | `task=occupancy` | classification (2) | 5 | 2 | 16 |
| Ozone | `task=ozone` | classification (2) | 72 | 2 | 32 |
| Person Activity | `task=person` | classification (7) | 7 | 7 | 32 |
| Power Consumption | `task=power` | regression | 6 | 1 | 32 |
| Traffic Volume | `task=traffic` | regression | 7 | 1 | 32 |
| Cheetah Motion Capture | `task=cheetah` | regression (AR) | 17 | 17 | variable |

## Configuration

The Hydra config lives in `conf/`. Key parameters with defaults:

```yaml
# conf/config.yaml
seed: 1
epochs: 200
batch_size: 128
lr: 5e-4
size: 32              # hidden units (num_units)
stretch_lo: 0.8       # time-stretch augmentation lower bound
stretch_hi: 1.2       # time-stretch augmentation upper bound
min_loops: 5          # minimum palindrome loops
min_loop_len: 500     # minimum total looped sequence length
burn_in: 30.0         # burn-in seconds for trainable IC
device: 'auto'        # auto | cpu | cuda | mps
compile: true         # torch.compile on CUDA
batched_ablations: null  # list of SRNN preset names for parallel bmm
output_dir: results/${task.name}/${model.name}_${size}
```

Override any value from the command line:

```bash
python train.py model=srnn task=har epochs=50 lr=1e-3 size=64 seed=42
```

## Training Pipeline

1. **Data loading** -- `data/datasets.py` loads numpy arrays `(N, T, features)`
2. **Augmentation** -- `data/transforms.py` applies PCHIP time-stretching and palindrome looping
3. **Model construction** -- `models/factory.py` builds `SequenceModel` wrapping the selected RNN cell
4. **Training loop** -- `train.py` runs Adam + warmup-hold-cosine LR schedule
5. **Checkpointing** -- Best model (by validation metric) and periodic checkpoints saved to `output_dir`
6. **Results** -- Single-row CSV with train/valid/test loss and metric

## Cloud (GCP)

Run experiments at scale on GCP VMs. See `cloud/` for infrastructure scripts.

### Prerequisites

The repo is private. VMs authenticate via an SSH deploy key stored in GCP Secret Manager:

1. Generate a deploy key: `ssh-keygen -t ed25519 -C "train-srnn-deploy-key" -f ~/.ssh/train-srnn-deploy -N ""`
2. Add the public key to GitHub: `TomRichner/train-srnn` > Settings > Deploy keys (read-only)
3. Store the private key in Secret Manager:
   ```bash
   gcloud secrets create train-srnn-deploy-key --project=liquidneuralnets --replication-policy=automatic
   gcloud secrets versions add train-srnn-deploy-key --project=liquidneuralnets --data-file=~/.ssh/train-srnn-deploy
   ```
4. Grant the Compute Engine default service account access:
   ```bash
   PROJECT_NUMBER=$(gcloud projects describe liquidneuralnets --format='value(projectNumber)')
   gcloud secrets add-iam-policy-binding train-srnn-deploy-key --project=liquidneuralnets \
       --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
       --role="roles/secretmanager.secretAccessor"
   ```

### Usage

```bash
# Build VM image (one-time)
bash cloud/build_image.sh

# Launch a single run
bash cloud/launch_run.sh my-experiment har srnn 1

# Launch batched ablations in the cloud
bash cloud/launch_run.sh my-run smnist srnn 1 "batched_ablations='[srnn-E-only,srnn-e-only-echo]' epochs=15"

# Launch full matrix (all models x all tasks x 5 seeds)
bash cloud/launch_all.sh full-run --seeds 5

# Monitor progress
bash cloud/monitor.sh full-run

# Collect results into tables
python cloud/collect_results.py full-run --output results.csv
```

VMs self-delete after training completes and results are uploaded to GCS.

## Post-run analysis

One command does it all — downloads a run from GCS, generates run-level loss/metric/LR/weight-evolution plots, and emits per-variant tau evolution, W_EI evolution, and effective-parameter tables into per-variant subfolders:

```bash
python scripts/postprocess.py <run_name>
# Optional flags:
#   --task <name>             # default: seeg
#   --seed <n>                # default: 1
#   --bucket gs://...         # default: from cloud/config.env
#   --skip-download           # use existing tmp/<run>/ contents
#   --variants v1,v2,...      # restrict per-variant outputs
```

Output layout:

```
tmp/<run>/
  init.pt, last.pt, epoch_*.pt, *.csv, *.json, training_log.txt
  curves_{skip,no-skip}.png         linear, semilogy, log_log, semilogy_direct
  lr_schedule.png                   linear / semilogy / log-log
  weight_evolution.png              top-16 most-changed params
  <variant>/
    tau_evolution.png               with grey overlay on masked-inactive panels
    W_EI_evolution.png              mean E vs I recurrent weights
    param_table.txt                 effective (post-transform) values, init vs final
```

Loss/metric curves are split skip vs no-skip because the autoregressive skip residual gives those variants a much lower y-scale baseline. `param_table.txt` reports **effective (post-transform) values** — what actually multiplies things on the right-hand side after softplus/exp/scale composition — with init/final mean & std for each parameter group.

For diagnostic gradient analysis (chunk-length sensitivity, direction consistency), see `scripts/grad_norm_probe.py` and `scripts/grad_cosine_consistency.py`.

A complete LaTeX specification of the full model — every parameter, buffer, mask, ODE RHS, and ablation gating — lives in [`FullModel.md`](FullModel.md) (rendered to `FullModel.pdf` via pandoc + lualatex). Use it as the authoritative reference for what each effective parameter is and how it composes from raw stored values.

## Project Structure

```
train-srnn/
  train.py                  # Hydra training entry point
  smoke_test.sh             # Quick 2-epoch test of all model x task combos
  requirements.txt
  conf/
    config.yaml             # Root config (defaults, hyperparameters)
    model/                  # 17 model configs (lstm, ltc, srnn, ablations...)
    task/                   # 9 task configs (har, smnist, gesture...)
  train_srnn/               # Python package
    models/
      sequence_model.py     # SequenceModel wrapper (unrolling, I/O masks, readout)
      factory.py            # build_model() / build_batched_model() from config
      srnn_cell.py          # SRNNCell + BatchedSRNNCell + SRNN_PRESETS
      ltc_cell.py           # LTCCell (3 ODE solvers)
      ctrnn_cell.py         # CTRNNCell, NODECell, CTGRUCell
    data/
      datasets.py           # 9 dataset loaders → numpy (N, T, F)
      transforms.py         # Time stretch, palindrome loop, train/eval wrappers
    utils/
      io_masks.py           # Neuron partitioning (input / inter / output)
      lr_schedule.py        # WarmupHoldCosineSchedule
      trainable_ic.py       # TrainableIC + compute_burn_in()
  cloud/
    config.env              # GCP project, zone, bucket, machine config
    startup.sh              # VM boot script (clone, train, upload, self-delete)
    build_image.sh          # Build PyTorch VM image
    launch_run.sh           # Launch single training VM
    launch_all.sh           # Launch full experiment matrix
    monitor.sh              # Monitor running/completed experiments
    collect_results.py      # Aggregate results from GCS
    experiments/            # Per-task VM configs (machine tier, batch size, epochs)
```

## Key Differences from TF 1.x Version

| | TF 1.x (`experiments_with_ltcs/`) | PyTorch (`train-srnn/`) |
|---|---|---|
| Config | argparse per script | Hydra YAML with CLI overrides |
| Models | `tf.nn.rnn_cell.RNNCell` | `nn.Module` with `(input, state) -> (output, state)` |
| Ablations | One model per process | `BatchedSRNNCell` tiles K variants via `torch.bmm` |
| Compilation | N/A | `torch.compile` on CUDA |
| Trainable IC | Deferred / partial | Fully integrated with burn-in |
| Data format | Time-major `(T, B, F)` | Batch-first `(B, T, F)` |
| Training loop | Per-script with copy-paste | Shared `train.py` + Hydra |
| Cloud args | Shell flags `--model lstm` | Hydra overrides `model=lstm` |
