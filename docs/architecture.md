# Architecture

How a training run flows through the code, and the shapes it carries.

## Entry point

`train.py` is short: seed, device, then

```
task    = build_task(cfg)                 # train_srnn/data
data    = task.load(cfg.task.data_dir)    # Dataset of (N, T, F) arrays
model   = build_model(cfg)                # train_srnn/models/factory.py
trainer = TRAINERS[cfg.task.trainer](cfg, model, task, data, device, run_dir)
trainer.fit()
```

`cfg` is a `TrainConfig` composed by Hydra from the dataclasses in
`train_srnn/config.py`. `conf/config.yaml` holds only the defaults list and
the run-directory setting; the run directory is
`$SRNN_RESULTS_DIR/<task>/<run_name>` and doubles as Hydra's output dir.

## Configuration

Three dataclass families are registered with Hydra's `ConfigStore`:

- `TrainConfig`: optimizer, schedule, BPTT and checkpointing knobs, device,
  AMP, closed-loop block, paths, `run_name`, `output_dir`.
- `TaskConfig` and subclasses: one per task (`HarConfig`, `Cheetah100Config`,
  ...). Per-task training knobs (`batch_size`, `window_len`, `bptt_len`,
  `bptt_chunk_len`, `trainer`, `h`) live here, so `task=cheetah100` selects
  the ring trainer and its chunking in one go.
- `ModelConfig` and subclasses: `SRNNModelConfig` (shared settings plus
  `variants` and `variant_seeds`), `LTCModelConfig`, `CTRNNModelConfig`,
  `NODEModelConfig`, `CTGRUModelConfig`, `LSTMModelConfig`.

Struct mode makes a misspelled override fail before anything runs.
`compose_config(overrides)` builds the same config outside `@hydra.main`
for tests and scripts.

## Tasks (`train_srnn/data`)

`Task.load(data_dir)` returns a `Dataset` of `(x, y)` arrays per split with
shape `(N, T, F)` (labels `(N, T)` or `(N,)`) and, for trace tasks, the
whole z-scored training trace `(T, C)`. The task also owns the loss and
metric and wraps batches:

- `ClassicTask.train_batch` applies the Hasani augmentation from
  `transforms.py` (PCHIP time stretch, palindrome loop, random window) and
  returns the readout index and the start of the gradient region.
- `TraceTask` (cheetah100, synthetic) truncates the training trace to a
  prime length, z-scores every split with train statistics, and cuts strided
  windows for evaluation. The target is the trace one sample ahead.

Tasks register in `TASKS` by name; `Cheetah100Task` serves both
`cheetah100` and `cheetah100_act` (observations plus the held control).

## Cells (`train_srnn/models`)

`RNNCell` fixes the contract `cell(x, state) -> (out, state)` with a flat
state vector, an optional `W_in_mask` buffer that zeroes the input rows of
non-input neurons, `hoist()` for per-pass constants, and
`constrain_parameters()`.

`SRNNCell` runs K variants at once. Every parameter has a leading K axis
and the state of variant k is `[a_E | a_I | b_E | b_I | x]`, padded to
the widest variant, with zero SFA and unit STD neutral contributions:

| block | shape | meaning |
|---|---|---|
| `a_E` | `(n_E, n_a_E_max)` | SFA variables per excitatory neuron and timescale |
| `a_I` | `(n_I, n_a_I_max)` | same for inhibitory neurons |
| `b_E`, `b_I` | `(n_E, n_b_E_max)`, `(n_I, n_b_I_max)` | available resource per STD timescale |
| `x` | `(N,)` | dendritic potential |

Ablations are multiplicative masks (`dales_mask`, `sfa_E_mask`,
`std_E_mask`, `echo_flags`, `per_neuron_mask`, `skip_flags`), so the forward
pass has no Python branch on the variant and compiles to one batched graph.
Per-neuron vectors are paired with per-variant scalars or log-gains; for
variants without per-neuron parameters the vector is detached in the
forward pass, so only the shared value trains. `hoist()` returns the
effective recurrent weight (Dale signs, softplus magnitudes, sparsity, gain)
once per pass instead of once per step.

The baselines (`LTCCell`, `CTRNNCell`, `NODECell`, `CTGRUCell`, `LSTMCell`)
are single-network cells with state `(B, S)`; explicit integrators come
from `ode.py`.

`SequenceModel` wraps a cell: neuron partition (input / interneuron /
output groups, the readout reads the output group), a `TrainableIC` that
burn-in overwrites with the state after a finite unforced simulation, `apply_readout` with the
per-variant residual skip, and two loops:

- `unroll(x_seg, state, alpha_seg=None, y_prev=None, hoisted=None, cell=None)`
  is the only per-timestep loop. Open loop returns the cell outputs;
  closed loop blends `x_in = (1 - alpha) x + alpha y_prev` and reads out
  every step.
- `forward(x, readout_idx, bptt_start_idx, bptt_chunk_len, grad_checkpoint, ...)`
  runs the warm-up under `no_grad`, then the gradient region in segments
  that are checkpointed and detached at chunk boundaries.

Outputs are `(K, B, O)` for an integer readout index and `(K, B, T, O)`
for a slice; single-network cells drop the K axis.

## Trainers (`train_srnn/training`)

`Trainer` owns the optimizer (Adam), the warmup-hold-cosine schedule sized
from `steps_per_epoch()`, the K-network loss (sum of K losses for backward,
per-network values for logging), `optimizer_step()` with per-variant
gradient clipping followed by `constrain_parameters()`, windowed evaluation,
burn-in, checkpoints, the history CSVs, resume, and `fit()`.

- `WindowedTrainer`: shuffled mini-batches of augmented windows, state
  reset from the IC per batch. Validation every epoch, checkpoint and test
  every `checkpoint_interval`.
- `ContinuousTrainer`: B readers at fixed phase offsets around the training
  trace as a ring. Each step every reader advances `bptt_chunk_len` samples;
  the state is detached but never reset, so slow adaptation variables carry
  across chunks and epochs. With the ring length coprime to
  `B * bptt_chunk_len` the chunk boundaries drift, so every sample pair
  eventually sits inside a gradient window. Evaluation, checkpoint, and test
  happen every `checkpoint_interval` epochs and on the last epoch. Only the
  cell is compiled, so evaluation keeps the eager cell.

Closed-loop teacher forcing is configured by `ClosedLoopConfig`; the
samplers in `closed_loop.py` draw a per-batch `(T, C)` schedule for the
windowed trainer and a per-reader `(B, T, C)` schedule for the ring.

## Run directory

```
$SRNN_RESULTS_DIR/<task>/<run_name>/
  .hydra/config.yaml        resolved config
  init.pt  epoch_NNN.pt  last.pt
  training_history.csv      epoch, optimizer_step, variant, train/valid loss and metric, lr
  test_history.csv          epoch, tag, variant, test loss and metric
  progress.json             for the cloud upload watcher
```

Checkpoints hold the model, optimizer, and scheduler state, the resolved
config, and `variant_names`, so `scripts/_runs.py:rebuild_model` can
reconstruct compatible runs for analysis. SRNN version 2 rejects legacy model
states; historical runs must be analyzed at their recorded source revision.

## Analysis (`scripts`)

`postprocess.py` downloads a run into `$SRNN_CACHE_DIR/<run>` and writes
learning curves, the learning-rate schedule, parameter evolution, per-variant
tau and weight plots and effective-parameter tables, a forward replay with
Benettin Lyapunov exponents, and a PDF report. `plot_adaptation_seeds.py`
makes the paired-seed figure; `simulate_srnn.py` runs variants open loop
under a step stimulus; the `grad_*` scripts probe gradient scale and
direction consistency.

## MATLAB-aligned experiment tools

`run_matlab_aligned.py` composes typed configuration and launches an isolated
worker for all condition/seed variants. It retains the ring trainer and
records source/data hashes, effective parameters, finite checks, and memory.
`--profile` executes one full epoch with the same scheduler configuration,
then evaluation and checkpoint reload. `summarize_matlab_aligned.py` requires
the complete validation grid, including optimizer step zero, and produces
paired learning curves and statistical summaries. See
[the protocol](matlab_alignment.md).
