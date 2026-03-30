# Code and Data Structure Reference

Complete reference for all code, data structures, tensor shapes, and cloud infrastructure in the PyTorch/Hydra refactor.

---

## Table of Contents

1. [Hydra Configuration System](#1-hydra-configuration-system)
2. [RNN Cell Implementations](#2-rnn-cell-implementations)
3. [Sequence Model Wrapper](#3-sequence-model-wrapper)
4. [Model Factory](#4-model-factory)
5. [Data Loading](#5-data-loading)
6. [Data Transforms and Augmentation](#6-data-transforms-and-augmentation)
7. [Utilities](#7-utilities)
8. [Training Loop](#8-training-loop)
9. [Cloud Infrastructure](#9-cloud-infrastructure)
10. [Tensor Shape Reference](#10-tensor-shape-reference)

---

## 1. Hydra Configuration System

All configuration is managed by Hydra, with YAML files under `conf/`. The root config composes a model config and a task config via Hydra defaults. Every value can be overridden from the command line.

### Root Config (`conf/config.yaml`)

```yaml
defaults:
  - model: srnn        # selects conf/model/srnn.yaml
  - task: har          # selects conf/task/har.yaml
  - _self_

seed: 1
epochs: 200
batch_size: 128
lr: 5e-4
size: 32              # maps to model.num_units and controls hidden size
stretch_lo: 0.8       # PCHIP time-stretch lower bound
stretch_hi: 1.2       # PCHIP time-stretch upper bound
min_loops: 5          # minimum palindrome loop repetitions
min_loop_len: 500     # minimum total timesteps after looping
burn_in: 30.0         # seconds of zero-input simulation for trainable IC
log_interval: 1
checkpoint_interval: 10
device: 'auto'        # auto | cpu | cuda | mps
compile: true         # torch.compile on CUDA devices
batched_ablations: null  # e.g. '[srnn,srnn-no-adapt,srnn-E-only]'
output_dir: results/${task.name}/${model.name}_${size}
```

### Model Configs (`conf/model/*.yaml`)

Each model config sets `name`, `type`, and model-specific hyperparameters. The `num_units` field is interpolated from the root `size` parameter via `${size}`.

**LSTM** (`lstm.yaml`):
```yaml
name: lstm
type: lstm
num_units: ${size}
```

**LTC** (`ltc.yaml`):
```yaml
name: ltc
type: ltc
num_units: ${size}
solver: semi_implicit     # semi_implicit | explicit | rk4
ode_unfolds: 6
erev_init_factor: 1.0
w_init_min: 0.01
w_init_max: 1.0
gleak_init_min: 1.0
gleak_init_max: 1.0
cm_init_min: 0.5
cm_init_max: 0.5
fix_vleak: false
fix_gleak: false
fix_cm: false
```

**LTC-RK** (`ltc_rk.yaml`): Inherits from ltc.yaml, overrides `solver: rk4`.

**LTC-EX** (`ltc_ex.yaml`): Inherits from ltc.yaml, overrides `solver: explicit`.

**CTRNN** (`ctrnn.yaml`):
```yaml
name: ctrnn
type: ctrnn
num_units: ${size}
global_feedback: true     # concatenate [input, state] as cell input
cell_clip: 0.0
unfolds: 6
delta_t: 0.1
fix_tau: true
tau: 1.0
```

**NODE** (`node.yaml`):
```yaml
name: node
type: node
num_units: ${size}
global_feedback: true
cell_clip: 0.0
unfolds: 6
h: 0.1                   # RK4 step size
fix_tau: true
tau: 1.0
```

**CTGRU** (`ctgru.yaml`):
```yaml
name: ctgru
type: ctgru
num_units: ${size}
M: 8                      # number of parallel timescales per neuron
tau_base: 1.0
cell_clip: -1.0            # disabled when < 0
```

**SRNN** (`srnn.yaml`):
```yaml
name: srnn
type: srnn
num_units: ${size}
dales: true               # Dale's law (E/I partitioning)
n_a_E: 3                  # SFA timescales for excitatory neurons
n_a_I: 3                  # SFA timescales for inhibitory neurons
n_b_E: 1                  # STD components for excitatory neurons
n_b_I: 1                  # STD components for inhibitory neurons
per_neuron: false          # shared vs per-neuron adaptation params
echo: false                # reservoir mode (freeze recurrent W)
solver: semi_implicit      # semi_implicit | explicit | rk4 | exponential
h: 0.04                   # ODE step size
ode_unfolds: 6
readout: synaptic          # synaptic | rate | dendritic
sparsity: 0.5
```

**SRNN ablation variants** inherit from `srnn.yaml` and override specific flags:

| Config | `dales` | `n_a_E` | `n_a_I` | `n_b_E` | `n_b_I` | `echo` | `per_neuron` |
|--------|---------|---------|---------|---------|---------|--------|-------------|
| `srnn` | true | 3 | 3 | 1 | 1 | false | false |
| `srnn_no_adapt` | true | 0 | 0 | 0 | 0 | false | false |
| `srnn_no_adapt_no_dales` | false | 0 | 0 | 0 | 0 | false | false |
| `srnn_e_only` | true | 3 | 0 | 1 | 0 | false | false |
| `srnn_sfa_only` | true | 3 | 3 | 0 | 0 | false | false |
| `srnn_std_only` | true | 0 | 0 | 1 | 1 | false | false |
| `srnn_echo` | true | 3 | 3 | 1 | 1 | true | false |
| `srnn_per_neuron` | true | 3 | 3 | 1 | 1 | false | true |
| `srnn_e_only_echo` | true | 3 | 0 | 1 | 0 | true | false |
| `srnn_e_only_per_neuron` | true | 3 | 0 | 1 | 0 | false | true |

### Task Configs (`conf/task/*.yaml`)

Each task config specifies the dataset name, input/output dimensions, sequence length, task type, and whether labels are per-timestep.

| Task | `input_size` | `output_size` | `seq_len` | `task_type` | `per_timestep_labels` |
|------|-------------|--------------|-----------|-------------|---------------------|
| har | 561 | 6 | 16 | classification | true |
| smnist | 28 | 10 | 28 | classification | false |
| gesture | 32 | 5 | 32 | classification | true |
| occupancy | 5 | 2 | 16 | classification | false |
| ozone | 72 | 2 | 32 | classification | false |
| person | 7 | 7 | 32 | classification | true |
| power | 6 | 1 | 32 | regression | true |
| traffic | 7 | 1 | 32 | regression | true |
| cheetah | 17 | 17 | null | regression | true |

The `data_dir` field in each task config defaults to `data/<task_name>` (relative to working directory).

---

## 2. RNN Cell Implementations

All cells follow a common interface:

```python
cell(input, state) -> (output, new_state)
# input:  (batch, input_size)
# state:  (batch, state_size)
# output: (batch, num_units)    — always num_units, even if state_size differs
# new_state: (batch, state_size)
```

Every cell has a `state_size` attribute (int) and optional `constrain_parameters()` method.

### 2.1 LTCCell (`models/ltc_cell.py`)

Liquid Time-Constant cell with conductance-based synapses, sigmoid gating, and learnable reversal potentials.

**Config dataclass: `LTCConfig`**

| Field | Default | Description |
|-------|---------|-------------|
| `num_units` | 32 | Hidden size |
| `solver` | "semi_implicit" | ODE solver: semi_implicit, explicit, rk4 |
| `ode_unfolds` | 6 | Sub-steps per timestep |
| `erev_init_factor` | 1.0 | Scale for reversal potential initialization |
| `w_init_min` / `w_init_max` | 0.01 / 1.0 | Synaptic weight init range |
| `gleak_init_min` / `gleak_init_max` | 1.0 / 1.0 | Leak conductance init range |
| `cm_init_min` / `cm_init_max` | 0.5 / 0.5 | Membrane capacitance init range |
| `fix_vleak` / `fix_gleak` / `fix_cm` | false | Freeze individual parameters |

**Learnable parameters:**

| Parameter | Shape | Description |
|-----------|-------|-------------|
| `sensory_mu` | `(input_size, N)` | Sensory synapse sigmoid center |
| `sensory_sigma` | `(input_size, N)` | Sensory synapse sigmoid width |
| `sensory_W` | `(input_size, N)` | Sensory synaptic weights |
| `sensory_erev` | `(input_size, N)` | Sensory reversal potentials |
| `mu` | `(N, N)` | Recurrent synapse sigmoid center |
| `sigma` | `(N, N)` | Recurrent synapse sigmoid width |
| `W` | `(N, N)` | Recurrent synaptic weights |
| `erev` | `(N, N)` | Recurrent reversal potentials |
| `vleak` | `(N,)` | Leak reversal potential |
| `gleak` | `(N,)` | Leak conductance |
| `cm_t` | `(N,)` | Membrane capacitance |

**Registered buffers:**

| Buffer | Shape | Description |
|--------|-------|-------------|
| `W_in_mask` | `(N,)` | Binary mask for input neurons (optional) |

**State:** `(batch, N)` -- membrane voltage `v`.

**ODE dynamics (semi-implicit form):**

```
g_sensory = sigmoid(sensory_mu, sensory_sigma, input) * sensory_W
g_recurrent = sigmoid(mu, sigma, v) * W

numerator = gleak * vleak + sum(g_sensory * sensory_erev) + sum(g_recurrent * erev)
denominator = gleak + sum(g_sensory) + sum(g_recurrent)

v_new = (numerator + v * cm / dt) / (denominator + cm / dt)
```

The RK4 and explicit solvers compute the raw `dv/dt = (numerator/denominator - v) / cm` and integrate accordingly.

**`constrain_parameters()`:** Clamps `gleak >= 1e-4`, `cm_t >= 1e-4`, and `W` / `sensory_W` to `[w_init_min, w_init_max]`.

### 2.2 SRNNCell (`models/srnn_cell.py`)

Spiking RNN with excitatory/inhibitory partitioning (Dale's law), spike-frequency adaptation (SFA), and short-term depression (STD).

**Config dataclass: `SRNNConfig`**

| Field | Default | Description |
|-------|---------|-------------|
| `num_units` | 32 | Hidden size (N). Split as n_E = N//2, n_I = N - n_E |
| `dales` | true | Enforce Dale's law (E/I weight sign constraint) |
| `n_a_E` / `n_a_I` | 3 / 3 | Number of SFA timescales for E/I neurons |
| `n_b_E` / `n_b_I` | 1 / 1 | Number of STD components for E/I neurons |
| `per_neuron` | false | Per-neuron vs shared adaptation parameters |
| `echo` | false | Freeze recurrent weights (reservoir mode) |
| `solver` | "semi_implicit" | semi_implicit, explicit, rk4, exponential |
| `h` | 0.04 | ODE step size (dt) |
| `ode_unfolds` | 6 | Sub-steps per timestep |
| `readout` | "synaptic" | Output type: synaptic, rate, dendritic |
| `sparsity` | 0.5 | Fraction of zero entries in recurrent W |

**Derived properties:**

- `n_E = num_units // 2` (excitatory neuron count)
- `n_I = num_units - n_E` (inhibitory neuron count)
- `state_size = n_E * n_a_E + n_I * n_a_I + n_E * n_b_E + n_I * n_b_I + num_units`

**State layout (packed flat):**

```
[a_E_flat | a_I_flat | b_E | b_I | x]
  n_E*n_a_E  n_I*n_a_I  n_E*n_b_E  n_I*n_b_I  N
```

- `a_E`: `(batch, n_E, n_a_E)` -- SFA variables for excitatory neurons
- `a_I`: `(batch, n_I, n_a_I)` -- SFA variables for inhibitory neurons
- `b_E`: `(batch, n_E, n_b_E)` -- STD variables for excitatory neurons
- `b_I`: `(batch, n_I, n_b_I)` -- STD variables for inhibitory neurons
- `x`: `(batch, N)` -- dendritic/membrane potential

**Learnable parameters:**

| Parameter | Shape (shared) | Shape (per_neuron) | Description |
|-----------|---------------|-------------------|-------------|
| `W_raw` | `(N, N)` | same | Recurrent weight matrix |
| `W_in` | `(N, input_size)` | same | Input weight matrix |
| `a_0` | `(N,)` | same | Firing threshold |
| `log_tau_d` | `(1,)` | `(N,)` | Dendritic time constant (log-space) |
| `log_tau_a_E_lo/hi` | `(1,)` each | `(n_E,)` each | SFA timescale endpoints (multi-timescale) |
| `log_tau_a_E` | `(1,)` | `(n_E,)` | SFA timescale (single-timescale, n_a_E=1) |
| `log_c_E` | `(1,)` | `(n_E,)` | SFA coupling strength |
| `c_0_E` | `(1,)` | `(n_E,)` | SFA baseline offset |
| (same pattern for `_I` variants) | | | |
| `log_tau_b_rec_E` | `(1,)` | `(n_E,)` | STD recovery time constant |
| `log_tau_b_rel_E` | `(1,)` | `(n_E,)` | STD release/facilitation time constant |
| (same pattern for `_I` variants) | | | |

**Registered buffers:**

| Buffer | Shape | Description |
|--------|-------|-------------|
| `sparsity_mask` | `(N, N)` | Binary mask for sparse connectivity |
| `W_in_mask` | `(N,)` | Binary mask restricting which neurons receive external input |

**Multi-timescale SFA interpolation:**

When `n_a_E >= 2`, the cell stores learnable endpoints `log_tau_a_E_lo` and `log_tau_a_E_hi`. At runtime, `_get_tau_a_E()` produces `n_a_E` evenly-spaced timescales:

```python
lo = softplus(log_tau_a_E_lo)
hi = softplus(log_tau_a_E_hi)
t = linspace(0, 1, n_a_E)   # [0.0, 0.5, 1.0] for n_a_E=3
tau = lo + (hi - lo) * t     # interpolated timescales
```

**Piecewise sigmoid activation:**

```python
def piecewise_sigmoid(x, S_a=0.9, S_c=0.0):
    # S_a = fraction of output range that is linear (0.9 = 90%)
    # Five regions: left saturated, left quadratic, linear, right quadratic, right saturated
    a = S_a / 2.0
    c = S_c
    x1 = c + a - 1.0    # left saturation boundary
    x2 = c - a          # left quadratic -> linear
    x3 = c + a          # linear -> right quadratic
    x4 = c + 1.0 - a   # right saturation boundary
```

**Effective weight matrix (Dale's law):**

```python
def _effective_W(self):
    W = softplus(W_raw) * sparsity_mask
    if dales:
        W[:, n_E:] = -W[:, n_E:]   # inhibitory columns are negative
    return W
```

**Forward pass per substep:**

1. Compute firing rate: `r = piecewise_sigmoid(x - threshold(a))`
   - `threshold(a) = a_0 + c_0 + c * sum(a, dim=-1)` (SFA raises threshold)
2. Compute synaptic output: `br = b * r` (STD modulates firing rate)
3. Recurrent drive: `u = W_eff @ br + W_in @ input`
4. Update dendritic potential: `dx/dt = (-x + u) / tau_d`
5. Update SFA: `da/dt = (-a + r) / tau_a` (each timescale independently)
6. Update STD: semi-implicit recovery/depression step

**Readout modes:**

- `synaptic`: output = `b * r` (STD-modulated firing rate)
- `rate`: output = `r` (raw firing rate)
- `dendritic`: output = `x` (membrane potential)

### 2.3 CTRNNCell (`models/ctrnn_cell.py`)

Continuous-time RNN with Euler integration.

**Config: `CTRNNConfig`**

| Field | Default | Description |
|-------|---------|-------------|
| `num_units` | 32 | Hidden size |
| `global_feedback` | true | Input is `[input, state]` concatenated |
| `cell_clip` | 0.0 | Hard clip state values (0 = disabled) |
| `unfolds` | 6 | Euler sub-steps |
| `delta_t` | 0.1 | Step size |
| `fix_tau` | true | Freeze time constant |
| `tau` | 1.0 | Membrane time constant |

**Parameters:** `W` (weight matrix), `bias`, `tau` (optional).

**State:** `(batch, N)` -- hidden activation.

**Update rule:** `x_new = x + delta_t * (-x + tanh(W @ concat(input, x) + bias)) / tau`

### 2.4 NODECell (`models/ctrnn_cell.py`)

Neural ODE with 4th-order Runge-Kutta integration. Same structure as CTRNNCell but uses RK4 substeps instead of Euler.

**Config: `NODEConfig`** -- same as CTRNNConfig but with `h` (RK4 step size) instead of `delta_t`.

**State:** `(batch, N)`.

### 2.5 CTGRUCell (`models/ctrnn_cell.py`)

Multi-timescale continuous-time GRU. Each neuron has M parallel timescale copies. Learned Dense layers produce data-dependent timescale selection weights via softmax.

**Config: `CTGRUConfig`**

| Field | Default | Description |
|-------|---------|-------------|
| `num_units` | 32 | Hidden size (N) |
| `M` | 8 | Number of parallel timescales per neuron |
| `tau_base` | 1.0 | Base timescale (timescales are `tau_base * 2^k` for k=0..M-1) |
| `cell_clip` | -1.0 | Disabled by default |

**Parameters:**

| Parameter | Shape | Description |
|-----------|-------|-------------|
| `tau_r_dense` | `Linear(fan_in, N*M)` | Data-dependent reset timescale selector |
| `tau_s_dense` | `Linear(fan_in, N*M)` | Data-dependent state timescale selector |
| `signal_dense` | `Linear(fan_in, N)` | GRU signal/candidate computation |

Where `fan_in = input_size + N` (concatenated input and collapsed state).

**Registered buffers:**

| Buffer | Shape | Description |
|--------|-------|-------------|
| `ln_tau_table` | `(1, 1, M)` | Log timescale values: `log(tau_base * 2^k)` |
| `exp_decay` | `(1, 1, M)` | Pre-computed exponential decay: `exp(-1/tau)` |

**State:** `(batch, N * M)` -- M copies of each neuron's hidden state.

**Forward:**

1. Collapse state from `(batch, N, M)` to `(batch, N)` by summing over M
2. Concatenate `[input, collapsed_state]` as fan_in
3. Compute timescale selection: `sf_r = -(tau_r_dense(fan_in) - ln_tau_table)^2`, softmax over M
4. Reset gate: `r = sum(sf_r * exp_decay * state_3d, dim=M)`
5. Candidate: `h_tilde = tanh(signal_dense([input, r]))`
6. State update with exponential interpolation per timescale
7. Output: collapsed state `(batch, N)`

---

## 3. Sequence Model Wrapper

**File:** `models/sequence_model.py`

### LSTMCellWrapper

Wraps `nn.LSTMCell` to match the `(input, state) -> (output, new_state)` interface. State is `[h, c]` concatenated: `state_size = num_units * 2`.

```python
class LSTMCellWrapper(nn.Module):
    # state_size = num_units * 2
    def forward(self, input, state):
        h, c = state.chunk(2, dim=-1)
        h_new, c_new = self.cell(input, (h, c))
        return h_new, torch.cat([h_new, c_new], dim=-1)
```

### SequenceModel

Wraps any RNN cell into a full sequence-to-prediction model.

**Constructor parameters:**

| Parameter | Description |
|-----------|-------------|
| `cell` | RNN cell module (any type above) |
| `input_size` | Features per timestep |
| `output_size` | Number of output classes or regression dims |
| `num_units` | Hidden size of the cell |
| `use_io_masks` | Enable neuron partitioning (input/inter/output) |
| `io_mask_seed` | Seed for neuron partition RNG |
| `trainable_ic` | Learn initial hidden state |
| `task_type` | "classification" or "regression" |

**Components:**

- **I/O masks** (if `use_io_masks`): `input_mask` and `output_mask` buffers from `generate_neuron_partition()`. Output mask selects ~25% of neurons for readout, reducing readout head input dimension.
- **Trainable IC** (if `trainable_ic`): `TrainableIC` module storing a learnable `(state_size,)` vector, expanded to `(batch, state_size)` at forward time.
- **Readout head**: `nn.Linear(effective_output_size, output_size)` where `effective_output_size` is either `num_units` (no mask) or `len(output_indices)` (~25% of `num_units`).

**Forward pass:**

```python
def forward(self, x, readout_idx=None, bptt_start_idx=None):
    # x: (batch, seq_len, features)
    # 1. Initialize state from TrainableIC or zeros
    # 2. Unroll cell over seq_len timesteps
    #    - If bptt_start_idx set: detach gradients for t < bptt_start_idx
    # 3. Select output at readout_idx (or last timestep)
    # 4. Apply output mask (select output neurons only)
    # 5. Linear readout -> (batch, output_size)
```

**`constrain_parameters()`:** Delegates to `cell.constrain_parameters()` if it exists. Called after each optimizer step in the training loop.

---

## 4. Model Factory

**File:** `models/factory.py`

### `build_model(cfg) -> SequenceModel`

1. Extracts `num_units`, `input_size`, `output_size`, `task_type` from Hydra config
2. Generates neuron partition via `generate_neuron_partition(num_units, seed)`
3. Creates `W_in_mask` from `make_input_mask(num_units, input_indices)` -- binary `(N,)` tensor
4. Calls `build_cell(cfg, W_in_mask)` to instantiate the appropriate cell
5. Wraps in `SequenceModel`

### `build_cell(cfg, W_in_mask=None) -> nn.Module`

Routes by `cfg.model.type`:

| Type | Cell Class | Config Dataclass |
|------|-----------|-----------------|
| `lstm` | `LSTMCellWrapper` | -- |
| `ltc` | `LTCCell` | `LTCConfig` |
| `ltc_rk` | `LTCCell` | `LTCConfig` (solver=rk4) |
| `ltc_ex` | `LTCCell` | `LTCConfig` (solver=explicit) |
| `ctrnn` | `CTRNNCell` | `CTRNNConfig` |
| `node` | `NODECell` | `NODEConfig` |
| `ctgru` | `CTGRUCell` | `CTGRUConfig` |
| `srnn*` | `SRNNCell` | `SRNNConfig` |

All cells receive `input_size` from the task config and `W_in_mask` from the factory.

### `build_batched_model(cfg, ablation_names) -> SequenceModel`

For running K SRNN ablation variants in parallel:

1. Looks up each name in `SRNN_PRESETS` dictionary
2. Verifies all share the same `solver`, `h`, `ode_unfolds`
3. Constructs `BatchedSRNNCell` with K configs stacked
4. Wraps in `SequenceModel`

### `_cfg_to_dataclass(model_cfg, dc_cls)`

Utility that extracts only the fields matching a dataclass signature from an OmegaConf DictConfig, avoiding errors from extra keys.

---

## 5. Data Loading

**File:** `data/datasets.py`

### Common Pattern

All loaders return:

```python
{
    "train": (x_train, y_train),   # numpy arrays
    "valid": (x_valid, y_valid),
    "test":  (x_test,  y_test),
    "meta":  {"input_size": int, "output_size": int, ...}
}
```

Data is always batch-first numpy: `x` has shape `(N, T, F)`, `y` has shape `(N,)` for sequence-level labels or `(N, T)` for per-timestep labels.

### Utility Functions

**`cut_in_sequences(data, labels, seq_len, inc=1)`**

Sliding window extraction:
- `data`: `(total_timesteps, features)`
- Returns: `(num_windows, seq_len, features)`, `(num_windows, seq_len)` or `(num_windows,)`

**`_split_75_10_15(x, y, seed)`** / **`_split_90_10(x, y, seed)`**

Deterministic shuffled splits using a seeded RNG.

### Dataset Loaders

| Loader | Source | Windowing | Labels | Split |
|--------|--------|-----------|--------|-------|
| `load_smnist` | torchvision MNIST | Row-by-row (28x28) | Sequence-level (digit class) | 90/10 train/valid from training set; MNIST test set |
| `load_har` | UCI HAR zip | 16-step windows, stride 1 (train) / 8 (test) | Per-timestep (6 activities) | Pre-split train/test; 90/10 train/valid |
| `load_gesture` | CSV files | 32-step windows, non-overlapping + 50% interleaved | Per-timestep (5 phases) | 75/10/15 |
| `load_occupancy` | CSV files | 16-step windows, stride 1 (train) / 8 (test) | Per-timestep (binary) | Pre-split train/test; 90/10 train/valid |
| `load_traffic` | CSV file | 32-step windows, stride 4 | Per-timestep (traffic volume) | 75/10/15 |
| `load_power` | CSV file | 32-step non-overlapping | Per-timestep (power consumption) | 75/10/15 |
| `load_ozone` | CSV files | 32-step windows, stride 4 | Sequence-level (binary ozone) | 75/10/15 |
| `load_person` | CSV files | 32-step windows, 50% overlap | Per-timestep (7 activities) | 75/10/15 |
| `load_cheetah` | npz files | 32-step windows, stride 10 | Per-timestep (17-dim autoregressive) | File-based splits |

**`load_dataset(task_name, data_dir)`**: Dispatcher that routes to the appropriate loader by task name string.

---

## 6. Data Transforms and Augmentation

**File:** `data/transforms.py`

The augmentation pipeline creates longer, varied sequences from the original data through time-stretching and palindrome looping. This forces RNNs to generalize across temporal scales rather than memorizing fixed-length patterns.

### Time Stretching

**`random_stretch_factor(lo, hi, rng)`**: Samples a stretch factor from a log-uniform distribution in `[lo, hi]`.

**`time_stretch(x, y, factor, per_timestep_labels)`**: Resamples a single sequence using PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) interpolation:
- Input `x`: `(T, F)` -> Output: `(T_new, F)` where `T_new = int(T * factor)`
- Labels: nearest-neighbor interpolation for integer labels, PCHIP for float labels

**`time_stretch_batch(batch_x, batch_y, factor, per_timestep_labels)`**: Vectorized batch version.

### Palindrome Looping

**`compute_n_loops(seq_len, min_loop_len, min_loops)`**: Determines how many forward+backward pairs to create:
- Each pair = 2 * seq_len timesteps
- Returns `max(min_loops, ceil(min_loop_len / (2 * seq_len)))`

**`palindrome_loop(x, y, n_loops, per_timestep_labels)`**: Creates `[forward, backward, forward, backward, ...]`:
- Input `x`: `(T, F)` -> Output: `(T * 2 * n_loops, F)`
- Labels follow the same palindrome pattern

**`palindrome_loop_batch(...)`**: Vectorized batch version.

### Random Window Extraction

**`random_window(x_looped, y_looped, loop_len, rng, n_bptt_loops, per_timestep_labels)`**:
- Extracts a contiguous window from the looped sequence
- Random offset within the first loop
- Window length = `(n_total_loops - 1) * loop_len`
- Returns `readout_idx` (random point in last loop) and `bptt_start_idx` (start of gradient computation)

### Training and Evaluation Wrappers

**`wrap_train_batch(batch_x, batch_y, rng, stretch_lo, stretch_hi, min_loops, min_loop_len, per_timestep_labels)`**:

```
Raw data (batch, T, F)
  -> time_stretch (random factor per batch)
  -> palindrome_loop (n_loops repetitions)
  -> random_window (extract training window)
  -> returns (aug_x, aug_y, readout_idx, bptt_start_idx)
```

**`wrap_eval_batch(batch_x, batch_y, min_loops, min_loop_len, per_timestep_labels)`**:

```
Raw data (batch, T, F)
  -> palindrome_loop only (no stretching, deterministic)
  -> readout at last timestep
  -> returns (looped_x, labels_at_readout, readout_idx)
```

### Data Flow Diagram

```
Training:
  (batch, T, F) --stretch--> (batch, T', F) --loop--> (batch, T'*2*n, F)
                 --window--> (batch, W, F) + readout_idx + bptt_start_idx

Evaluation:
  (batch, T, F) --loop--> (batch, T*2*n, F) + readout_idx = last timestep
```

---

## 7. Utilities

### Neuron Partitioning (`utils/io_masks.py`)

**`generate_neuron_partition(n, seed, frac_input=0.25, frac_output=0.25)`**:
- Randomly assigns each neuron to one of three groups: input (25%), interneuron (50%), output (25%)
- Returns three index arrays: `(input_indices, inter_indices, output_indices)`
- Used both for W_in_mask (restricting external input) and output mask (selecting readout neurons)

**`make_input_mask(n, input_indices)`**: Returns binary `(n,)` array with 1s at input neuron positions.

**`make_output_mask(n, output_indices)`**: Returns binary `(n,)` array with 1s at output neuron positions.

### Learning Rate Schedule (`utils/lr_schedule.py`)

**`WarmupHoldCosineSchedule`** (extends `torch.optim.lr_scheduler._LRScheduler`):

Three-phase LR schedule applied per optimizer step (not per epoch):

```
LR
^
|         max_lr
|        ________
|       /        \
|      /          \  cosine
|     / warmup     \  decay
|    /              \
|___/                \___  end_lr
|
+-----|------|---------|---> steps
   warmup   hold     end
   (20%)    (70%)   (100%)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `total_steps` | required | Total training iterations |
| `max_lr` | 5e-3 | Peak learning rate |
| `start_lr` | 1e-8 | Initial learning rate |
| `end_lr` | max_lr / 20 | Final learning rate |
| `warmup_frac` | 0.2 | Fraction of steps for warmup phase |
| `hold_frac` | 0.7 | Fraction of steps at which hold ends and decay begins |

### Trainable Initial Conditions (`utils/trainable_ic.py`)

**`TrainableIC(nn.Module)`**: Stores a learnable `(state_dim,)` parameter vector. Forward expands to `(batch, state_dim)` via `unsqueeze(0).expand(batch_size, -1)`.

**`compute_burn_in(cell, input_size, burn_in_seconds, device)`**:
- Runs the cell with zero input for `burn_in_seconds / dt_per_step` timesteps
- Returns the resulting `(state_dim,)` state as an initial condition
- Used to initialize `TrainableIC` to a dynamically stable fixed point

---

## 8. Training Loop

**File:** `train.py`

### Main Function (`main(cfg)`)

Decorated with `@hydra.main(config_path="conf", config_name="config")`.

**Steps:**

1. **Seed**: `torch.manual_seed(cfg.seed)`, `np.random.seed(cfg.seed)`
2. **Device**: Auto-detect CUDA > MPS > CPU
3. **Data**: `load_dataset(cfg.task.name, cfg.task.data_dir)` -> numpy arrays
4. **Model**: `build_model(cfg)` or `build_batched_model(cfg, cfg.batched_ablations)` -> `.to(device)`
5. **Compile**: `torch.compile(model)` if CUDA and `cfg.compile`
6. **Optimizer**: `Adam(lr=cfg.lr)` + `WarmupHoldCosineSchedule`
7. **Loss**: `CrossEntropyLoss` (classification) or `MSELoss` (regression)
8. **Burn-in**: If `cfg.burn_in > 0` and model has IC, run `compute_burn_in()` and copy result
9. **Training loop**: For each epoch:
   - `run_epoch(training=True)` -- forward, loss, backward, optimizer step, scheduler step
   - `model.constrain_parameters()` -- e.g. LTC weight clipping
   - `run_epoch(training=False)` -- validation
   - Checkpoint best model and periodic snapshots
10. **Test**: Final evaluation on test set
11. **Results CSV**: Write single-row CSV with all metrics

### `run_epoch(model, data_x, data_y, optimizer, scheduler, criterion, cfg, rng, device, training)`

Returns `(avg_loss, avg_metric)`.

**Per-batch steps:**

1. Sample batch indices (shuffled for training, sequential for eval)
2. Apply augmentation:
   - Training: `wrap_train_batch()` -> `(aug_x, aug_y, readout_idx, bptt_start_idx)`
   - Eval: `wrap_eval_batch()` -> `(looped_x, labels_at_readout, readout_idx)`
3. For per-timestep label tasks during training: extract `batch_y = batch_y[:, readout_idx]`
4. Convert to tensors on device
5. Forward: `model(batch_x_t, readout_idx=readout_idx, bptt_start_idx=bptt_start)`
6. Loss: CrossEntropyLoss or MSELoss
7. Backward + optimizer step (training only)
8. Accumulate loss and metric (accuracy for classification, negative MAE for regression)

### Output Files

Written to `cfg.output_dir`:

- `checkpoint_best.pt` -- model + optimizer state at best validation metric
- `checkpoint_epoch{N}.pt` -- periodic checkpoints
- `{model_name}_{size}.csv` -- single-row results CSV:

```csv
best_epoch,train_loss,train_accuracy,valid_loss,valid_accuracy,test_loss,test_accuracy
42,0.123456,0.945000,0.234567,0.912000,0.256789,0.908000
```

---

## 9. Cloud Infrastructure

### Architecture Overview

```
Local machine                          GCP
  |                                     |
  |  launch_run.sh / launch_all.sh      |
  |  -------------------------------->  gcloud compute instances create
  |                                     |
  |                                     VM boots -> startup.sh runs:
  |                                       1. Clone repo from GitHub
  |                                       2. Download dataset from GCS
  |                                       3. pip install requirements
  |                                       4. python3 train.py model=X task=Y ...
  |                                       5. Upload results to GCS
  |                                       6. Self-delete VM
  |                                     |
  |  monitor.sh                         |
  |  <------------------------------->  gcloud compute instances list
  |                                     gcloud storage ls
  |                                     |
  |  collect_results.py                 |
  |  <-------------------------------- gcloud storage cat (CSV files)
```

### GCP Configuration (`cloud/config.env`)

```bash
GCP_PROJECT="liquidneuralnets"
GCP_ZONE="us-central1-b"
GCP_BUCKET="gs://liquidneuralnets-experiments"
GCP_IMAGE_FAMILY="srnn-pytorch"
GCP_USE_SPOT="true"

DEFAULT_MACHINE_TYPE_FAMILY="n4d"
DEFAULT_MACHINE_TIER="standard-8"
BOOT_DISK_SIZE="15GB"
BOOT_DISK_TYPE="hyperdisk-balanced"    # n4d requires hyperdisk (not pd-standard/pd-balanced)

MAX_CONCURRENT_VMS=32

MODELS="lstm ltc ltc_rk ltc_ex ctrnn node ctgru srnn srnn_no_adapt srnn_no_adapt_no_dales srnn_e_only srnn_sfa_only srnn_std_only srnn_echo srnn_per_neuron srnn_e_only_echo srnn_e_only_per_neuron"
EXPERIMENTS="har gesture occupancy smnist traffic power ozone person cheetah"
DEFAULT_SEEDS=5
```

### Experiment Configs (`cloud/experiments/*.env`)

Per-task configuration overriding defaults. Each file sets training hyperparameters and VM sizing.

| Task | `ARGS` | `MACHINE_TIER` | `N_SEEDS` |
|------|--------|----------------|-----------|
| har | `epochs=200 size=32 batch_size=128` | standard-2 | 5 |
| smnist | `epochs=200 size=32 batch_size=128` | standard-2 | 5 |
| gesture | `epochs=200 size=32 batch_size=16` | standard-2 | 5 |
| occupancy | `epochs=200 size=32 batch_size=128` | standard-2 | 5 |
| ozone | `epochs=200 size=32 batch_size=16` | standard-2 | 5 |
| person | `epochs=200 size=32 batch_size=128` | standard-2 | 5 |
| power | `epochs=200 size=32 batch_size=128` | standard-2 | 5 |
| traffic | `epochs=200 size=32 batch_size=128` | standard-2 | 5 |
| cheetah | `epochs=200 size=32 batch_size=16` | standard-2 | 5 |

### VM Lifecycle (`cloud/startup.sh`)

The startup script runs automatically when a GCP VM boots. It reads experiment parameters from VM metadata tags set by `launch_run.sh`.

**Metadata tags:**

| Tag | Example | Description |
|-----|---------|-------------|
| `run-name` | `full-run` | Experiment batch identifier |
| `experiment` | `har` | Task name |
| `model` | `srnn` | Model config name |
| `seed` | `1` | Random seed |
| `bucket` | `gs://liquidneuralnets-experiments` | GCS bucket |
| `train-args` | `epochs=200 size=32` | Extra Hydra CLI overrides |

**Execution steps:**

1. Read all metadata tags via `curl` to GCE metadata server
2. Clone repo from GitHub (3 retries)
3. Download dataset from GCS (unless smnist, which downloads via torchvision)
4. Activate Python venv (`/opt/python-venv` if pre-built image, else create new)
5. Install requirements (pip)
6. Set `PYTHONPATH=/tmp/workdir` for package imports
7. Run: `python3 train.py model=$MODEL task=$EXPERIMENT seed=$SEED $TRAIN_ARGS`
8. On exit (success or failure):
   - Upload results, logs, and metadata JSON to GCS
   - Self-delete VM via `gcloud compute instances delete`

### GCS Directory Structure

```
gs://liquidneuralnets-experiments/
  datasets/
    har/                    # Raw dataset files
    smnist/
    gesture/
    ...
  results-pytorch/
    <run-name>/
      <model>/
        <experiment>/
          seed<N>/
            run_metadata.json        # exit_code, timestamps, train_args
            training_log.txt         # stdout/stderr from training
            results/
              <experiment>/
                <model>_<size>/
                  checkpoint_best.pt
                  checkpoint_epoch0.pt
                  <model>_<size>.csv  # results CSV
```

### Launch Scripts

**`launch_run.sh <run_name> <experiment> <model> <seed> [extra hydra args...]`**

Launches a single training VM:
1. Constructs VM name: `{run_name}-{model}-{experiment}-seed{seed}` (underscores replaced with hyphens)
2. Checks if VM already exists (skip if so)
3. Checks concurrency limit
4. Loads experiment-specific overrides from `experiments/{experiment}.env`
5. Creates VM with metadata tags and startup script
6. Uses spot/preemptible instances if configured

**`launch_all.sh <run_name> [--seeds N] [--models "m1 m2 ..."] [--dry-run]`**

Launches the full experiment matrix:
1. Builds job list: all `(experiment, model, seed)` combinations
2. Phase 1: Burst-launch in batches of 8 until concurrency limit
3. Phase 2: Poll for free VM slots, launch one at a time as capacity opens
4. Respects `MAX_CONCURRENT_VMS` from config.env

### Monitoring (`cloud/monitor.sh`)

```bash
./monitor.sh <run_name> [experiment] [model]
```

Displays:
- List of currently running VMs
- Completion matrix: model (rows) x experiment (columns)
- Legend: `5` = all seeds done, `3+run` = 3 done + running, `.` = not started

### Results Collection (`cloud/collect_results.py`)

```bash
python collect_results.py <run_name> [--output results.csv]
```

- Downloads all result CSVs from GCS
- Computes mean +/- std across seeds
- Outputs formatted table and optional aggregated CSV

### VM Image (`cloud/build_image.sh`)

Builds a reusable GCP image with pre-installed dependencies:
1. Creates temporary `e2-standard-4` VM
2. SSH retry loop (20 attempts, 15s apart) for reliable connection
3. Installs: Python 3, pip, venv, PyTorch (CPU), Hydra, NumPy, SciPy, pandas, h5py, tqdm, torchvision
4. Saves as image in the `srnn-pytorch` family
5. Deletes temporary VM

---

## 10. Tensor Shape Reference

### Cell State Shapes

| Cell | `state_size` | State Tensor Shape |
|------|-------------|-------------------|
| LSTMCellWrapper | `num_units * 2` | `(batch, num_units * 2)` -- `[h, c]` concatenated |
| LTCCell | `num_units` | `(batch, num_units)` -- membrane voltage |
| CTRNNCell | `num_units` | `(batch, num_units)` -- hidden activation |
| NODECell | `num_units` | `(batch, num_units)` -- hidden activation |
| CTGRUCell | `num_units * M` | `(batch, num_units * M)` -- M copies per neuron |
| SRNNCell | `n_E*n_a_E + n_I*n_a_I + n_E*n_b_E + n_I*n_b_I + N` | `(batch, state_size)` -- packed flat |

### SRNN State Components (unpacked)

For `num_units=32` (n_E=16, n_I=16), `n_a_E=3, n_a_I=3, n_b_E=1, n_b_I=1`:

| Component | Shape | Size | Description |
|-----------|-------|------|-------------|
| `a_E` | `(batch, 16, 3)` | 48 | SFA variables, excitatory |
| `a_I` | `(batch, 16, 3)` | 48 | SFA variables, inhibitory |
| `b_E` | `(batch, 16, 1)` | 16 | STD variables, excitatory |
| `b_I` | `(batch, 16, 1)` | 16 | STD variables, inhibitory |
| `x` | `(batch, 32)` | 32 | Membrane potential |
| **Total** | | **160** | `state_size` |

### BatchedSRNNCell Shapes

For K ablation variants:

| Tensor | Shape | Description |
|--------|-------|-------------|
| Input | `(K, batch, input_size)` | Tiled input (broadcast from single batch) |
| State | `(K, batch, max_state_dim)` | Padded to max across variants |
| W_raw | `(K, N, N)` | Stacked recurrent weights |
| W_in | `(K, N, input_size)` | Stacked input weights |
| Output | `(K, batch, N)` | Per-variant hidden output |

### Training Data Flow Shapes

Example: HAR task (input_size=561, seq_len=16, batch_size=128)

| Stage | x Shape | y Shape |
|-------|---------|---------|
| Raw data | `(N, 16, 561)` | `(N, 16)` |
| After time stretch (factor ~1.1) | `(128, 17, 561)` | `(128, 17)` |
| After palindrome loop (5 loops) | `(128, 170, 561)` | `(128, 170)` |
| After random window | `(128, ~136, 561)` | `(128, ~136)` |
| To device (float32) | `(128, ~136, 561)` | `(128,)` -- extracted at readout_idx |
| Model output | `(128, 6)` | -- |

### I/O Mask Shapes

For `num_units=32`:

| Mask | Shape | ~Active | Applied Where |
|------|-------|---------|--------------|
| `W_in_mask` | `(32,)` | ~8 (25%) | Multiplied with input weight rows in cell |
| `input_mask` | `(32,)` | ~8 (25%) | (Same partition, used by SequenceModel) |
| `output_mask` | `(32,)` | ~8 (25%) | Selects neurons for readout head input |
| Readout head | `Linear(8, output_size)` | -- | Reduced from 32 to 8 inputs |
