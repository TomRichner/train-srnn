# Known Issues

Tracked limitations and bugs. When working in these areas, either avoid the pitfall or plan around it explicitly.

---

## 1. `per_neuron=False` is not enforced in `BatchedSRNNCell`

**Summary.** Every variant trained through `BatchedSRNNCell` (i.e. anything using `batched_ablations=...` via `build_batched_model`) effectively runs as **per_neuron=True**, regardless of each variant's `SRNNConfig.per_neuron` flag. The `per_neuron=False` presets (`srnn`, `srnn-echo`, `srnn-no-adapt`, etc.) all silently learn per-neuron time constants in batched mode.

**Why this happens.** In the single-cell `SRNNCell` (`train_srnn/models/srnn_cell.py:194, 199, 212, 224, 236, 248, 257`), the time-constant parameters have shape `(1,)` when `per_neuron=False` and `(n_E,)` / `(n_I,)` / `(N,)` when `True`. Shape `(1,)` structurally forces all neurons to share the parameter, so "non-per-neuron" is enforced by construction.

In `BatchedSRNNCell` (same file, constructor from line 729), **all** time-constant parameters are stored at per-neuron shape unconditionally:

- `log_tau_d`: `(K, N)` (line 805)
- `log_tau_a_E`, `log_c_E`, `c_0_E`: `(K, n_E, max_n_a_E)` (lines 809–825)
- `log_tau_a_I`, `log_c_I`, `c_0_I`: `(K, n_I, max_n_a_I)`
- `log_tau_b_rec_E`, `log_tau_b_rel_E`: `(K, n_E)`
- `log_tau_b_rec_I`, `log_tau_b_rel_I`: `(K, n_I)`

There is no mechanism (no post-step tying, no mean broadcast, no frozen-shared parameter) that keeps a non-per-neuron variant's neurons tied together during training. At init the `torch.full(...)` calls set all N entries identically, but the optimizer pushes them apart from the first gradient step.

**Grepping `per_neuron` in `srnn_cell.py`** confirms the flag is referenced only inside `SRNNCell` and the preset dictionary. It never appears inside `BatchedSRNNCell`.

**What this means for mixing.** Nothing special — the behavior is the same whether:
- All variants in the batch are `per_neuron=True` (works as intended),
- All variants are `per_neuron=False` (each trains as per-neuron — wrong),
- The batch mixes the two (per-neuron ones work, non-per-neuron ones train as per-neuron).

Mixing per-neuron and non-per-neuron presets in a batch is not a new bug; the non-per-neuron ones were already running as per-neuron.

**Practical consequences.**
- Running `srnn-per-neuron` side-by-side with `srnn` in a batched ablation is redundant — both will end up training as per-neuron.
- A rigorous `per_neuron=True` vs `per_neuron=False` comparison **must** use two separate single-cell training runs (`build_model`, not `build_batched_model`).
- Results from prior batched ablation runs that included non-per-neuron variants should be interpreted with this in mind: the "srnn" etc. baselines in those runs are effectively `srnn-per-neuron`.

**If/when a fix is attempted,** options include:
- Register a post-step hook per variant that averages the per-neuron values and broadcasts the mean back for `per_neuron=False` variants.
- Add a buffer that masks gradients for non-per-neuron variants so only a shared component updates.
- Accept the current behavior as "batched ablations are per-neuron by nature" and update docs/presets to reflect it (drop non-per-neuron presets from the batched catalog).

---

## 2. `echo=True` only freezes the recurrent weights, not the other intrinsic parameters

**Summary.** The original design intent of the `echo` / reservoir-mode presets (`srnn-echo`, `srnn-e-only-echo`) was "freeze everything except `W_in` and the readout" — i.e. a proper reservoir where the recurrent matrix, all time constants, adaptation/depression parameters, and the firing threshold are fixed, and only the input projection and the output head learn. What's actually implemented freezes **only** the recurrent matrix `W_raw`. Every other intrinsic parameter still trains.

**Where the limitation lives.**

`SRNNCell` (`train_srnn/models/srnn_cell.py:179–180`):
```python
if config.echo:
    self.W_raw.requires_grad_(False)
```

`BatchedSRNNCell` (`train_srnn/models/srnn_cell.py:887–890`):
```python
if any(c.echo for c in configs):
    echo_grad_mask = 1.0 - echo_flags  # (K, 1, 1): 0 for echo, 1 for non-echo
    self.register_buffer("_echo_grad_mask", echo_grad_mask)
    self.W_raw.register_hook(lambda grad: grad * self._echo_grad_mask)
```

Both touch `W_raw` only. No `requires_grad_(False)` calls and no grad hooks are applied to the other learnable parameters.

**What's supposed to be frozen (per intent) but isn't.** All of:
- `log_tau_d` (dendritic time constant)
- `log_tau_global` (global timescale multiplier)
- `log_tau_a_E`, `log_tau_a_I`, `log_tau_a_E_lo`/`hi` (SFA timescales)
- `log_c_E`, `log_c_I`, `c_0_E`, `c_0_I` (SFA adaptation weights and offsets)
- `log_tau_b_rec_E`, `log_tau_b_rel_E`, `log_tau_b_rec_I`, `log_tau_b_rel_I` (STD timescales)
- `a_0` (firing threshold)

**What is frozen correctly.** `W_raw` only. The echo grad mask in batched mode is per-variant and applies cleanly (`echo_flags: (K, 1, 1)` broadcasting against `W_raw: (K, N, N)` gradient), so mixing echo and non-echo variants in one batched run does not cross-contaminate `W_raw` training.

**Practical consequences.**
- Results labeled "echo" in this codebase correspond to "frozen recurrent matrix, trainable intrinsic dynamics (taus, adaptation, depression, threshold) and trainable input+readout" — a hybrid, not a classical reservoir.
- Reservoir-vs-trained comparisons are weaker than intended; the echo condition has substantial learned capacity beyond just the input/output weights.
- When citing against ESN / reservoir-computing literature, do not label current results as "reservoir" or "ESN" without qualification.

**Fix sketch for later.**
- **Single cell**: after parameter creation, when `config.echo` is true, also call `.requires_grad_(False)` on `log_tau_d`, `log_tau_global`, `log_tau_a_E`, `log_tau_a_I`, `log_tau_a_E_lo`, `log_tau_a_E_hi`, `log_c_E`, `log_c_I`, `c_0_E`, `c_0_I`, `log_tau_b_rec_E`, `log_tau_b_rel_E`, `log_tau_b_rec_I`, `log_tau_b_rel_I`, and `a_0`.
- **Batched cell**: extend the grad-hook pattern — register a `grad * _echo_grad_mask_broadcasted` hook on each of the above parameters, with the mask reshaped to broadcast against the respective `(K, ...)` parameter shape.
- Readout head lives in `SequenceModel` and is already per-variant, so it needs no change.
- Fix interacts with §1: even after applying the above freezes, non-per-neuron semantics for `per_neuron=False` variants would still not hold — but for echo variants that point becomes moot because all the per-neuron-shaped params are frozen at init values anyway.

---

## 3. `smnist` task is row-wise (28×28), not canonical pixel-wise sMNIST (784×1)

**Summary.** `conf/task/smnist.yaml` defines sMNIST with `seq_len: 28, input_size: 28` — the model sees each MNIST image as 28 timesteps of 28-dim row vectors, scanning top-to-bottom. This is the formulation used in the 2021 LTC paper (Hasani et al.) and some LNN follow-ups, but it is **not** the canonical "Sequential MNIST" benchmark from the RNN long-range memory literature.

**Canonical sMNIST** (Le, Jaitly, Hinton 2015 IRNN paper, subsequently used by uRNN, IndRNN, LMU, S4, Mamba, etc.) serializes the image into one pixel per timestep: `seq_len: 784, input_size: 1`. This is a genuinely hard long-range memory task; LSTMs ~98.9%, modern SSMs ~99.6%.

**Row-wise (current)** is a 28-step problem. LSTMs hit >99% easily and long-range memory is not meaningfully tested. Continuous-time models (LTC/SRNN) also have little opportunity to demonstrate a memory advantage at 28 steps.

**Why keep it.** Intentionally retained to enable direct comparison against the numbers reported in the 2021 LTC paper and related LNN work, which used the same row-wise setup.

**Planned addition.** A separate `serial_smnist` task with `seq_len: 784, input_size: 1` will be added later so both formulations can be compared. This will require:
- A new loader (or a flag on the existing one) that flattens `(28, 28)` → `(784, 1)` instead of treating rows as timesteps.
- A new task YAML (`conf/task/serial_smnist.yaml`).
- Awareness that 784-step BPTT exposes the no-gradient-clipping issue (see §3 if added) and is much more expensive per epoch.

**Practical consequences.**
- Do not cite results on the current `smnist` task as "sequential MNIST" without qualification — specify row-wise.
- When benchmarking against external SSM/long-RNN papers, use `serial_smnist` (once added), not the current `smnist`.
