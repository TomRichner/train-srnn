# Known Issues

Tracked limitations and bugs. When working in these areas, either avoid the pitfall or plan around it explicitly.

---

## 1. `per_neuron=False` is not enforced in `BatchedSRNNCell` — **FIXED**

**Status.** Fixed via a scalar+vector+mask decomposition in `BatchedSRNNCell` (commit following this doc update). Historical description kept below for context.

**How it's fixed.** Every previously per-neuron-shaped parameter is split into two learnable components plus a per-variant gradient mask:

- `<name>_vec` — same per-neuron shape as before `(K, N)` / `(K, n_E, max_n_a_E)` etc.
- `<name>_gain` (softplus'd params) or `<name>_scalar` (direct params) — shape `(K,)`.
- `_install_vec_mask(...)` in `BatchedSRNNCell.__init__` registers a buffer `_<name>_vec_mask` of shape `(K, 1, ...)` derived from each preset's `per_neuron` flag, and installs a `Tensor.register_hook` on `<name>_vec` that multiplies incoming gradients by the mask (same idiom as the existing echo `W_raw` freeze).

Effective values are computed in helper methods:

- Softplus'd positive params (log-space): `tau = exp(log_gain) · softplus(vec)` — a multiplicative log-gain outside `softplus`.
  - Helpers: `_tau_d()`, `_tau_a_E()`, `_tau_a_I()`, `_tau_b_rec_E()`, `_tau_b_rel_E()`, `_tau_b_rec_I()`, `_tau_b_rel_I()`, `_c_E()`, `_c_I()`.
- Direct params (`a_0`, `c_0_E`, `c_0_I`): `effective = vec + scalar` — a shared additive shift across neurons.
  - Helpers: `_a_0()`, `_c_0_E()`, `_c_0_I()`.

Semantics per preset:

- `per_neuron=False` → `_vec` gradient masked to zero → vec stays frozen at its (identical-across-neurons) init; only the shared `_gain`/`_scalar` moves. **True cross-neuron linking.**
- `per_neuron=True`  → both components train; the scalar captures shared direction, vec captures per-neuron deviations. Strictly richer than the original per_neuron=True (which only had `(N,)` per-neuron parameters).

Init values preserve exact pre-fix behavior: `log_gain=0` → multiplicative identity, `scalar=0` → additive identity. An untrained model produces bit-identical outputs to pre-fix for any preset.

**New W gains (bonus).** Alongside the fix, two per-variant scalar gains are added — `W_raw_gain` and `W_in_gain`, shape `(K,)`, init 1.0. They multiply the full `W_eff` and the full `W_in` matrix respectively. Pooled optimization directions (classical Echo-State-Network spectral-radius / input-gain knobs, made differentiable). Always trainable for all variants including `echo=True` — the reservoir *structure* is frozen by the existing `W_raw` hook, but its overall *gain* is learnable. Equivalent `W_raw_gain`/`W_in_gain` scalars are also added to single-cell `SRNNCell`.

**Implications.**
- Batched `per_neuron=False` variants now genuinely share their 12 linkable parameters across neurons; ablations between `srnn` (shared) and `srnn-per-neuron` (per-neuron) are now meaningful.
- Prior batched-ablation results where non-per-neuron variants were included should be re-interpreted: those "srnn" baselines were effectively `srnn-per-neuron`.
- The `factory.build_batched_model` API is unchanged; `build_model` (single-cell) is unaffected aside from adding the two new W gains.

---

### Historical description (kept for context)

Every variant trained through `BatchedSRNNCell` effectively ran as per_neuron=True, regardless of each variant's `SRNNConfig.per_neuron` flag. The `per_neuron=False` presets all silently learned per-neuron time constants.

The root cause was that in `BatchedSRNNCell.__init__`, all time-constant parameters were stored at per-neuron shape unconditionally:

- `log_tau_d`: `(K, N)`
- `log_tau_a_E`, `log_c_E`, `c_0_E`: `(K, n_E, max_n_a_E)`
- `log_tau_a_I`, `log_c_I`, `c_0_I`: `(K, n_I, max_n_a_I)`
- `log_tau_b_rec_E`, `log_tau_b_rel_E`: `(K, n_E)`
- `log_tau_b_rec_I`, `log_tau_b_rel_I`: `(K, n_I)`

Plus `a_0` at `(K, N)` which was *always* per-neuron in both cells (not just batched).

At init the `torch.full(...)` calls set all N entries identically, but the optimizer pushed them apart from the first gradient step. The single-cell `SRNNCell` avoided this by using shape `(1,)` for `per_neuron=False`, which structurally forced all neurons to share the parameter.

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

---

## 4. `TrainableIC` receives no gradient when `bptt_start_idx > 0`

**Summary.** The project has three independent "warmup"-adjacent mechanisms that interact in a non-obvious way. The `TrainableIC` parameter is initialised by an unforced burn-in at train start, but as soon as there is any forward-only prefix inside the per-batch window (i.e. `bptt_start_idx > 0`, which is the standard setting), the IC parameter receives **zero gradient** and is effectively frozen at its burn-in value for the rest of training.

### The four concepts and how they interact

All four operate at different scopes. Keep them distinct when reasoning about dynamics:

| Mechanism | Scope | What it does | Governed by |
|---|---|---|---|
| **`compute_burn_in`** | once, at train start | runs the cell with zero input for `burn_in_seconds / dt_per_step` steps; copies the final state into `TrainableIC.ic` | `cfg.burn_in` (default 10.0 s), cell's `dt_per_step` |
| **`TrainableIC`** | per batch, at `t=0` | every forward pass starts from `self.ic` (expanded to batch) instead of zeros | `SequenceModel.__init__(trainable_ic=True)` (default on) |
| **Palindrome loop** | per batch, data-side | repeats the image fwd/bwd/fwd/bwd so the `window_len`-long sample has real content everywhere, not padding | `window_len`, `seq_len` (auto-sized inside `wrap_train_batch`) |
| **Forward-only warmup (BPTT truncation)** | per batch, compute-graph-side | first `window_len - bptt_len` timesteps of the unroll run under `torch.no_grad()` so dynamics settle without a gradient graph | `cfg.window_len`, `cfg.bptt_len` |
| **Random window offset + random readout idx** | per batch, data-side | `wrap_train_batch` picks a random offset into the palindrome-looped sequence and a random readout timestep within the last loop, adding stochasticity | `rng` in `wrap_train_batch` |

### Why the IC freezes with forward-only warmup

In `SequenceModel.forward`:

```python
state = self.ic(batch_size)                    # tensor is linked to self.ic via autograd
for t in range(seq_len):
    inp = x[:, t, :]
    if bptt_start_idx is not None and t < bptt_start_idx:
        with torch.no_grad():                  # ← breaks the graph
            output, state = self.cell(inp, state)
    else:
        output, state = self.cell(inp, state)
```

Inside `torch.no_grad()`, the `state` tensor returned by the cell has `requires_grad=False`. The autograd link between that new `state` and `self.ic` is severed. When the BPTT region starts at `bptt_start_idx`, the state input to the cell is a plain tensor — gradients computed on the loss cannot propagate back through it to `self.ic`. Result: `self.ic.grad` is `None` every step and the optimizer never updates it.

### Practical effect

- `TrainableIC.ic` is *de facto* a frozen buffer set by `compute_burn_in`, not a learned parameter, whenever `window_len > bptt_len`.
- For the current sMNIST setting (`window_len=112, bptt_len=56`), the IC is frozen at the unforced fixed point.
- This is usually fine: after 56 timesteps of driven palindrome-cycle dynamics, the state has been largely "washed out" of its dependence on the initial condition, so the IC mostly doesn't matter past the warmup. The burn-in gave it a reasonable value; that's all it needs.
- It does mean `trainable_ic=True` is behaviourally equivalent to `trainable_ic=False` + using the burn-in state directly, whenever `bptt_start_idx > 0`.

### Remedy in use

The `freeze_ic_after_burnin` config flag (default **true** in `conf/config.yaml`) calls `model.ic.ic.requires_grad_(False)` after `compute_burn_in` copies the fixed-point state into the parameter. This makes the frozen-IC behaviour explicit rather than implicit, so:

- `param_count` reporting no longer includes a trainable parameter that in fact never updates.
- Anyone reading the code or checkpoints can see at a glance that the IC is a fixed, burn-in-derived starting state, not a learned one.
- Optimizer state (Adam moments) isn't tracked for a parameter whose gradient is always `None`.

Set `freeze_ic_after_burnin=false` only when deliberately training the IC under full-window BPTT (`bptt_len == window_len`) — that is the only configuration where the flag's value matters in practice.

### Fixed IC drifts from the fixed point as training proceeds

`compute_burn_in` at train start finds the unforced fixed point of the **initial** network. As `W_raw`, time constants, thresholds, adaptation weights, etc. update, the *current* network's fixed point moves — but the frozen IC stays at the initial location. After many epochs, the IC can be far from any stable attractor of the current network.

How much this matters depends on whether the per-batch forward-only warmup (`window_len - bptt_len` steps) is long enough to let the current network's dynamics settle from the stale IC into a meaningful regime before BPTT begins. For fast-decaying LSTM/CTRNN dynamics this is usually fine. For SRNN/LTC with multi-second SFA or depression timescales and a short warmup, it can matter.

**Remedy: `burn_in_every` config flag.** Re-runs `compute_burn_in` at the start of every N epochs (skipping epoch 0, which is handled by the init burn-in). Each call overwrites `model.ic.ic.data` with the current network's unforced fixed point. This happens outside autograd, so `freeze_ic_after_burnin` is unaffected — the IC is still a non-learned buffer, just one that gets refreshed periodically.

Defaults:

```yaml
burn_in: 10.0            # seconds of simulated time per burn-in call
burn_in_every: 1         # refresh every N epochs; 0 = only at init
freeze_ic_after_burnin: true
```

Cost per refresh for SRNN (`h=0.02`, batch=1): 500 zero-input cell calls. At `burn_in_every: 1` over 50 epochs that adds a few minutes on CPU. For cheaper tracking, set `burn_in_every: 10` to align with `checkpoint_interval` and pay roughly 1/10th the cost.

### When to actually train the IC

If you want the IC parameter to learn:
- Set `bptt_len == window_len` (equivalent: `bptt_start_idx = 0`), so there is no forward-only prefix and gradients flow to the IC through every step.
- This is only advisable for short-sequence / short-window settings where the full unroll fits in memory, and when the IC genuinely influences the readout (e.g. fast-decay dynamics or short windows where the initial state still matters at readout time).
- At long `window_len` the IC signal would be numerically swamped by the recurrent dynamics anyway, so training it rarely helps.

### When this actually matters for results

- For a fair "does the model learn its IC?" ablation, compare `trainable_ic=True bptt_len=window_len` vs `trainable_ic=False bptt_len=window_len`. Comparing with truncated BPTT is not informative either way.
- Plots of parameter statistics that include `ic.*` entries should be read with the understanding that `ic` drift over training is zero under the default settings — any observed "trained IC" is just the burn-in output.

### Fix sketch if we ever want the IC to train under truncated BPTT

- Replace `with torch.no_grad():` with an explicit `state = state.detach()` wrapper that only severs the gradient once, **after** the first no-grad cell call — still loses the IC connection.
- A real fix would need either: (a) running the forward-only prefix with gradients on but using gradient checkpointing to control memory; (b) a separate gradient path where `self.ic`'s gradient is computed through a short fully-tracked prefix and then summed in; or (c) accepting that IC + truncated BPTT is incompatible and exposing this in docs + removing `trainable_ic` from the default when `bptt_len < window_len`.
