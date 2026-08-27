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

- `isp_tau_d`: `(K, N)`
- `isp_tau_a_E`, `isp_c_E`, `c_0_E`: `(K, n_E, max_n_a_E)`
- `isp_tau_a_I`, `isp_c_I`, `c_0_I`: `(K, n_I, max_n_a_I)`
- `isp_tau_b_rec_E`, `isp_tau_b_rel_E`: `(K, n_E)`
- `isp_tau_b_rec_I`, `isp_tau_b_rel_I`: `(K, n_I)`

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
- `isp_tau_d` (dendritic time constant)
- `isp_tau_global` (global timescale multiplier)
- `isp_tau_a_E`, `isp_tau_a_I`, `isp_tau_a_E_lo`/`hi` (SFA timescales)
- `isp_c_E`, `isp_c_I`, `c_0_E`, `c_0_I` (SFA adaptation weights and offsets)
- `isp_tau_b_rec_E`, `isp_tau_b_rel_E`, `isp_tau_b_rec_I`, `isp_tau_b_rel_I` (STD timescales)
- `a_0` (firing threshold)

**What is frozen correctly.** `W_raw` only. The echo grad mask in batched mode is per-variant and applies cleanly (`echo_flags: (K, 1, 1)` broadcasting against `W_raw: (K, N, N)` gradient), so mixing echo and non-echo variants in one batched run does not cross-contaminate `W_raw` training.

**Practical consequences.**
- Results labeled "echo" in this codebase correspond to "frozen recurrent matrix, trainable intrinsic dynamics (taus, adaptation, depression, threshold) and trainable input+readout" — a hybrid, not a classical reservoir.
- Reservoir-vs-trained comparisons are weaker than intended; the echo condition has substantial learned capacity beyond just the input/output weights.
- When citing against ESN / reservoir-computing literature, do not label current results as "reservoir" or "ESN" without qualification.

**Fix sketch for later.**
- **Single cell**: after parameter creation, when `config.echo` is true, also call `.requires_grad_(False)` on `isp_tau_d`, `isp_tau_global`, `isp_tau_a_E`, `isp_tau_a_I`, `isp_tau_a_E_lo`, `isp_tau_a_E_hi`, `isp_c_E`, `isp_c_I`, `c_0_E`, `c_0_I`, `isp_tau_b_rec_E`, `isp_tau_b_rel_E`, `isp_tau_b_rec_I`, `isp_tau_b_rel_I`, and `a_0`.
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

---

## 5. `W_in` initialization is not fan-in scaled

**Summary.** `W_in` is initialized with a fixed σ = 0.1 regardless of `input_size`, so per-neuron input-drive variance scales linearly with the number of input channels. This produces a 3× spread in init drive magnitude across the supported tasks — purely a function of `task.input_size`, not anything the user controls. The trainable `W_in_gain` scalar (init 1.0) eventually compensates, but training starts from a dimensionality-dependent operating point.

**Where the limitation lives.**

`BatchedSRNNCell.__init__` (`train_srnn/models/srnn_cell.py:818`):
```python
self.W_in = nn.Parameter(torch.randn(self.K, N, input_size) * 0.1)
```

The single-cell `SRNNCell` has the analogous `randn * 0.1` init (search `W_in` in `srnn_cell.py`). Neither path applies a `1/√input_size` factor.

**Per-neuron drive variance at init.** With z-scored inputs (e.g. SEEG via `load_seeg`'s per-channel z-score on train stats, `datasets.py:673–678`), each input channel is ≈ N(0, 1). The pre-mask drive at neuron i is `Σⱼ W_in[i,j]·inputs[j]`, variance `input_size · 0.01`:

| Task            | `input_size` | drive σ |
|-----------------|-------------:|--------:|
| seeg            | 89           | 0.94    |
| smnist (rowwise)| 28           | 0.53    |
| cheetah         | 17           | 0.41    |
| HAR             | 9            | 0.30    |
| occupancy       | 5            | 0.22    |
| serial_smnist (planned) | 1    | 0.10    |

The masked W_in (only the ~25% input partition receives drive) doesn't change this — the mask is applied at forward time, not init.

**`W_out` is fine.** The readout uses `nn.init.kaiming_uniform_(... a=math.sqrt(5))` (`sequence_model.py:130`), which is exactly `nn.Linear`'s default and gives variance `1/(3·fan_in)` where `fan_in = effective_output_size` (~N/4). Per-output-channel logit variance at init is ≈ `E[r²]/3`, independent of `N` and `output_size`. No fix needed.

**Practical consequences.**
- Cross-task ablations are not on equal footing at init — a wide-input task (seeg) starts saturated, a narrow-input task (HAR, occupancy) starts under-driven, before training has done anything.
- The `W_in_gain` trainable scalar partially absorbs the mismatch (training history typically shows it moves several %), but the early-epoch loss landscape differs by task in ways orthogonal to the model's actual capacity.
- A `serial_smnist` task with `input_size = 1` (planned, see §3) would land at drive σ = 0.1 — likely too small to see input over recurrent activity at init.

**Fix sketch.**
- Single cell and batched cell: change init to `randn(...) / math.sqrt(input_size)` so per-neuron drive σ ≈ 1.0 at init regardless of task. Equivalent to the standard `nn.Linear` Kaiming default applied to the input projection.
- Existing checkpoints replay correctly (state dict load is unaffected); only fresh runs would converge differently.
- `W_in_gain` semantics are unchanged — it remains a learnable post-multiplier; it would just start from a sensible scale.
- Optionally, since `W_in_mask` zeros ~75% of the matrix at every forward, the *effective* fan-in is the size of the input partition (~N/4 entries per row when computing the drive). If that's the better target, scale by `1/√n_input_neurons` instead — but `1/√input_size` is the conventional choice and matches what `nn.Linear` would do.

**Why not fix now.** All current SEEG/HAR/etc. results were obtained under the existing init. Switching the W_in scale changes the init operating point and would invalidate cross-run comparisons until everything is re-run. Worth doing before the next batch of cross-task experiments, not in the middle of an existing series.

---

## 6. `_install_vec_mask` gradient hooks don't fire under `torch.utils.checkpoint(use_reentrant=False)`

`BatchedSRNNCell.__init__:_install_vec_mask` (`srnn_cell.py:978-985`) registers `Tensor.register_hook` callbacks on the `*_vec` parameters that multiply incoming gradients by a `(K, ...)` per-variant mask. The mask is zero for non-per-neuron variants, freezing those parameters at init.

**The bug.** `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)` does not reliably trigger `Tensor.register_hook` callbacks on parameters used inside the checkpointed segment during the re-forward backward pass. As a result, `*_vec` parameters of non-per-neuron variants accumulate non-zero `.grad` and drift away from their init during training when checkpointing is enabled.

**Reproduction.** Run a K-batched training step with at least one non-per-neuron variant (e.g. `srnn-no-adapt`) under `grad_checkpoint=True`. Compare `cell.a_0_vec.grad` to a non-checkpointed run on the same inputs — they will differ on the non-per-neuron K slice. The closed-loop gradient-equivalence tests in `scripts/test_closed_loop_grad_checkpoint.py` exercise the no-bug regime (all-per-neuron variants).

**Affects.** Both open-loop and closed-loop forward paths under `grad_checkpoint=True`. Latent in cl250 production runs because cl250 uses all-per-neuron variants (mask=ones, hook is a no-op). Would manifest in any future run that mixes per-neuron and non-per-neuron variants under checkpointing.

**Fix sketch.** Replace the gradient-hook masking with a forward-time multiplication: compute effective values as `param * mask` inside the per-helper accessors (`_a_0()`, `_tau_d()`, `_c_E()`, etc. in `BatchedSRNNCell`). Mathematically equivalent for forward output (the `_vec` init is identical-across-neurons + scalar shifts produce per-K offsets; with mask=0 the vec contribution is zero, matching the no-grad-drift behavior of the hook). Backward gradient through the multiplication naturally produces zero on `*_vec` for masked variants, with no hook needed. Removes the latent checkpoint-incompatibility entirely.

**Why not fix now.** Out of scope for the closed-loop checkpoint enablement work. Tracked here so the next time `_install_vec_mask` is touched, the fix lands cleanly.

---

## 7. `grad_checkpoint=True` + `amp=bf16` + closed-loop fails with saved-tensor metadata mismatch — **FIXED**

**Status.** Fixed in commit `e12b26f` by wrapping `_run_segment` and `_cl_run_segment` bodies in `torch.autocast(..., cache_enabled=False)` when autocast is active. See `SequenceModel._no_autocast_cache_ctx` in `train_srnn/models/sequence_model.py`.

**Root cause** (confirmed via `torch.utils.checkpoint(... debug=True)` op trace on CUDA bf16): PyTorch's autocast cache stores fp32→bf16 casts of weight tensors so repeated uses of the same weight don't re-cast. The original forward populates this cache as it runs. The recompute pass starts with an empty cache and re-executes the casts, producing extra `aten._to_copy.default(..., dtype=torch.bfloat16)` saved tensors. The saved-tensor count then diverges between forward and recompute, triggering `CheckpointError` from the metadata sanity check at op 91 of the trace (the first divergence point).

**Fix.** Disable the autocast cache for the duration of each segment's forward by wrapping it in `torch.autocast(device_type=..., dtype=..., cache_enabled=False)`. With cache disabled, both forward and recompute re-cast every weight use, producing identical saved-tensor counts. Cost: a small extra cast per weight use within a segment — negligible vs. the matmul itself.

**Historical description (kept for context):**

`torch.utils.checkpoint.checkpoint(..., use_reentrant=False)` does not reliably re-apply the ambient `torch.autocast` context during recomputation. Tensors saved during the original forward (in bf16, because autocast was active) don't match the recomputed forward (running in fp32 because the autocast context isn't propagated). The non-reentrant checkpoint's saved-tensor metadata sanity check raises `CheckpointError: tensor saved during forward is now a different size or dtype during recomputation`.

**Reproduction.** Train with `closed_loop.enabled=true grad_checkpoint=true amp=bf16` on a CUDA device. Crashes early in epoch 0 with metadata mismatches like `saved {shape: (K, B, N), dtype: bfloat16}` vs `recomputed {shape: (K, N), dtype: float32}`. CPU runs (`amp=off`) and bf16-only runs without checkpointing both work fine.

**Affects.** Closed-loop + checkpoint paths under bf16 autocast. The open-loop checkpoint path may also have this issue but cl250 didn't trigger it (or triggered it silently — the open-loop path doesn't have a gradient-equivalence test to catch it). The closed-loop path triggers it because the per-step `_readout_one` includes ops (output_mask boolean indexing, einsum + bias add + skip residual) whose dtypes shift more visibly under autocast.

**Workaround.** Run closed-loop + checkpoint with `amp=off`. At `size=150 bs=12 bptt_chunk_len=125` on L4 this fits comfortably; at production scale (`size=300 bs=24 bptt=2500`) the memory hit may force smaller batch_size to compensate.

**Investigation history.** Tried the obvious fix in commit `213fd31` (subsequently reverted): pass `context_fn` to `torch.utils.checkpoint.checkpoint(...)` that re-applies `torch.autocast(...)` for both forward and recompute. **Did not fix the CUDA failure.** A re-run with the patch in place produced the same metadata-mismatch error, with positions of the mismatches indicating that recompute saves a *different count* of tensors than forward (the displayed error positions are misaligned in a way consistent with one saved tensor extra/missing during recompute), not just different dtypes.

CPU bf16 autocast tests added in `scripts/test_closed_loop_grad_checkpoint.py:test_grad_equivalence_under_cpu_autocast_bf16_*` pass both with and without the `context_fn` patch — PyTorch handles CPU autocast in `torch.utils.checkpoint` internals automatically, so CPU is not a useful regression detector for the CUDA failure. Those tests are kept for general regression coverage.

**Hypotheses still open** (any may be the root cause):
1. CUDA-specific cache state in autocast (op cache for matmul cast) differs between forward and recompute, producing a different number of saved tensors.
2. Boolean indexing `out_t[..., self.output_mask.bool()]` in `_readout_one` produces autograd-graph differences under bf16 autocast that don't appear under fp32 or under CPU bf16.
3. The K-batched `expand` + `bmm` interaction in `_batched_input_drive` saves cached intermediates whose dtype depends on autocast state in a way that drifts between forward and recompute.

**Workaround for now.** Run closed-loop + checkpoint with `amp=off`. At `size=150 bs=12 bptt_chunk_len=125` on L4 this fits comfortably. At production scale (`size=300 bs=24 bptt=2500`) the memory hit may force smaller batch_size; can also drop `bptt_chunk_len` further (e.g. 64 or 32) which directly reduces per-chunk activation memory without needing checkpointing.

**Real fix probably requires.** Reproducing locally with CUDA, running with `torch.utils.checkpoint.set_checkpoint_debug_enabled(True)` to get the per-op trace of what's saved, then narrowing to the offending op. May also require filing a PyTorch issue if the bug is upstream. Out of scope for the closed-loop training work; tracked here for the next person who hits the symptom.

---

## 8. `torch.compile(mode="reduce-overhead")` and `mode="max-autotune")` are unsupported in continuous training

**Summary.** Both modes attempt to wrap each compiled-cell invocation in a CUDA graph (via `cudagraph_trees`). On the BPTT-over-Python-loop training pattern this codebase uses (T cell calls inside `_forward_chunk_pure_tf` / `_forward_chunk_closed_loop` followed by one `loss.backward()`), this is **structurally incompatible** with PyTorch's current `cudagraph_trees` implementation. Default mode (kernel fusion / Inductor codegen, no CUDA graphs) works; only the graph-capture modes fail.

**Symptom.** Run `compile=true compile_cell=true compile_mode=reduce-overhead` (or `max-autotune`) and either:
- Forward fails with `static input data pointer changed` / aliasing errors (no `cudagraph_mark_step_begin()` between iterations), or
- Backward fails with `RuntimeError: Error: accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run` (with `cudagraph_mark_step_begin()` between iterations — the call that fixes the forward).

The `train_srnn/utils/cell_loop.py:mark_cudagraph_step()` calls in our loop bodies ARE called — they are required for the forward to not alias — but they invalidate intermediates that the deferred backward still needs. There is no setting of "yes invalidate forward outputs / no don't invalidate backward intermediates" available in the API.

**Root cause** (PyTorch limitation, not ours). `cudagraph_trees`' allocator can recycle a graph's output buffers between invocations OR keep saved-for-backward intermediates alive across invocations, but not both for the same workload. PyTorch docs: *"Memory for activations that are saved in the forward cannot be reclaimed in the backward."* The trees machinery's training heuristic assumes forward + backward + step *per* invocation, not T forwards followed by one backward.

**Status upstream.** Open issues with no maintainer fix as of PyTorch 2.11:
- pytorch/pytorch [#148439](https://github.com/pytorch/pytorch/issues/148439) — accessing overwritten output
- pytorch/pytorch [#158551](https://github.com/pytorch/pytorch/issues/158551) — clone + mark_step_begin both insufficient
- pytorch/pytorch [#169545](https://github.com/pytorch/pytorch/issues/169545) — compile + cudagraph + gradient accumulation fails

**What works today.**
- `compile=false` — eager mode, no compile.
- `compile=true compile_mode=null` (default mode) — Inductor kernel fusion + Triton codegen, no CUDA graphs. Confirmed working at K=2/30 (~11s/epoch steady-state on L4) and K=6/300 (production).

**What we did anyway.** The cell_loop refactor (`train_srnn/utils/cell_loop.py`, commit `065dfb6`) replaced `outputs.append(out) ... torch.stack(outputs, dim=-2)` with a pre-allocated `torch.empty(..., T, F)` buffer and slice-assign `out_seq[..., t, :] = out`, plus `mark_cudagraph_step()` at the top of each loop body. This is the pattern PyTorch's docs recommend for CUDA-graph-friendly loops — we are aligned with the API contract; the bug is below us. If/when PyTorch fixes the underlying limitation, our code is already in the correct shape.

**If we ever need CUDA graphs in training** — known-working alternatives, none implemented here:
1. **Manual `torch.cuda.graph()` capture wrapped in a custom `torch.autograd.Function`** — NVIDIA's RNN-T pattern (`docs.nvidia.com/dl-cuda-graph/examples/rnnt.html`). Captures the entire BPTT chunk's forward and backward as two graphs sharing a memory pool, exposed to autograd as one fused op. ~300–500 LOC; the only documented working pattern.
2. **Larger `torch.compile` scope** — compile `_forward_chunk_pure_tf` (the whole chunk loop) instead of just `cell`. Dynamo unrolls the Python loop into one ~12,500-op graph; if the trace completes, the chunk becomes a single forward + single backward, which `cudagraph_trees` *does* handle. Compile-time may blow up at production scale (size=300, K=6); needs a 1-day spike to verify feasibility. **Cheapest first experiment if launch overhead is measured to dominate.**
3. **Fused custom Triton kernel for the SRNN cell forward** — replaces ~50 separate ops with one kernel. Eliminates launch overhead at the source, no interaction with cuda graphs. ~weeks of work; defer until cell architecture is stable.

**Decision.** Not worth pursuing until a measured wall-clock breakdown at production K=6 size=300 B=24 (or larger B) shows kernel-launch overhead is ≥ 20% of step time. See `tmp/notes/cuda_graphs_options.md` for the full analysis. The `compile_mode` knobs remain in `conf/config.yaml` for forward compatibility but should be left at `null` (default mode) until that measurement happens.

---

## 9. Per-step host syncs in the continuous trainer's per-K loss/metric path

**Summary.** The continuous trainer issues `2·K` forced host syncs **per training step** in its per-K bookkeeping. At K=6 with B=48 (15 steps/epoch on full T) that's ~180 syncs/epoch. Each sync blocks the host until prior CUDA work finishes, serializing the trainer's Python loop with the GPU's queue.

**Where.** `train_srnn/training/continuous.py`:
- `_compute_loss` (line 301): `per_k = [l.item() for l in losses]` — K calls to `Tensor.item()` per chunk step.
- `_per_k_metric` (line 315): list-comp doing `float(... .item())` per K — another K calls per chunk step.

Both are called inside the per-step inner loop of `run_continuous_training`. Each `.item()` walks down to `aten._local_scalar_dense` which forces `cudaStreamSynchronize`. The K losses are needed only for **per-epoch** logging and CSV emission, not for backward (the `loss.sum()` used for `loss.backward()` stays on GPU).

**Impact.** Bounded — at K=6 size=300 B=48 production we see GPU compute util at 91% / memory util at 100% (i.e. queue is mostly full despite the syncs), so eliminating them would buy on the order of 5–10% wall-clock, not 50%. The cost is more visible at smaller scales where the queue can drain between syncs.

**Fix sketch.** Keep per-K losses and metrics as **GPU tensors** through the epoch:
- `_compute_loss`: return `(loss_for_backward, per_k_loss_tensor)` where `per_k_loss_tensor` is `torch.stack(losses)` shape `(K,)`, no `.item()` calls.
- `_per_k_metric`: return a `(K,)` tensor of negative-MAE values, computed via vectorized abs+mean over the K axis (no Python comprehension).
- In `run_continuous_training`, accumulate `epoch_loss_sum_k` / `epoch_metric_sum_k` as in-place tensor adds (`tensor += per_k_loss_tensor`), starting from a single zero `(K,)` tensor at epoch start.
- At epoch end, do **one** `tensor.tolist()` (or per-K `.item()`) for logging — that's K syncs per epoch, not per step.

The `loss.item()` at line 525 in the non-K path is fine (one sync per step at K=None, much smaller workload).

**Why not fix now.** Production runs already saturate the GPU; fixing this is a modest cleanup, not a critical bug. Tracked here so the next time the trainer's metrics pipeline is touched, the fix lands.

---

## 10. `_effective_W` is rebuilt every cell call (250× per chunk)

**Summary.** `SRNNCell._effective_W` (line 296) and `BatchedSRNNCell._effective_W` (line 1108) materialize the full effective recurrent weight matrix from `W_raw`, `dales_sign`/`dales_signs`, `sparsity_mask`/`sparsity_masks`, `W_raw_gain`, and `softplus(...)` on **every cell call**. With `chunk_len=250` and a single optimizer step per chunk, the matrix is rebuilt 250× per step even though `W_raw` only changes once per `optimizer.step()`.

**Cost.** At K=6 N=300 (production), `W_eff` is `(6, 300, 300) = 540K` bf16 elements = 1.08 MB per materialization. Per chunk step: 250 × 1.08 MB ≈ 270 MB of avoidable scratch traffic for the matrix construction (read W_raw, mask, signs, write W_eff, all element-wise on N²). Per epoch (15 steps): ~4 GB. Plus the `softplus` + element-wise ops themselves run 250×.

**Why this exists.** When the cell was designed, the inner ode_unfolds loop genuinely needed a freshly-broadcast W_eff for each substep (different state, same params). Hoisting outside the cell wasn't trivial because `_effective_W` lives on the cell module and uses parameters owned by it. The 250× repeat across timesteps within one chunk is an artifact of the per-timestep cell-call pattern that the trainer uses.

**Fix options** (none required; tracked for opportunistic improvement):

1. **Hoist W_eff into the chunk function.** Compute `W_eff = cell._effective_W()` once at the top of `_forward_chunk_pure_tf` / `_forward_chunk_closed_loop` and pass it into the cell as an arg. Requires modifying the cell forward signature `cell(input, state) -> (output, state)` to accept an optional pre-computed W_eff, falling back to `self._effective_W()` if not provided. Mechanically clean (~30 LOC). Saves 249/250 of the rebuild traffic per chunk step.
2. **Caching with version counter.** Store `_W_eff_cached` on the cell + a version int that increments on every parameter mutation (via `register_post_accumulate_grad_hook` on `W_raw` or in optimizer wrapping). Only recompute when stale. More invasive than (1) and the stale-check itself isn't free.
3. **Ignore.** Inductor under default-mode `torch.compile` may already CSE the redundant materializations within one compiled trace, in which case the extra HBM traffic is mostly an illusion. Worth checking via `TORCH_LOGS=output_code` before doing (1).

**Worth fixing?** Probably yes via option (1) once measured savings are ≥ 5%, but verify (3) first to avoid duplicating an Inductor optimization. Same caveat as §9: the GPU is already saturated, so the absolute wall-clock win is bounded.

---

## 11. Mixed `softplus` / `exp` parameterization of positive scalars

**Summary.** `BatchedSRNNCell` uses two different positivity transforms for what are mathematically the same kind of object (a non-negative scalar that gets multiplied into the ODE). Per-axis bases and the global timescale go through `softplus`; per-class gains go through `exp`. There is no functional reason for the asymmetry — both transforms are smooth, monotone, $\mathbb{R} \to \mathbb{R}_{>0}$, and either could carry the whole parameterization. The split is conventional / aesthetic.

**Where the limitation lives.**

`BatchedSRNNCell._tau_global`, `_tau_d`, `_tau_a_E`, `_tau_a_I`,
`_tau_b_rec_E`, `_tau_b_rel_E`, `_tau_b_rec_I`, `_tau_b_rel_I`, `_c_E`,
`_c_I` (`srnn_cell.py:1135-1224`):

```python
def _tau_d(self) -> torch.Tensor:
    """(K, N) = tau_global · exp(log_tau_d_gain) · softplus(isp_tau_d_vec)"""
    gain = torch.exp(self.log_tau_d_gain).view(self.K, 1)
    return self._tau_global().unsqueeze(-1) * gain * F.softplus(self.isp_tau_d_vec)
```

i.e. `softplus` on the per-neuron base and on `isp_tau_global` (via `_tau_global`), `exp` on the per-class log-gain. Same pattern for every other timescale and for the SFA coupling `c`.

**Why the asymmetry exists.**

- The per-axis base values (e.g. $\tau_d \approx 0.1$ s, $c \approx 0.05$) are small positive numbers stored at $\sigma^{+,-1}(\text{value})$. Around these init values, softplus is approximately linear in its raw argument (slope $\sigma(\ell) \approx \text{value}$), giving Adam steps that map cleanly to additive changes in seconds.
- The per-class gain is a multiplier centered at $1$ (multiplicative identity). With `exp` and raw init $0$, the gain is exactly $1$ at training step $0$ and explores log-space symmetrically (a Δlog of ±log 2 doubles or halves the gain by equal magnitude moves). With softplus you'd init at $\sigma^{+,-1}(1) \approx 0.541$ and get asymmetric exploration around 1.

Both differences are stylistic, not load-bearing.

**Practical consequences.**

- Reading code that builds an effective parameter requires knowing which transform applies to which factor.
- Plots of *raw* parameters mix two different scales: `isp_tau_d_vec` lives in inv-softplus space (where `value = softplus(raw)`), while `log_tau_d_gain` lives in true log-space (where `value = exp(raw)`). Comparing magnitudes across these names is misleading.
- The prefix now distinguishes them: `isp_*` for inverse-softplus, `log_*_gain` for true log. Pre-rename checkpoints (param keys still spelled `log_*_vec` / `log_tau_global` etc.) will not load into the renamed model without a key-translation pass — no shim is shipped.
- Postprocess analysis (`scripts/postprocess.py:effective_taus`) and the implementation section of `docs/equations.md` (\S 5.4-5.5) have to spell out the mixed transform every time they convert raw params to effective values.

**Fix sketch (not planned).**

A unified parameterization would replace every `softplus(log_X_vec)` with `exp(log_X_vec)` (cleaner) or every `exp(log_X_gain)` with `softplus(log_X_gain)` (matches the prefix). The all-`exp` version is cleaner mathematically:

$$\tau_d = \exp\!\big(\ell_{\tau_g} + g_{\tau_d} + \ell^{\text{vec}}_{\tau_d, i}\big)$$

i.e. the three multiplicative pieces become additive in log-space, and the "log_" prefix becomes accurate everywhere. Init values would change ($\ell^{\text{vec}}_{\tau_d, i} = \log(0.1) \approx -2.30$ instead of $\sigma^{+,-1}(0.1) \approx -2.25$), and the linear-near-init behaviour of softplus would be lost — Adam steps in raw space would become *geometric* in $\tau_d$ rather than approximately linear. That's a real semantic change, not a no-op refactor; whether it speeds or slows convergence is empirical.

**Why not fix now.** The asymmetry is benign — every analysis script and the docs already account for it. Switching parameterization invalidates checkpoints (the saved `isp_tau_d_vec` values would be interpreted under the wrong transform) and changes the optimization geometry, so it would invalidate cross-run comparisons until everything is re-run. Tracked here so the next major refactor of the cell parameterization can adopt a unified transform.

---

## 12. `--skip-refresh` is all-or-nothing in `cloud/submit.sh`

**Summary.** `cloud/submit.sh`'s `--skip-refresh` flag bundles two unrelated refreshes into a single switch: (a) `git fetch` + `git checkout` to update the repo on the VM, and (b) `gcloud storage cp -r $BUCKET/datasets/$EXPERIMENT/*` to re-download the task dataset. There is no way to skip just one.

**Where the limitation lives.**

`cloud/startup_gpu.sh` (around lines 184-189):
```bash
# Step 2: Download dataset from GCS (skipped on skip-refresh — reuse on-disk copy)
if [[ "$SKIP_REFRESH" == "1" ]]; then
    echo "skip-refresh=1; reusing on-disk dataset at train_srnn/data/$EXPERIMENT"
else
    gcloud storage cp -r "$BUCKET/datasets/$EXPERIMENT/*" "train_srnn/data/$EXPERIMENT/" || true
fi
```
The same `$SKIP_REFRESH` flag also gates the git fetch/checkout block above it.

**Practical consequences.**

- When pushing a new code commit but reusing an unchanged dataset (the common case for iterative training-arg sweeps after a code change), every dispatch re-downloads the full dataset. For SEEG that's ~680 MB (b4 + b5 filtered .mat) per dispatch, costing ~30-60 s of startup time and bucket egress for no reason.
- Conversely, when re-uploading a dataset (e.g. a new filter tag) but not touching code, you'd want to skip git but force a dataset refresh — also impossible.
- The reverse cases (`gcloud storage cp` is essentially free if the local copy is already there because `gcloud storage cp` doesn't checksum-verify by default; it overwrites unconditionally) make the wasted work small but non-zero.

**Fix options.**

1. **Rsync the dataset (preferred).** Replace the unconditional `gcloud storage cp -r` with `gcloud storage rsync` (or `gsutil -m rsync`), which checksum-compares each object and only transfers what changed. Same `gcloud storage cp ... | true` failure semantics; one-line change in `cloud/startup_gpu.sh:189`. Makes the dataset block effectively free when nothing has changed, so `--skip-refresh`'s only remaining purpose is "skip git fetch to test against an older code commit while debugging" — a much narrower use case.

2. **Split into `--skip-data` and `--skip-git`.** Two independent flags; `--skip-refresh` becomes an alias that sets both. Plumbing:
   - `cloud/submit.sh`: parse the two flags, write `SKIP_DATA` / `SKIP_GIT` metadata keys.
   - `cloud/startup_gpu.sh`: gate the two blocks independently.
   - `cloud/launch_run_gpu.sh`: only relevant for first launch where neither block can be skipped; could ignore or warn.

   More invasive but gives explicit control. Mostly redundant if (1) is in place.

**Why not fix now.** Cosmetic — the unwanted dataset re-copy adds <1 minute to a multi-hour training run. Tracked here so the next time `cloud/submit.sh` or `cloud/startup_gpu.sh` is touched, the rsync swap (option 1) lands cleanly.

---

## 13. `_per_k_metric` operator precedence bug (1-D output tasks only)

**Summary.** `train_srnn/training/continuous.py:329` computes the per-variant
metric as

```python
float(-torch.mean(torch.abs(
    logits[k].squeeze(-1) if logits[k].shape[-1] == 1 else logits[k] - target
)).item())
```

The conditional binds looser than the subtraction, so when
`logits[k].shape[-1] == 1` the expression reduces to `mean(|pred|)` — the
target is dropped entirely — instead of `mean(|pred - target|)`. The
`K is None` branch at `:333` is written correctly.

**Not currently triggered.** It only fires for single-output regression in
K-batched continuous mode. seeg has C=89 and cheetah100 has C=17 (or 23 with
`include_actions`), so every task that uses the continuous trainer today has
`shape[-1] > 1` and takes the correct branch.

**Impact if triggered.** The reported `train_metric` would be the mean absolute
*prediction magnitude* rather than the error, so it would look plausible while
being unrelated to accuracy. Loss is unaffected — only the metric.

**Fix.** Parenthesise the subtraction:
`torch.abs((logits[k].squeeze(-1) if ... else logits[k]) - target)`. Left
untouched for now because no active task exercises it and changing it would
alter historical metric values for nothing.

---

## 14. Gradient clipping is global across batched-ablation variants

**Summary.** `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)`
computes **one norm over every parameter in the model** and rescales all
gradients by a single factor. In `batched_ablations` mode every trainable
tensor is `(K, ...)`-shaped, so that single factor is shared by all K variants:
one variant with a large gradient throttles the step size of all the others.

This is the only place batched and individually-trained variants are not
equivalent. Everything else is properly isolated — there are no shared
trainable tensors (verified: every parameter is `(K, ...)`), so
`∂(Σⱼ lossⱼ)/∂θₖ = ∂lossₖ/∂θₖ`, and Adam's update is elementwise.

**Where.**
- `train.py:257-259` (windowed loop)
- `train_srnn/training/continuous.py:526-527` (ring trainer)

**Scale of the problem.** The global norm grows roughly as `√K` relative to a
single variant's, so clipping engages sooner in batched mode than it would for
any variant run alone. Per-variant norms are also uneven — measured at the end
of `ring6-400e` (K=6, size=300):

```
srnn-skip                     0.513   <- 2.8x the smallest
srnn-no-adapt-no-dales-skip   0.393
srnn-no-adapt-no-dales        0.245
srnn-no-adapt                 0.225
srnn-no-dales-skip            0.217
srnn                          0.181
global                        0.779
```

So if the threshold were crossed, `srnn-skip` would be driving it while `srnn`
got throttled for a gradient that is not its own.

**Did it actually bite?** No. On `ring6-400e` the global norm was 0.444 at init
and 0.720-0.808 across ten ring positions at the end, against `grad_clip: 1.0`
— never clipped, so those results are unaffected. But the norm *grew* over
training and ended 1.24x below the threshold, so a longer run, a larger `size`,
more variants, or a higher lr could all cross it.

**Mitigation in place.** Clipping is now **disabled automatically when
`batched_ablations` is active** (K is not None), with a one-time log line.
Single-variant runs are unaffected and still honour `cfg.grad_clip`.

**Fix sketch.** Clip each variant's slice independently. Every trainable tensor
is `(K, ...)`, so the slices are already separable and no host sync is needed:

```python
def clip_grad_norm_per_variant(model, max_norm, K):
    """Per-variant analogue of clip_grad_norm_ for BatchedSRNNCell models."""
    sq, grads = None, []
    for p in model.parameters():
        if p.grad is None:
            continue
        g = p.grad
        assert g.shape[0] == K          # holds for every batched parameter
        s = (g.reshape(K, -1) ** 2).sum(1)
        sq = s if sq is None else sq + s
        grads.append(g)
    norms = sq.sqrt()
    scale = (max_norm / (norms + 1e-6)).clamp(max=1.0)
    for g in grads:
        g.mul_(scale.view(-1, *([1] * (g.dim() - 1))))
    return norms                        # also useful to log per-variant
```

Cost is one extra reduction per parameter. Returning `norms` would additionally
give per-variant gradient-norm logging, which does not exist today.

**Why not fix now.** Switching to per-variant clipping changes the optimisation
for any run where the global threshold *would* have been crossed, making new
results non-comparable with `ring6-400e` and earlier batched runs for no
present benefit — clipping never fired in those. Disabling it in batched mode
is the conservative interim: it makes batched and individual training exactly
equivalent, which is the property the ablation comparison depends on.
