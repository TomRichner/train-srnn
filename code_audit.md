# SRNN Code Audit: MATLAB `SRNNModel2` vs PyTorch `srnn-E-only`

> **Scope**: Side-by-side comparison of `SRNNModel2('n_a_E', 3, 'n_b_E', 1)` (MATLAB, well-tested)
> against `SRNNConfig(n_a_E=3, n_a_I=0, n_b_E=1, n_b_I=0)` (PyTorch `srnn_cell.py`).
>
> **Note**: The current `srnn-E-only` preset has `n_a_E=1`. This audit assumes the fix to `n_a_E=3`.
>
> **Date**: 2026-04-08

---

## Table 1: Initial Conditions

State layout: `S = [a_E(:); a_I(:); b_E(:); b_I(:); x(:)]`

| State Variable | MATLAB | PyTorch | Match? | Notes |
|---|---|---|---|---|
| **a_E** (SFA, E) | `zeros(n_E * n_a_E, 1)` | `zeros(batch, n_E * n_a_E)` | ✅ | Both start at zero |
| **a_I** (SFA, I) | *absent* (n_a_I=0) | *absent* (n_a_I=0) | ✅ | — |
| **b_E** (STD, E) | **`ones(n_E, 1)`** | **`zeros(batch, n_E)`** | ❌ | **PyTorch starts fully depressed (0), MATLAB starts fully available (1). b=0 is non-physical.** |
| **b_I** (STD, I) | *absent* (n_b_I=0) | *absent* (n_b_I=0) | ✅ | — |
| **x** (dendritic) | `0.1 * randn(n, 1)` | `zeros(batch, N)` | ⚠️ | MATLAB uses small random perturbation; PyTorch starts at exact zero. Acceptable for training (gradients break symmetry), but affects fidelity comparison. |

> [!WARNING]
> **Critical**: `b_E = 0` means all excitatory synapses start fully depressed. The STD equation `db/dt = (1−b)/τ_rec − b·r/τ_rel` will recover toward 1, but the initial transient is qualitatively wrong compared to the MATLAB model.

---

## Table 2: Parameters

All values verified by running both codebases. MATLAB values from `model.get_params()` after `build()`.
PyTorch values from inspecting `SRNNCell` with `SRNNConfig(n_a_E=3, n_a_I=0, n_b_E=1, n_b_I=0)`.

### Network Architecture

| Parameter | Symbol | MATLAB | PyTorch | Match? | Notes |
|---|---|---|---|---|---|
| Total neurons | N | 300 | 32 (default) | — | Expected: different scale for training. Not a bug. |
| E fraction | f | 0.5 | `N//2` → 0.5 | ✅ | |
| E neuron count | n_E | 150 | 16 | — | Derived from N and f |
| I neuron count | n_I | 150 | 16 | — | Derived |
| SFA timescales (E) | n_a_E | **3** | **1** (preset) | ❌ | Preset `srnn-E-only` has 1, should be 3 |
| SFA timescales (I) | n_a_I | 0 | 0 | ✅ | |
| STD channels (E) | n_b_E | 1 | 1 | ✅ | |
| STD channels (I) | n_b_I | 0 | 0 | ✅ | |
| Dale's law | — | Built into W via RMT | `dales=True`, softplus enforcement | ✅ | Different mechanism, same constraint |
| Sparsity | α | 1/3 (indegree=100, n=300) | 0.5 (random mask) | — | Different construction; expected |

### Time Constants

| Parameter | Symbol | MATLAB | PyTorch | Match? | Notes |
|---|---|---|---|---|---|
| Dendritic τ | τ_d | 0.1 s (scalar) | softplus(inv_softplus(0.1)) = 0.1 s | ✅ | PyTorch can be per-neuron if `per_neuron=True` |
| SFA τ (E) | τ_a_E | **logspace(0.25, 10, 3)** = [0.25, 1.581, 10.0] | **linspace(0.25, 10, 3)** = [0.25, 5.125, 10.0] | ❌ | **Middle timescale differs 3.2×** (1.58 vs 5.13). MATLAB uses log-spacing; PyTorch uses linear interpolation. |
| STD recovery τ | τ_rec | 1.0 s | softplus(inv_softplus(1.0)) = 1.0 s | ✅ | |
| STD release τ | τ_rel | 0.25 s | softplus(inv_softplus(0.25)) = 0.25 s | ✅ | |

### Adaptation Coupling

| Parameter | Symbol | MATLAB | PyTorch | Match? | Notes |
|---|---|---|---|---|---|
| SFA coupling (E) | c_E | 0.15/3 = **0.0500** (scalar, same for all k) | softplus(−3.0) = **0.0486** (per-timescale vector) | ⚠️ | 2.8% lower. PyTorch allows per-timescale c_E (more flexible). Total coupling: MATLAB 0.15 vs PyTorch 0.146. |
| SFA resting point (E) | c_0_E | 0.0 | 0.0 | ✅ | |
| SFA coupling (I) | c_I | 0.05 (unused, n_a_I=0) | N/A | — | |
| SFA resting point (I) | c_0_I | 0.0 (unused) | N/A | — | |

### Activation Function

| Parameter | Symbol | MATLAB | PyTorch | Match? | Notes |
|---|---|---|---|---|---|
| Function | φ | PiecewiseSigmoid | `piecewise_sigmoid` | ✅ | Same 5-piece function |
| Linear fraction | S_a / q_φ | 0.9 | 0.9 | ✅ | |
| Center / threshold | S_c / a_0 | S_c = 0.35 (inside φ) | a_0 = 0.35 (subtracted before φ, S_c=0.0) | ✅ | **Mathematically equivalent**: φ(x; S_c=0.35) = φ(x−0.35; S_c=0). Verified numerically. |

### Solver

| Parameter | Symbol | MATLAB | PyTorch | Match? | Notes |
|---|---|---|---|---|---|
| Solver type | — | ode45 (adaptive RK45) or fused (semi-implicit) | semi_implicit (fixed-step) | — | Different solver classes; expected |
| Outer step size | h | 1/fs = 1/400 = 0.0025 s | 0.04 s | ⚠️ | PyTorch 16× larger outer step |
| Sub-steps per step | — | fused_substeps = 6 | ode_unfolds = 6 | ✅ | Same sub-stepping concept |
| Effective dt | dt | 0.0025/6 = **0.000417 s** | 0.04/6 = **0.00667 s** | ⚠️ | PyTorch sub-step is **16× larger** than MATLAB fused sub-step |
| Sampling rate | f_s | 400 Hz | 1/h = 25 Hz | — | Training uses coarser time resolution |

### Input / Readout

| Parameter | Symbol | MATLAB | PyTorch | Match? | Notes |
|---|---|---|---|---|---|
| Input weights | W_in | **eye(n)** (fixed identity) | **randn(N, input_size) × 0.1** (learned) | — | Expected: PyTorch learns input mapping |
| Readout mode | — | `firing_rate` (default) | `synaptic` (b·r) | ⚠️ | Different default readout. MATLAB plots r; PyTorch outputs b·r. |

### Weight Matrix

| Parameter | Symbol | MATLAB | PyTorch | Match? | Notes |
|---|---|---|---|---|---|
| Construction | W | RMT (Harris 2023) with E/I statistics | `randn(N,N) * sqrt(2/N)` + Dale's softplus | — | Fundamentally different; expected for analysis vs training |
| Scaling | level_of_chaos | 1.0 | N/A (no explicit scaling) | — | |
| Spectral radius | R | ~2.37 (theoretical) | Emergent from init + training | — | |

---

## Table 3: Equations

### Dendritic Potential

| | MATLAB | PyTorch |
|---|---|---|
| **ODE** | $\frac{dx_i}{dt} = \frac{-x_i + u_i + \sum_j w_{ij} b_j r_j}{\tau_d}$ | Same |
| **Semi-implicit** | $x_{n+1} = \frac{x_n + \frac{dt}{\tau_d}(W(b_{n+1} \cdot r_n) + u)}{1 + \frac{dt}{\tau_d}}$ | $x_{n+1} = \frac{x_n + \frac{dt}{\tau_d}(W(b_n \cdot r_n) + u)}{1 + \frac{dt}{\tau_d}}$ |
| **Difference** | Uses **updated** b (from step 2) in x update | Uses **old** b in x update |
| **Source** | `SRNNModel2.m` L1251–1261 | `srnn_cell.py` L456–459 |

### Firing Rate

| | MATLAB | PyTorch |
|---|---|---|
| **Equation** | $r_i = \phi(x_i^{eff})$ where $x_i^{eff} = x_i - c_E \sum_k a_{ik}$ | $r_i = \phi(x_i^{eff} - a_0)$ where $x_i^{eff} = x_i - \sum_k c_{E,k} \cdot a_{ik}$ |
| **Threshold** | Built into φ via S_c = 0.35 | Explicit subtraction: a_0 = 0.35, S_c = 0.0 |
| **Equivalence** | $\phi(z; S_c{=}0.35) \equiv \phi(z - 0.35; S_c{=}0)$ | ✅ **Verified numerically** |

### Spike-Frequency Adaptation (SFA)

| | MATLAB | PyTorch |
|---|---|---|
| **ODE** | $\frac{da_{ik}}{dt} = \frac{c_{0} + r_i - a_{ik}}{\tau_{a,k}}$ | $\frac{da_{ik}}{dt} = \frac{-a_{ik} + c_{0,k} + r_i}{\tau_{a,k}}$ |
| **Match** | ✅ Same equation | |
| **Semi-implicit** | $a_{n+1} = \frac{a_n + \frac{dt}{\tau_a}(c_0 + r_n)}{1 + \frac{dt}{\tau_a}}$ | $a_{n+1} = \frac{a_n + \frac{dt}{\tau_a}(c_0 + r_n)}{1 + \frac{dt}{\tau_a}}$ |
| **Match** | ✅ Same | |
| **c_E application** | Scalar: $c_E \cdot \sum_k a_{ik}$ | Per-timescale: $\sum_k c_{E,k} \cdot a_{ik}$ |
| | Uniform c → equivalent when all c_E,k equal | More flexible for training |

### Short-Term Depression (STD)

| | MATLAB | PyTorch |
|---|---|---|
| **ODE** | $\frac{db_i}{dt} = \frac{1 - b_i}{\tau_{rec}} - \frac{b_i r_i}{\tau_{rel}}$ | Same |
| **Match** | ✅ | |
| **Semi-implicit** | $b_{n+1} = \frac{b_n + \frac{dt}{\tau_{rec}}}{1 + dt(\frac{1}{\tau_{rec}} + \frac{r_n}{\tau_{rel}})}$ | Same |
| **Match** | ✅ | |
| **Post-update clamp** | `b = max(0, min(1, b))` | None | 
| | Safety clamp to [0, 1] | No clamping |

### Semi-Implicit Update Order

| Step | MATLAB (`fused_step`) | PyTorch (`_step_semi_implicit`) |
|---|---|---|
| 1 | Compute r from current state | Compute r from current state |
| 2 | Update a (semi-implicit) | Compute b·r and W·(b·r) using **old** b |
| 3 | Update b (semi-implicit), **clamp** | Update x using **old** b·r |
| 4 | Rebuild b vector with **new** b | Update a (semi-implicit) |
| 5 | Update x using **new** b | Update b (semi-implicit) |

> [!IMPORTANT]
> MATLAB updates b *before* x and uses the freshly updated b in the x recurrence. PyTorch computes everything from the old state, then updates all variables. This is a minor but systematic numerical difference that accumulates over time.

---

## Discrepancy Summary

| # | Issue | Severity | Status | Description |
|---|---|---|---|---|
| **D1** | `n_a_E` preset value | **High** | ✅ FIXED | `srnn-E-only` preset had `n_a_E=1`, now updated to `n_a_E=3` to match MATLAB. Also fixed `srnn-e-only-echo` and `srnn-e-only-per-neuron`. |
| **D2** | `b_E` initial condition | **High** | ✅ FIXED | `init_state()` now initializes `b_E=1`, `b_I=1` (fully available synapses) in both `SRNNCell` and `BatchedSRNNCell`. |
| **D3** | `tau_a_E` spacing | **Medium** | ✅ FIXED | `_get_tau_a_E()` and `_get_tau_a_I()` now use log-space interpolation: `exp(log(lo) + (log(hi) - log(lo)) * t)`. Verified: `[0.25, 1.581, 10.0]` matches MATLAB `logspace`. |
| **D4** | Semi-implicit update order | **Low** | Open | MATLAB uses updated b in the x update; PyTorch uses old b. Same formula, different coupling. Accumulates over time but each step's error is O(dt²). |
| **D5** | `c_E` initial value | **Low** | Open | MATLAB: 0.0500, PyTorch: softplus(−3.0) = 0.0486. 2.8% relative difference. Acceptable for trainable parameters. |
| **D6** | `b_E` clamping | **Low** | ✅ FIXED | Added `b.clamp(0, 1)` after all STD updates in all solvers (semi-implicit, explicit, RK4, exponential) for both `SRNNCell` and `BatchedSRNNCell`. |
| **D7** | `x` initial condition | **Low** | ✅ FIXED | `init_state()` now initializes `x = 0.1 * randn(...)` in both `SRNNCell` and `BatchedSRNNCell`, matching MATLAB. |
| **D8** | Readout default | **Info** | N/A | MATLAB default readout is `firing_rate` (r), PyTorch default is `synaptic` (b·r). Not a bug — PyTorch uses synaptic output as features. |
| **D9** | `W_raw` initialization | **Critical** | Open | PyTorch uses Kaiming init (`randn * sqrt(2/N)`) + `softplus` for Dale's law. `softplus(~0) ≈ 0.693` maps small raw weights to large effective weights. Measured spectral radius = **9.38** vs MATLAB RMT = **0.58** (16× too large). Network is far past edge of chaos, causing saturating firing rates. Correct RMT scaling is `randn / sqrt(N)`, but softplus defeats any raw-space scaling. |

---

## Resolution Log

All fixes applied to `models/srnn_cell.py` on 2026-04-08 (commit `d37d499`).

### D1: `n_a_E` preset — FIXED
Updated `SRNN_PRESETS` dict: `srnn-E-only`, `srnn-e-only-echo`, and `srnn-e-only-per-neuron` now use `n_a_E=3` (was `1`), matching the MATLAB `SRNNModel2` default configuration.

### D2: `b_E` initial condition — FIXED
Rewrote `SRNNCell.init_state()` to construct the state vector piece-by-piece:
- `a_E`, `a_I` → zeros (no adaptation)
- `b_E`, `b_I` → **ones** (fully available synapses)
- `x` → `0.1 * randn` (see D7)

Same fix applied to `BatchedSRNNCell.init_state()` using `unpack_state` / `pack_state`.

### D3: `tau_a_E` log-spacing — FIXED
Changed `_get_tau_a_E()` and `_get_tau_a_I()` from linear interpolation to log-space interpolation:
```python
# Before (linear):  lo + (hi - lo) * t  →  [0.25, 5.125, 10.0]
# After  (log):     exp(log(lo) + (log(hi) - log(lo)) * t)  →  [0.25, 1.581, 10.0]
```
Verified: PyTorch output `[0.25, 1.5811, 10.0]` matches MATLAB `logspace(log10(0.25), log10(10), 3)` to 5 decimal places.

### D6: `b` clamping — FIXED
Added `b.clamp(0.0, 1.0)` after every STD update across all solver variants:
- `SRNNCell`: `_step_semi_implicit`, `_step_explicit`, `_step_rk4`, `_step_exponential`
- `BatchedSRNNCell`: `_batched_step_semi_implicit`, `_batched_step_explicit`, `_batched_step_rk4`, `_batched_step_exponential`

### D7: `x` initial condition — FIXED
Both `SRNNCell.init_state()` and `BatchedSRNNCell.init_state()` now initialize `x = 0.1 * randn(...)` instead of zeros, matching MATLAB `initialize_state`.

### Remaining Open Items
- **D4** (update order): Not fixed. Would require reordering the semi-implicit solver. Low priority — error is O(dt²) per step.
- **D5** (c_E value): Not fixed. 2.8% difference is acceptable since c_E is a trainable parameter.
- **D8** (readout): Not a bug — intentional design difference.
- **D9** (W init): **Critical.** `W_raw` uses Kaiming init (`randn(N,N) * sqrt(2/N)`) meant for deep-net gradient flow, not RMT-style dynamical systems. After `softplus` (Dale's law enforcement), the effective spectral radius is 9.38 — far above MATLAB's RMT value of 0.58. The `softplus` nonlinearity maps any small value to `ln(2) ≈ 0.693`, so scaling `W_raw` alone cannot fix this. Options:
  1. Initialize in inverse-softplus space: `W_raw = inv_softplus(target)` where `target ~ |N(0, 1/sqrt(N))|`
  2. Post-hoc rescale: compute `W_eff`, measure spectral radius, divide by `ρ/ρ_target`
  3. Use a different Dale's law parameterization (e.g., `W = diag(sign) @ |W_raw|` instead of softplus)

---

## Side Note: `parameter_table.md` Typo

The existing MATLAB documentation at `Intersect-LNNs-SRNNs/Docs/SRNN_docs/parameter_table.md` lists `τ_rel = 1/2 s`, but the code (`SRNNModel2.m` L33) and verified parameter extraction both show `tau_b_E_rel = 0.25 s`. The table value should be corrected to `1/4 s`.
