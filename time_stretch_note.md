# Time Stretch Augmentation: Design Note

## Overview

This note documents the relationship between the ODE integration parameters (`h`, `ode_unfolds`) and the time-stretch data augmentation, and proposes a cleaner design that eliminates redundancy.

## Current System

Each call to `SRNNCell.forward` (one input timestep) does:

```
dt = h / ode_unfolds
for _ in range(ode_unfolds):
    x_new = (x + (dt/τ_d)(u + Wbr)) / (1 + dt/τ_d)   # semi-implicit
```

- **`h`**: Simulated time per input timestep. Think of it as `1/fs` where `fs` is the ODE sampling rate.
- **`ode_unfolds`**: Number of ODE sub-steps per input timestep (numerical accuracy).
- **`dt = h / ode_unfolds`**: Actual ODE step size.

With defaults `h=0.01, ode_unfolds=4`: the ODE step size is `dt=0.0025`, and each input sample spans `0.01s` of simulated time.

Time-stretch augmentation resamples the input from `T` to `T_new = round(T × factor)` timesteps via PCHIP interpolation. The model still integrates `h` seconds per timestep, so total simulated time = `T_new × h`.

## The Redundancy

Three knobs all affect the ratio of network dynamics to input presentation rate:

| Parameter | Effect |
|---|---|
| `h` | More simulated time per input sample |
| `tau_global` | Scales all network time constants (learnable) |
| `stretch_factor` | More/fewer input samples for the same content |

Since `tau_global` is learnable, the optimizer will find the right `h/tau` ratio regardless of the absolute value of `h`. This makes `h` a **convention** (unit of time), not a tuning knob.

## Proposed Design: `fs`-Based, No Unfolds

Set `ode_unfolds = 1` and think of `h = 1/fs`:

- **`fs` is the ODE sampling rate** — a fixed architectural choice (e.g. `fs = 400 Hz` → `h = 0.0025`).
- **Stretch factor is the sole timescale augmentation knob** — controls how fast data is played into the model.
- **`tau_global` (learnable)** adapts the network's intrinsic timescale to the data.

### Migrating from the Old Config

Old: `h = 0.01, ode_unfolds = 4` → actual ODE step `dt = 0.0025`

To preserve the same ODE step size and numerical precision:

```
New: h = 0.0025, ode_unfolds = 1
```

But now each input sample spans only `0.0025s` instead of `0.01s`. To recover the same dynamics, we need **4× more input samples** — a stretch factor of **4.0**:

| | Old | New (equivalent) |
|---|---|---|
| `h` | 0.01 | 0.0025 |
| `ode_unfolds` | 4 | 1 |
| `dt` (ODE step) | 0.0025 | 0.0025 |
| Stretch factor | 1.0 | 4.0 |
| Input samples | T | 4T |
| Sim time | T × 0.01 | 4T × 0.0025 = T × 0.01 ✓ |

**Note:** these are not bitwise identical. The old system holds the input constant (zero-order hold) across 4 sub-steps. The new system PCHIP-interpolates to 4× more samples, so the input smoothly varies every step. This is arguably more physically natural.

### Why `ode_unfolds = 1` Is Safe

The semi-implicit solver (default) uses the update:

```
x_new = (x + α·drive) / (1 + α),    where α = dt/τ
```

This is **A-stable** (unconditionally stable). The `1/(1+α)` factor ensures bounded updates for any step size. The same holds for the SFA and STD semi-implicit updates. Accuracy degrades for very large `dt/τ`, but the dynamics can never blow up.

The exponential solver (`exp(-dt/τ)` decay) is also unconditionally stable.

Only the explicit Euler solver requires `dt/τ < 2` for stability — but this is not the default and would need care with large `h`.

## Current Bias: Slow Timescales Are Over-Represented

Currently, time stretching is applied **per-batch** (one factor for all samples), and the number of forward/BPTT steps scales linearly with the stretch factor:

| Quantity | factor=0.5 | factor=1.0 | factor=4.0 |
|---|---|---|---|
| Seq length after stretch | T/2 | T | 4T |
| Forward steps (win_len) | ~512 | ~1024 | ~4096 |
| BPTT steps | ~256 | ~512 | ~2048 |

Slow timescales get proportionally more gradient signal. This is an 8:1 bias across a `[0.5, 4.0]` stretch range.

## Proposed Fix: Fixed Step Count, Per-Sample Stretch

1. **Per-sample stretch**: Each sample in a batch draws its own stretch factor. The model sees a mixture of timescales every batch.

2. **Fixed total steps** (`M`): All samples are palindrome-looped and windowed to exactly `M` timesteps, regardless of stretch factor. Fast-stretched samples cycle through more palindrome repetitions; slow-stretched samples see fewer.

3. **Fixed BPTT depth**: Backpropagate through the last `bptt_steps` timesteps (a constant), not the last N palindrome cycles.

This gives:
- **Uniform compute per sample** — no bias toward any timescale
- **Constant wall-clock per batch** — no variable-length batches
- **Per-sample diversity** — the model trains on a mixture of speeds every batch

## Summary

| Component | Current | Proposed |
|---|---|---|
| `ode_unfolds` | 4 (config param) | 1 (hardcoded) |
| `h` | 0.01 | `1/fs`, fixed (e.g. 0.0025 for fs=400) |
| stretch per batch | 1 factor, all samples | Per-sample factors |
| Forward/BPTT steps | Scale with stretch | Fixed count `M` |
| Speedup | — | ~4× (fewer ODE evals per timestep) |
