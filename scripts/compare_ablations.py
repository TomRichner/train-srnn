"""compare_ablations.py — Compare SRNN ablation variants via BatchedSRNNCell.

Runs K SRNN preset variants in parallel through the same network and stimulus
using BatchedSRNNCell (torch.bmm), then plots internal dynamics side by side.

Usage:
    python scripts/compare_ablations.py
"""

import dataclasses
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from train_srnn.models.srnn_cell import (
    SRNNConfig, BatchedSRNNCell, SRNN_PRESETS, piecewise_sigmoid,
)
from train_srnn.models.rmt_matrix import RMTMatrix

# ── Which variants to compare (change this list) ─────────────────────────
# All non-per-neuron presets; solver/h/ode_unfolds get overridden below to
# satisfy BatchedSRNNCell's "all configs share solver/h/ode_unfolds" check.
VARIANT_NAMES = [
    name for name, cfg in SRNN_PRESETS.items()
    if not cfg.per_neuron
]

# ── Shared simulation parameters ─────────────────────────────────────────

N = 300
F_EXC = 0.5
N_E = int(N * F_EXC)
ALPHA = 1.0 / 3.0     # Connection density (matches factory.py default)
LEVEL_OF_CHAOS = 1.0
RNG_SEED_NETWORK = 7
RNG_SEED_STIMULUS = 8

T = 50.0
SOLVER = "rk4"
H = 1.0 / 400          # Match test_srnn_defaults_pyOnly.py (FS=400)
ODE_UNFOLDS = 1

# Stimulus defaults
N_STEPS_STIM = 3
STIM_DENSITY_E = 0.15
STIM_DENSITY_I = 0.0
STIM_AMP = 0.5
NO_STIM_PATTERN = [True, False, True]

# ── Build configs from presets ───────────────────────────────────────────

configs = []
for name in VARIANT_NAMES:
    cfg = dataclasses.replace(
        SRNN_PRESETS[name],
        num_units=N, solver=SOLVER, h=H, ode_unfolds=ODE_UNFOLDS,
    )
    configs.append(cfg)
K = len(configs)

print(f"Variants ({K}): {VARIANT_NAMES}")
for i, cfg in enumerate(configs):
    print(f"  [{i}] {VARIANT_NAMES[i]}: n_a_E={cfg.n_a_E}, n_a_I={cfg.n_a_I}, "
          f"n_b_E={cfg.n_b_E}, n_b_I={cfg.n_b_I}, dales={cfg.dales}, echo={cfg.echo}")

# ── Build W via RMTMatrix ────────────────────────────────────────────────

rmt = RMTMatrix(
    n=N, f=F_EXC, density=ALPHA,
    seed=RNG_SEED_NETWORK,
    level_of_chaos=LEVEL_OF_CHAOS,
)
W = rmt.build()
print(f"\n{rmt.summary()}")

exports = [rmt.export_for_srnn(dales=cfg.dales) for cfg in configs]

# ── Construct BatchedSRNNCell ────────────────────────────────────────────

cell = BatchedSRNNCell(configs, input_size=N, rmt_exports=exports)

# Override W_in to identity (matching MATLAB / existing scripts)
with torch.no_grad():
    for ki in range(K):
        cell.W_in[ki].copy_(torch.eye(N))

# ── Generate Step Stimulus ───────────────────────────────────────────────

rng_stim = np.random.default_rng(RNG_SEED_STIMULUS)

n_outer_steps = int(T / H)
t_vec = np.arange(n_outer_steps) * H

step_period = int(T / N_STEPS_STIM)
step_length = round(step_period / H)

stim_amplitudes = STIM_AMP * rng_stim.standard_normal((N, N_STEPS_STIM))

sparse_mask = np.zeros((N, N_STEPS_STIM), dtype=bool)
sparse_mask[:N_E, :] = rng_stim.random((N_E, N_STEPS_STIM)) < STIM_DENSITY_E
sparse_mask[N_E:, :] = rng_stim.random((N - N_E, N_STEPS_STIM)) < STIM_DENSITY_I

stim_amplitudes *= sparse_mask
for k, silent in enumerate(NO_STIM_PATTERN):
    if silent:
        stim_amplitudes[:, k] = 0.0

u_ex = np.zeros((N, n_outer_steps))
for step_idx in range(N_STEPS_STIM):
    start = step_idx * step_length
    end = min((step_idx + 1) * step_length, n_outer_steps)
    if start >= n_outer_steps:
        break
    u_ex[:, start:end] = stim_amplitudes[:, step_idx:step_idx + 1]

u_ex_tensor = torch.tensor(u_ex, dtype=torch.float32)

dt = H / ODE_UNFOLDS
print(f"\nSolver: {SOLVER}, h={H}s, unfolds={ODE_UNFOLDS}, dt={dt:.6f}s")
print(f"Simulation: {n_outer_steps} steps, T={T}s")

# ── Forward Simulation ───────────────────────────────────────────────────

torch.manual_seed(42)
state = cell.init_state(batch_size=1)

n_E = cell.n_E
max_n_a_E = cell.max_n_a_E
max_n_a_I = cell.max_n_a_I

r_hist = np.zeros((K, n_outer_steps, N))
x_hist = np.zeros((K, n_outer_steps, N))
a_E_hist = np.zeros((K, n_outer_steps, n_E, max(max_n_a_E, 1)))
b_E_hist = np.zeros((K, n_outer_steps, n_E))
br_hist = np.zeros((K, n_outer_steps, N))

print(f"\nRunning forward simulation ({n_outer_steps} steps, {K} variants via bmm)...")
with torch.no_grad():
    for t_idx in range(n_outer_steps):
        u_t = u_ex_tensor[:, t_idx].unsqueeze(0)  # (1, N) — broadcasts to (K, 1, N)
        output, state = cell(u_t, state)

        # Use cell.get_diagnostics — single source of truth that handles
        # the SFA / STD / std_zero_floor math identically to the solver step.
        diag = cell.get_diagnostics(state, u_t)
        a_E = diag["a_E"]      # (K, 1, n_E, max_n_a_E)
        b_full = diag["b_full"]  # (K, 1, N) — already includes std_zero_floor rescaling
        r = diag["r"]          # (K, 1, N)
        x = diag["x"]          # (K, 1, N)

        # Store (squeeze batch dim)
        r_hist[:, t_idx] = r[:, 0].numpy()
        x_hist[:, t_idx] = x[:, 0].numpy()
        if max_n_a_E > 0:
            a_E_hist[:, t_idx] = a_E[:, 0].numpy()
        # E-side b_full only (n_E columns)
        b_E_hist[:, t_idx] = b_full[:, 0, :n_E].numpy()
        br_hist[:, t_idx] = (b_full * r)[:, 0].numpy()

# ── Sanity Checks ────────────────────────────────────────────────────────

print("\n=== Sanity Checks ===")
for ki, name in enumerate(VARIANT_NAMES):
    assert not np.any(np.isnan(r_hist[ki])), f"NaN in {name} firing rates!"
    assert not np.any(np.isnan(x_hist[ki])), f"NaN in {name} dendritic state!"
    print(f"  {name}: r range [{r_hist[ki].min():.4f}, {r_hist[ki].max():.4f}]")
print("  All checks passed.")

# ── Colormaps ────────────────────────────────────────────────────────────

EXCITATORY_BASE = np.array([
    [1.00, 0.00, 0.00], [1.00, 0.75, 0.00], [0.85, 0.20, 0.45],
    [0.90, 0.10, 0.60], [0.90, 0.55, 0.00], [0.55, 0.27, 0.27],
    [0.86, 0.08, 0.24], [0.60, 0.15, 0.45],
])

INHIBITORY_BASE = np.array([
    [0.00, 0.45, 0.74], [0.00, 0.75, 1.00], [0.20, 0.47, 0.62],
    [0.00, 0.50, 0.50], [0.30, 0.75, 0.93], [0.25, 0.62, 0.75],
    [0.00, 0.80, 0.80], [0.15, 0.55, 0.65],
])


def make_colormap(base_palette, n_colors):
    n_base = len(base_palette)
    if n_colors <= n_base:
        indices = np.round(np.linspace(0, n_base - 1, n_colors)).astype(int)
        return base_palette[indices]
    from scipy.interpolate import PchipInterpolator
    x_base = np.linspace(0, 1, n_base)
    x_new = np.linspace(0, 1, n_colors)
    cmap = np.zeros((n_colors, 3))
    for ch in range(3):
        interp = PchipInterpolator(x_base, base_palette[:, ch])
        cmap[:, ch] = interp(x_new)
    return np.clip(cmap, 0, 1)


def plot_lines_with_colormap(ax, t, data, cmap):
    n_lines = data.shape[0]
    n_colors = len(cmap)
    for i in range(n_lines):
        ax.plot(t, data[i], color=cmap[i % n_colors], linewidth=0.4, alpha=0.7)


# ── Plot (6 x K grid) ───────────────────────────────────────────────────

cmap_E = make_colormap(EXCITATORY_BASE, 8)
cmap_I = make_colormap(INHIBITORY_BASE, 8)

panel_labels = ["stim", "dendrite", "firing rate", "synaptic output", "adaptation", "depression"]
n_panels = len(panel_labels)

per_col = 10 if K <= 3 else max(2.5, 30.0 / K)
fig, axes = plt.subplots(n_panels, K, figsize=(per_col * K, 16), sharex=True, sharey="row")
if K == 1:
    axes = axes[:, np.newaxis]

fig.suptitle(
    f"SRNN Ablation Comparison (BatchedSRNNCell, bmm)\n"
    f"N={N}, {SOLVER}, h={H}s, unfolds={ODE_UNFOLDS}, "
    f"\u03c1={rmt.spectral_radius:.2f}",
    fontsize=14, fontweight="bold", y=0.99,
)

u_E = u_ex[:N_E, :]
u_I = u_ex[N_E:, :]

for col, name in enumerate(VARIANT_NAMES):
    axes[0, col].set_title(name, fontsize=13, fontweight="bold")

    x_E = x_hist[col, :, :N_E].T
    x_I = x_hist[col, :, N_E:].T
    r_E = r_hist[col, :, :N_E].T
    r_I = r_hist[col, :, N_E:].T
    br_E = br_hist[col, :, :N_E].T
    br_I = br_hist[col, :, N_E:].T
    a_E_summed = a_E_hist[col].sum(axis=2).T   # (n_E, T)
    b_E_plot = b_E_hist[col].T                  # (n_E, T)

    panel_data = [
        (u_E, u_I),
        (x_E, x_I),
        (r_E, r_I),
        (br_E, br_I),
        (a_E_summed, None),
        (b_E_plot, None),
    ]

    for row, (label, (data_E, data_I)) in enumerate(zip(panel_labels, panel_data)):
        ax = axes[row, col]
        if data_I is not None:
            plot_lines_with_colormap(ax, t_vec, data_I, cmap_I)
        plot_lines_with_colormap(ax, t_vec, data_E, cmap_E)
        if col == 0:
            ax.set_ylabel(label)
        if label in ("firing rate", "synaptic output", "depression"):
            ax.set_ylim(0, 1)
            ax.set_yticks([0, 1])

# Bottom row x-axis
for col in range(K):
    axes[-1, col].set_xlabel("Time (s)")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("scripts/compare_ablations.png", dpi=150, bbox_inches="tight")
print(f"\nFigure saved to scripts/compare_ablations.png")
plt.show()
