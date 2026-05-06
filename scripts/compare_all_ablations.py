"""compare_all_ablations.py — Visual survey of all SRNN ablation variants.

Runs all 13 meaningful SRNN presets in parallel via BatchedSRNNCell and plots
3 key panels (firing rate, synaptic output, adaptation) per variant in a
wrapped grid layout.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/compare_all_ablations.py
    PYTHONPATH=. .venv/bin/python scripts/compare_all_ablations.py --N 64
    PYTHONPATH=. .venv/bin/python scripts/compare_all_ablations.py --N 128 --no-show
"""

import argparse
import dataclasses
import math
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from train_srnn.models.srnn_cell import (
    SRNNConfig, BatchedSRNNCell, SRNN_PRESETS, piecewise_sigmoid,
)
from train_srnn.models.rmt_matrix import RMTMatrix

# ── CLI args ─────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--N", type=int, default=300, help="Number of neurons")
parser.add_argument("--no-show", action="store_true", help="Save PNG without displaying")
args = parser.parse_args()

# ── All meaningful variants (skip solver-only: srnn-explicit, srnn-rk4) ──

VARIANT_NAMES = [
    "srnn",
    "srnn-per-neuron",
    "srnn-echo",
    "srnn-no-adapt",
    "srnn-no-adapt-no-dales",
    "srnn-sfa-only",
    "srnn-std-only",
    "srnn-E-only",
    "srnn-e-only-echo",
    "srnn-e-only-per-neuron",
    "srnn-multi-sfa",
    "srnn-multi-sfa-E",
    "srnn-no-dales",
]

# ── Shared simulation parameters ─────────────────────────────────────────

N = args.N
F_EXC = 0.5
N_E = int(N * F_EXC)
ALPHA = 1.0 / 3.0     # Connection density (matches factory.py default)
LEVEL_OF_CHAOS = 1.0
RNG_SEED_NETWORK = 7
RNG_SEED_STIMULUS = 8

T = 50.0
SOLVER = "rk4"
H = 1.0 / 400
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

print(f"Variants ({K}):")
for i, cfg in enumerate(configs):
    print(f"  [{i:2d}] {VARIANT_NAMES[i]:25s}  n_a_E={cfg.n_a_E} n_a_I={cfg.n_a_I} "
          f"n_b_E={cfg.n_b_E} n_b_I={cfg.n_b_I} dales={cfg.dales} echo={cfg.echo}")

# ── Build W via RMTMatrix ────────────────────────────────────────────────

rmt = RMTMatrix(
    n=N, f=F_EXC, density=ALPHA,
    seed=RNG_SEED_NETWORK,
    level_of_chaos=LEVEL_OF_CHAOS,
)
W = rmt.build()
print(f"\nSpectral radius: {rmt.spectral_radius:.4f}")

exports = [rmt.export_for_srnn(dales=cfg.dales) for cfg in configs]

# ── Construct BatchedSRNNCell ────────────────────────────────────────────

cell = BatchedSRNNCell(configs, input_size=N, rmt_exports=exports)

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
print(f"Solver: {SOLVER}, h={H:.4f}s, unfolds={ODE_UNFOLDS}, dt={dt:.6f}s")
print(f"Simulation: {n_outer_steps} steps, T={T}s")

# ── Forward Simulation ───────────────────────────────────────────────────

torch.manual_seed(42)
state = cell.init_state(batch_size=1)

n_E = cell.n_E
n_I = cell.n_I
max_n_a_E = cell.max_n_a_E
max_n_a_I = cell.max_n_a_I

a_E_hist = np.zeros((K, n_outer_steps, n_E, max(max_n_a_E, 1)))
a_I_hist = np.zeros((K, n_outer_steps, n_I, max(max_n_a_I, 1)))
b_E_hist = np.zeros((K, n_outer_steps, n_E))
b_I_hist = np.zeros((K, n_outer_steps, n_I))
br_hist = np.zeros((K, n_outer_steps, N))

print(f"\nRunning forward simulation ({n_outer_steps} steps, {K} variants via bmm)...")
with torch.no_grad():
    for t_idx in range(n_outer_steps):
        u_t = u_ex_tensor[:, t_idx].unsqueeze(0)
        output, state = cell(u_t, state)

        a_E, a_I, b_E, b_I, x = cell.unpack_state(state)

        # Recompute effective potential and firing rate
        x_eff = x.clone()
        if max_n_a_E > 0:
            c_E = F.softplus(cell.isp_c_E)
            c_E_masked = c_E * cell.sfa_E_mask
            sfa_E_contrib = (c_E_masked.unsqueeze(1) * a_E).sum(-1)
            x_eff = torch.cat([x[:, :, :n_E] - sfa_E_contrib, x_eff[:, :, n_E:]], dim=-1)

        if max_n_a_I > 0:
            c_I = F.softplus(cell.isp_c_I)
            c_I_masked = c_I * cell.sfa_I_mask
            sfa_I_contrib = (c_I_masked.unsqueeze(1) * a_I).sum(-1)
            x_eff = torch.cat([x_eff[:, :, :n_E], x_eff[:, :, n_E:] - sfa_I_contrib], dim=-1)

        r = piecewise_sigmoid(x_eff - cell.a_0.unsqueeze(1))

        b_full_E = b_E * cell.std_E_mask.unsqueeze(1) + (1.0 - cell.std_E_mask.unsqueeze(1))
        b_full_I = b_I * cell.std_I_mask.unsqueeze(1) + (1.0 - cell.std_I_mask.unsqueeze(1))
        b_full = torch.cat([b_full_E, b_full_I], dim=-1)

        if max_n_a_E > 0:
            a_E_hist[:, t_idx] = a_E[:, 0].numpy()
        if max_n_a_I > 0:
            a_I_hist[:, t_idx] = a_I[:, 0].numpy()
        b_E_hist[:, t_idx] = b_full_E[:, 0].numpy()
        b_I_hist[:, t_idx] = b_full_I[:, 0].numpy()
        br_hist[:, t_idx] = (b_full * r)[:, 0].numpy()

# ── Sanity Checks ────────────────────────────────────────────────────────

print("\n=== Sanity Checks ===")
for ki, name in enumerate(VARIANT_NAMES):
    assert not np.any(np.isnan(br_hist[ki])), f"NaN in {name} synaptic output!"
    print(f"  {name:25s}: br range [{br_hist[ki].min():.4f}, {br_hist[ki].max():.4f}]")
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

# MATLAB 'lines' colormap for no-dales variants (no E/I connotation)
LINES_BASE = np.array([
    [0.0660, 0.4430, 0.7450],
    [0.8660, 0.3290, 0.0000],
    [0.9290, 0.6940, 0.1250],
    [0.5210, 0.0860, 0.8190],
    [0.2310, 0.6660, 0.1960],
    [0.1840, 0.7450, 0.9370],
    [0.8190, 0.0150, 0.5450],
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


# ── Plot: 3 panels per variant in a wrapped grid ────────────────────────

cmap_E = make_colormap(EXCITATORY_BASE, 8)
cmap_I = make_colormap(INHIBITORY_BASE, 8)
cmap_lines = make_colormap(LINES_BASE, 8)

# Track which variants have dales=False
is_no_dales = [not cfg.dales for cfg in configs]

PANELS_PER_VARIANT = 3
panel_labels = ["synaptic output", "depression", "adaptation"]
NCOLS = 5
n_meta_rows = math.ceil(K / NCOLS)
total_rows = n_meta_rows * PANELS_PER_VARIANT
total_cols = NCOLS

fig, axes = plt.subplots(
    total_rows, total_cols,
    figsize=(5 * total_cols, 3 * total_rows),
    squeeze=False,
)

fig.suptitle(
    f"All SRNN Ablation Variants (BatchedSRNNCell, K={K})\n"
    f"N={N}, {SOLVER}, h={H:.4f}s, "
    f"\u03c1={rmt.spectral_radius:.2f}",
    fontsize=16, fontweight="bold", y=1.0,
)

for ki, name in enumerate(VARIANT_NAMES):
    meta_row = ki // NCOLS
    col = ki % NCOLS
    row_base = meta_row * PANELS_PER_VARIANT

    # Select colormaps: neutral for no-dales, E/I for dales variants
    if is_no_dales[ki]:
        cm_first = cmap_lines
        cm_second = cmap_lines
    else:
        cm_first = cmap_E
        cm_second = cmap_I

    br_first = br_hist[ki, :, :N_E].T
    br_second = br_hist[ki, :, N_E:].T
    b_first = b_E_hist[ki].T
    b_second = b_I_hist[ki].T
    a_first = a_E_hist[ki].sum(axis=2).T
    a_second = a_I_hist[ki].sum(axis=2).T

    # (first_half, cm_first, second_half, cm_second) per panel
    panel_data = [
        (br_first, cm_first, br_second, cm_second),
        (b_first, cm_first, b_second, cm_second),
        (a_first, cm_first, a_second, cm_second),
    ]

    for pi, (label, (d1, c1, d2, c2)) in enumerate(zip(panel_labels, panel_data)):
        ax = axes[row_base + pi, col]
        plot_lines_with_colormap(ax, t_vec, d2, c2)
        plot_lines_with_colormap(ax, t_vec, d1, c1)

        if pi == 0:
            ax.set_title(name, fontsize=10, fontweight="bold")
        if col == 0:
            ax.set_ylabel(label, fontsize=9)
        if label in ("synaptic output", "depression"):
            ax.set_ylim(0, 1)
            ax.set_yticks([0, 1])
        if pi < PANELS_PER_VARIANT - 1:
            ax.tick_params(axis='x', labelbottom=False)
        else:
            ax.set_xlabel("Time (s)", fontsize=8)
        ax.tick_params(labelsize=7)

# Hide empty slots
for ki in range(K, n_meta_rows * NCOLS):
    meta_row = ki // NCOLS
    col = ki % NCOLS
    row_base = meta_row * PANELS_PER_VARIANT
    for pi in range(PANELS_PER_VARIANT):
        axes[row_base + pi, col].set_visible(False)

plt.tight_layout(rect=[0, 0, 1, 0.97])
out_path = f"scripts/compare_all_ablations_N{N}.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nFigure saved to {out_path}")
if args.no_show:
    plt.close(fig)
else:
    plt.show()
