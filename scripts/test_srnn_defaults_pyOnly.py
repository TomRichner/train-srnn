"""test_srnn_defaults_pyOnly.py — Pure-Python SRNN forward simulation.

Reproduces MATLAB test_SRNN2_defaults.m using:
  - RMTMatrix for W initialization (no MATLAB .mat file needed)
  - SRNNCell with n_a_E=3, n_b_E=1, N=300
  - StepStimulus-equivalent input (3 steps, 15% E density, amp=0.5)
  - RK4 solver at 400 Hz

Usage:
    python scripts/test_srnn_defaults_pyOnly.py
"""

import sys
sys.path.insert(0, ".")

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from models.srnn_cell import SRNNConfig, SRNNCell, piecewise_sigmoid
from models.rmt_matrix import RMTMatrix

# ── MATLAB-matching defaults ───────────────────────────────────────────────

N = 300
F_EXC = 0.5              # Fraction excitatory
N_E = int(N * F_EXC)
INDEGREE = 100
LEVEL_OF_CHAOS = 1.0
RNG_SEED_NETWORK = 1     # MATLAB rng_seeds(1)
RNG_SEED_STIMULUS = 2    # MATLAB rng_seeds(2)

T = 50.0                 # Simulation time (seconds)
FS = 400                 # Sampling frequency (Hz)
H = 1.0 / FS             # Outer timestep
ODE_UNFOLDS = 1           # Sub-steps per outer step

# Stimulus defaults (from StepStimulus.m)
N_STEPS_STIM = 3
STIM_DENSITY_E = 0.15
STIM_DENSITY_I = 0.0
STIM_AMP = 0.5
NO_STIM_PATTERN = [True, False, True]  # odd steps silent (MATLAB: 1:2:end)

# ── Build W via RMTMatrix ──────────────────────────────────────────────────

rmt = RMTMatrix(
    n=N, f=F_EXC, indegree=INDEGREE,
    seed=RNG_SEED_NETWORK,
    level_of_chaos=LEVEL_OF_CHAOS,
)
W = rmt.build()

print(rmt.summary())
print()

# Export for SRNNCell (dales=True → softplus-inverse space)
export = rmt.export_for_srnn(dales=True)

# ── Configure SRNNCell ─────────────────────────────────────────────────────

cfg = SRNNConfig(
    num_units=N,
    n_a_E=3, n_a_I=0,
    n_b_E=1, n_b_I=0,
    solver="rk4",
    h=H,
    ode_unfolds=ODE_UNFOLDS,
    dales=True,
    sparsity=0.0,  # Sparsity handled by RMT mask, not random
)

cell = SRNNCell(cfg, input_size=N)

# Inject RMT weights and sparsity mask
with torch.no_grad():
    cell.W_raw.copy_(export["W_init"])
    cell.W_in.copy_(torch.eye(N))  # W_in = eye(n), matching MATLAB

# Replace the random sparsity mask with the RMT one
cell.sparsity_mask = export["sparsity_mask"]

# ── Generate Step Stimulus (matching StepStimulus.m) ───────────────────────

rng_stim = np.random.default_rng(RNG_SEED_STIMULUS)

n_outer_steps = int(T / H)
t_vec = np.arange(n_outer_steps) * H

step_period = int(T / N_STEPS_STIM)  # fix() in MATLAB
step_length = round(step_period * FS)

# Random amplitudes
stim_amplitudes = STIM_AMP * rng_stim.standard_normal((N, N_STEPS_STIM))

# Sparse mask with separate E/I densities
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

active_E = sparse_mask[:N_E, 1].sum()
print(f"Stimulus: {N_STEPS_STIM} steps, E density={STIM_DENSITY_E}, I density={STIM_DENSITY_I}")
print(f"  Active E neurons in step 2: {active_E}")
print(f"Solver: {cfg.solver}, h={H}s, unfolds={ODE_UNFOLDS}, dt={H/ODE_UNFOLDS:.6f}s")
print(f"Simulation: {n_outer_steps} steps, T={T}s")

# ── Verify W round-trip ────────────────────────────────────────────────────

with torch.no_grad():
    W_eff = cell._effective_W().numpy()
    rho = np.max(np.abs(np.linalg.eigvals(W_eff)))
    print(f"\nW_eff spectral radius: {rho:.4f} (MATLAB: {rmt.spectral_radius:.4f})")

# ── Colormaps (match MATLAB SRNNModel2) ─────────────────────────────────────

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


# ── Forward Simulation ────────────────────────────────────────────────────

state = cell.init_state(batch_size=1)

r_history = np.zeros((n_outer_steps, N))
a_E_history = np.zeros((n_outer_steps, N_E, cfg.n_a_E))
b_E_history = np.zeros((n_outer_steps, N_E))
x_history = np.zeros((n_outer_steps, N))
br_history = np.zeros((n_outer_steps, N))

print(f"\nRunning forward simulation ({n_outer_steps} steps)...")
with torch.no_grad():
    for t_idx in range(n_outer_steps):
        u_t = u_ex_tensor[:, t_idx].unsqueeze(0)
        output, state = cell(u_t, state)

        a_E, a_I, b_E, b_I, x = cell.unpack_state(state)

        # Recompute firing rate
        x_eff = x.clone()
        if a_E is not None:
            c_E = F.softplus(cell.log_c_E)
            x_eff_E = x[:, :N_E] - (c_E * a_E).sum(-1)
            x_eff = torch.cat([x_eff_E, x[:, N_E:]], dim=-1)
        r = piecewise_sigmoid(x_eff - cell.a_0)

        b_full = torch.ones_like(r)
        if b_E is not None:
            b_full[:, :N_E] = b_E

        r_history[t_idx] = r[0].numpy()
        if a_E is not None:
            a_E_history[t_idx] = a_E[0].numpy()
        if b_E is not None:
            b_E_history[t_idx] = b_E[0].numpy()
        x_history[t_idx] = x[0].numpy()
        br_history[t_idx] = (b_full * r)[0].numpy()

# ── Sanity Checks ─────────────────────────────────────────────────────────

print("\n=== Sanity Checks ===")
assert not np.any(np.isnan(r_history)), "NaN in firing rates!"
assert not np.any(np.isnan(x_history)), "NaN in dendritic state!"
assert np.all(r_history >= 0) and np.all(r_history <= 1), "r outside [0,1]!"
assert np.all(b_E_history >= 0) and np.all(b_E_history <= 1), "b outside [0,1]!"
print("  All checks passed: no NaN, r∈[0,1], b∈[0,1]")

# ── Prepare data for plotting ─────────────────────────────────────────────

cmap_E = make_colormap(EXCITATORY_BASE, 8)
cmap_I = make_colormap(INHIBITORY_BASE, 8)

u_E = u_ex[:N_E, :]
u_I = u_ex[N_E:, :]
x_E = x_history[:, :N_E].T
x_I = x_history[:, N_E:].T
r_E = r_history[:, :N_E].T
r_I = r_history[:, N_E:].T
br_E = br_history[:, :N_E].T
br_I = br_history[:, N_E:].T
a_E_summed = a_E_history.sum(axis=2).T
b_E_plot = b_E_history.T

# ── Plot (matching MATLAB layout) ─────────────────────────────────────────

n_panels = 6
fig, axes = plt.subplots(n_panels, 1, figsize=(14, 14), sharex=True)
fig.suptitle(
    f"PyTorch SRNN (RMT init): n_a_E=3, n_b_E=1, N={N}, "
    f"ρ={rmt.spectral_radius:.2f}",
    fontsize=14, fontweight="bold", y=0.98,
)

# Panel 1: External input
ax = axes[0]
plot_lines_with_colormap(ax, t_vec, u_I, cmap_I)
plot_lines_with_colormap(ax, t_vec, u_E, cmap_E)
ax.set_ylabel("stim")
yl = ax.get_ylim()
ax.set_ylim(yl[0] - 0.05, yl[1])
ax.set_yticks([-1, 0, 1])

# Panel 2: Dendritic state
ax = axes[1]
plot_lines_with_colormap(ax, t_vec, x_I, cmap_I)
plot_lines_with_colormap(ax, t_vec, x_E, cmap_E)
ax.set_ylabel("dendrite")

# Panel 3: Firing rate
ax = axes[2]
plot_lines_with_colormap(ax, t_vec, r_I, cmap_I)
plot_lines_with_colormap(ax, t_vec, r_E, cmap_E)
ax.set_ylabel("firing rate")
ax.set_ylim(0, 1)
ax.set_yticks([0, 1])

# Panel 4: Synaptic output
ax = axes[3]
plot_lines_with_colormap(ax, t_vec, br_I, cmap_I)
plot_lines_with_colormap(ax, t_vec, br_E, cmap_E)
ax.set_ylabel("synaptic output")
ax.set_ylim(0, 1)
ax.set_yticks([0, 1])

# Panel 5: Adaptation
ax = axes[4]
plot_lines_with_colormap(ax, t_vec, a_E_summed, cmap_E)
ax.set_ylabel("adaptation")

# Panel 6: Depression
ax = axes[5]
plot_lines_with_colormap(ax, t_vec, b_E_plot, cmap_E)
ax.set_ylabel("depression")
ax.set_ylim(0, 1)
ax.set_yticks([0, 1])

# X-axis formatting
for ax in axes[:-1]:
    ax.tick_params(axis='x', labelbottom=False, length=0)
    ax.spines['bottom'].set_visible(False)

# Time scale bar
ax = axes[-1]
ax.set_xlabel("")
xlims = ax.get_xlim()
ylims = ax.get_ylim()
bar_len = round(0.1 * (xlims[1] - xlims[0]))
if bar_len < 1:
    bar_len = 0.1 * (xlims[1] - xlims[0])
x_end = xlims[0] + 0.95 * (xlims[1] - xlims[0])
x_start = x_end - bar_len
y_pos = ylims[0] + 0.10 * (ylims[1] - ylims[0])
ax.plot([x_start, x_end], [y_pos, y_pos], 'k-', linewidth=4)
ax.text((x_start + x_end) / 2, ylims[0] + 0.03 * (ylims[1] - ylims[0]),
        f"{bar_len:g} seconds", ha='center', va='top', fontsize=9)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("scripts/test_srnn_defaults_pyOnly.png", dpi=150, bbox_inches="tight")
print(f"\nFigure saved to scripts/test_srnn_defaults_pyOnly.png")
plt.show()
