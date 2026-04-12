"""test_srnn_defaults.py — Forward simulation of PyTorch SRNN with default parameters.

Mirrors MATLAB test_SRNN2_defaults.m: runs the SRNN (E-only adaptation,
n_a_E=3, n_b_E=1, N=300) forward with a step stimulus and plots all 300
neurons using the same panel layout and E/I colormaps as MATLAB.

Usage:
    python scripts/test_srnn_defaults.py
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from train_srnn.models.srnn_cell import SRNNConfig, SRNNCell, piecewise_sigmoid

# ── Configuration ──────────────────────────────────────────────────────────

N = 300               # Network size (match MATLAB)
N_E = N // 2          # 150 E, 150 I
SEED = 1
T = 50.0              # Total simulation time (seconds)
H = 1.0 / 400         # Outer timestep (400 Hz, match MATLAB fs)
ODE_UNFOLDS = 1       # Sub-steps per outer step
N_STEPS_STIM = 3      # Number of stimulus step periods
STIM_DENSITY_E = 0.15 # Fraction of E neurons receiving input
STIM_AMP = 0.5        # Stimulus amplitude
NO_STIM_PATTERN = [True, False, True]  # Steps 1,3 silent; step 2 active

# ── Colormaps (match MATLAB SRNNModel2) ────────────────────────────────────

EXCITATORY_BASE = np.array([
    [1.00, 0.00, 0.00],
    [1.00, 0.75, 0.00],
    [0.85, 0.20, 0.45],
    [0.90, 0.10, 0.60],
    [0.90, 0.55, 0.00],
    [0.55, 0.27, 0.27],
    [0.86, 0.08, 0.24],
    [0.60, 0.15, 0.45],
])

INHIBITORY_BASE = np.array([
    [0.00, 0.45, 0.74],
    [0.00, 0.75, 1.00],
    [0.20, 0.47, 0.62],
    [0.00, 0.50, 0.50],
    [0.30, 0.75, 0.93],
    [0.25, 0.62, 0.75],
    [0.00, 0.80, 0.80],
    [0.15, 0.55, 0.65],
])


def make_colormap(base_palette, n_colors):
    """Interpolate base palette to n_colors, matching MATLAB pchip interp."""
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
    """Plot each row of data (n_neurons, n_t) with cycling colormap colors."""
    n_lines = data.shape[0]
    n_colors = len(cmap)
    for i in range(n_lines):
        color_idx = i % n_colors
        ax.plot(t, data[i], color=cmap[color_idx], linewidth=0.4, alpha=0.7)


# ── Setup ──────────────────────────────────────────────────────────────────

# Load MATLAB W and stimulus (including decomposed W components)
MATLAB_MAT = "/Users/tom/Desktop/local_code/Intersect-LNNs-SRNNs/Matlab/SRNN/scripts/matlab_W_and_stim.mat"
from scipy.io import loadmat
mat = loadmat(MATLAB_MAT, squeeze_me=True)
W_matlab = mat["W"]          # (300, 300) — already has E/I signs + sparsity
u_ex_matlab = mat["u_ex"]    # (300, 20001) — at 400 Hz
t_ex_matlab = mat["t_ex"]    # (20001,) — time vector at 400 Hz
S0_matlab = mat["S0"]        # initial state vector
N_mat = int(mat["n"])
f_mat = float(mat["f"])

# Decomposed W components for dales=True path
W_init_matlab = mat["W_init"]            # (300, 300) — softplus-inverse of |W|
sparsity_mask_matlab = mat["sparsity_mask"]  # (300, 300) — binary mask
dales_sign_matlab = mat["dales_sign"]    # (300,) — +1 E, -1 I
n_E_matlab = int(mat["n_E"])

assert N_mat == N, f"MATLAB n={N_mat} != script N={N}"
N_E = int(N * f_mat)
assert N_E == n_E_matlab, f"N_E mismatch: {N_E} vs {n_E_matlab}"

print(f"Loaded MATLAB W: shape={W_matlab.shape}, spectral radius={np.max(np.abs(np.linalg.eigvals(W_matlab))):.3f}")
print(f"Loaded MATLAB stimulus: shape={u_ex_matlab.shape}, t=[{t_ex_matlab[0]:.1f}, {t_ex_matlab[-1]:.1f}]s")
print(f"Loaded decomposed W: W_init={W_init_matlab.shape}, sparsity={sparsity_mask_matlab.mean():.4f}")

torch.manual_seed(SEED)
np.random.seed(SEED + 1)

# Build rmt_export dict from MATLAB decomposed data
rmt_export = {
    "W_init": torch.tensor(W_init_matlab, dtype=torch.float32),
    "sparsity_mask": torch.tensor(sparsity_mask_matlab, dtype=torch.float32),
    "dales_sign": torch.tensor(dales_sign_matlab, dtype=torch.float32),
    "n_E": n_E_matlab,
    "dales": True,
}

cfg = SRNNConfig(
    num_units=N,
    n_a_E=3, n_a_I=0,
    n_b_E=1, n_b_I=0,
    solver="rk4",
    h=H,
    ode_unfolds=ODE_UNFOLDS,
    dales=True,
)

cell = SRNNCell(cfg, input_size=N, rmt_export=rmt_export)

# Override W_in to identity (matching MATLAB)
with torch.no_grad():
    cell.W_in.copy_(torch.eye(N))

# ── Resample MATLAB stimulus to match our h ────────────────────────────────

n_outer_steps = int(T / H)
t_vec = np.arange(n_outer_steps) * H

# Nearest-neighbor resample from 400 Hz MATLAB stimulus to our timestep
# Find the MATLAB time index closest to each of our timesteps
matlab_indices = np.searchsorted(t_ex_matlab, t_vec, side="left")
matlab_indices = np.clip(matlab_indices, 0, len(t_ex_matlab) - 1)
u_ex = u_ex_matlab[:, matlab_indices]  # (N, n_outer_steps)

u_ex_tensor = torch.tensor(u_ex, dtype=torch.float32)

# Verify W round-trip: decomposed → effective should match MATLAB W
with torch.no_grad():
    W_eff = cell._effective_W().numpy()
    rho = np.max(np.abs(np.linalg.eigvals(W_eff)))
    rho_matlab = np.max(np.abs(np.linalg.eigvals(W_matlab)))
    print(f"\nW_eff spectral radius: {rho:.4f} (MATLAB: {rho_matlab:.4f})")
    # Check reconstruction error
    err = np.max(np.abs(W_eff - W_matlab))
    print(f"Max |W_eff - W_matlab|: {err:.2e}")

print(f"Config: solver={cfg.solver}, h={H}s, unfolds={ODE_UNFOLDS}, dt={H/ODE_UNFOLDS:.6f}s")
print(f"Simulation: {n_outer_steps} outer steps, T={T}s")

# ── Forward Simulation ────────────────────────────────────────────────────

state = cell.init_state(batch_size=1)

# Storage
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

        # Build b_full for synaptic output
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

# ── Prepare colormaps ──────────────────────────────────────────────────────

cmap_E = make_colormap(EXCITATORY_BASE, 8)
cmap_I = make_colormap(INHIBITORY_BASE, 8)

# Split stored arrays into E and I (neurons x time)
u_E = u_ex[:N_E, :]       # (N_E, n_t)
u_I = u_ex[N_E:, :]       # (N_I, n_t)
x_E = x_history[:, :N_E].T   # (N_E, n_t)
x_I = x_history[:, N_E:].T
r_E = r_history[:, :N_E].T
r_I = r_history[:, N_E:].T
br_E = br_history[:, :N_E].T
br_I = br_history[:, N_E:].T
a_E_summed = a_E_history.sum(axis=2).T  # sum over timescales → (N_E, n_t)
b_E_plot = b_E_history.T  # (N_E, n_t)

# ── Plot (matching MATLAB layout) ─────────────────────────────────────────

n_panels = 6  # stim, dendrite, firing rate, synaptic output, adaptation, depression
fig, axes = plt.subplots(n_panels, 1, figsize=(14, 14), sharex=True)
fig.suptitle("PyTorch SRNN: E-only adaptation (n_a_E=3, n_b_E=1, N=300)",
             fontsize=14, fontweight="bold", y=0.98)

# Panel 1: External input (stimulus)
ax = axes[0]
plot_lines_with_colormap(ax, t_vec, u_I, cmap_I)
plot_lines_with_colormap(ax, t_vec, u_E, cmap_E)
ax.set_ylabel("stim")
yl = ax.get_ylim()
ax.set_ylim(yl[0] - 0.05, yl[1])
ax.set_yticks([-1, 0, 1])

# Panel 2: Dendritic state (x)
ax = axes[1]
plot_lines_with_colormap(ax, t_vec, x_I, cmap_I)
plot_lines_with_colormap(ax, t_vec, x_E, cmap_E)
ax.set_ylabel("dendrite")

# Panel 3: Firing rate (r)
ax = axes[2]
plot_lines_with_colormap(ax, t_vec, r_I, cmap_I)
plot_lines_with_colormap(ax, t_vec, r_E, cmap_E)
ax.set_ylabel("firing rate")
ax.set_ylim(0, 1)
ax.set_yticks([0, 1])

# Panel 4: Synaptic output (b·r)
ax = axes[3]
plot_lines_with_colormap(ax, t_vec, br_I, cmap_I)
plot_lines_with_colormap(ax, t_vec, br_E, cmap_E)
ax.set_ylabel("synaptic output")
ax.set_ylim(0, 1)
ax.set_yticks([0, 1])

# Panel 5: Adaptation (sum of a_E across timescales — E only, since n_a_I=0)
ax = axes[4]
plot_lines_with_colormap(ax, t_vec, a_E_summed, cmap_E)
ax.set_ylabel("adaptation")

# Panel 6: Depression (b — E only, since n_b_I=0)
ax = axes[5]
plot_lines_with_colormap(ax, t_vec, b_E_plot, cmap_E)
ax.set_ylabel("depression")
ax.set_ylim(0, 1)
ax.set_yticks([0, 1])

# Shared x-axis formatting: hide tick labels on all but last
for ax in axes[:-1]:
    ax.tick_params(axis='x', labelbottom=False, length=0)
    ax.spines['bottom'].set_visible(False)

# Time scale bar on last panel (matching MATLAB)
ax = axes[-1]
ax.set_xlabel("")  # MATLAB uses scale bar instead of xlabel
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
plt.savefig("scripts/test_srnn_defaults.png", dpi=150, bbox_inches="tight")
print(f"\nFigure saved to scripts/test_srnn_defaults.png")
plt.show()
