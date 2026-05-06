"""compare_solvers.py — Compare semi-implicit vs RK4 SRNN forward simulation.

Based on test_srnn_defaults_pyOnly.py. Runs the same stimulus through both
solvers and plots them side by side.

Usage:
    python scripts/compare_solvers.py
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from train_srnn.models.srnn_cell import SRNNConfig, SRNNCell, piecewise_sigmoid
from train_srnn.models.rmt_matrix import RMTMatrix

# ── MATLAB-matching defaults ───────────────────────────────────────────────

N = 300
F_EXC = 0.5
N_E = int(N * F_EXC)
INDEGREE = 100
LEVEL_OF_CHAOS = 1.0
RNG_SEED_NETWORK = 7
RNG_SEED_STIMULUS = 8

T = 50.0
H = 0.01
ODE_UNFOLDS = 4

# Stimulus defaults
N_STEPS_STIM = 3
STIM_DENSITY_E = 0.15
STIM_DENSITY_I = 0.0
STIM_AMP = 0.5
NO_STIM_PATTERN = [True, False, True]

# ── Build W via RMTMatrix ──────────────────────────────────────────────────

rmt = RMTMatrix(
    n=N, f=F_EXC, indegree=INDEGREE,
    seed=RNG_SEED_NETWORK,
    level_of_chaos=LEVEL_OF_CHAOS,
)
W = rmt.build()
export = rmt.export_for_srnn(dales=True)

print(rmt.summary())

# ── Generate Step Stimulus ────────────────────────────────────────────────

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

# ── Colormaps ─────────────────────────────────────────────────────────────

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


# ── Run simulation with a given solver ────────────────────────────────────

def run_simulation(solver_name, h=H, ode_unfolds=ODE_UNFOLDS):
    cfg = SRNNConfig(
        num_units=N,
        n_a_E=3, n_a_I=0,
        n_b_E=1, n_b_I=0,
        solver=solver_name,
        h=h,
        ode_unfolds=ode_unfolds,
        dales=True,
    )

    cell = SRNNCell(cfg, input_size=N, rmt_export=export)
    with torch.no_grad():
        cell.W_in.copy_(torch.eye(N))

    state = cell.init_state(batch_size=1)

    r_history = np.zeros((n_outer_steps, N))
    a_E_history = np.zeros((n_outer_steps, N_E, cfg.n_a_E))
    b_E_history = np.zeros((n_outer_steps, N_E))
    x_history = np.zeros((n_outer_steps, N))
    br_history = np.zeros((n_outer_steps, N))

    dt = h / ode_unfolds
    print(f"\nRunning {solver_name} (h={h}, unfolds={ode_unfolds}, dt={dt:.6f}s)...")
    with torch.no_grad():
        for t_idx in range(n_outer_steps):
            u_t = u_ex_tensor[:, t_idx].unsqueeze(0)
            output, state = cell(u_t, state)

            a_E, a_I, b_E, b_I, x = cell.unpack_state(state)

            x_eff = x.clone()
            if a_E is not None:
                c_E = F.softplus(cell.isp_c_E)
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

    # Sanity checks
    assert not np.any(np.isnan(r_history)), f"NaN in {solver_name} firing rates!"
    assert not np.any(np.isnan(x_history)), f"NaN in {solver_name} dendritic state!"
    print(f"  {solver_name}: no NaN, r range [{r_history.min():.4f}, {r_history.max():.4f}]")

    return {
        "r": r_history, "a_E": a_E_history, "b_E": b_E_history,
        "x": x_history, "br": br_history,
    }


# ── Run both solvers ──────────────────────────────────────────────────────

results_rk4 = run_simulation("rk4")
results_semi = run_simulation("semi_implicit")

# ── Plot side by side ─────────────────────────────────────────────────────

cmap_E = make_colormap(EXCITATORY_BASE, 8)
cmap_I = make_colormap(INHIBITORY_BASE, 8)

panel_labels = ["stim", "dendrite", "firing rate", "synaptic output", "adaptation", "depression"]
n_panels = len(panel_labels)

fig, axes = plt.subplots(n_panels, 2, figsize=(20, 16), sharex=True, sharey="row")
fig.suptitle(
    f"SRNN Solver Comparison: RK4 vs Semi-Implicit\n"
    f"n_a_E=3, n_b_E=1, N={N}, h={H}s, unfolds={ODE_UNFOLDS}, "
    f"\u03c1={rmt.spectral_radius:.2f}",
    fontsize=14, fontweight="bold", y=0.99,
)

for col, (solver_name, res) in enumerate([("RK4", results_rk4), ("Semi-Implicit", results_semi)]):
    axes[0, col].set_title(solver_name, fontsize=13, fontweight="bold")

    u_E = u_ex[:N_E, :]
    u_I = u_ex[N_E:, :]
    x_E = res["x"][:, :N_E].T
    x_I = res["x"][:, N_E:].T
    r_E = res["r"][:, :N_E].T
    r_I = res["r"][:, N_E:].T
    br_E = res["br"][:, :N_E].T
    br_I = res["br"][:, N_E:].T
    a_E_summed = res["a_E"].sum(axis=2).T
    b_E_plot = res["b_E"].T

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
for col in range(2):
    axes[-1, col].set_xlabel("Time (s)")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("scripts/compare_solvers.png", dpi=150, bbox_inches="tight")
print(f"\nFigure saved to scripts/compare_solvers.png")
plt.show()
