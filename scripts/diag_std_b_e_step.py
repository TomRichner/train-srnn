"""Diagnostic: do STD b_E(t) state variables actually go below 0.5?

User observation: in trained models (e.g. alpha-ramp-150e), the STD synaptic-
depression state `b_E(t)` rarely depresses below 0.5, even though MATLAB
simulations of equivalent dynamics readily go below 0.5.

This script isolates *cell-implementation* concerns: with a fresh untrained
``srnn-std-e-only`` cell at default init values
(``tau_b_rec_E = 1.0 s``, ``tau_b_rel_E = 0.25 s``), under a strong sustained
step input applied via identity W_in to 15% of E neurons, do the driven
neurons' b_E values depress below 0.5?

Theoretical asymptote at sustained max firing (r=1):
    b_E_ss = tau_rel / (tau_rec * r + tau_rel)
            = 0.25 / (1.0 * 1 + 0.25) = 0.20

So well-driven neurons should approach 0.2; sub-saturation neurons should
land somewhere between 0.2 and 1.0. The 0.5 line is the sanity threshold —
if NO neuron crosses it, something is wrong in the cell forward.

We test all three solvers (euler, rk4, semi_implicit) to detect any
solver-specific bug.

Run: PYTHONPATH=. python scripts/diag_std_b_e_step.py
"""
from __future__ import annotations

import os
import sys
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from train_srnn.models.rmt_matrix import RMTMatrix
from train_srnn.models.srnn_cell import SRNN_PRESETS, SRNNCell, piecewise_sigmoid


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

NUM_UNITS = 300
# Note: SRNNCell uses n_E = num_units // 2 hard-coded (the `frac_E` field in
# model YAML is filtered out by _cfg_to_dataclass and not used). So with
# num_units=300, n_E = 150 (not 225).
H = 0.004
ODE_UNFOLDS = 1
T_SECONDS = 5.0
T = int(T_SECONDS / H)        # 1250 timesteps
FRAC_ACTIVE = 0.15
SEED = 0
B = 1
SOLVERS = ["explicit", "rk4", "semi_implicit"]  # explicit = forward Euler
STD_ZERO_FLOOR = os.environ.get("STD_ZERO_FLOOR", "0") == "1"
OUT_DIR = os.path.join(REPO_ROOT, "tmp", "diag_std")
OUT_TAG = "_zerofloor" if STD_ZERO_FLOOR else ""


# ---------------------------------------------------------------------------
# Build the input vector once (reused across solvers)
# ---------------------------------------------------------------------------

def make_input(num_units: int, n_E: int, frac_active: float, seed: int):
    """Return (input_vec (num_units,), active_idx (n_active,), amplitudes (n_active,))."""
    rng = np.random.default_rng(seed)
    n_active = max(1, int(round(n_E * frac_active)))
    # Pick `n_active` unique E-indices from [0, n_E).
    perm = rng.permutation(n_E)
    active_idx = np.sort(perm[:n_active])
    amplitudes = np.abs(rng.standard_normal(n_active)).astype(np.float32)

    input_vec = torch.zeros(num_units, dtype=torch.float32)
    input_vec[active_idx] = torch.from_numpy(amplitudes)
    return input_vec, active_idx, amplitudes


# ---------------------------------------------------------------------------
# Run one forward simulation per solver
# ---------------------------------------------------------------------------

def run_solver(solver: str, input_vec: torch.Tensor) -> dict:
    """Build a fresh `srnn-std-e-only` cell with the given solver, run
    forward T steps with constant `input_vec` step, return trajectories.
    """
    n_E = NUM_UNITS // 2  # hard-coded in SRNNCell.n_E property

    cfg = replace(
        SRNN_PRESETS["srnn-std-e-only"],
        num_units=NUM_UNITS,
        solver=solver,
        h=H,
        ode_unfolds=ODE_UNFOLDS,
        std_zero_floor=STD_ZERO_FLOOR,
    )

    # Reset seed *before* cell construction so all three solvers see
    # identical W_raw / random parameter init. Without this, parameter
    # randomness confounds the solver comparison.
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Mask shape (N,): cell internally reshapes to (N, 1) for broadcasting.
    # All-ones means every output neuron is allowed to receive input.
    W_in_mask = torch.ones(NUM_UNITS)

    # Build RMTMatrix to get the recurrent W initial values + sparsity mask.
    rmt = RMTMatrix(n=NUM_UNITS, density=1.0 / 3.0, seed=SEED, level_of_chaos=1.0)
    rmt.build()
    rmt_export = rmt.export_for_srnn(dales=cfg.dales)

    cell = SRNNCell(config=cfg, input_size=NUM_UNITS, rmt_export=rmt_export,
                     W_in_mask=W_in_mask)

    # Override W_in with identity for clean per-neuron input drive.
    assert cell.W_in.shape == (NUM_UNITS, NUM_UNITS), (
        f"W_in shape {tuple(cell.W_in.shape)} != ({NUM_UNITS}, {NUM_UNITS})"
    )
    cell.W_in.data.copy_(torch.eye(NUM_UNITS))
    cell.eval()

    # Sanity-check state layout for std-e-only mode.
    state = cell.init_state(batch_size=B)
    assert state.shape == (B, n_E + NUM_UNITS), (
        f"state shape {tuple(state.shape)} != ({B}, {n_E + NUM_UNITS})"
    )
    # b_E init should be 1.0
    assert torch.allclose(state[:, :n_E], torch.ones(B, n_E)), \
        "b_E init is not 1.0 — state layout assumption may be wrong"

    b_E_traj = torch.zeros(T, n_E)
    x_E_traj = torch.zeros(T, n_E)
    r_E_traj = torch.zeros(T, n_E)

    inp = input_vec.unsqueeze(0)  # (1, num_units)

    # Reconstruct b_min for the readout-side rescaling. b_min depends on the
    # tau params, which are constant during this run, so compute once.
    tau_rec_E = cell._tau_b_rec_E().detach()  # (1, n_E) or (n_E,)
    tau_rel_E = cell._tau_b_rel_E().detach()
    b_min_E = (tau_rel_E / (tau_rec_E + tau_rel_E)).reshape(-1)[:n_E]

    with torch.no_grad():
        for t in range(T):
            _out, state = cell(inp, state)
            b_E = state[0, :n_E]
            x = state[0, n_E:n_E + NUM_UNITS]
            x_E = x[:n_E]
            # Reconstruct firing rate diagnostic. The cell uses:
            #     r = piecewise_sigmoid(x_eff - a_0)
            # where x_eff = x for std-only (no SFA subtraction). a_0 init = 0.
            # This may diverge from the cell's internal r if a_0 has drifted,
            # but it's correct for fresh init.
            a_0 = cell.a_0.detach()
            r_E = piecewise_sigmoid(x_E - a_0[:n_E])
            b_E_traj[t] = b_E
            x_E_traj[t] = x_E
            r_E_traj[t] = r_E

    # b_full reconstruction (what gates W_out): rescaled when flag on,
    # equals raw b_E when flag off (since std_E_mask=1 for std-e-only).
    if STD_ZERO_FLOOR:
        b_full_traj = (b_E_traj - b_min_E) / (1.0 - b_min_E)
    else:
        b_full_traj = b_E_traj.clone()

    return {
        "b_E": b_E_traj.numpy(),
        "b_full": b_full_traj.numpy(),
        "x_E": x_E_traj.numpy(),
        "r_E": r_E_traj.numpy(),
        "b_min_E": float(b_min_E.mean()),
        "n_E": n_E,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(results: dict, active_idx, amplitudes, out_path: str):
    n_solvers = len(SOLVERS)
    fig, axes = plt.subplots(3, n_solvers, figsize=(5.5 * n_solvers, 11.0),
                              sharex=True, sharey="row")

    t_axis = np.arange(T) * H

    # Color driven neurons by amplitude (viridis: low=dark, high=bright).
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=amplitudes.min(), vmax=amplitudes.max())

    # Pick a few non-driven E neurons (random) to overlay in faint gray.
    rng = np.random.default_rng(SEED + 1)
    n_E = next(iter(results.values()))["n_E"]
    nondriven_pool = np.setdiff1d(np.arange(n_E), active_idx)
    n_show_nondriven = min(10, len(nondriven_pool))
    nondriven_idx = rng.choice(nondriven_pool, size=n_show_nondriven, replace=False)

    for col, solver in enumerate(SOLVERS):
        res = results[solver]
        b = res["b_E"]
        bf = res["b_full"]
        r = res["r_E"]
        b_min = res["b_min_E"]

        ax_b = axes[0, col]
        ax_bf = axes[1, col]
        ax_r = axes[2, col]

        # Non-driven (gray) — under
        for j in nondriven_idx:
            ax_b.plot(t_axis, b[:, j], color="0.85", lw=0.5, alpha=0.6)
            ax_bf.plot(t_axis, bf[:, j], color="0.85", lw=0.5, alpha=0.6)
            ax_r.plot(t_axis, r[:, j], color="0.85", lw=0.5, alpha=0.6)

        # Driven (color by amplitude)
        for j_local, j in enumerate(active_idx):
            color = cmap(norm(amplitudes[j_local]))
            ax_b.plot(t_axis, b[:, j], color=color, lw=1.0, alpha=0.85)
            ax_bf.plot(t_axis, bf[:, j], color=color, lw=1.0, alpha=0.85)
            ax_r.plot(t_axis, r[:, j], color=color, lw=1.0, alpha=0.85)

        # Reference lines on raw b_E panel
        ax_b.axhline(0.5, color="red",  ls="--", lw=0.7, alpha=0.7,
                     label="b_E = 0.5 threshold")
        ax_b.axhline(b_min, color="blue", ls=":", lw=0.7, alpha=0.7,
                     label=f"b_E_ss at r=1 (= {b_min:.2f})")

        # Reference lines on b_full panel
        if STD_ZERO_FLOOR:
            ax_bf.axhline(0.0, color="blue", ls=":", lw=0.7, alpha=0.7,
                          label="b_full floor = 0 (zero_floor ON)")
        else:
            ax_bf.axhline(b_min, color="blue", ls=":", lw=0.7, alpha=0.7,
                          label=f"b_full floor = {b_min:.2f}")

        ax_b.set_ylim(0.0, 1.05)
        ax_bf.set_ylim(-0.05, 1.05)
        ax_r.set_ylim(-0.02, 1.05)
        flag_tag = " (zero_floor ON)" if STD_ZERO_FLOOR else ""
        ax_b.set_title(f"{solver}: raw b_E(t) state")
        ax_bf.set_title(f"{solver}: b_full(t) readout-side gain{flag_tag}")
        ax_r.set_title(f"{solver}: r_E(t)")
        ax_b.grid(alpha=0.25)
        ax_bf.grid(alpha=0.25)
        ax_r.grid(alpha=0.25)
        if col == 0:
            ax_b.set_ylabel("b_E")
            ax_bf.set_ylabel("b_full")
            ax_r.set_ylabel("r_E")
            ax_b.legend(loc="lower right", fontsize=7)
            ax_bf.legend(loc="lower right", fontsize=7)
        ax_r.set_xlabel("time (s)")

    # Colorbar for input amplitude
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.6,
                        location="right", pad=0.02)
    cbar.set_label("input step amplitude  |randn()|")

    fig.suptitle(
        f"STD b_E diagnostic — fresh srnn-std-e-only, identity W_in, step input "
        f"(N={NUM_UNITS}, n_E={n_E}, n_active={len(active_idx)}, h={H}s, "
        f"T={T_SECONDS:.1f}s)",
        fontsize=11,
    )
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(results: dict, active_idx):
    n_E = next(iter(results.values()))["n_E"]
    nondriven_pool = np.setdiff1d(np.arange(n_E), active_idx)
    print()
    print(f"Diagnostic summary  (driven neurons: {len(active_idx)}, "
          f"non-driven: {len(nondriven_pool)})")
    print(f"  Theoretical b_E_ss at r=1: 0.20  (tau_rel=0.25, tau_rec=1.0)")
    print()
    cols = ("solver", "min(b_E)_drv", "frac<0.5_drv@end",
            "mean_r_E_drv@end", "min(b_E)_nondrv", "max_r_E_nondrv@end")
    fmt = "  {:<16s} {:>12s} {:>16s} {:>16s} {:>16s} {:>20s}"
    print(fmt.format(*cols))
    print("  " + "-" * 100)
    for solver in SOLVERS:
        res = results[solver]
        b = res["b_E"]
        r = res["r_E"]
        b_drv = b[:, active_idx]
        r_drv = r[:, active_idx]
        b_non = b[:, nondriven_pool]
        r_non = r[:, nondriven_pool]
        row = (
            solver,
            f"{b_drv.min():.3f}",
            f"{(b_drv[-1] < 0.5).mean():.2f}",
            f"{r_drv[-1].mean():.3f}",
            f"{b_non.min():.3f}",
            f"{r_non[-1].max():.3f}",
        )
        print(fmt.format(*row))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    n_E = NUM_UNITS // 2

    input_vec, active_idx, amplitudes = make_input(
        NUM_UNITS, n_E, FRAC_ACTIVE, SEED
    )
    print(f"input: num_units={NUM_UNITS}, n_E={n_E}, n_active={len(active_idx)}")
    print(f"  amplitude stats: min={amplitudes.min():.3f}, "
          f"mean={amplitudes.mean():.3f}, max={amplitudes.max():.3f}")
    print(f"  active_idx (first 10): {active_idx[:10].tolist()}")

    results = {}
    for solver in SOLVERS:
        print(f"\n[{solver}] running {T} steps ({T_SECONDS:.1f}s) ...")
        results[solver] = run_solver(solver, input_vec)
        b_drv_min = results[solver]["b_E"][:, active_idx].min()
        print(f"  done. min(b_E) over driven neurons across whole trace: "
              f"{b_drv_min:.4f}")

    out_path = os.path.join(OUT_DIR, f"std_b_e_step{OUT_TAG}.png")
    plot_results(results, active_idx, amplitudes, out_path)
    print_summary(results, active_idx)


if __name__ == "__main__":
    main()
