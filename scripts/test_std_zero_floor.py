"""Synthetic saturation test for the std_zero_floor flag.

Builds two single-variant BatchedSRNNCells with identical weights — one with
``std_zero_floor=True``, one with ``False`` — drives them with a strong
constant input until firing rate saturates, and plots:

  (a) raw b_E(t)        — identical between flag on/off (same ODE, same params)
  (b) b_full(t)         — what the readout sees: rescaled to [0,1] when flag on,
                           bottoms at b_min ≈ 0.2 when flag off
  (c) br = b_full · r   — synaptic output, the quantity that gates W_out

Also asserts:
  • bit-identity of _compute_b_full vs legacy formula when flag=False
  • flag-OFF asymptote ≈ τ_rel/(τ_rec + τ_rel) ≈ 0.20 at IC values
  • flag-ON asymptote ≈ 0

Output: tmp/test_std_zero_floor.png
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from train_srnn.models.rmt_matrix import RMTMatrix  # noqa: E402
from train_srnn.models.srnn_cell import BatchedSRNNCell, SRNNConfig  # noqa: E402

OUT = REPO / "tmp" / "test_std_zero_floor.png"


def build_cell(std_zero_floor: bool, exp: dict) -> BatchedSRNNCell:
    cfg = SRNNConfig(
        num_units=32,
        n_a_E=0,
        n_a_I=0,
        n_b_E=1,
        n_b_I=0,
        std_zero_floor=std_zero_floor,
        solver="semi_implicit",
        h=0.02,
        ode_unfolds=4,
    )
    return BatchedSRNNCell([cfg], input_size=4, rmt_exports=[exp], W_in_mask=None)


def run(cell: BatchedSRNNCell, n_steps: int, u_amp: float):
    """Drive cell for n_steps with constant input u_amp; capture per-step
    diagnostics. Returns dict of (T,) numpy arrays for one neuron's trace."""
    state = cell.init_state(1)
    u = torch.full((1, 4), u_amp)

    b_E_trace = np.zeros(n_steps, dtype=np.float32)
    b_full_trace = np.zeros(n_steps, dtype=np.float32)
    br_trace = np.zeros(n_steps, dtype=np.float32)
    r_trace = np.zeros(n_steps, dtype=np.float32)

    with torch.no_grad():
        for t in range(n_steps):
            _, state = cell(u, state)
            diag = cell.get_diagnostics(state, u)
            # pick excitatory neuron 0 of variant 0
            b_E_trace[t] = float(diag["b_E"][0, 0, 0])
            b_full_trace[t] = float(diag["b_full"][0, 0, 0])
            br_trace[t] = float(diag["br"][0, 0, 0])
            r_trace[t] = float(diag["r"][0, 0, 0])

    return {
        "b_E": b_E_trace,
        "b_full": b_full_trace,
        "br": br_trace,
        "r": r_trace,
    }


def main():
    torch.manual_seed(0)
    rmt = RMTMatrix(n=32, density=0.5, seed=1, level_of_chaos=1.0)
    rmt.build()
    exp = rmt.export_for_srnn(dales=True)

    cell_off = build_cell(std_zero_floor=False, exp=exp)
    cell_on = build_cell(std_zero_floor=True, exp=exp)
    # Match weights exactly (same RMT seed already gives same buffers, but
    # synchronize trainable params via state-dict copy).
    cell_on.load_state_dict(cell_off.state_dict())

    # Isolate the b ODE from network feedback so the test reproduces the
    # closed-form analysis in `math/std_asymtote.md`:
    #   - zero W_raw so x → u at steady state (no recurrent contribution)
    #   - replace W_in with ones so every E neuron sees drive = N_in · u_amp,
    #     guaranteeing all of them saturate r = 1 (vs randn W_in which leaves
    #     some neurons under threshold)
    # Both cells get the same patched weights, so any differences in b_E
    # between flag-on/off can only come from the rescaling, which the
    # b ODE doesn't see in this isolated regime.
    with torch.no_grad():
        for c in (cell_off, cell_on):
            c.W_raw.zero_()
            c.W_in.fill_(1.0)
            c.W_raw_gain.fill_(1.0)
            c.W_in_gain.fill_(1.0)

    # Theoretical floor at IC values — single source of truth from the helpers
    tau_rec = cell_off._tau_b_rec_E()
    tau_rel = cell_off._tau_b_rel_E()
    b_min_theory = float((tau_rel / (tau_rec + tau_rel)).mean())
    print(f"theoretical b_min at IC = τ_rel/(τ_rec+τ_rel) = {b_min_theory:.4f}")

    # ── Test 1: bit-identity when flag=False ──────────────────────────────
    state = cell_off.init_state(1)
    u_dummy = torch.full((1, 4), 1.0)
    for _ in range(20):
        _, state = cell_off(u_dummy, state)
    a_E, a_I, b_E, b_I, x = cell_off.unpack_state(state)
    b_full_new = cell_off._compute_b_full(b_E, b_I)
    b_full_E_legacy = (
        b_E * cell_off.std_E_mask.unsqueeze(1)
        + (1.0 - cell_off.std_E_mask.unsqueeze(1))
    )
    b_full_I_legacy = (
        b_I * cell_off.std_I_mask.unsqueeze(1)
        + (1.0 - cell_off.std_I_mask.unsqueeze(1))
    )
    b_full_legacy = torch.cat([b_full_E_legacy, b_full_I_legacy], dim=-1)
    bit_identical = torch.equal(b_full_new, b_full_legacy)
    print(f"bit-identity new vs legacy (flag=False): {bit_identical}")
    assert bit_identical, "Default-off code path must be bit-identical to legacy"

    # ── Test 2: saturation trajectories ───────────────────────────────────
    n_steps = 1000
    u_amp = 5.0
    print(f"running {n_steps} steps with constant u={u_amp} ...")
    trace_off = run(cell_off, n_steps, u_amp)
    trace_on = run(cell_on, n_steps, u_amp)

    # Asymptotic minima (last 10% of trajectory)
    tail = slice(int(0.9 * n_steps), n_steps)
    b_full_min_off = float(trace_off["b_full"][tail].min())
    b_full_min_on = float(trace_on["b_full"][tail].min())
    b_E_min_off = float(trace_off["b_E"][tail].min())
    b_E_min_on = float(trace_on["b_E"][tail].min())

    print(f"flag OFF: b_E asymptote = {b_E_min_off:.4f} "
          f"(theory {b_min_theory:.4f}), b_full asymptote = {b_full_min_off:.4f}")
    print(f"flag ON:  b_E asymptote = {b_E_min_on:.4f} "
          f"(should match flag-OFF: same ODE), b_full asymptote = {b_full_min_on:.4f}")

    # b_E trajectories should agree closely (small drift only from the random
    # x-IC inside each cell's init_state; in this isolated regime the b ODE
    # depends on r only, which is the same for both flags).
    raw_diff = np.abs(trace_off["b_E"] - trace_on["b_E"]).max()
    print(f"max |b_E(off) - b_E(on)| over trajectory: {raw_diff:.3e}")

    # Floor assertions
    assert abs(b_full_min_off - b_min_theory) < 0.01, (
        f"flag-OFF floor {b_full_min_off:.4f} should match theory "
        f"{b_min_theory:.4f}"
    )
    assert b_full_min_on < 0.01, (
        f"flag-ON floor {b_full_min_on:.4f} should be near 0"
    )
    print("PASS: all assertions hold")

    # ── Plot ──────────────────────────────────────────────────────────────
    h = float(cell_off.h)
    t = np.arange(n_steps) * h

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

    axes[0].plot(t, trace_off["b_E"], color="C0", label="flag OFF (raw b_E)", lw=1.5)
    axes[0].plot(t, trace_on["b_E"], color="C1", label="flag ON (raw b_E)",
                 lw=1.0, ls="--")
    axes[0].axhline(b_min_theory, color="0.5", lw=0.8, ls=":",
                    label=f"b_min theory = {b_min_theory:.3f}")
    axes[0].set_ylabel("b_E(t)\n(state)")
    axes[0].set_title(
        "Test: identical b_E ODE under both flags (lines should overlap); "
        "rescaling only changes the readout"
    )
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].grid(alpha=0.3)

    axes[1].plot(t, trace_off["b_full"], color="C0", label="flag OFF", lw=1.5)
    axes[1].plot(t, trace_on["b_full"], color="C1", label="flag ON", lw=1.5)
    axes[1].axhline(b_min_theory, color="0.5", lw=0.8, ls=":",
                    label=f"b_min = {b_min_theory:.3f}")
    axes[1].axhline(0.0, color="0.7", lw=0.6, ls=":")
    axes[1].set_ylabel("b_full(t)\n(readout-side gain)")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend(loc="upper right", fontsize=9)
    axes[1].grid(alpha=0.3)

    axes[2].plot(t, trace_off["br"], color="C0", label="flag OFF: br = b_full·r", lw=1.5)
    axes[2].plot(t, trace_on["br"], color="C1", label="flag ON: br = b_full·r", lw=1.5)
    axes[2].plot(t, trace_on["r"], color="0.4", lw=0.8, ls="--", label="r(t)")
    axes[2].set_ylabel("synaptic output")
    axes[2].set_xlabel("time (s)")
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].legend(loc="upper right", fontsize=9)
    axes[2].grid(alpha=0.3)

    fig.suptitle(
        f"std_zero_floor synthetic saturation test "
        f"(constant input u={u_amp}, {n_steps} steps × h={h}s)",
        fontsize=11,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=120)
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
