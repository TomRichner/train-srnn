"""Open-loop simulation of SRNN variants under a step stimulus.

Runs the variants side by side in one SRNNCell with identity input weights
and plots synaptic output, depression, and adaptation per variant. Use it to
eyeball what an ablation does to the dynamics, or to compare solvers.

    python scripts/simulate_srnn.py                                   # default variant set
    python scripts/simulate_srnn.py --variants srnn,srnn-no-adapt --N 64
    python scripts/simulate_srnn.py --solvers rk4,semi_implicit --variants srnn-e-only
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from train_srnn import paths
from train_srnn.models import variants as V
from train_srnn.models.rmt_matrix import RMTMatrix
from train_srnn.models.srnn_cell import SRNNCell, SRNNConfig
from train_srnn.utils.stimulus import step_stimulus

DEFAULT_VARIANTS = ["srnn-no-adapt", "srnn-sfa1-std1", "srnn-sfa3-std2"]
BASE_FLAGS = dict(dales=True, n_a_E=3, n_a_I=3, n_b_E=2, n_b_I=2, per_neuron=False, echo=False, skip=False)

E_COLORS = np.array([[1.0, 0.0, 0.0], [1.0, 0.75, 0.0], [0.85, 0.2, 0.45], [0.9, 0.1, 0.6],
                     [0.9, 0.55, 0.0], [0.55, 0.27, 0.27], [0.86, 0.08, 0.24], [0.6, 0.15, 0.45]])
I_COLORS = np.array([[0.0, 0.45, 0.74], [0.0, 0.75, 1.0], [0.2, 0.47, 0.62], [0.0, 0.5, 0.5],
                     [0.3, 0.75, 0.93], [0.25, 0.62, 0.75], [0.0, 0.8, 0.8], [0.15, 0.55, 0.65]])
NEUTRAL_COLORS = np.array([[0.066, 0.443, 0.745], [0.866, 0.329, 0.0], [0.929, 0.694, 0.125],
                           [0.521, 0.086, 0.819], [0.231, 0.666, 0.196], [0.184, 0.745, 0.937]])


def build_cell(names: list[str], N: int, solver: str, h: float, seed: int, density: float) -> SRNNCell:
    configs = [SRNNConfig(num_units=N, solver=solver, h=h, ode_unfolds=1, init_seed=V.make_variant(n, BASE_FLAGS, seed).seed,
                          **{k: getattr(V.make_variant(n, BASE_FLAGS, seed), k) for k in V.FLAGS})
               for n in names]
    exports = []
    scale = 1 / math.sqrt(500 * .2 * 1.8)
    for cfg in configs:
        rmt = RMTMatrix(n=N, f=.5, density=density, seed=cfg.init_seed,
                        mu_E_tilde=7 * scale, mu_I_tilde=-7 * scale,
                        sigma_E_tilde=1.5 * scale, sigma_I_tilde=1.5 * scale)
        rmt.build()
        exports.append(rmt.export_for_srnn(dales=cfg.dales))
    cell = SRNNCell(configs, input_size=N, rmt_exports=exports)
    with torch.no_grad():
        cell.W_in.copy_(torch.eye(N).expand(cell.K, N, N))
    cell.variant_names = names
    return cell


@torch.no_grad()
def simulate(cell: SRNNCell, u: np.ndarray) -> dict[str, np.ndarray]:
    """Per-step diagnostics, each ``(K, T, ...)``."""
    torch.manual_seed(0)
    state = cell.init_state(1)
    hist = {k: [] for k in ("br", "b_full", "a_E", "a_I", "x")}
    for t in range(u.shape[0]):
        _, state = cell(torch.as_tensor(u[t], dtype=state.dtype, device=state.device).unsqueeze(0), state)
        d = cell.get_diagnostics(state)
        for k in hist:
            hist[k].append(d[k][:, 0].numpy())
    return {k: np.stack(v, axis=1) for k, v in hist.items()}


def plot_lines(ax, t, data, colors):
    for i in range(data.shape[0]):
        ax.plot(t, data[i], color=colors[i % len(colors)], lw=0.4, alpha=0.7)


def plot_variants(cell, hist, t, title, out: Path, ncols: int = 4):
    K, n_E = cell.K, cell.n_E
    panels = ["synaptic output", "depression", "mean SFA state"]
    rows = math.ceil(K / ncols)
    fig, axes = plt.subplots(rows * 3, ncols, figsize=(4.5 * ncols, 2.6 * rows * 3), squeeze=False)
    for k, name in enumerate(cell.variant_names):
        r0, c = 3 * (k // ncols), k % ncols
        dales = bool(cell.configs[k].dales)
        cE, cI = (E_COLORS, I_COLORS) if dales else (NEUTRAL_COLORS, NEUTRAL_COLORS)
        series = [(hist["br"][k, :, :n_E].T, hist["br"][k, :, n_E:].T),
                  (hist["b_full"][k, :, :n_E].T, hist["b_full"][k, :, n_E:].T),
                  (hist["a_E"][k, ..., :cell.configs[k].n_a_E].sum(-1).T / max(cell.configs[k].n_a_E, 1),
                   hist["a_I"][k, ..., :cell.configs[k].n_a_I].sum(-1).T / max(cell.configs[k].n_a_I, 1))]
        for p, (label, (dE, dI)) in enumerate(zip(panels, series)):
            ax = axes[r0 + p, c]
            plot_lines(ax, t, dI, cI)
            plot_lines(ax, t, dE, cE)
            if p == 0:
                ax.set_title(name, fontsize=10)
            if c == 0:
                ax.set_ylabel(label, fontsize=9)
            if p != 2:
                ax.set_ylim(0, 1)
            ax.tick_params(labelsize=7)
        axes[r0 + 2, c].set_xlabel("time (s)", fontsize=8)
    for k in range(K, rows * ncols):
        for p in range(3):
            axes[3 * (k // ncols) + p, k % ncols].set_visible(False)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    ap.add_argument("--solvers", default="sra1", help="comma-separated; one figure per solver")
    ap.add_argument("--N", type=int, default=500)
    ap.add_argument("--T", type=float, default=50.0, help="simulated seconds")
    ap.add_argument("--fs", type=float, default=400.0, help="steps per second")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--density", type=float, default=0.2)
    ap.add_argument("--out-dir", type=Path, default=paths.cache_dir() / "simulations")
    args = ap.parse_args()

    names = [s.strip() for s in args.variants.split(",") if s.strip()]
    h = 1.0 / args.fs
    n_steps = int(args.T * args.fs)
    u = step_stimulus(args.N, args.N // 2, n_steps)
    t = np.arange(n_steps) * h
    for solver in [s.strip() for s in args.solvers.split(",")]:
        cell = build_cell(names, args.N, solver, h, args.seed, args.density)
        hist = simulate(cell, u)
        assert np.isfinite(hist["br"]).all(), f"{solver}: non-finite synaptic output"
        for k, name in enumerate(names):
            print(f"{solver:14s} {name:28s} br in [{hist['br'][k].min():.3f}, {hist['br'][k].max():.3f}]")
        plot_variants(cell, hist, t, f"SRNN variants, N={args.N}, {solver}, h={h:.4f} s",
                      args.out_dir / f"variants_N{args.N}_{solver}.png")


if __name__ == "__main__":
    main()
