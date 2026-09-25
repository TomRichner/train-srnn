"""Network activity on a test trace: predictions, rates, SFA, STD and synaptic output.

    uv run python scripts/plot_activity.py <run_dir> --data-root "$SRNN_HOME/data" \
        --test cheetah100_warp_multi/test.npz --checkpoints init,last --out <dir>

For one seed per condition (``--seed``), the checkpoint's networks are driven by the
z-scored test trace in teacher-forcing mode, after ``--warmup-s`` seconds of the same
trace, and ``--dur`` seconds are plotted: target and prediction for two channels, the
rates ``r``, the STD product ``prod(b)``, the SFA feedback ``(c/K) sum(a)`` and the
dendritic state ``x`` of a few E and I neurons, plus the distribution of synaptic output
``r * prod(b)`` over all neurons and plotted samples. Writes ``activity_<ckpt>.png``,
``synaptic_<ckpt>.png`` and ``activity_<ckpt>.json`` (population statistics).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from _runs import rebuild_model  # noqa: E402
from eval_speed import train_stats  # noqa: E402

COLORS = {"no-adapt": "#555555", "sfa1-std1": "#0072B2", "sfa3-std2": "#DAA520",
          "sfa3-std1": "#009E73", "sfa1-std2": "#CC79A7"}
CHANNELS = (2, 11)   # back-thigh angle and its angular velocity


def condition(name: str) -> str:
    return re.match(r"srnn-(.*)-seed\d+$", name).group(1)


@torch.no_grad()
def simulate(model, x: torch.Tensor, ks: list[int], n_show: int):
    """Step the cell over ``x`` (T, C); record diagnostics of variants ``ks``."""
    cell = model.cell
    state = model.initial_state(1)
    W_eff = cell.hoist()
    n_E = cell.n_E
    show = list(range(n_show)) + list(range(n_E, n_E + n_show))
    rec = {k: [] for k in ("pred", "r", "b", "sfa", "x", "br_all", "r_all")}
    for t in range(x.shape[0]):
        out, state = cell(x[t:t + 1], state, W_eff)
        pred = model.apply_readout(out, x[t:t + 1])[:, 0]            # (K, O)
        d = cell.get_diagnostics(state)
        c_E = (cell._c("E") / cell.sfa_E_count)                      # (K, n_E)
        c_I = (cell._c("I") / cell.sfa_I_count)
        sfa = torch.cat([c_E * (cell.sfa_E_mask * d["a_E"][:, 0]).sum(-1),
                         c_I * (cell.sfa_I_mask * d["a_I"][:, 0]).sum(-1)], dim=-1)
        rec["pred"].append(pred[ks])
        rec["r"].append(d["r"][ks, 0][:, show])
        rec["b"].append(d["b_full"][ks, 0][:, show])
        rec["sfa"].append(sfa[ks][:, show])
        rec["x"].append(d["x"][ks, 0][:, show])
        rec["br_all"].append(d["br"][ks, 0])
        rec["r_all"].append(d["r"][ks, 0])
    return {k: torch.stack(v, dim=1).cpu().numpy() for k, v in rec.items()}   # (k, T, ...)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--test", default="cheetah100_warp_multi/test.npz")
    p.add_argument("--checkpoints", default="init,last")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--warmup-s", type=float, default=10.0)
    p.add_argument("--dur", type=float, default=6.0)
    p.add_argument("--start-s", type=float, default=30.0, help="where in the trace the warm-up begins")
    p.add_argument("--n-show", type=int, default=6, help="E and I neurons plotted per network")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    for tag in args.checkpoints.split(","):
        ckpt = torch.load(args.run_dir / f"{tag}.pt", map_location="cpu", weights_only=False)
        model, cfg, names = rebuild_model(ckpt, device=args.device)
        ks = [i for i, n in enumerate(names) if n.endswith(f"-seed{args.seed}")]
        conds = [condition(names[k]).replace("-no-dales", "") for k in ks]
        palette = plt.get_cmap("tab10")
        colors = {c: COLORS.get(c, palette(i % 10)) for i, c in enumerate(conds)}
        mu, sd = train_stats(cfg, args.data_root)
        with np.load(args.data_root / args.test) as z:
            obs = z["obs"].astype(np.float32)
            rate = z["rate"] if "rate" in z.files else np.ones(len(obs))
        fs = cfg.task.sample_rate_hz
        i0, n_warm, n_plot = int(args.start_s * fs), int(args.warmup_s * fs), int(args.dur * fs)
        seg = (obs[i0:i0 + n_warm + n_plot + 1] - mu) / sd
        x = torch.tensor(seg, device=args.device)
        rec = simulate(model, x[:-1], ks, args.n_show)
        keep = slice(n_warm, n_warm + n_plot)
        t = np.arange(n_plot) / fs
        target = seg[1:][keep]

        rows = ["prediction", "r", "prod(b)", "SFA (c/K)Σa", "x"]
        fig, axes = plt.subplots(len(rows), len(ks), figsize=(3.0 * len(ks), 11), sharex=True,
                                 squeeze=False)
        n = args.n_show
        for j, (k, cond) in enumerate(zip(ks, conds)):
            ax = axes[0][j]
            for ci, ls in zip(CHANNELS, ("-", "--")):
                ax.plot(t, target[:, ci], color="k", lw=0.8, ls=ls)
                ax.plot(t, rec["pred"][j][keep][:, ci], color=colors[cond], lw=1, ls=ls)
            mse = float(((rec["pred"][j][keep] - target) ** 2).mean())
            ax.set_title(f"{cond}\nMSE {mse:.3f} (plotted {args.dur:g} s)", fontsize=9)
            for row, key in enumerate(("r", "b", "sfa", "x"), start=1):
                ax = axes[row][j]
                vals = rec[key][j][keep]
                ax.plot(t, vals[:, :n], color="#C0392B", lw=0.6, alpha=0.8)
                ax.plot(t, vals[:, n:], color="#2471A3", lw=0.6, alpha=0.8)
        for row, label in enumerate(rows):
            axes[row][0].set_ylabel(label)
        axes[2][0].set_ylim(-0.02, 1.02)
        for ax in axes[-1]:
            ax.set_xlabel("time (s)")
        r_seg = rate[i0 + n_warm:i0 + n_warm + n_plot]
        fig.suptitle(f"{args.run_dir.name} {tag}.pt on {args.test} (rate {r_seg.min():.2f}-"
                     f"{r_seg.max():.2f}x); black = target ch {CHANNELS[0]} (solid), "
                     f"ch {CHANNELS[1]} (dashed); red E, blue I", fontsize=9)
        fig.tight_layout()
        fig.savefig(args.out / f"activity_{tag}.png", dpi=110)
        plt.close(fig)

        stats = {}
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
        bins = np.logspace(-4, 0, 60)
        for j, (k, cond) in enumerate(zip(ks, conds)):
            br = rec["br_all"][j][keep].ravel()
            r = rec["r_all"][j][keep].ravel()
            stats[cond] = {"mean_r": float(r.mean()), "median_r": float(np.median(r)),
                           "mean_br": float(br.mean()), "median_br": float(np.median(br)),
                           "frac_r_saturated": float((r > 0.99).mean()),
                           "frac_r_silent": float((r < 0.01).mean())}
            axes[0].hist(np.clip(br, 1e-4, 1), bins=bins, histtype="step", density=True,
                         color=colors[cond], label=cond)
            axes[1].hist(r, bins=50, range=(0, 1), histtype="step", density=True,
                         color=colors[cond], label=cond)
        axes[0].set_xscale("log")
        axes[0].set_xlabel("synaptic output r·Πb")
        axes[1].set_xlabel("rate r")
        axes[0].legend(fontsize=7)
        fig.suptitle(f"{args.run_dir.name} {tag}.pt, seed {args.seed}: all neurons, plotted window",
                     fontsize=9)
        fig.tight_layout()
        fig.savefig(args.out / f"synaptic_{tag}.png", dpi=110)
        plt.close(fig)
        (args.out / f"activity_{tag}.json").write_text(json.dumps(stats, indent=1) + "\n")
        print(tag, json.dumps({c: {k: round(v, 3) for k, v in s.items()} for c, s in stats.items()}))


if __name__ == "__main__":
    main()
