"""Replot grad_norms.csv by parameter group: readout / recurrent+input / dendritic / SFA / STD / global."""
import csv
from collections import defaultdict
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _runs import cache_dir  # noqa: E402

OUT = cache_dir() / "grad_probe"
rows = list(csv.DictReader(open(OUT / "grad_norms.csv")))

GROUPS = [
    ("Output (chunk-invariant)", [
        ("readout_weight", "C0"),
        ("readout_bias",   "C1"),
    ]),
    ("Input + recurrent weights", [
        ("cell.W_in",        "C0"),
        ("cell.W_in_gain",   "C1"),
        ("cell.W_raw",       "C2"),
        ("cell.log_W_raw_gain",  "C3"),
    ]),
    ("Dendritic / fast", [
        ("cell.log_tau_d_gain", "C0"),
        ("cell.isp_tau_d_vec",  "C1"),
        ("cell.a_0_scalar",     "C2"),
        ("cell.a_0_vec",        "C3"),
    ]),
    ("SFA E", [
        ("cell.log_tau_a_E_gain", "C0"),
        ("cell.isp_tau_a_E_vec",  "C1"),
        ("cell.log_c_E_gain",     "C2"),
        ("cell.isp_c_E_vec",      "C3"),
    ]),
    ("SFA I", [
        ("cell.log_tau_a_I_gain", "C0"),
        ("cell.isp_tau_a_I_vec",  "C1"),
        ("cell.log_c_I_gain",     "C2"),
        ("cell.isp_c_I_vec",      "C3"),
    ]),
    ("STD E", [
        ("cell.log_tau_b_rec_E_gain", "C0"),
        ("cell.isp_tau_b_rec_E_vec",  "C1"),
        ("cell.log_tau_b_rel_E_gain", "C2"),
        ("cell.isp_tau_b_rel_E_vec",  "C3"),
    ]),
    ("STD I", [
        ("cell.log_tau_b_rec_I_gain", "C0"),
        ("cell.isp_tau_b_rec_I_vec",  "C1"),
        ("cell.log_tau_b_rel_I_gain", "C2"),
        ("cell.isp_tau_b_rel_I_vec",  "C3"),
    ]),
]


def plot_for_variant(variant):
    fig, axes = plt.subplots(2, 4, figsize=(18, 9), squeeze=False)
    flat = axes.flatten()
    ymin, ymax = 1e-12, 0
    for ax, (title, params) in zip(flat, GROUPS):
        for pname, color in params:
            xs, ys = [], []
            for r in rows:
                if r["variant"] != variant or r["param"] != pname:
                    continue
                gn = float(r["grad_norm"])
                xs.append(int(r["chunk_len"]))
                ys.append(gn)
            if not xs:
                continue
            order = np.argsort(xs)
            xs = np.array(xs)[order]
            ys = np.array(ys)[order]
            label = pname.replace("cell.", "")
            # Plot only nonzero values; show zero-grad entries as a label note
            nz = ys > 0
            if nz.any():
                ax.loglog(xs[nz], ys[nz], "o-", color=color, label=label, lw=1.4, ms=4)
                ymax = max(ymax, ys[nz].max())
                ymin = min(ymin, ys[nz][ys[nz] > 0].min())
            else:
                ax.plot([], [], "o-", color=color, label=f"{label} (≡0, grad-masked)")
        ax.set_title(title); ax.set_xlabel("bptt_chunk_len"); ax.set_ylabel("|∇θ|₂")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7, loc="best")

    # hide unused panel
    for j in range(len(GROUPS), len(flat)):
        flat[j].axis("off")

    # uniform y-axis across panels (loglog)
    if ymin > 0 and ymax > 0:
        for ax in flat[: len(GROUPS)]:
            ax.set_ylim(ymin * 0.5, ymax * 2)

    fig.suptitle(f"|∇θ|₂ vs bptt_chunk_len, variant {variant}", fontsize=13)
    plt.tight_layout()
    out = OUT / f"grad_vs_chunk_groups_{variant}.png"
    plt.savefig(out, dpi=120); plt.close(fig)
    print(f"wrote {out}")


variants = sorted({r["variant"] for r in rows})
for v in variants:
    plot_for_variant(v)
