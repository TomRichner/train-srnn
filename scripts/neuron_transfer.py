"""Small-signal transfer function of one SRNN neuron: input drive to synaptic output.

    uv run python scripts/neuron_transfer.py --out <dir> [--r0 0.25] [--time-scale 1,0.25]

Linearized around a constant operating rate ``r0`` (no recurrence), with ``s = 2 pi i f``:

    dendrite   X/U     = 1 / (1 + s tau_d)
    SFA        R/X     = g / (1 + g (c/K) sum_k 1 / (1 + s tau_a_k))
    STD        Theta/R = P0 [1 - r0 sum_m (1/tau_rel_m) / (s + lambda_m)],
               lambda_m = 1/tau_rec_m + r0/tau_rel_m,  b_m0 = 1/(1 + r0 tau_rec_m/tau_rel_m)

``g`` is the activation slope (1 in the linear range of ``piecewise_sigmoid``) and ``P0``
the steady depression product (``geo`` takes its M-th root and scales the STD term by
1/M). Plots gain and phase of Theta/U for several adaptation configurations and time
scalings, with the cheetah gait fundamental (2.42 Hz) and harmonics marked. A phase
that stays roughly constant over a band is the fractional-order (constant phase
advance) signature that a multiple-timescale ladder can produce.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

GAIT_HZ = 2.42
LADDER3 = (0.25, float(np.sqrt(0.25 * 10.0)), 10.0)


def transfer(f, *, tau_d=0.1, sfa=(), c=0.5, std=(), r0=0.25, g=1.0, geo=False):
    s = 2j * np.pi * np.asarray(f)
    H = 1.0 / (1.0 + s * tau_d)
    if sfa:
        H = H * g / (1.0 + g * (c / len(sfa)) * sum(1.0 / (1.0 + s * t) for t in sfa))
    else:
        H = H * g
    if std:
        b0 = [1.0 / (1.0 + r0 * rec / rel) for rec, rel in std]
        P0 = float(np.prod(b0))
        term = sum((1.0 / rel) / (s + 1.0 / rec + r0 / rel) for rec, rel in std)
        if geo:
            H = H * P0 ** (1.0 / len(std)) * (1.0 - r0 * term / len(std))
        else:
            H = H * P0 * (1.0 - r0 * term)
    return H


def configs(scale: float) -> dict:
    k = scale
    ladder = tuple(t * k for t in LADDER3)
    one = ((2.0 * k, 0.25 * k),)
    two = ((2.0 * k, 0.25 * k), (4.0 * k, 0.5 * k))
    return {
        "no adaptation": dict(),
        "SFA1 / STD1": dict(sfa=(0.25 * k,), std=one),
        "SFA3 / STD2 (product)": dict(sfa=ladder, std=two),
        "SFA3 / STD2 geo": dict(sfa=ladder, std=two, geo=True),
        "SFA3 / STD1": dict(sfa=ladder, std=one),
        "SFA3 only": dict(sfa=ladder),
        "SFA1 only": dict(sfa=(0.25 * k,)),
        "STD2 only (product)": dict(std=two),
        "STD1 only": dict(std=one),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--r0", type=float, default=0.25)
    p.add_argument("--time-scale", default="1,0.25", help="factors applied to every adaptation time")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    f = np.logspace(-3, np.log10(50), 400)
    scales = [float(x) for x in args.time_scale.split(",")]
    colors = dict(zip(configs(1).keys(), ["#555555", "#0072B2", "#DAA520", "#E69F00", "#009E73",
                                          "#56B4E9", "#9ACD32", "#CC79A7", "#AA4499"]))
    fig, axes = plt.subplots(2, len(scales), figsize=(6 * len(scales), 7), sharex=True, squeeze=False)
    summary = {}
    for col, k in enumerate(scales):
        rows = {}
        for name, kw in configs(k).items():
            H = transfer(f, r0=args.r0, **kw)
            axes[0][col].loglog(f, np.abs(H), color=colors[name], label=name)
            phase = np.degrees(np.angle(H))
            axes[1][col].semilogx(f, phase, color=colors[name], label=name)
            Hg = transfer([GAIT_HZ, 2 * GAIT_HZ], r0=args.r0, **kw)
            rows[name] = {"gain_gait": float(abs(Hg[0])), "phase_gait_deg": float(np.degrees(np.angle(Hg[0]))),
                          "phase_2gait_deg": float(np.degrees(np.angle(Hg[1]))),
                          "max_phase_lead_deg": float(phase.max()),
                          "f_max_lead_hz": float(f[phase.argmax()])}
        summary[f"time_scale_{k:g}"] = rows
        for ax in axes[:, col]:
            for h in (1, 2, 3):
                ax.axvline(GAIT_HZ * h, color="0.7", lw=0.8, ls=":" if h > 1 else "-")
            ax.grid(alpha=0.3, which="both")
        axes[0][col].set_title(f"adaptation times x {k:g} (r0 = {args.r0})", fontsize=10)
        axes[1][col].set_xlabel("frequency (Hz); grey = gait 2.42 Hz and harmonics")
    axes[0][0].set_ylabel("|synaptic output / input|")
    axes[1][0].set_ylabel("phase (deg); > 0 = lead")
    axes[1][-1].legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(args.out / "neuron_transfer.png", dpi=120)
    (args.out / "neuron_transfer.json").write_text(json.dumps(summary, indent=1) + "\n")
    for k, rows in summary.items():
        print(k)
        for name, r in rows.items():
            print(f"  {name:24s} gain@gait {r['gain_gait']:.3f} phase@gait {r['phase_gait_deg']:+6.1f} "
                  f"@2gait {r['phase_2gait_deg']:+6.1f}  max lead {r['max_phase_lead_deg']:+5.1f} at "
                  f"{r['f_max_lead_hz']:.3f} Hz")


if __name__ == "__main__":
    main()
