"""Extract all log_tau_* parameters from srnn_e_only checkpoints and plot.

Produces two figures:
  1. tau_global-only  (single scalar over epochs)
  2. tau_x and tau_global * tau_x  (side-by-side) for each tau family.
"""
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

CKPTS = [
    ("init",     "tmp/cmp-20ep/srnn_e_only_ckpts/init.pt",      -1),
    ("epoch 0",  "tmp/cmp-20ep/srnn_e_only_ckpts/epoch_000.pt",  0),
    ("epoch 10", "tmp/cmp-20ep/srnn_e_only_ckpts/epoch_010.pt", 10),
    ("epoch 19", "tmp/cmp-20ep/srnn_e_only_ckpts/last.pt",      19),
]


def load_sd(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        return ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    return ckpt.state_dict()


# Collect state dicts.
epochs = [c[2] for c in CKPTS]
labels = [c[0] for c in CKPTS]
sds = [load_sd(p) for _, p, _ in CKPTS]

# Find all log_tau_* keys (stripping the top-level wrapper like "cell.").
tau_keys = sorted({k for sd in sds for k in sd if "log_tau" in k})
print("Discovered tau keys:")
for k in tau_keys:
    shape = tuple(sds[0][k].shape)
    print(f"  {k}   shape={shape}")
print()


def tau_value(sd, key):
    """Return softplus(log_tau_*), returning a scalar or 1-D array."""
    t = sd[key]
    val = F.softplus(t).detach().cpu().numpy()
    return val


# Extract tau_global scalar across checkpoints.
tg_key = [k for k in tau_keys if k.endswith("log_tau_global")][0]
tau_global_vals = np.array([tau_value(sd, tg_key).item() for sd in sds])
print("tau_global over epochs:", tau_global_vals)
print()

# Organize remaining tau keys into families.
families = [
    ("tau_d",         [k for k in tau_keys if k.endswith("log_tau_d")]),
    ("tau_a_E (lo/hi)", [k for k in tau_keys if "log_tau_a_E_lo" in k or "log_tau_a_E_hi" in k]),
    ("tau_b_rec_E",   [k for k in tau_keys if k.endswith("log_tau_b_rec_E")]),
    ("tau_b_rel_E",   [k for k in tau_keys if k.endswith("log_tau_b_rel_E")]),
]

n_families = len([f for f in families if f[1]])
fig, axes = plt.subplots(n_families, 2, figsize=(11, 3.2 * n_families),
                         squeeze=False)
fig.suptitle("srnn_e_only on sMNIST — tau parameters over training\n"
             "left: tau_x (per-family)    right: effective tau = tau_global × tau_x",
             fontsize=12, y=1.0)

row_i = 0
for fam_name, keys in families:
    if not keys:
        continue
    ax_raw = axes[row_i, 0]
    ax_eff = axes[row_i, 1]
    for key in keys:
        # Gather values across checkpoints.
        vals_per_ckpt = [tau_value(sd, key) for sd in sds]
        # If parameter has vector shape, each checkpoint gives a vector;
        # plot each element as a line (muted colour).
        arr = np.stack([np.atleast_1d(v) for v in vals_per_ckpt])  # (n_ckpt, n)
        n_elements = arr.shape[1]
        short_name = key.split(".")[-1]
        for j in range(n_elements):
            raw = arr[:, j]
            eff = raw * tau_global_vals
            tag = short_name if n_elements == 1 else f"{short_name}[{j}]"
            ax_raw.plot(epochs, raw, marker="o", lw=1.2, alpha=0.85,
                        label=tag if j < 4 else None)
            ax_eff.plot(epochs, eff, marker="o", lw=1.2, alpha=0.85,
                        label=tag if j < 4 else None)

    for ax, ylabel in [(ax_raw, f"{fam_name} (s)"),
                       (ax_eff, f"effective {fam_name} (s)")]:
        ax.set_xticks([-1, 0, 10, 19])
        ax.set_xticklabels(labels, rotation=0, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best")
    row_i += 1

axes[-1, 0].set_xlabel("epoch")
axes[-1, 1].set_xlabel("epoch")
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("tmp/cmp-20ep/all_taus.png", dpi=120, bbox_inches="tight")
print("saved tmp/cmp-20ep/all_taus.png")
