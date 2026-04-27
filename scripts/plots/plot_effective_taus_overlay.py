"""Single combined plot: all effective taus (tau_global * softplus(log_tau_x))
on one axis, log-y, so their relative speeds are directly comparable."""
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
    return ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt)) \
        if isinstance(ckpt, dict) else ckpt.state_dict()


epochs = [c[2] for c in CKPTS]
labels = [c[0] for c in CKPTS]
sds = [load_sd(p) for _, p, _ in CKPTS]

tg = np.array([F.softplus(sd["cell.log_tau_global"]).item() for sd in sds])

def eff(key):
    """Return effective tau = tau_global * softplus(log_tau_x) scalar per checkpoint."""
    return np.array([
        (F.softplus(sd[key]).mean().item()) * tg[i]
        for i, sd in enumerate(sds)
    ])

series = {
    "tau_global":                  tg,
    "tau_d (membrane)":             eff("cell.log_tau_d"),
    "tau_a_E_lo (SFA short)":       eff("cell.log_tau_a_E_lo"),
    "tau_a_E_hi (SFA long)":        eff("cell.log_tau_a_E_hi"),
    "tau_b_rec_E (STD recovery)":   eff("cell.log_tau_b_rec_E"),
    "tau_b_rel_E (STD release)":    eff("cell.log_tau_b_rel_E"),
}

fig, ax = plt.subplots(figsize=(9.5, 5.5))
colors = {
    "tau_global":                  "black",
    "tau_d (membrane)":             "tab:blue",
    "tau_a_E_lo (SFA short)":       "tab:green",
    "tau_a_E_hi (SFA long)":        "tab:olive",
    "tau_b_rec_E (STD recovery)":   "tab:red",
    "tau_b_rel_E (STD release)":    "tab:orange",
}
styles = {
    "tau_global": "--",
}
for name, vals in series.items():
    ax.plot(epochs, vals, marker="o", lw=2,
            color=colors[name], ls=styles.get(name, "-"), label=name)

# Reference lines for scale context
ax.axhline(0.02, color="gray", ls=":", alpha=0.6)
ax.text(19.2, 0.02, "h=0.02s", color="gray", fontsize=8, va="bottom")
ax.axhline(56 * 0.02, color="gray", ls=":", alpha=0.6)
ax.text(19.2, 56 * 0.02, "bptt = 1.12s", color="gray", fontsize=8, va="bottom")

ax.set_yscale("log")
ax.set_xlabel("epoch")
ax.set_ylabel("effective tau (seconds)")
ax.set_title("srnn_e_only — all effective taus (= tau_global × tau_x)\n"
             "sMNIST, seed 1, cmp-20ep, N=32")
ax.set_xticks([-1, 0, 10, 19])
ax.set_xticklabels(labels)
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
plt.tight_layout()
plt.savefig("tmp/cmp-20ep/effective_taus_overlay.png", dpi=120, bbox_inches="tight")
print("saved tmp/cmp-20ep/effective_taus_overlay.png")
print()
for name, vals in series.items():
    print(f"  {name:45s}  init={vals[0]:.4f}  final={vals[-1]:.4f}  ratio={vals[-1]/vals[0]:.2f}")
