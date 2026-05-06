"""Extract and plot isp_tau_global (as effective tau_global via softplus)
across the 4 available checkpoints of srnn_e_only on cmp-20ep."""
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

CKPTS = [
    ("init",       "tmp/cmp-20ep/srnn_e_only_ckpts/init.pt",     None),
    ("epoch 0",    "tmp/cmp-20ep/srnn_e_only_ckpts/epoch_000.pt", 0),
    ("epoch 10",   "tmp/cmp-20ep/srnn_e_only_ckpts/epoch_010.pt", 10),
    ("epoch 19",   "tmp/cmp-20ep/srnn_e_only_ckpts/last.pt",      19),
]

def get_isp_tau_global(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    # State dict may be nested under "model_state_dict" or flat.
    if isinstance(ckpt, dict):
        sd = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    else:
        sd = ckpt.state_dict()
    # Find any key ending with 'isp_tau_global'.
    matches = [k for k in sd.keys() if k.endswith("isp_tau_global")]
    assert len(matches) == 1, f"expected one match, got {matches}"
    return sd[matches[0]].item()

rows = []
for name, path, ep in CKPTS:
    lg = get_isp_tau_global(path)
    tau = F.softplus(torch.tensor(lg)).item()
    rows.append((name, ep, lg, tau))
    print(f"{name:10s}  isp_tau_global={lg:+.4f}  softplus→tau_global={tau:.4f}")

# Plot: x=epoch (use -1 for 'init'), y=effective tau_global
xs = [-1 if ep is None else ep for _, ep, _, _ in rows]
ys = [tau for _, _, _, tau in rows]
labels = [r[0] for r in rows]

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(xs, ys, marker="o", lw=2)
for x, y, lab in zip(xs, ys, labels):
    ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(8, 4), fontsize=9)
ax.set_xlabel("epoch")
ax.set_ylabel("effective tau_global (softplus of parameter)")
ax.set_title("srnn_e_only — tau_global over training (cmp-20ep, seed 1)")
ax.axhline(1.0, ls="--", color="gray", alpha=0.5, label="init value = 1.0")
ax.set_xticks([-1, 0, 10, 19])
ax.set_xticklabels(["init", "0", "10", "19"])
ax.grid(alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig("tmp/cmp-20ep/tau_global.png", dpi=120)
print("\nsaved tmp/cmp-20ep/tau_global.png")
