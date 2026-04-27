"""Plot the LR schedule recorded in training_history.csv (overnight9h run)."""
import csv
from pathlib import Path
import matplotlib.pyplot as plt

RUN_DIR = Path(__file__).resolve().parents[2] / "tmp" / "overnight9h"

rows = list(csv.DictReader(open(RUN_DIR / "training_history.csv")))
# all variants share the same schedule; take the first variant
v0 = rows[0]["variant"]
ep, lr = [], []
for r in rows:
    if r["variant"] != v0:
        continue
    ep.append(int(r["epoch"]))
    lr.append(float(r["lr"]))

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(ep, lr, marker="o", ms=3); axes[0].set_title("LR schedule (linear)")
axes[1].semilogy(ep, lr, marker="o", ms=3); axes[1].set_title("LR schedule (semilogy)")
for ax in axes:
    ax.set_xlabel("epoch"); ax.set_ylabel("lr"); ax.grid(alpha=0.3, which="both")
plt.tight_layout()
out = RUN_DIR / "lr_schedule.png"
plt.savefig(out, dpi=120)
print(f"wrote {out}  (peak={max(lr):.3e}, final={lr[-1]:.3e})")
