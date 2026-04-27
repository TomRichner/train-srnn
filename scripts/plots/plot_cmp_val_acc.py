"""Plot validation accuracy vs epoch for all 6 models on cmp-20ep smnist run."""
import pandas as pd
import matplotlib.pyplot as plt

MODELS = ["lstm", "ltc", "srnn", "srnn_no_adapt", "srnn_e_only", "srnn_e_only_echo"]

fig, ax = plt.subplots(figsize=(9, 5.5))
for m in MODELS:
    df = pd.read_csv(f"tmp/cmp-20ep/{m}.csv")
    ax.plot(df["epoch"], df["valid_metric"], marker="o", lw=1.5, label=m)

ax.set_xlabel("epoch")
ax.set_ylabel("validation accuracy")
ax.set_title("sMNIST val accuracy — cmp-20ep (seed 1, N=32, window_len=112, bptt_len=56)")
ax.set_ylim(0, 1)
ax.set_xticks(range(0, 21, 2))
ax.grid(alpha=0.3)
ax.legend(loc="upper left", fontsize=9)
plt.tight_layout()
plt.savefig("tmp/cmp-20ep/val_acc.png", dpi=120)
print("saved tmp/cmp-20ep/val_acc.png")
