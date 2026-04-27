"""Plot LSTM training curves: loss, val accuracy, learning rate."""
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("tmp/lstm_training.csv")

fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)

ax = axes[0]
ax.plot(df["epoch"], df["train_loss"], label="train", lw=1.5)
ax.plot(df["epoch"], df["valid_loss"], label="valid", lw=1.5)
ax.set_ylabel("loss")
ax.set_title("LSTM on sMNIST (N=32, seed=1, 50 epochs)")
ax.legend()
ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(df["epoch"], df["train_metric"], label="train acc", lw=1.5)
ax.plot(df["epoch"], df["valid_metric"], label="valid acc", lw=1.5)
ax.set_ylabel("accuracy")
ax.set_ylim(0, 1)
ax.legend()
ax.grid(alpha=0.3)

ax = axes[2]
ax.plot(df["epoch"], df["lr"], color="tab:green", lw=1.5)
ax.set_ylabel("learning rate")
ax.set_xlabel("epoch")
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("tmp/lstm_curves.png", dpi=120)
print("saved tmp/lstm_curves.png")
print(f"\nepochs: {len(df)}")
print(f"final train_loss={df.iloc[-1]['train_loss']:.3f} valid_loss={df.iloc[-1]['valid_loss']:.3f}")
print(f"final train_acc={df.iloc[-1]['train_metric']:.3f} valid_acc={df.iloc[-1]['valid_metric']:.3f}")
print(f"lr range: {df['lr'].min():.2e} → {df['lr'].max():.2e}")
