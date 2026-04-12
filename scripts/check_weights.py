"""Compare PyTorch SRNN weight distribution to MATLAB RMT reference."""
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from train_srnn.models.srnn_cell import SRNNConfig, SRNNCell
from train_srnn.models.rmt_matrix import RMTMatrix

torch.manual_seed(42)

rmt = RMTMatrix(n=300, density=1.0/3.0, seed=42, level_of_chaos=1.0)
rmt.build()
export = rmt.export_for_srnn(dales=True)

cfg = SRNNConfig(num_units=300, n_a_E=3, n_a_I=0, n_b_E=1, n_b_I=0, dales=True)
cell = SRNNCell(cfg, input_size=300, rmt_export=export)

with torch.no_grad():
    W = cell._effective_W().numpy()

N_E = 150
W_EE = W[:N_E, :N_E]
W_EI = W[:N_E, N_E:]
W_IE = W[N_E:, :N_E]
W_II = W[N_E:, N_E:]

print("=== PyTorch W statistics ===")
print(f"W shape: {W.shape}")
print(f"W_EE (E->E): mean={W_EE.mean():.4f}, std={W_EE.std():.4f}, range=[{W_EE.min():.4f}, {W_EE.max():.4f}]")
print(f"W_EI (I->E): mean={W_EI.mean():.4f}, std={W_EI.std():.4f}, range=[{W_EI.min():.4f}, {W_EI.max():.4f}]")
print(f"W_IE (E->I): mean={W_IE.mean():.4f}, std={W_IE.std():.4f}, range=[{W_IE.min():.4f}, {W_IE.max():.4f}]")
print(f"W_II (I->I): mean={W_II.mean():.4f}, std={W_II.std():.4f}, range=[{W_II.min():.4f}, {W_II.max():.4f}]")
print(f"Spectral radius: {np.max(np.abs(np.linalg.eigvals(W))):.4f}")
print(f"Sparsity: {(W == 0).sum() / W.size:.2%}")
print(f"Frobenius norm: {np.linalg.norm(W, 'fro'):.4f}")

print()
print("=== MATLAB RMT reference (n=300, indegree=100, loc=1.0) ===")
alpha = 100 / 300
R_theory = np.sqrt(alpha) * 1.0
print(f"Theoretical spectral radius R = sqrt(alpha)*loc = {R_theory:.4f}")
print(f"Column std = 1/sqrt(indegree) = {1/np.sqrt(100):.4f}")

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle("PyTorch W_eff weight distributions (Dale's law)", fontsize=13)

axes[0, 0].hist(W_EE.flatten(), bins=80, color="red", alpha=0.7)
axes[0, 0].set_title(f"E->E (excitatory), mean={W_EE.mean():.3f}")
axes[0, 0].axvline(0, color="k", ls="--", lw=0.5)

axes[0, 1].hist(W_EI.flatten(), bins=80, color="blue", alpha=0.7)
axes[0, 1].set_title(f"I->E (inhibitory), mean={W_EI.mean():.3f}")
axes[0, 1].axvline(0, color="k", ls="--", lw=0.5)

axes[1, 0].hist(W_IE.flatten(), bins=80, color="orange", alpha=0.7)
axes[1, 0].set_title(f"E->I (excitatory), mean={W_IE.mean():.3f}")
axes[1, 0].axvline(0, color="k", ls="--", lw=0.5)

axes[1, 1].hist(W_II.flatten(), bins=80, color="cyan", alpha=0.7)
axes[1, 1].set_title(f"I->I (inhibitory), mean={W_II.mean():.3f}")
axes[1, 1].axvline(0, color="k", ls="--", lw=0.5)

plt.tight_layout()
plt.savefig("scripts/weight_histograms.png", dpi=150)
print(f"\nSaved scripts/weight_histograms.png")
plt.show()
