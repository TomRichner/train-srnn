"""Test RMTMatrix: verify spectral properties, sparsity, and export round-trip."""
import numpy as np
import matplotlib.pyplot as plt
from train_srnn.models.rmt_matrix import RMTMatrix

# ── Build with MATLAB defaults ─────────────────────────────────────────────

rmt = RMTMatrix(n=300, f=0.5, indegree=100, seed=42, level_of_chaos=1.0)
W = rmt.build()

print(rmt.summary())
print()

# ── Spectral radius check ─────────────────────────────────────────────────

rho_empirical = rmt.spectral_radius
R_theory = rmt.R
print(f"Spectral radius: empirical={rho_empirical:.4f}, theory={R_theory:.4f}, "
      f"ratio={rho_empirical/R_theory:.3f}")

# ── Sparsity check ────────────────────────────────────────────────────────

alpha = rmt.alpha
S_mean = rmt.S.mean()
print(f"Sparsity: alpha={alpha:.4f}, S.mean={S_mean:.4f}")

# ── Dale's law check ──────────────────────────────────────────────────────

n_E = rmt.n_E
E_col_mean = W[:, :n_E].mean()
I_col_mean = W[:, n_E:].mean()
print(f"Dale's: E col mean={E_col_mean:.4f} (>0), I col mean={I_col_mean:.4f} (<0)")

# ── Export round-trip (dales=True) ─────────────────────────────────────────

import torch
import torch.nn.functional as F

export = rmt.export_for_srnn(dales=True)
W_init = export["W_init"]
sparsity_mask = export["sparsity_mask"]
dales_sign = export["dales_sign"]

# Reconstruct: W_eff = dales_sign * softplus(W_init) * sparsity
W_pos = F.softplus(W_init)
W_signed = W_pos * dales_sign[None, :]  # broadcast sign across rows
W_reconstructed = (W_signed * sparsity_mask).numpy()

# Compare to original W
max_err = np.max(np.abs(W_reconstructed - W.astype(np.float32)))
mean_err = np.mean(np.abs(W_reconstructed - W.astype(np.float32)))
print(f"\nExport round-trip (dales=True):")
print(f"  max |W_reconstructed - W| = {max_err:.2e}")
print(f"  mean |W_reconstructed - W| = {mean_err:.2e}")

# ── Export round-trip (dales=False) ────────────────────────────────────────

export_nod = rmt.export_for_srnn(dales=False)
W_init_nod = export_nod["W_init"]
sparsity_mask_nod = export_nod["sparsity_mask"]

# For dales=False, W_init holds real values; sparsity already in W but mask separate
# Reconstruction: W_eff = W_init * sparsity_mask (if not baked in)
# Actually W already has sparsity baked in from build, but the mask is separate for training
# Just verify the mask matches
S_from_export = sparsity_mask_nod.numpy()
S_from_build = rmt.S.astype(np.float32)
assert np.allclose(S_from_export, S_from_build), "Sparsity mask mismatch!"
print(f"\nExport (dales=False): sparsity mask matches ✓")

# ── Seed determinism ──────────────────────────────────────────────────────

rmt2 = RMTMatrix(n=300, f=0.5, indegree=100, seed=42, level_of_chaos=1.0)
W2 = rmt2.build()
assert np.allclose(W, W2), "Seed determinism failed!"
print(f"Seed determinism: ✓")

# ── Plot ───────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Spectrum
rmt.plot_spectrum(ax=axes[0])

# Weight histogram by quadrant
n_E = rmt.n_E
W_EE = W[:n_E, :n_E]
W_EI = W[:n_E, n_E:]
W_IE = W[n_E:, :n_E]
W_II = W[n_E:, n_E:]

axes[1].hist(W_EE[W_EE != 0].flatten(), bins=60, color="red", alpha=0.5, label="E→E")
axes[1].hist(W_EI[W_EI != 0].flatten(), bins=60, color="blue", alpha=0.5, label="I→E")
axes[1].hist(W_IE[W_IE != 0].flatten(), bins=60, color="orange", alpha=0.5, label="E→I")
axes[1].hist(W_II[W_II != 0].flatten(), bins=60, color="cyan", alpha=0.5, label="I→I")
axes[1].set_title("Non-zero weight distributions")
axes[1].legend(fontsize=8)
axes[1].axvline(0, color="k", ls="--", lw=0.5)

# Weight matrix image
im = axes[2].imshow(W, cmap="RdBu_r", vmin=-np.percentile(np.abs(W), 98),
                     vmax=np.percentile(np.abs(W), 98), aspect="auto")
axes[2].axhline(n_E - 0.5, color="k", lw=0.5)
axes[2].axvline(n_E - 0.5, color="k", lw=0.5)
axes[2].set_title("W matrix (E|I)")
plt.colorbar(im, ax=axes[2], shrink=0.8)

plt.tight_layout()
plt.savefig("scripts/test_rmt_matrix.png", dpi=150)
print(f"\nSaved scripts/test_rmt_matrix.png")
plt.show()
