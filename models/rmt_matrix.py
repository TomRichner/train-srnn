"""rmt_matrix.py — Random Matrix Theory weight matrix generator.

Python port of MATLAB RMTMatrix + RMTConnectivity (Harris et al. 2023).
Generates recurrent weight matrices with E/I population structure,
controlled sparsity, and precise spectral radius via RMT theory.

Usage:
    rmt = RMTMatrix(n=300, indegree=100, seed=42)
    W = rmt.build()
    export = rmt.export_for_srnn(dales=True)
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch


def _softplus_inv_np(x: np.ndarray) -> np.ndarray:
    """Inverse of softplus in numpy: log(exp(x) - 1). Requires x > 0."""
    return np.log(np.expm1(x))


class RMTMatrix:
    """Random Matrix Theory weight matrix generator (Harris et al. 2023).

    Combines MATLAB's RMTMatrix (core W construction) and RMTConnectivity
    (parameter defaults, level_of_chaos scaling, abscissa rescaling).

    Parameters
    ----------
    n : int
        Network size.
    f : float
        Fraction of excitatory neurons (default 0.5).
    indegree : int or None
        Expected in-degree per neuron. None → fully connected (indegree=n).
    seed : int
        RNG seed for reproducibility.
    level_of_chaos : float
        Global scaling for W (default 1.0).
    mu_E_tilde, mu_I_tilde : float or None
        Normalized population means (tilde notation). None → defaults.
    sigma_E_tilde, sigma_I_tilde : float or None
        Normalized population std devs. None → defaults.
    E_W : float
        Mean offset added to both mu_E_tilde and mu_I_tilde.
    zrs_mode : str
        Zero-row-sum mode: 'none', 'ZRS', 'SZRS', 'Partial_SZRS'.
    rescale_by_abscissa : bool
        If True, rescale W so spectral abscissa = level_of_chaos.
    """

    VALID_ZRS_MODES = ("none", "ZRS", "SZRS", "Partial_SZRS")

    def __init__(
        self,
        n: int,
        f: float = 0.5,
        indegree: Optional[int] = None,
        seed: int = 42,
        level_of_chaos: float = 1.0,
        mu_E_tilde: Optional[float] = None,
        mu_I_tilde: Optional[float] = None,
        sigma_E_tilde: Optional[float] = None,
        sigma_I_tilde: Optional[float] = None,
        E_W: float = 0.0,
        zrs_mode: str = "none",
        rescale_by_abscissa: bool = False,
    ):
        if zrs_mode not in self.VALID_ZRS_MODES:
            raise ValueError(
                f"Invalid zrs_mode '{zrs_mode}'. Valid: {self.VALID_ZRS_MODES}"
            )

        self.n = n
        self.f = f
        self.indegree = indegree if indegree is not None else n
        self.seed = seed
        self.level_of_chaos = level_of_chaos
        self.E_W = E_W
        self.zrs_mode = zrs_mode
        self.rescale_by_abscissa = rescale_by_abscissa

        # Compute normalization factor F
        alph = self.indegree / self.n
        F = 1.0 / math.sqrt(self.n * alph * (2.0 - alph))

        # Set tilde parameters (defaults from RMTConnectivity.build)
        self.mu_E_tilde = mu_E_tilde if mu_E_tilde is not None else 3.0 * F
        self.mu_I_tilde = mu_I_tilde if mu_I_tilde is not None else -4.0 * F
        self.sigma_E_tilde = sigma_E_tilde if sigma_E_tilde is not None else F
        self.sigma_I_tilde = sigma_I_tilde if sigma_I_tilde is not None else F

        # Outputs (populated by build())
        self.W: Optional[np.ndarray] = None
        self.A: Optional[np.ndarray] = None
        self.S: Optional[np.ndarray] = None
        self._is_built = False

    # ── Properties ─────────────────────────────────────────────────────────

    @property
    def n_E(self) -> int:
        """Number of excitatory neurons."""
        return round(self.f * self.n)

    @property
    def n_I(self) -> int:
        """Number of inhibitory neurons."""
        return self.n - self.n_E

    @property
    def alpha(self) -> float:
        """Connection probability = indegree / n."""
        return self.indegree / self.n

    @property
    def F(self) -> float:
        """Normalization factor F = 1 / sqrt(N * alpha * (2 - alpha))."""
        alph = self.alpha
        return 1.0 / math.sqrt(self.n * alph * (2.0 - alph))

    # ── Sparse statistics (Harris 2023 Eq 15-16) ──────────────────────────

    @property
    def mu_se(self) -> float:
        """Sparse excitatory mean."""
        return self.alpha * (self.mu_E_tilde + self.E_W)

    @property
    def mu_si(self) -> float:
        """Sparse inhibitory mean."""
        return self.alpha * (self.mu_I_tilde + self.E_W)

    @property
    def sigma_se_sq(self) -> float:
        """Sparse excitatory variance."""
        alph = self.alpha
        mu_eff = self.mu_E_tilde + self.E_W
        return alph * (1.0 - alph) * mu_eff**2 + alph * self.sigma_E_tilde**2

    @property
    def sigma_si_sq(self) -> float:
        """Sparse inhibitory variance."""
        alph = self.alpha
        mu_eff = self.mu_I_tilde + self.E_W
        return alph * (1.0 - alph) * mu_eff**2 + alph * self.sigma_I_tilde**2

    # ── Theoretical predictions (Harris 2023 Eq 17-18) ─────────────────────

    @property
    def lambda_O(self) -> float:
        """Theoretical outlier eigenvalue."""
        return self.n * (self.f * self.mu_se + (1.0 - self.f) * self.mu_si)

    @property
    def R(self) -> float:
        """Theoretical spectral radius (before level_of_chaos scaling)."""
        return math.sqrt(
            self.n * (self.f * self.sigma_se_sq + (1.0 - self.f) * self.sigma_si_sq)
        ) * self.level_of_chaos

    # ── Empirical spectral properties (after build) ────────────────────────

    @property
    def spectral_radius(self) -> float:
        """Empirical spectral radius of W."""
        self._assert_built()
        return float(np.max(np.abs(np.linalg.eigvals(self.W))))

    @property
    def spectral_abscissa(self) -> float:
        """Empirical spectral abscissa max(Re(eig(W)))."""
        self._assert_built()
        return float(np.max(np.real(np.linalg.eigvals(self.W))))

    # ── Build ──────────────────────────────────────────────────────────────

    def build(self) -> np.ndarray:
        """Generate the weight matrix W.

        Returns
        -------
        W : np.ndarray (n, n)
            The recurrent weight matrix.
        """
        rng = np.random.default_rng(self.seed)

        n = self.n
        n_E = self.n_E
        alph = self.alpha

        # Base Gaussian matrix
        self.A = rng.standard_normal((n, n))

        # Sparsity mask
        if alph < 1.0:
            self.S = (rng.random((n, n)) < alph).astype(np.float64)
        else:
            self.S = np.ones((n, n), dtype=np.float64)

        # Variance structure D = diag(sigma_tilde)
        D_vec = np.empty(n)
        D_vec[:n_E] = self.sigma_E_tilde
        D_vec[n_E:] = self.sigma_I_tilde

        # Low-rank mean structure M = u @ v.T
        v = np.empty(n)
        v[:n_E] = self.mu_E_tilde + self.E_W
        v[n_E:] = self.mu_I_tilde + self.E_W
        u = np.ones(n)

        # A @ D (equivalent to A * D_vec[None, :] for diagonal D)
        AD = self.A * D_vec[np.newaxis, :]
        M = u[:, np.newaxis] * v[np.newaxis, :]  # outer product

        # Apply ZRS mode
        if self.zrs_mode == "none":
            W_dense = AD + M
            W_rmt = self.S * W_dense

        elif self.zrs_mode == "ZRS":
            if alph < 1.0:
                import warnings
                warnings.warn(
                    "Using ZRS with sparse matrix destroys sparsity. Consider SZRS.",
                    stacklevel=2,
                )
            P = np.eye(n) - np.ones((n, n)) / n
            W_rmt = AD @ P + M
            if alph < 1.0:
                W_rmt = self.S * W_rmt

        elif self.zrs_mode == "SZRS":
            W_base = self.S * (AD + M)
            row_sums = W_base.sum(axis=1)
            row_counts = self.S.sum(axis=1)
            row_counts[row_counts == 0] = 1.0
            W_bar_i = row_sums / row_counts
            B = self.S * W_bar_i[:, np.newaxis]
            W_rmt = W_base - B

        elif self.zrs_mode == "Partial_SZRS":
            J_base = self.S * AD
            M_base = self.S * M
            J_row_sums = J_base.sum(axis=1)
            row_counts = self.S.sum(axis=1)
            row_counts[row_counts == 0] = 1.0
            J_bar_i = J_row_sums / row_counts
            B_partial = self.S * J_bar_i[:, np.newaxis]
            W_rmt = (J_base - B_partial) + M_base

        # Apply level_of_chaos scaling
        self.W = self.level_of_chaos * W_rmt

        # Optional abscissa rescaling
        if self.rescale_by_abscissa:
            eigs = np.linalg.eigvals(self.W)
            abscissa_0 = np.max(np.real(eigs))
            if abs(abscissa_0) > np.finfo(float).eps:
                self.W = self.W * (self.level_of_chaos / abscissa_0)

        self._is_built = True
        return self.W

    # ── Export for SRNNCell ─────────────────────────────────────────────────

    def export_for_srnn(self, dales: bool = True) -> dict:
        """Export W decomposed into SRNNCell-compatible tensors.

        Parameters
        ----------
        dales : bool
            If True, decompose W into softplus-inverse magnitudes + sign mask.
            If False, return W directly with real signed values.

        Returns
        -------
        dict with:
            'W_init' : torch.Tensor (N, N)
                In softplus-inverse space (dales=True) or real values (dales=False).
            'sparsity_mask' : torch.Tensor (N, N)
                Binary 0/1 mask. Always provided regardless of dales.
            'dales_sign' : torch.Tensor (N,)
                +1 for E columns, -1 for I columns.
            'n_E' : int
        """
        self._assert_built()

        sparsity_mask = torch.tensor(self.S, dtype=torch.float32)

        dales_sign = torch.ones(self.n)
        dales_sign[self.n_E:] = -1.0

        if dales:
            # Decompose: W_eff = sign * softplus(W_init) * sparsity
            W_abs = np.abs(self.W)
            W_abs = np.maximum(W_abs, 1e-7)  # clamp for softplus_inv safety
            W_init = torch.tensor(
                _softplus_inv_np(W_abs), dtype=torch.float32
            )
        else:
            # No softplus — store real signed values; sparsity applied separately
            W_init = torch.tensor(self.W, dtype=torch.float32)

        return {
            "W_init": W_init,
            "sparsity_mask": sparsity_mask,
            "dales_sign": dales_sign,
            "n_E": self.n_E,
        }

    # ── Plotting ───────────────────────────────────────────────────────────

    def plot_spectrum(self, ax=None):
        """Plot eigenvalue spectrum with theoretical Girko circle overlay.

        Parameters
        ----------
        ax : matplotlib Axes or None
            If None, creates a new figure.

        Returns
        -------
        ax : matplotlib Axes
        """
        self._assert_built()
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(6, 6))

        eigs = np.linalg.eigvals(self.W)

        ax.scatter(eigs.real, eigs.imag, s=3, alpha=0.6, color="steelblue", label="eigenvalues")

        # Theoretical Girko circle (radius R)
        theta = np.linspace(0, 2 * np.pi, 200)
        R = self.R
        ax.plot(R * np.cos(theta), R * np.sin(theta), "r--", lw=1.5,
                label=f"R = {R:.3f} (theory)")

        # Mark outlier
        ax.axvline(self.lambda_O, color="green", ls=":", lw=1,
                   label=f"λ_O = {self.lambda_O:.3f}")

        ax.set_xlabel("Re(λ)")
        ax.set_ylabel("Im(λ)")
        ax.set_title(
            f"RMTMatrix spectrum (N={self.n}, α={self.alpha:.2f}, "
            f"ρ={self.spectral_radius:.3f})"
        )
        ax.set_aspect("equal")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        return ax

    # ── Summary ────────────────────────────────────────────────────────────

    def summary(self) -> str:
        """Return a formatted summary string."""
        lines = [
            f"RMTMatrix(n={self.n}, f={self.f}, indegree={self.indegree})",
            f"  alpha = {self.alpha:.4f}",
            f"  F = {self.F:.6f}",
            f"  mu_E_tilde = {self.mu_E_tilde:.6f}, mu_I_tilde = {self.mu_I_tilde:.6f}",
            f"  sigma_E_tilde = {self.sigma_E_tilde:.6f}, sigma_I_tilde = {self.sigma_I_tilde:.6f}",
            f"  E_W = {self.E_W}",
            f"  level_of_chaos = {self.level_of_chaos}",
            f"  zrs_mode = '{self.zrs_mode}'",
            f"  Theoretical R = {self.R:.4f}",
            f"  Theoretical lambda_O = {self.lambda_O:.4f}",
        ]
        if self._is_built:
            lines.append(f"  Empirical spectral radius = {self.spectral_radius:.4f}")
            lines.append(f"  Empirical spectral abscissa = {self.spectral_abscissa:.4f}")
            lines.append(f"  Sparsity (S.mean) = {self.S.mean():.4f}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"RMTMatrix(n={self.n}, f={self.f}, indegree={self.indegree}, "
            f"level_of_chaos={self.level_of_chaos}, built={self._is_built})"
        )

    # ── Internal ───────────────────────────────────────────────────────────

    def _assert_built(self):
        if not self._is_built:
            raise RuntimeError("RMTMatrix has not been built. Call .build() first.")
