"""Random recurrent connectivity with E/I structure and a controlled spectrum.

Implements the construction of Harris, Meffin, Burkitt & Peterson (2023),
"Effect of sparsity on network stability in random neural networks obeying
Dale's law", Phys. Rev. Research 5, 043132: W = S * (A D + u v^T), where A
is standard Gaussian, D scales the E and I columns, u v^T sets the column
means, and S is a Bernoulli sparsity mask. The population means and
standard deviations are given in units of F = 1 / sqrt(N alpha (2 - alpha)),
which keeps the bulk spectral radius near one at every density.

    rmt = RMTMatrix(n=300, density=1/3, seed=42)
    W = rmt.build()
    export = rmt.export_for_srnn(dales=True)
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch


def _softplus_inv_np(x: np.ndarray) -> np.ndarray:
    return np.log(np.expm1(x))


class RMTMatrix:
    """Sample one recurrent matrix.

    Args:
        n: network size.
        f: fraction of excitatory neurons.
        indegree / density: expected in-degree, or connection probability
            (give one; default fully connected).
        seed: RNG seed.
        level_of_chaos: overall scale of W.
        mu_E_tilde, mu_I_tilde, sigma_E_tilde, sigma_I_tilde: population
            means and standard deviations; defaults 3F, -4F, F, F.
        E_W: offset added to both means.
    """

    def __init__(self, n: int, f: float = 0.5, indegree: Optional[int] = None,
                 density: Optional[float] = None, seed: int = 42, level_of_chaos: float = 1.0,
                 mu_E_tilde: Optional[float] = None, mu_I_tilde: Optional[float] = None,
                 sigma_E_tilde: Optional[float] = None, sigma_I_tilde: Optional[float] = None,
                 E_W: float = 0.0):
        if density is not None and indegree is not None:
            raise ValueError("Specify density or indegree, not both")
        self.n, self.f, self.seed = n, f, seed
        self.indegree = round(density * n) if density is not None else (indegree if indegree is not None else n)
        self.level_of_chaos, self.E_W = level_of_chaos, E_W
        F = self.F
        self.mu_E_tilde = mu_E_tilde if mu_E_tilde is not None else 3.0 * F
        self.mu_I_tilde = mu_I_tilde if mu_I_tilde is not None else -4.0 * F
        self.sigma_E_tilde = sigma_E_tilde if sigma_E_tilde is not None else F
        self.sigma_I_tilde = sigma_I_tilde if sigma_I_tilde is not None else F
        self.W: Optional[np.ndarray] = None
        self.A: Optional[np.ndarray] = None
        self.S: Optional[np.ndarray] = None

    @property
    def n_E(self) -> int:
        return int(self.f * self.n)   # truncates like the cell's n_E = N // 2 at f = 0.5

    @property
    def n_I(self) -> int:
        return self.n - self.n_E

    @property
    def alpha(self) -> float:
        """Connection probability."""
        return self.indegree / self.n

    @property
    def F(self) -> float:
        a = self.alpha
        return 1.0 / math.sqrt(self.n * a * (2.0 - a))

    # Sparse-population statistics and the resulting spectrum (Harris et al. 2023, eqs. 15-18).

    @property
    def mu_se(self) -> float:
        return self.alpha * (self.mu_E_tilde + self.E_W)

    @property
    def mu_si(self) -> float:
        return self.alpha * (self.mu_I_tilde + self.E_W)

    @property
    def sigma_se_sq(self) -> float:
        a, mu = self.alpha, self.mu_E_tilde + self.E_W
        return a * (1.0 - a) * mu ** 2 + a * self.sigma_E_tilde ** 2

    @property
    def sigma_si_sq(self) -> float:
        a, mu = self.alpha, self.mu_I_tilde + self.E_W
        return a * (1.0 - a) * mu ** 2 + a * self.sigma_I_tilde ** 2

    @property
    def lambda_O(self) -> float:
        """Predicted outlier eigenvalue."""
        return self.n * (self.f * self.mu_se + (1.0 - self.f) * self.mu_si)

    @property
    def R(self) -> float:
        """Predicted bulk spectral radius."""
        return math.sqrt(self.n * (self.f * self.sigma_se_sq + (1.0 - self.f) * self.sigma_si_sq)) * self.level_of_chaos

    @property
    def spectral_radius(self) -> float:
        self._assert_built()
        return float(np.max(np.abs(np.linalg.eigvals(self.W))))

    @property
    def spectral_abscissa(self) -> float:
        self._assert_built()
        return float(np.max(np.real(np.linalg.eigvals(self.W))))

    def build(self) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        n, n_E, a = self.n, self.n_E, self.alpha
        self.A = rng.standard_normal((n, n))
        self.S = (rng.random((n, n)) < a).astype(np.float64) if a < 1.0 else np.ones((n, n))
        D = np.empty(n)
        D[:n_E], D[n_E:] = self.sigma_E_tilde, self.sigma_I_tilde
        v = np.empty(n)
        v[:n_E], v[n_E:] = self.mu_E_tilde + self.E_W, self.mu_I_tilde + self.E_W
        AD = self.A * D[np.newaxis, :]
        M = np.ones(n)[:, np.newaxis] * v[np.newaxis, :]
        self.W = self.level_of_chaos * (self.S * (AD + M))
        return self.W

    def export_for_srnn(self, dales: bool = True, dales_init: bool = True) -> dict:
        """Tensors for ``SRNNCell``: ``W_init``, ``sparsity_mask`` (N, N), ``dales_sign`` (N,).

        With ``dales`` the magnitude |W| is stored in inverse-softplus space and
        the column sign separately, so the effective weight is
        ``sign * softplus(W_init) * mask``; entries whose sampled sign disagrees
        with their column are flipped. Without enforcement ``W_init`` stores signed weights directly.
        ``dales_init`` independently projects the initial matrix onto column signs.
        """
        self._assert_built()
        dales_sign = torch.ones(self.n)
        dales_sign[self.n_E:] = -1.0
        initial = np.abs(self.W) * dales_sign.numpy()[None, :] if dales_init else self.W
        if dales:
            W_init = torch.tensor(_softplus_inv_np(np.maximum(np.abs(initial), 1e-7)), dtype=torch.float32)
        else:
            W_init = torch.tensor(initial, dtype=torch.float32)
        return {"W_init": W_init, "sparsity_mask": torch.tensor(self.S, dtype=torch.float32),
                "dales_sign": dales_sign, "n_E": self.n_E, "dales": dales}

    def summary(self) -> str:
        lines = [f"RMTMatrix(n={self.n}, f={self.f}, indegree={self.indegree}, "
                 f"level_of_chaos={self.level_of_chaos})",
                 f"  alpha={self.alpha:.4f}  F={self.F:.6f}  predicted R={self.R:.4f}  "
                 f"lambda_O={self.lambda_O:.4f}"]
        if self.W is not None:
            lines.append(f"  spectral radius={self.spectral_radius:.4f}  "
                         f"abscissa={self.spectral_abscissa:.4f}  density={self.S.mean():.4f}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (f"RMTMatrix(n={self.n}, f={self.f}, indegree={self.indegree}, "
                f"level_of_chaos={self.level_of_chaos}, built={self.W is not None})")

    def _assert_built(self) -> None:
        if self.W is None:
            raise RuntimeError("call build() first")
