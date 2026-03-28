"""PyTorch SRNN cell with batched ablation support via bmm.

Port from TensorFlow 1.x. Key innovation: multiple SRNN ablation variants
can be tiled into GPU-parallel compute using torch.bmm(), enabling efficient
ablation studies without sequential evaluation.

All forward-pass code is torch.compile-friendly (no Python-level branching
on tensor values).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def inv_softplus(x: float) -> float:
    """Inverse of softplus: log(exp(x) - 1)."""
    return math.log(math.expm1(x))


def piecewise_sigmoid(x: torch.Tensor, S_a: float = 0.9, S_c: float = 0.0) -> torch.Tensor:
    """Bounded [0, 1] activation with linear center and quadratic edges.

    S_a controls the fraction of the output range that is linear (default 0.9
    means 90% linear, 10% quadratic rounding at edges).  Matches the TF 1.x
    ``piecewise_sigmoid`` from ``srnn_model.py``.

    Parameters
    ----------
    x : Tensor
        Input values.
    S_a : float
        Width of the linear region as a fraction of [0, 1] output range.
    S_c : float
        Center of the linear region.
    """
    a = S_a / 2.0
    c = S_c
    k = 0.5 / (1.0 - 2.0 * a) if abs(1.0 - 2.0 * a) > 1e-8 else 0.0

    x1 = c + a - 1.0   # left saturation boundary
    x2 = c - a          # left quadratic → linear transition
    x3 = c + a          # linear → right quadratic transition
    x4 = c + 1.0 - a   # right saturation boundary

    # Piecewise: 0 | quadratic rise | linear | quadratic cap | 1
    # All branches computed and blended via masks for torch.compile.
    quad_rise = k * (x - x1) * (x - x1)
    linear = (x - c) + 0.5
    quad_cap = 1.0 - k * (x - x4) * (x - x4)

    out = torch.where(x < x1, torch.zeros_like(x), quad_rise)
    out = torch.where(x >= x2, linear, out)
    out = torch.where(x > x3, quad_cap, out)
    out = torch.where(x > x4, torch.ones_like(x), out)
    return out


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SRNNConfig:
    """Configuration for a single SRNN variant."""

    num_units: int = 32
    dales: bool = True
    n_a_E: int = 3       # SFA timescales for E (0=off, 1=single, >=2=multi)
    n_a_I: int = 3       # SFA timescales for I
    n_b_E: int = 1       # STD for E (0=off, 1=on)
    n_b_I: int = 1       # STD for I
    per_neuron: bool = False
    echo: bool = False    # Reservoir mode (freeze W)
    solver: str = "semi_implicit"
    h: float = 0.04
    ode_unfolds: int = 6
    readout: str = "synaptic"
    sparsity: float = 0.5

    @property
    def n_E(self) -> int:
        return self.num_units // 2

    @property
    def n_I(self) -> int:
        return self.num_units - self.n_E

    @property
    def state_size(self) -> int:
        """Total flat state dimension."""
        return (
            self.n_E * self.n_a_E
            + self.n_I * self.n_a_I
            + self.n_E * self.n_b_E
            + self.n_I * self.n_b_I
            + self.num_units
        )


# ---------------------------------------------------------------------------
# Preset ablation variants
# ---------------------------------------------------------------------------

SRNN_PRESETS: dict[str, SRNNConfig] = {
    "srnn": SRNNConfig(),
    "srnn-per-neuron": SRNNConfig(per_neuron=True),
    "srnn-echo": SRNNConfig(echo=True),
    "srnn-no-adapt": SRNNConfig(n_a_E=0, n_a_I=0, n_b_E=0, n_b_I=0),
    "srnn-no-adapt-no-dales": SRNNConfig(dales=False, n_a_E=0, n_a_I=0, n_b_E=0, n_b_I=0),
    "srnn-sfa-only": SRNNConfig(n_a_E=1, n_a_I=1, n_b_E=0, n_b_I=0),
    "srnn-std-only": SRNNConfig(n_a_E=0, n_a_I=0, n_b_E=1, n_b_I=1),
    "srnn-E-only": SRNNConfig(n_a_E=1, n_a_I=0, n_b_E=1, n_b_I=0),
    "srnn-e-only-echo": SRNNConfig(n_a_E=1, n_a_I=0, n_b_E=1, n_b_I=0, echo=True),
    "srnn-e-only-per-neuron": SRNNConfig(n_a_E=1, n_a_I=0, n_b_E=1, n_b_I=0, per_neuron=True),
    "srnn-multi-sfa": SRNNConfig(n_a_E=2, n_a_I=2),
    "srnn-multi-sfa-E": SRNNConfig(n_a_E=2, n_a_I=0, n_b_E=1, n_b_I=0),
    "srnn-no-dales": SRNNConfig(dales=False),
    "srnn-explicit": SRNNConfig(solver="explicit"),
    "srnn-rk4": SRNNConfig(solver="rk4"),
}


# ---------------------------------------------------------------------------
# Single SRNNCell
# ---------------------------------------------------------------------------

class SRNNCell(nn.Module):
    """Single SRNN variant as an RNNCell-compatible module.

    State is a flat (batch, state_size) tensor packed as:
        [a_E_flat, a_I_flat, b_E, b_I, x]
    """

    def __init__(
        self,
        config: SRNNConfig,
        input_size: int,
        W_in_mask: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.config = config
        self.input_size = input_size
        N = config.num_units
        n_E = config.n_E
        n_I = config.n_I

        # Optional input mask: (N,) binary, applied to W_in rows
        if W_in_mask is not None:
            self.register_buffer("W_in_mask", W_in_mask.view(-1, 1))  # (N, 1)
        else:
            self.W_in_mask: Optional[torch.Tensor] = None

        # ---- Recurrent weight ----
        W_init = torch.randn(N, N) * math.sqrt(2.0 / N)
        if config.dales and config.sparsity > 0:
            mask = (torch.rand(N, N) > config.sparsity).float()
            self.register_buffer("sparsity_mask", mask)
            W_init = W_init * mask
        else:
            self.register_buffer("sparsity_mask", None)

        self.W_raw = nn.Parameter(W_init)
        if config.echo:
            self.W_raw.requires_grad_(False)

        # ---- Input weight ----
        self.W_in = nn.Parameter(torch.randn(N, input_size) * 0.1)

        # ---- Threshold ----
        self.a_0 = nn.Parameter(torch.full((N,), 0.35))

        # ---- Dendritic time constant ----
        tau_d_shape = (N,) if config.per_neuron else (1,)
        self.log_tau_d = nn.Parameter(torch.full(tau_d_shape, inv_softplus(0.1)))

        # ---- SFA parameters for E ----
        if config.n_a_E > 0:
            base_E = (n_E,) if config.per_neuron else (1,)
            if config.n_a_E == 1:
                self.log_tau_a_E = nn.Parameter(
                    torch.full((*base_E, 1), inv_softplus(1.0)))
                self.log_tau_a_E_lo = None
                self.log_tau_a_E_hi = None
            else:
                # Store lo/hi endpoints; interpolate n_a_E values at runtime
                self.log_tau_a_E = None
                self.log_tau_a_E_lo = nn.Parameter(
                    torch.full((*base_E, 1), inv_softplus(0.25)))
                self.log_tau_a_E_hi = nn.Parameter(
                    torch.full((*base_E, 1), inv_softplus(10.0)))
            c_E_shape = (n_E, config.n_a_E) if config.per_neuron else (1, config.n_a_E)
            self.log_c_E = nn.Parameter(torch.full(c_E_shape, -3.0))
            self.c_0_E = nn.Parameter(torch.zeros(c_E_shape))
        else:
            self.log_tau_a_E = None
            self.log_tau_a_E_lo = None
            self.log_tau_a_E_hi = None
            self.log_c_E = None
            self.c_0_E = None

        # ---- SFA parameters for I ----
        if config.n_a_I > 0:
            base_I = (n_I,) if config.per_neuron else (1,)
            if config.n_a_I == 1:
                self.log_tau_a_I = nn.Parameter(
                    torch.full((*base_I, 1), inv_softplus(1.0)))
                self.log_tau_a_I_lo = None
                self.log_tau_a_I_hi = None
            else:
                self.log_tau_a_I = None
                self.log_tau_a_I_lo = nn.Parameter(
                    torch.full((*base_I, 1), inv_softplus(0.25)))
                self.log_tau_a_I_hi = nn.Parameter(
                    torch.full((*base_I, 1), inv_softplus(10.0)))
            c_I_shape = (n_I, config.n_a_I) if config.per_neuron else (1, config.n_a_I)
            self.log_c_I = nn.Parameter(torch.full(c_I_shape, -3.0))
            self.c_0_I = nn.Parameter(torch.zeros(c_I_shape))
        else:
            self.log_tau_a_I = None
            self.log_tau_a_I_lo = None
            self.log_tau_a_I_hi = None
            self.log_c_I = None
            self.c_0_I = None

        # ---- STD parameters for E ----
        if config.n_b_E > 0:
            b_shape = (n_E,) if config.per_neuron else (1,)
            self.log_tau_b_rec_E = nn.Parameter(torch.full(b_shape, inv_softplus(1.0)))
            self.log_tau_b_rel_E = nn.Parameter(torch.full(b_shape, inv_softplus(0.25)))
        else:
            self.log_tau_b_rec_E = None
            self.log_tau_b_rel_E = None

        # ---- STD parameters for I ----
        if config.n_b_I > 0:
            b_shape = (n_I,) if config.per_neuron else (1,)
            self.log_tau_b_rec_I = nn.Parameter(torch.full(b_shape, inv_softplus(1.0)))
            self.log_tau_b_rel_I = nn.Parameter(torch.full(b_shape, inv_softplus(0.25)))
        else:
            self.log_tau_b_rec_I = None
            self.log_tau_b_rel_I = None

    # ---- Weight construction ----

    def _effective_W(self) -> torch.Tensor:
        """Build the effective weight matrix, applying Dale's law if enabled."""
        cfg = self.config
        if cfg.dales:
            W_pos = F.softplus(self.W_raw)
            W_eff = W_pos.clone()
            W_eff[:, cfg.n_E:] = -W_pos[:, cfg.n_E:]
            if self.sparsity_mask is not None:
                W_eff = W_eff * self.sparsity_mask
            return W_eff
        else:
            return self.W_raw

    # ---- State packing / unpacking ----

    def pack_state(
        self,
        a_E: Optional[torch.Tensor],
        a_I: Optional[torch.Tensor],
        b_E: Optional[torch.Tensor],
        b_I: Optional[torch.Tensor],
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Pack state components into a flat (batch, state_size) tensor."""
        parts: list[torch.Tensor] = []
        if a_E is not None:
            parts.append(a_E.flatten(start_dim=1))
        if a_I is not None:
            parts.append(a_I.flatten(start_dim=1))
        if b_E is not None:
            parts.append(b_E)
        if b_I is not None:
            parts.append(b_I)
        parts.append(x)
        return torch.cat(parts, dim=-1)

    def unpack_state(self, state: torch.Tensor):
        """Unpack flat state into (a_E, a_I, b_E, b_I, x)."""
        cfg = self.config
        idx = 0
        a_E = a_I = b_E = b_I = None

        if cfg.n_a_E > 0:
            sz = cfg.n_E * cfg.n_a_E
            a_E = state[:, idx:idx + sz].reshape(-1, cfg.n_E, cfg.n_a_E)
            idx += sz
        if cfg.n_a_I > 0:
            sz = cfg.n_I * cfg.n_a_I
            a_I = state[:, idx:idx + sz].reshape(-1, cfg.n_I, cfg.n_a_I)
            idx += sz
        if cfg.n_b_E > 0:
            b_E = state[:, idx:idx + cfg.n_E]
            idx += cfg.n_E
        if cfg.n_b_I > 0:
            b_I = state[:, idx:idx + cfg.n_I]
            idx += cfg.n_I
        x = state[:, idx:idx + cfg.num_units]
        return a_E, a_I, b_E, b_I, x

    @property
    def state_size(self) -> int:
        return self.config.state_size

    def init_state(self, batch_size: int, device: torch.device = None) -> torch.Tensor:
        """Return a zero-initialized flat state."""
        if device is None:
            device = self.W_raw.device
        return torch.zeros(batch_size, self.config.state_size, device=device)

    # ---- Tau interpolation helpers (matches TF _make_tau_range) ----

    def _get_tau_a_E(self) -> torch.Tensor:
        """Get SFA time constants for E neurons, interpolating if multi-timescale."""
        if self.log_tau_a_E is not None:
            return F.softplus(self.log_tau_a_E)
        # Multi-timescale: interpolate n_a_E values between lo and hi
        lo = F.softplus(self.log_tau_a_E_lo)  # (dim, 1)
        hi = F.softplus(self.log_tau_a_E_hi)  # (dim, 1)
        n = self.config.n_a_E
        t = torch.linspace(0.0, 1.0, n, device=lo.device)  # (n,)
        return lo + (hi - lo) * t  # (dim, n)

    def _get_tau_a_I(self) -> torch.Tensor:
        """Get SFA time constants for I neurons, interpolating if multi-timescale."""
        if self.log_tau_a_I is not None:
            return F.softplus(self.log_tau_a_I)
        lo = F.softplus(self.log_tau_a_I_lo)
        hi = F.softplus(self.log_tau_a_I_hi)
        n = self.config.n_a_I
        t = torch.linspace(0.0, 1.0, n, device=lo.device)
        return lo + (hi - lo) * t

    # ---- ODE right-hand side ----

    def _compute_rhs(
        self,
        x: torch.Tensor,
        a_E: Optional[torch.Tensor],
        a_I: Optional[torch.Tensor],
        b_E: Optional[torch.Tensor],
        b_I: Optional[torch.Tensor],
        u: torch.Tensor,
        W_eff: torch.Tensor,
    ):
        """Compute derivatives (dx, da_E, da_I, db_E, db_I) for one sub-step.

        Returns the values needed by each solver (slightly different usage
        depending on the solver, but the core math is the same).
        """
        cfg = self.config
        N = cfg.num_units
        n_E = cfg.n_E

        # 1. Effective potential after SFA subtraction
        x_eff = x
        if a_E is not None:
            c_E = F.softplus(self.log_c_E)  # (n_E, n_a_E) or (1, n_a_E)
            x_eff_E = x[:, :n_E] - (c_E * a_E).sum(-1)
            x_eff = torch.cat([x_eff_E, x_eff[:, n_E:]], dim=-1)
        if a_I is not None:
            c_I = F.softplus(self.log_c_I)
            x_eff_I = x_eff[:, n_E:] - (c_I * a_I).sum(-1)
            x_eff = torch.cat([x_eff[:, :n_E], x_eff_I], dim=-1)

        # 2. Firing rate
        r = piecewise_sigmoid(x_eff - self.a_0)  # (batch, N)

        # 3. Synaptic output with depression
        b_full = torch.ones_like(r)
        if b_E is not None:
            b_full = torch.cat([b_E, b_full[:, n_E:]], dim=-1)
        if b_I is not None:
            b_full = torch.cat([b_full[:, :n_E], b_I], dim=-1)
        br = b_full * r  # (batch, N)

        # 4. Recurrent drive
        Wbr = br @ W_eff.T  # (batch, N)

        # 5. Dendritic potential derivative
        tau_d = F.softplus(self.log_tau_d)  # (N,) or (1,)
        dx = (-x + u + Wbr) / tau_d

        # 6. SFA derivatives
        da_E = da_I = None
        if a_E is not None:
            tau_a_E = self._get_tau_a_E()  # (dim, n_a_E), interpolated if multi
            r_E = r[:, :n_E].unsqueeze(-1)  # (batch, n_E, 1)
            da_E = (-a_E + self.c_0_E + r_E) / tau_a_E

        if a_I is not None:
            tau_a_I = self._get_tau_a_I()
            r_I = r[:, n_E:].unsqueeze(-1)
            da_I = (-a_I + self.c_0_I + r_I) / tau_a_I

        # 7. STD derivatives
        db_E = db_I = None
        if b_E is not None:
            tau_b_rec_E = F.softplus(self.log_tau_b_rec_E)
            tau_b_rel_E = F.softplus(self.log_tau_b_rel_E)
            r_E_flat = r[:, :n_E]
            db_E = (1.0 - b_E) / tau_b_rec_E - r_E_flat * b_E / tau_b_rel_E

        if b_I is not None:
            tau_b_rec_I = F.softplus(self.log_tau_b_rec_I)
            tau_b_rel_I = F.softplus(self.log_tau_b_rel_I)
            r_I_flat = r[:, n_E:]
            db_I = (1.0 - b_I) / tau_b_rec_I - r_I_flat * b_I / tau_b_rel_I

        return dx, da_E, da_I, db_E, db_I, r, b_full

    # ---- Solvers ----

    def _step_semi_implicit(
        self, dt: float, x, a_E, a_I, b_E, b_I, u, W_eff,
    ):
        """Semi-implicit (linearly-implicit Euler) step."""
        cfg = self.config
        n_E = cfg.n_E

        # Effective potential
        x_eff = x
        if a_E is not None:
            c_E = F.softplus(self.log_c_E)
            x_eff_E = x[:, :n_E] - (c_E * a_E).sum(-1)
            x_eff = torch.cat([x_eff_E, x_eff[:, n_E:]], dim=-1)
        if a_I is not None:
            c_I = F.softplus(self.log_c_I)
            x_eff_I = x_eff[:, n_E:] - (c_I * a_I).sum(-1)
            x_eff = torch.cat([x_eff[:, :n_E], x_eff_I], dim=-1)

        r = piecewise_sigmoid(x_eff - self.a_0)

        b_full = torch.ones_like(r)
        if b_E is not None:
            b_full = torch.cat([b_E, b_full[:, n_E:]], dim=-1)
        if b_I is not None:
            b_full = torch.cat([b_full[:, :n_E], b_I], dim=-1)
        br = b_full * r
        Wbr = br @ W_eff.T

        # Semi-implicit x update
        tau_d = F.softplus(self.log_tau_d)
        alpha_x = dt / tau_d
        x_new = (x + alpha_x * (u + Wbr)) / (1.0 + alpha_x)

        # Semi-implicit SFA update
        a_E_new = a_I_new = None
        if a_E is not None:
            tau_a_E = self._get_tau_a_E()  # (dim, n_a_E), interpolated if multi
            alpha_a_E = dt / tau_a_E
            r_E = r[:, :n_E].unsqueeze(-1)
            a_E_new = (a_E + alpha_a_E * (self.c_0_E + r_E)) / (1.0 + alpha_a_E)

        if a_I is not None:
            tau_a_I = self._get_tau_a_I()
            alpha_a_I = dt / tau_a_I
            r_I = r[:, n_E:].unsqueeze(-1)
            a_I_new = (a_I + alpha_a_I * (self.c_0_I + r_I)) / (1.0 + alpha_a_I)

        # Semi-implicit STD update
        b_E_new = b_I_new = None
        if b_E is not None:
            tau_b_rec_E = F.softplus(self.log_tau_b_rec_E)
            tau_b_rel_E = F.softplus(self.log_tau_b_rel_E)
            r_E_flat = r[:, :n_E]
            b_E_new = (b_E + dt / tau_b_rec_E) / (
                1.0 + dt * (1.0 / tau_b_rec_E + r_E_flat / tau_b_rel_E)
            )

        if b_I is not None:
            tau_b_rec_I = F.softplus(self.log_tau_b_rec_I)
            tau_b_rel_I = F.softplus(self.log_tau_b_rel_I)
            r_I_flat = r[:, n_E:]
            b_I_new = (b_I + dt / tau_b_rec_I) / (
                1.0 + dt * (1.0 / tau_b_rec_I + r_I_flat / tau_b_rel_I)
            )

        return x_new, a_E_new, a_I_new, b_E_new, b_I_new, r, b_full

    def _step_explicit(self, dt, x, a_E, a_I, b_E, b_I, u, W_eff):
        """Forward Euler step."""
        dx, da_E, da_I, db_E, db_I, r, b_full = self._compute_rhs(
            x, a_E, a_I, b_E, b_I, u, W_eff
        )
        x_new = x + dt * dx
        a_E_new = (a_E + dt * da_E) if a_E is not None else None
        a_I_new = (a_I + dt * da_I) if a_I is not None else None
        b_E_new = (b_E + dt * db_E) if b_E is not None else None
        b_I_new = (b_I + dt * db_I) if b_I is not None else None
        return x_new, a_E_new, a_I_new, b_E_new, b_I_new, r, b_full

    def _step_rk4(self, dt, x, a_E, a_I, b_E, b_I, u, W_eff):
        """Classical 4th-order Runge-Kutta step."""

        def _add(a, b, scale):
            if a is None:
                return None
            return a + scale * b

        # k1
        k1 = self._compute_rhs(x, a_E, a_I, b_E, b_I, u, W_eff)
        # k2
        k2 = self._compute_rhs(
            _add(x, k1[0], 0.5 * dt),
            _add(a_E, k1[1], 0.5 * dt),
            _add(a_I, k1[2], 0.5 * dt),
            _add(b_E, k1[3], 0.5 * dt),
            _add(b_I, k1[4], 0.5 * dt),
            u, W_eff,
        )
        # k3
        k3 = self._compute_rhs(
            _add(x, k2[0], 0.5 * dt),
            _add(a_E, k2[1], 0.5 * dt),
            _add(a_I, k2[2], 0.5 * dt),
            _add(b_E, k2[3], 0.5 * dt),
            _add(b_I, k2[4], 0.5 * dt),
            u, W_eff,
        )
        # k4
        k4 = self._compute_rhs(
            _add(x, k3[0], dt),
            _add(a_E, k3[1], dt),
            _add(a_I, k3[2], dt),
            _add(b_E, k3[3], dt),
            _add(b_I, k3[4], dt),
            u, W_eff,
        )

        def _rk4_combine(y, k1v, k2v, k3v, k4v):
            if y is None:
                return None
            return y + (dt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)

        x_new = _rk4_combine(x, k1[0], k2[0], k3[0], k4[0])
        a_E_new = _rk4_combine(a_E, k1[1], k2[1], k3[1], k4[1])
        a_I_new = _rk4_combine(a_I, k1[2], k2[2], k3[2], k4[2])
        b_E_new = _rk4_combine(b_E, k1[3], k2[3], k3[3], k4[3])
        b_I_new = _rk4_combine(b_I, k1[4], k2[4], k3[4], k4[4])
        return x_new, a_E_new, a_I_new, b_E_new, b_I_new, k1[5], k1[6]

    def _step_exponential(self, dt, x, a_E, a_I, b_E, b_I, u, W_eff):
        """Exponential Euler step: exact decay factor for the linear part."""
        cfg = self.config
        n_E = cfg.n_E

        # Compute firing rate and recurrent drive (same as explicit)
        dx, da_E, da_I, db_E, db_I, r, b_full = self._compute_rhs(
            x, a_E, a_I, b_E, b_I, u, W_eff
        )

        # Exponential update for x: x_new = x*exp(-dt/tau) + (1-exp(-dt/tau)) * driving
        tau_d = F.softplus(self.log_tau_d)
        decay_x = torch.exp(-dt / tau_d)
        b_rebuilt = torch.ones_like(r)
        if b_E is not None:
            b_rebuilt = torch.cat([b_E, b_rebuilt[:, n_E:]], dim=-1)
        if b_I is not None:
            b_rebuilt = torch.cat([b_rebuilt[:, :n_E], b_I], dim=-1)
        br = b_rebuilt * r
        Wbr = br @ W_eff.T
        x_new = x * decay_x + (1.0 - decay_x) * (u + Wbr)

        # Exponential update for SFA
        a_E_new = a_I_new = None
        if a_E is not None:
            tau_a_E = self._get_tau_a_E()
            decay_a_E = torch.exp(-dt / tau_a_E)
            r_E = r[:, :n_E].unsqueeze(-1)
            a_E_new = a_E * decay_a_E + (1.0 - decay_a_E) * (self.c_0_E + r_E)

        if a_I is not None:
            tau_a_I = self._get_tau_a_I()
            decay_a_I = torch.exp(-dt / tau_a_I)
            r_I = r[:, n_E:].unsqueeze(-1)
            a_I_new = a_I * decay_a_I + (1.0 - decay_a_I) * (self.c_0_I + r_I)

        # STD: use explicit Euler for the nonlinear STD equation
        b_E_new = (b_E + dt * db_E) if b_E is not None else None
        b_I_new = (b_I + dt * db_I) if b_I is not None else None

        return x_new, a_E_new, a_I_new, b_E_new, b_I_new, r, b_full

    # ---- Forward ----

    def forward(
        self, inputs: torch.Tensor, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        inputs : (batch, input_size)
        state  : (batch, state_size)

        Returns
        -------
        output : (batch, num_units)  -- readout
        new_state : (batch, state_size)
        """
        cfg = self.config
        a_E, a_I, b_E, b_I, x = self.unpack_state(state)
        W_eff = self._effective_W()
        # Apply W_in_mask to input weight rows (zero input to non-input neurons)
        W_in_eff = self.W_in
        if self.W_in_mask is not None:
            W_in_eff = self.W_in * self.W_in_mask  # (N, input_size) * (N, 1)
        u = inputs @ W_in_eff.T  # (batch, N)
        dt = cfg.h / cfg.ode_unfolds

        # Select solver -- Python branching on *config strings* is fine
        # (static at trace time for torch.compile).
        if cfg.solver == "semi_implicit":
            step_fn = self._step_semi_implicit
        elif cfg.solver == "explicit":
            step_fn = self._step_explicit
        elif cfg.solver == "rk4":
            step_fn = self._step_rk4
        elif cfg.solver == "exponential":
            step_fn = self._step_exponential
        else:
            raise ValueError(f"Unknown solver: {cfg.solver}")

        r_last = None
        b_last = None
        for _ in range(cfg.ode_unfolds):
            x, a_E, a_I, b_E, b_I, r_last, b_last = step_fn(
                dt, x, a_E, a_I, b_E, b_I, u, W_eff
            )

        # Readout
        if cfg.readout == "synaptic":
            output = b_last * r_last
        elif cfg.readout == "rate":
            output = r_last
        elif cfg.readout == "dendritic":
            output = x
        else:
            raise ValueError(f"Unknown readout: {cfg.readout}")

        new_state = self.pack_state(a_E, a_I, b_E, b_I, x)
        return output, new_state


# ---------------------------------------------------------------------------
# BatchedSRNNCell  -- K ablation variants in parallel via bmm
# ---------------------------------------------------------------------------

class BatchedSRNNCell(nn.Module):
    """Run K SRNN ablation variants simultaneously using bmm.

    All K variants must share the same num_units and input_size.
    States are padded to a common max_state_dim so they can be stacked.

    The forward pass is fully torch.compile-friendly: ablation differences
    are encoded as multiplicative masks (buffers) rather than Python branches.
    """

    def __init__(
        self,
        configs: list[SRNNConfig],
        input_size: int,
        W_in_mask: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        assert len(configs) > 0
        N = configs[0].num_units
        for c in configs:
            assert c.num_units == N, "All configs must share num_units"

        self.K = len(configs)
        self.N = N
        self.input_size = input_size
        self.configs = configs

        n_E = N // 2
        n_I = N - n_E
        self.n_E = n_E
        self.n_I = n_I

        # Max SFA/STD dimensions across all configs
        self.max_n_a_E = max(c.n_a_E for c in configs)
        self.max_n_a_I = max(c.n_a_I for c in configs)
        self.max_n_b_E = max(c.n_b_E for c in configs)
        self.max_n_b_I = max(c.n_b_I for c in configs)

        self.max_state_dim = (
            n_E * self.max_n_a_E
            + n_I * self.max_n_a_I
            + n_E * self.max_n_b_E
            + n_I * self.max_n_b_I
            + N
        )

        # ---- Per-variant parameters stacked as (K, ...) ----
        # Recurrent weights: (K, N, N)
        W_stack = []
        sparsity_masks = []
        for c in configs:
            w = torch.randn(N, N) * math.sqrt(2.0 / N)
            if c.dales and c.sparsity > 0:
                m = (torch.rand(N, N) > c.sparsity).float()
                w = w * m
                sparsity_masks.append(m)
            else:
                sparsity_masks.append(torch.ones(N, N))
            W_stack.append(w)
        self.W_raw = nn.Parameter(torch.stack(W_stack))  # (K, N, N)
        self.register_buffer("sparsity_masks", torch.stack(sparsity_masks))  # (K, N, N)

        # Input weights: (K, N, input_size)
        self.W_in = nn.Parameter(torch.randn(self.K, N, input_size) * 0.1)

        # W_in_mask: shared across all K ablations (same neuron partition)
        if W_in_mask is not None:
            # (N, 1) for broadcast with W_in (K, N, input_size)
            self.register_buffer("W_in_mask", W_in_mask.view(1, -1, 1))  # (1, N, 1)
        else:
            self.W_in_mask: Optional[torch.Tensor] = None

        # Threshold: (K, N)
        self.a_0 = nn.Parameter(torch.full((self.K, N), 0.35))

        # Dendritic time constant: (K, N)
        self.log_tau_d = nn.Parameter(torch.full((self.K, N), inv_softplus(0.1)))

        # ---- SFA E params: (K, n_E, max_n_a_E) ----
        if self.max_n_a_E > 0:
            log_tau_init = torch.zeros(self.K, n_E, self.max_n_a_E)
            log_c_init = torch.full((self.K, n_E, self.max_n_a_E), -3.0)
            c_0_init = torch.zeros(self.K, n_E, self.max_n_a_E)
            for ki, c in enumerate(configs):
                if c.n_a_E == 1:
                    log_tau_init[ki, :, 0] = inv_softplus(1.0)
                elif c.n_a_E >= 2:
                    # Interpolate n_a_E values between lo and hi (matches TF _make_tau_range)
                    lo_val = inv_softplus(0.25)
                    hi_val = inv_softplus(10.0)
                    for ai in range(c.n_a_E):
                        t = ai / (c.n_a_E - 1) if c.n_a_E > 1 else 0.5
                        log_tau_init[ki, :, ai] = lo_val + (hi_val - lo_val) * t
                # else (n_a_E == 0): leave at 0 (masked out)
            self.log_tau_a_E = nn.Parameter(log_tau_init)
            self.log_c_E = nn.Parameter(log_c_init)
            self.c_0_E = nn.Parameter(c_0_init)
        else:
            self.log_tau_a_E = None
            self.log_c_E = None
            self.c_0_E = None

        # ---- SFA I params: (K, n_I, max_n_a_I) ----
        if self.max_n_a_I > 0:
            log_tau_init = torch.zeros(self.K, n_I, self.max_n_a_I)
            log_c_init = torch.full((self.K, n_I, self.max_n_a_I), -3.0)
            c_0_init = torch.zeros(self.K, n_I, self.max_n_a_I)
            for ki, c in enumerate(configs):
                if c.n_a_I == 1:
                    log_tau_init[ki, :, 0] = inv_softplus(1.0)
                elif c.n_a_I >= 2:
                    lo_val = inv_softplus(0.25)
                    hi_val = inv_softplus(10.0)
                    for ai in range(c.n_a_I):
                        t = ai / (c.n_a_I - 1) if c.n_a_I > 1 else 0.5
                        log_tau_init[ki, :, ai] = lo_val + (hi_val - lo_val) * t
            self.log_tau_a_I = nn.Parameter(log_tau_init)
            self.log_c_I = nn.Parameter(log_c_init)
            self.c_0_I = nn.Parameter(c_0_init)
        else:
            self.log_tau_a_I = None
            self.log_c_I = None
            self.c_0_I = None

        # ---- STD E params: (K, n_E) ----
        if self.max_n_b_E > 0:
            self.log_tau_b_rec_E = nn.Parameter(
                torch.full((self.K, n_E), inv_softplus(1.0))
            )
            self.log_tau_b_rel_E = nn.Parameter(
                torch.full((self.K, n_E), inv_softplus(0.25))
            )
        else:
            self.log_tau_b_rec_E = None
            self.log_tau_b_rel_E = None

        # ---- STD I params: (K, n_I) ----
        if self.max_n_b_I > 0:
            self.log_tau_b_rec_I = nn.Parameter(
                torch.full((self.K, n_I), inv_softplus(1.0))
            )
            self.log_tau_b_rel_I = nn.Parameter(
                torch.full((self.K, n_I), inv_softplus(0.25))
            )
        else:
            self.log_tau_b_rec_I = None
            self.log_tau_b_rel_I = None

        # ---- Ablation masks (buffers) ----
        # Dale's mask: (K, 1, 1) -- 1.0 if Dale's active
        dales_mask = torch.tensor([float(c.dales) for c in configs]).reshape(self.K, 1, 1)
        self.register_buffer("dales_mask", dales_mask)

        # Echo mask for freezing W gradients selectively -- handled via
        # per-variant hook, but we store for reference.
        echo_flags = torch.tensor([float(c.echo) for c in configs]).reshape(self.K, 1, 1)
        self.register_buffer("echo_flags", echo_flags)

        # SFA masks: (K, 1, 1) -- broadcast-friendly
        if self.max_n_a_E > 0:
            sfa_E_mask = torch.zeros(self.K, 1, self.max_n_a_E)
            for ki, c in enumerate(configs):
                for ai in range(c.n_a_E):
                    sfa_E_mask[ki, :, ai] = 1.0
            self.register_buffer("sfa_E_mask", sfa_E_mask)
        else:
            self.register_buffer("sfa_E_mask", torch.zeros(self.K, 1, 1))

        if self.max_n_a_I > 0:
            sfa_I_mask = torch.zeros(self.K, 1, self.max_n_a_I)
            for ki, c in enumerate(configs):
                for ai in range(c.n_a_I):
                    sfa_I_mask[ki, :, ai] = 1.0
            self.register_buffer("sfa_I_mask", sfa_I_mask)
        else:
            self.register_buffer("sfa_I_mask", torch.zeros(self.K, 1, 1))

        # STD masks: (K, 1)
        std_E_mask = torch.tensor([float(c.n_b_E > 0) for c in configs]).reshape(self.K, 1)
        std_I_mask = torch.tensor([float(c.n_b_I > 0) for c in configs]).reshape(self.K, 1)
        self.register_buffer("std_E_mask", std_E_mask)
        self.register_buffer("std_I_mask", std_I_mask)

        # Readout mode encoded as integer: 0=synaptic, 1=rate, 2=dendritic
        readout_map = {"synaptic": 0, "rate": 1, "dendritic": 2}
        readout_ids = torch.tensor(
            [readout_map[c.readout] for c in configs], dtype=torch.long
        )
        self.register_buffer("readout_ids", readout_ids)

        # Solver: all configs must share solver for batched execution
        solvers = set(c.solver for c in configs)
        assert len(solvers) == 1, (
            f"BatchedSRNNCell requires all configs to use the same solver, got {solvers}"
        )
        self.solver = configs[0].solver

        # ODE params: all configs must share h and ode_unfolds
        hs = set(c.h for c in configs)
        unfolds = set(c.ode_unfolds for c in configs)
        assert len(hs) == 1 and len(unfolds) == 1, (
            "BatchedSRNNCell requires all configs to share h and ode_unfolds"
        )
        self.h = configs[0].h
        self.ode_unfolds = configs[0].ode_unfolds

    # ---- Weight construction ----

    def _effective_W(self) -> torch.Tensor:
        """Build effective (K, N, N) weight matrix applying Dale's per variant."""
        W_pos = F.softplus(self.W_raw)  # (K, N, N)
        # For Dale's variants: positive E cols, negative I cols
        # For non-Dale's: use raw weights directly
        # Blend via dales_mask
        W_dale = W_pos.clone()
        W_dale[:, :, self.n_E:] = -W_pos[:, :, self.n_E:]
        W_dale = W_dale * self.sparsity_masks

        # dales_mask: (K, 1, 1)
        W_eff = self.dales_mask * W_dale + (1.0 - self.dales_mask) * self.W_raw
        return W_eff

    # ---- State packing / unpacking ----

    def unpack_state(self, state: torch.Tensor):
        """Unpack (K, batch, max_state_dim) -> components.

        Returns a_E, a_I, b_E, b_I, x -- all may be zero-padded for
        inactive ablation slots; masks handle zeroing.
        """
        K, B = state.shape[0], state.shape[1]
        n_E, n_I, N = self.n_E, self.n_I, self.N
        idx = 0

        # a_E: (K, batch, n_E, max_n_a_E)
        if self.max_n_a_E > 0:
            sz = n_E * self.max_n_a_E
            a_E = state[:, :, idx:idx + sz].reshape(K, B, n_E, self.max_n_a_E)
            idx += sz
        else:
            a_E = torch.zeros(K, B, n_E, 1, device=state.device)

        # a_I: (K, batch, n_I, max_n_a_I)
        if self.max_n_a_I > 0:
            sz = n_I * self.max_n_a_I
            a_I = state[:, :, idx:idx + sz].reshape(K, B, n_I, self.max_n_a_I)
            idx += sz
        else:
            a_I = torch.zeros(K, B, n_I, 1, device=state.device)

        # b_E: (K, batch, n_E)
        if self.max_n_b_E > 0:
            b_E = state[:, :, idx:idx + n_E]
            idx += n_E
        else:
            b_E = torch.ones(K, B, n_E, device=state.device)

        # b_I: (K, batch, n_I)
        if self.max_n_b_I > 0:
            b_I = state[:, :, idx:idx + n_I]
            idx += n_I
        else:
            b_I = torch.ones(K, B, n_I, device=state.device)

        # x: (K, batch, N)
        x = state[:, :, idx:idx + N]

        return a_E, a_I, b_E, b_I, x

    def pack_state(self, a_E, a_I, b_E, b_I, x) -> torch.Tensor:
        """Pack components back into (K, batch, max_state_dim)."""
        parts: list[torch.Tensor] = []
        if self.max_n_a_E > 0:
            parts.append(a_E.flatten(start_dim=2))  # (K, B, n_E*max_n_a_E)
        if self.max_n_a_I > 0:
            parts.append(a_I.flatten(start_dim=2))
        if self.max_n_b_E > 0:
            parts.append(b_E)
        if self.max_n_b_I > 0:
            parts.append(b_I)
        parts.append(x)
        return torch.cat(parts, dim=-1)

    @property
    def state_size(self) -> int:
        return self.max_state_dim

    def init_state(self, batch_size: int, device: torch.device = None) -> torch.Tensor:
        """Return zero state (K, batch, max_state_dim)."""
        if device is None:
            device = self.W_raw.device
        return torch.zeros(self.K, batch_size, self.max_state_dim, device=device)

    # ---- BMM recurrent drive ----

    def _batched_recurrent_drive(
        self, br: torch.Tensor, W_eff: torch.Tensor
    ) -> torch.Tensor:
        """Compute Wbr = br @ W_eff.T for all K variants and B batch elements.

        Parameters
        ----------
        br    : (K, B, N)
        W_eff : (K, N, N)

        Returns
        -------
        Wbr : (K, B, N)
        """
        K, B, N = br.shape
        # Expand W to (K*B, N, N) by repeating each variant B times
        W_exp = W_eff.unsqueeze(1).expand(K, B, N, N).reshape(K * B, N, N)
        # br to (K*B, N, 1)
        br_exp = br.reshape(K * B, N, 1)
        # bmm: (K*B, N, N) @ (K*B, N, 1) -> (K*B, N, 1)
        # We want br @ W.T, i.e. W.T @ br as column vector
        Wbr = torch.bmm(W_exp.transpose(-2, -1), br_exp).squeeze(-1)  # (K*B, N)
        return Wbr.reshape(K, B, N)

    # ---- Batched input drive ----

    def _batched_input_drive(self, inputs: torch.Tensor) -> torch.Tensor:
        """Compute u = inputs @ W_in.T for all K variants.

        Parameters
        ----------
        inputs : (K, B, input_size) or (B, input_size)

        Returns
        -------
        u : (K, B, N)
        """
        if inputs.dim() == 2:
            # Broadcast: (B, input_size) -> (K, B, input_size)
            inputs = inputs.unsqueeze(0).expand(self.K, -1, -1)
        K, B, D = inputs.shape
        # Apply W_in_mask to input weight rows before computing drive
        W_in = self.W_in  # (K, N, input_size)
        if self.W_in_mask is not None:
            W_in = W_in * self.W_in_mask  # (K, N, D) * (1, N, 1)
        # (K*B, 1, D) @ (K*B, D, N) -> (K*B, 1, N) -> (K, B, N)
        W_in_exp = W_in.unsqueeze(1).expand(K, B, self.N, D).reshape(K * B, self.N, D)
        inp_exp = inputs.reshape(K * B, D, 1)
        u = torch.bmm(W_in_exp, inp_exp).squeeze(-1).reshape(K, B, self.N)
        return u

    # ---- Semi-implicit step (batched) ----

    def _batched_step_semi_implicit(self, dt, x, a_E, a_I, b_E, b_I, u, W_eff):
        """Semi-implicit step for all K variants in parallel."""
        K = self.K
        n_E, n_I, N = self.n_E, self.n_I, self.N

        # Effective potential with SFA
        x_eff = x.clone()
        if self.max_n_a_E > 0:
            c_E = F.softplus(self.log_c_E)  # (K, n_E, max_n_a_E)
            # Mask inactive timescales
            c_E_masked = c_E * self.sfa_E_mask  # (K, n_E, max_n_a_E) * (K, 1, max_n_a_E)
            sfa_E_contrib = (c_E_masked.unsqueeze(1) * a_E).sum(-1)  # (K, B, n_E)
            x_eff_E = x[:, :, :n_E] - sfa_E_contrib
            x_eff = torch.cat([x_eff_E, x_eff[:, :, n_E:]], dim=-1)

        if self.max_n_a_I > 0:
            c_I = F.softplus(self.log_c_I)
            c_I_masked = c_I * self.sfa_I_mask
            sfa_I_contrib = (c_I_masked.unsqueeze(1) * a_I).sum(-1)  # (K, B, n_I)
            x_eff_I = x_eff[:, :, n_E:] - sfa_I_contrib
            x_eff = torch.cat([x_eff[:, :, :n_E], x_eff_I], dim=-1)

        # Firing rate
        r = piecewise_sigmoid(x_eff - self.a_0.unsqueeze(1))  # (K, B, N)

        # Depression
        # For inactive STD: b stays 1.0 (no depression)
        # std_E_mask: (K, 1), std_I_mask: (K, 1)
        b_full_E = b_E * self.std_E_mask.unsqueeze(1) + (1.0 - self.std_E_mask.unsqueeze(1))
        b_full_I = b_I * self.std_I_mask.unsqueeze(1) + (1.0 - self.std_I_mask.unsqueeze(1))
        b_full = torch.cat([b_full_E, b_full_I], dim=-1)  # (K, B, N)
        br = b_full * r

        # Recurrent drive via bmm
        Wbr = self._batched_recurrent_drive(br, W_eff)  # (K, B, N)

        # Semi-implicit x update
        tau_d = F.softplus(self.log_tau_d).unsqueeze(1)  # (K, 1, N)
        alpha_x = dt / tau_d
        x_new = (x + alpha_x * (u + Wbr)) / (1.0 + alpha_x)

        # SFA E update
        if self.max_n_a_E > 0:
            tau_a_E = F.softplus(self.log_tau_a_E)  # (K, n_E, max_n_a_E)
            alpha_a_E = dt / tau_a_E  # (K, n_E, max_n_a_E)
            r_E = r[:, :, :n_E].unsqueeze(-1)  # (K, B, n_E, 1)
            c_0_E = self.c_0_E.unsqueeze(1)  # (K, 1, n_E, max_n_a_E)
            alpha_a_E_b = alpha_a_E.unsqueeze(1)  # (K, 1, n_E, max_n_a_E)
            a_E_updated = (a_E + alpha_a_E_b * (c_0_E + r_E)) / (1.0 + alpha_a_E_b)
            # Mask: keep old (zero) for inactive timescales
            sfa_E_mask_b = self.sfa_E_mask.unsqueeze(1)  # (K, 1, 1, max_n_a_E)
            a_E_new = a_E * (1.0 - sfa_E_mask_b) + a_E_updated * sfa_E_mask_b
        else:
            a_E_new = a_E

        # SFA I update
        if self.max_n_a_I > 0:
            tau_a_I = F.softplus(self.log_tau_a_I)
            alpha_a_I = dt / tau_a_I
            r_I = r[:, :, n_E:].unsqueeze(-1)
            c_0_I = self.c_0_I.unsqueeze(1)
            alpha_a_I_b = alpha_a_I.unsqueeze(1)
            a_I_updated = (a_I + alpha_a_I_b * (c_0_I + r_I)) / (1.0 + alpha_a_I_b)
            sfa_I_mask_b = self.sfa_I_mask.unsqueeze(1)
            a_I_new = a_I * (1.0 - sfa_I_mask_b) + a_I_updated * sfa_I_mask_b
        else:
            a_I_new = a_I

        # STD E update
        if self.max_n_b_E > 0:
            tau_b_rec_E = F.softplus(self.log_tau_b_rec_E).unsqueeze(1)  # (K, 1, n_E)
            tau_b_rel_E = F.softplus(self.log_tau_b_rel_E).unsqueeze(1)
            r_E_flat = r[:, :, :n_E]
            b_E_updated = (b_E + dt / tau_b_rec_E) / (
                1.0 + dt * (1.0 / tau_b_rec_E + r_E_flat / tau_b_rel_E)
            )
            std_E_m = self.std_E_mask.unsqueeze(1)  # (K, 1, 1)
            b_E_new = b_E * (1.0 - std_E_m) + b_E_updated * std_E_m
        else:
            b_E_new = b_E

        # STD I update
        if self.max_n_b_I > 0:
            tau_b_rec_I = F.softplus(self.log_tau_b_rec_I).unsqueeze(1)
            tau_b_rel_I = F.softplus(self.log_tau_b_rel_I).unsqueeze(1)
            r_I_flat = r[:, :, n_E:]
            b_I_updated = (b_I + dt / tau_b_rec_I) / (
                1.0 + dt * (1.0 / tau_b_rec_I + r_I_flat / tau_b_rel_I)
            )
            std_I_m = self.std_I_mask.unsqueeze(1)
            b_I_new = b_I * (1.0 - std_I_m) + b_I_updated * std_I_m
        else:
            b_I_new = b_I

        return x_new, a_E_new, a_I_new, b_E_new, b_I_new, r, b_full

    # ---- Batched RHS for explicit / RK4 / exponential solvers ----

    def _batched_compute_rhs(self, x, a_E, a_I, b_E, b_I, u, W_eff):
        """Compute derivatives for all K variants. Returns (dx, da_E, da_I, db_E, db_I, r, b_full)."""
        n_E, n_I, N = self.n_E, self.n_I, self.N

        # Effective potential
        x_eff = x.clone()
        if self.max_n_a_E > 0:
            c_E = F.softplus(self.log_c_E)
            c_E_masked = c_E * self.sfa_E_mask
            sfa_E_contrib = (c_E_masked.unsqueeze(1) * a_E).sum(-1)
            x_eff = torch.cat([x[:, :, :n_E] - sfa_E_contrib, x_eff[:, :, n_E:]], dim=-1)

        if self.max_n_a_I > 0:
            c_I = F.softplus(self.log_c_I)
            c_I_masked = c_I * self.sfa_I_mask
            sfa_I_contrib = (c_I_masked.unsqueeze(1) * a_I).sum(-1)
            x_eff = torch.cat([x_eff[:, :, :n_E], x_eff[:, :, n_E:] - sfa_I_contrib], dim=-1)

        r = piecewise_sigmoid(x_eff - self.a_0.unsqueeze(1))

        b_full_E = b_E * self.std_E_mask.unsqueeze(1) + (1.0 - self.std_E_mask.unsqueeze(1))
        b_full_I = b_I * self.std_I_mask.unsqueeze(1) + (1.0 - self.std_I_mask.unsqueeze(1))
        b_full = torch.cat([b_full_E, b_full_I], dim=-1)
        br = b_full * r

        Wbr = self._batched_recurrent_drive(br, W_eff)

        tau_d = F.softplus(self.log_tau_d).unsqueeze(1)
        dx = (-x + u + Wbr) / tau_d

        # SFA derivatives
        if self.max_n_a_E > 0:
            tau_a_E = F.softplus(self.log_tau_a_E).unsqueeze(1)
            r_E = r[:, :, :n_E].unsqueeze(-1)
            c_0_E = self.c_0_E.unsqueeze(1)
            da_E = ((-a_E + c_0_E + r_E) / tau_a_E) * self.sfa_E_mask.unsqueeze(1)
        else:
            da_E = torch.zeros_like(a_E)

        if self.max_n_a_I > 0:
            tau_a_I = F.softplus(self.log_tau_a_I).unsqueeze(1)
            r_I = r[:, :, n_E:].unsqueeze(-1)
            c_0_I = self.c_0_I.unsqueeze(1)
            da_I = ((-a_I + c_0_I + r_I) / tau_a_I) * self.sfa_I_mask.unsqueeze(1)
        else:
            da_I = torch.zeros_like(a_I)

        # STD derivatives
        if self.max_n_b_E > 0:
            tau_b_rec_E = F.softplus(self.log_tau_b_rec_E).unsqueeze(1)
            tau_b_rel_E = F.softplus(self.log_tau_b_rel_E).unsqueeze(1)
            r_E_flat = r[:, :, :n_E]
            db_E = ((1.0 - b_E) / tau_b_rec_E - r_E_flat * b_E / tau_b_rel_E) * self.std_E_mask.unsqueeze(1)
        else:
            db_E = torch.zeros_like(b_E)

        if self.max_n_b_I > 0:
            tau_b_rec_I = F.softplus(self.log_tau_b_rec_I).unsqueeze(1)
            tau_b_rel_I = F.softplus(self.log_tau_b_rel_I).unsqueeze(1)
            r_I_flat = r[:, :, n_E:]
            db_I = ((1.0 - b_I) / tau_b_rec_I - r_I_flat * b_I / tau_b_rel_I) * self.std_I_mask.unsqueeze(1)
        else:
            db_I = torch.zeros_like(b_I)

        return dx, da_E, da_I, db_E, db_I, r, b_full

    def _batched_step_explicit(self, dt, x, a_E, a_I, b_E, b_I, u, W_eff):
        """Forward Euler step for all K variants."""
        dx, da_E, da_I, db_E, db_I, r, b_full = self._batched_compute_rhs(
            x, a_E, a_I, b_E, b_I, u, W_eff
        )
        return (
            x + dt * dx,
            a_E + dt * da_E,
            a_I + dt * da_I,
            b_E + dt * db_E,
            b_I + dt * db_I,
            r, b_full,
        )

    def _batched_step_rk4(self, dt, x, a_E, a_I, b_E, b_I, u, W_eff):
        """RK4 step for all K variants."""
        k1 = self._batched_compute_rhs(x, a_E, a_I, b_E, b_I, u, W_eff)
        k2 = self._batched_compute_rhs(
            x + 0.5 * dt * k1[0], a_E + 0.5 * dt * k1[1],
            a_I + 0.5 * dt * k1[2], b_E + 0.5 * dt * k1[3],
            b_I + 0.5 * dt * k1[4], u, W_eff,
        )
        k3 = self._batched_compute_rhs(
            x + 0.5 * dt * k2[0], a_E + 0.5 * dt * k2[1],
            a_I + 0.5 * dt * k2[2], b_E + 0.5 * dt * k2[3],
            b_I + 0.5 * dt * k2[4], u, W_eff,
        )
        k4 = self._batched_compute_rhs(
            x + dt * k3[0], a_E + dt * k3[1],
            a_I + dt * k3[2], b_E + dt * k3[3],
            b_I + dt * k3[4], u, W_eff,
        )
        s = dt / 6.0
        return (
            x + s * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]),
            a_E + s * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]),
            a_I + s * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]),
            b_E + s * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3]),
            b_I + s * (k1[4] + 2 * k2[4] + 2 * k3[4] + k4[4]),
            k1[5], k1[6],
        )

    def _batched_step_exponential(self, dt, x, a_E, a_I, b_E, b_I, u, W_eff):
        """Exponential Euler step for all K variants."""
        dx, da_E, da_I, db_E, db_I, r, b_full = self._batched_compute_rhs(
            x, a_E, a_I, b_E, b_I, u, W_eff
        )
        n_E = self.n_E

        # Exponential x
        tau_d = F.softplus(self.log_tau_d).unsqueeze(1)
        decay_x = torch.exp(-dt / tau_d)
        br = b_full * r
        Wbr = self._batched_recurrent_drive(br, W_eff)
        x_new = x * decay_x + (1.0 - decay_x) * (u + Wbr)

        # Exponential SFA
        if self.max_n_a_E > 0:
            tau_a_E = F.softplus(self.log_tau_a_E).unsqueeze(1)
            decay_a_E = torch.exp(-dt / tau_a_E)
            r_E = r[:, :, :n_E].unsqueeze(-1)
            c_0_E = self.c_0_E.unsqueeze(1)
            a_E_updated = a_E * decay_a_E + (1.0 - decay_a_E) * (c_0_E + r_E)
            sfa_E_mask_b = self.sfa_E_mask.unsqueeze(1)
            a_E_new = a_E * (1.0 - sfa_E_mask_b) + a_E_updated * sfa_E_mask_b
        else:
            a_E_new = a_E

        if self.max_n_a_I > 0:
            tau_a_I = F.softplus(self.log_tau_a_I).unsqueeze(1)
            decay_a_I = torch.exp(-dt / tau_a_I)
            r_I = r[:, :, n_E:].unsqueeze(-1)
            c_0_I = self.c_0_I.unsqueeze(1)
            a_I_updated = a_I * decay_a_I + (1.0 - decay_a_I) * (c_0_I + r_I)
            sfa_I_mask_b = self.sfa_I_mask.unsqueeze(1)
            a_I_new = a_I * (1.0 - sfa_I_mask_b) + a_I_updated * sfa_I_mask_b
        else:
            a_I_new = a_I

        # STD: explicit Euler for nonlinear equation
        b_E_new = b_E + dt * db_E
        b_I_new = b_I + dt * db_I

        return x_new, a_E_new, a_I_new, b_E_new, b_I_new, r, b_full

    # ---- Forward ----

    def forward(
        self,
        inputs: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        inputs : (K, batch, input_size) or (batch, input_size)
        state  : (K, batch, max_state_dim)

        Returns
        -------
        output    : (K, batch, num_units)
        new_state : (K, batch, max_state_dim)
        """
        a_E, a_I, b_E, b_I, x = self.unpack_state(state)
        W_eff = self._effective_W()  # (K, N, N)
        u = self._batched_input_drive(inputs)  # (K, B, N)
        dt = self.h / self.ode_unfolds

        # Solver selection -- static Python branching on config string, compile-safe
        if self.solver == "semi_implicit":
            step_fn = self._batched_step_semi_implicit
        elif self.solver == "explicit":
            step_fn = self._batched_step_explicit
        elif self.solver == "rk4":
            step_fn = self._batched_step_rk4
        elif self.solver == "exponential":
            step_fn = self._batched_step_exponential
        else:
            raise ValueError(f"Unknown solver: {self.solver}")

        r_last = None
        b_last = None
        for _ in range(self.ode_unfolds):
            x, a_E, a_I, b_E, b_I, r_last, b_last = step_fn(
                dt, x, a_E, a_I, b_E, b_I, u, W_eff,
            )

        # Readout: encode all three variants, select via mask
        # readout_ids: (K,)  -> 0=synaptic, 1=rate, 2=dendritic
        out_synaptic = b_last * r_last          # (K, B, N)
        out_rate = r_last                       # (K, B, N)
        out_dendritic = x                       # (K, B, N)

        # Build selection masks: (K, 1, 1)
        is_synaptic = (self.readout_ids == 0).float().reshape(self.K, 1, 1)
        is_rate = (self.readout_ids == 1).float().reshape(self.K, 1, 1)
        is_dendritic = (self.readout_ids == 2).float().reshape(self.K, 1, 1)

        output = (
            is_synaptic * out_synaptic
            + is_rate * out_rate
            + is_dendritic * out_dendritic
        )

        new_state = self.pack_state(a_E, a_I, b_E, b_I, x)
        return output, new_state
