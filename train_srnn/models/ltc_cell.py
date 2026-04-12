"""Liquid Time-Constant (LTC) cell — PyTorch port.

Each neuron has trainable time constants, leak conductances, and
membrane capacitances.  Synaptic connections are gated by a custom
sigmoid whose centre (mu) and width (sigma) are also learned.

Three ODE solvers are supported: semi-implicit (default), explicit
Euler, and 4th-order Runge-Kutta.  All paths are data-dependent
Python-control-flow-free and therefore ``torch.compile``-friendly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class LTCConfig:
    num_units: int = 32
    solver: str = "semi_implicit"  # semi_implicit | explicit | rk4
    ode_unfolds: int = 6
    erev_init_factor: float = 1.0
    w_init_min: float = 0.01
    w_init_max: float = 1.0
    gleak_init_min: float = 1.0
    gleak_init_max: float = 1.0
    cm_init_min: float = 0.5
    cm_init_max: float = 0.5
    fix_vleak: bool = False
    fix_gleak: bool = False
    fix_cm: bool = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sigmoid(v_pre: Tensor, mu: Tensor, sigma: Tensor) -> Tensor:
    """Custom sigmoid gating.

    Args:
        v_pre: (batch, D_in)
        mu:    (D_in, num_units)
        sigma: (D_in, num_units)

    Returns:
        (batch, D_in, num_units)
    """
    v = v_pre.unsqueeze(-1)  # (batch, D_in, 1)
    return torch.sigmoid(sigma * (v - mu))  # (batch, D_in, num_units)


def _erev_init(rows: int, cols: int, factor: float) -> Tensor:
    """Half +1, half -1, times *factor*."""
    signs = 2 * torch.randint(0, 2, (rows, cols)).float() - 1
    return signs * factor


# ---------------------------------------------------------------------------
# LTCCell
# ---------------------------------------------------------------------------

class LTCCell(nn.Module):
    """Liquid Time-Constant RNN cell.

    Parameters
    ----------
    input_size : int
        Dimensionality of each input vector.
    config : LTCConfig, optional
        Hyper-parameters.  Defaults are identical to the original TF cell.
    W_in_mask : Tensor | None
        Optional binary mask of shape ``(num_units,)`` applied to sensory
        activations (broadcast as ``(1, 1, num_units)``).
    """

    def __init__(
        self,
        input_size: int,
        config: LTCConfig | None = None,
        W_in_mask: Optional[Tensor] = None,
    ) -> None:
        super().__init__()
        cfg = config or LTCConfig()
        self.input_size = input_size
        self.num_units = cfg.num_units
        self.state_size = cfg.num_units
        self.solver = cfg.solver
        self.ode_unfolds = cfg.ode_unfolds

        N = cfg.num_units
        I = input_size

        # -- sensory (input) layer ----------------------------------------
        self.sensory_mu = nn.Parameter(torch.empty(I, N).uniform_(0.3, 0.8))
        self.sensory_sigma = nn.Parameter(torch.empty(I, N).uniform_(3.0, 8.0))
        self.sensory_W = nn.Parameter(torch.empty(I, N).uniform_(cfg.w_init_min, cfg.w_init_max))
        self.sensory_erev = nn.Parameter(_erev_init(I, N, cfg.erev_init_factor))

        # -- recurrent layer -----------------------------------------------
        self.mu = nn.Parameter(torch.empty(N, N).uniform_(0.3, 0.8))
        self.sigma = nn.Parameter(torch.empty(N, N).uniform_(3.0, 8.0))
        self.W = nn.Parameter(torch.empty(N, N).uniform_(cfg.w_init_min, cfg.w_init_max))
        self.erev = nn.Parameter(_erev_init(N, N, cfg.erev_init_factor))

        # -- neuron properties ---------------------------------------------
        self.vleak = nn.Parameter(
            torch.empty(N).uniform_(-0.2, 0.2),
            requires_grad=not cfg.fix_vleak,
        )

        if cfg.gleak_init_max > cfg.gleak_init_min:
            gleak_init = torch.empty(N).uniform_(cfg.gleak_init_min, cfg.gleak_init_max)
        else:
            gleak_init = torch.full((N,), cfg.gleak_init_min)
        self.gleak = nn.Parameter(gleak_init, requires_grad=not cfg.fix_gleak)

        if cfg.cm_init_max > cfg.cm_init_min:
            cm_init = torch.empty(N).uniform_(cfg.cm_init_min, cfg.cm_init_max)
        else:
            cm_init = torch.full((N,), cfg.cm_init_min)
        self.cm_t = nn.Parameter(cm_init, requires_grad=not cfg.fix_cm)

        # -- optional mask (not a parameter) --------------------------------
        if W_in_mask is not None:
            # Store as (1, 1, num_units) for broadcasting over (B, I, N)
            self.register_buffer("W_in_mask", W_in_mask.view(1, 1, -1))
        else:
            self.W_in_mask: Optional[Tensor] = None

    # ------------------------------------------------------------------
    # Constraint helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def constrain_parameters(self) -> None:
        """Clamp trainable parameters to their valid ranges.

        Call this after ``optimizer.step()`` in the training loop.
        """
        self.cm_t.clamp_(1e-6, 1000.0)
        self.gleak.clamp_(1e-5, 1000.0)
        self.W.clamp_(1e-5, 1000.0)
        self.sensory_W.clamp_(1e-5, 1000.0)

    # ------------------------------------------------------------------
    # Sensory pre-computation (shared by all solvers)
    # ------------------------------------------------------------------

    def _sensory_input(self, inputs: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute sensory (input-layer) numerator and denominator terms.

        Returns
        -------
        w_num_sensory : (batch, num_units)
        w_den_sensory : (batch, num_units)
        """
        sensory_w_act = self.sensory_W * _sigmoid(inputs, self.sensory_mu, self.sensory_sigma)
        # (batch, input_size, num_units)

        if self.W_in_mask is not None:
            sensory_w_act = sensory_w_act * self.W_in_mask  # (1,1,N) broadcast

        sensory_rev_act = sensory_w_act * self.sensory_erev
        w_num_sensory = sensory_rev_act.sum(dim=1)  # (batch, N)
        w_den_sensory = sensory_w_act.sum(dim=1)    # (batch, N)
        return w_num_sensory, w_den_sensory

    # ------------------------------------------------------------------
    # ODE solvers
    # ------------------------------------------------------------------

    def _ode_step_semi_implicit(self, inputs: Tensor, state: Tensor) -> Tensor:
        v_pre = state
        w_num_sensory, w_den_sensory = self._sensory_input(inputs)

        for _ in range(self.ode_unfolds):
            w_act = self.W * _sigmoid(v_pre, self.mu, self.sigma)
            rev_act = w_act * self.erev
            w_num = rev_act.sum(dim=1) + w_num_sensory
            w_den = w_act.sum(dim=1) + w_den_sensory

            numerator = self.cm_t * v_pre + self.gleak * self.vleak + w_num
            denominator = self.cm_t + self.gleak + w_den
            v_pre = numerator / denominator

        return v_pre

    def _f_prime(self, inputs: Tensor, state: Tensor) -> Tensor:
        """Compute dv/dt for the explicit / RK4 solvers."""
        sensory_w_act = self.sensory_W * _sigmoid(inputs, self.sensory_mu, self.sensory_sigma)
        if self.W_in_mask is not None:
            sensory_w_act = sensory_w_act * self.W_in_mask

        w_act = self.W * _sigmoid(state, self.mu, self.sigma)

        sensory_in = (self.sensory_erev * sensory_w_act).sum(dim=1)
        synapse_in = (self.erev * w_act).sum(dim=1)

        w_reduced_sensory = sensory_w_act.sum(dim=1)
        w_reduced_synapse = w_act.sum(dim=1)

        sum_in = (
            sensory_in
            - state * w_reduced_sensory
            + synapse_in
            - state * w_reduced_synapse
        )
        return (1.0 / self.cm_t) * (self.gleak * (self.vleak - state) + sum_in)

    def _ode_step_explicit(self, inputs: Tensor, state: Tensor) -> Tensor:
        v_pre = state
        for _ in range(self.ode_unfolds):
            f_prime = self._f_prime(inputs, v_pre)
            v_pre = v_pre + 0.1 * f_prime
        return v_pre

    def _ode_step_rk4(self, inputs: Tensor, state: Tensor) -> Tensor:
        h = 0.1
        for _ in range(self.ode_unfolds):
            k1 = h * self._f_prime(inputs, state)
            k2 = h * self._f_prime(inputs, state + 0.5 * k1)
            k3 = h * self._f_prime(inputs, state + 0.5 * k2)
            k4 = h * self._f_prime(inputs, state + k3)
            state = state + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        return state

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, inputs: Tensor, state: Tensor) -> Tuple[Tensor, Tensor]:
        """Single-step forward.

        Args:
            inputs: (batch, input_size)
            state:  (batch, num_units)  — voltage *v*

        Returns:
            output: (batch, num_units)
            new_state: (batch, num_units)
        """
        if self.solver == "semi_implicit":
            new_state = self._ode_step_semi_implicit(inputs, state)
        elif self.solver == "explicit":
            new_state = self._ode_step_explicit(inputs, state)
        elif self.solver == "rk4":
            new_state = self._ode_step_rk4(inputs, state)
        else:
            raise ValueError(f"Unknown solver: {self.solver!r}")
        return new_state, new_state

    def init_state(self, batch_size: int, device: torch.device | None = None) -> Tensor:
        """Return a zero-initialised hidden state."""
        return torch.zeros(batch_size, self.num_units, device=device)
