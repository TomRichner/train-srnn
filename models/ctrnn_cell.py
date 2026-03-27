"""Continuous-time RNN cells — PyTorch port.

Three cell types:

* **CTRNNCell** — Euler-integrated CTRNN with optional global feedback.
* **NODECell** — Neural ODE using 4th-order Runge-Kutta.
* **CTGRUCell** — Multi-timescale continuous-time GRU (M parallel timescales).

All cells are ``torch.compile``-friendly: no data-dependent control flow,
no in-place mutation of tensors on the compute graph, and all constants
live in registered buffers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ======================================================================
# Configs
# ======================================================================

@dataclass
class CTRNNConfig:
    num_units: int = 32
    global_feedback: bool = True
    cell_clip: float = 0.0
    unfolds: int = 6
    delta_t: float = 0.1
    fix_tau: bool = True
    tau: float = 1.0


@dataclass
class NODEConfig:
    num_units: int = 32
    global_feedback: bool = True
    cell_clip: float = 0.0
    unfolds: int = 6
    h: float = 0.1
    fix_tau: bool = True
    tau: float = 1.0


@dataclass
class CTGRUConfig:
    num_units: int = 32
    M: int = 8          # number of parallel timescales
    tau_base: float = 1.0


# ======================================================================
# CTRNNCell
# ======================================================================

class CTRNNCell(nn.Module):
    """Continuous-Time RNN cell (Euler integration).

    Parameters
    ----------
    input_size : int
        Feature dimension of each input vector.
    config : CTRNNConfig, optional
        Hyper-parameters.
    W_in_mask : Tensor | None
        Optional ``(num_units,)`` binary mask applied after the dense
        projection (broadcast as ``(1, num_units)``).
    """

    def __init__(
        self,
        input_size: int,
        config: CTRNNConfig | None = None,
        W_in_mask: Optional[Tensor] = None,
    ) -> None:
        super().__init__()
        cfg = config or CTRNNConfig()
        self.num_units = cfg.num_units
        self.global_feedback = cfg.global_feedback
        self.cell_clip = cfg.cell_clip
        self.unfolds = cfg.unfolds
        self.delta_t = cfg.delta_t

        N = cfg.num_units
        fan_in = (input_size + N) if cfg.global_feedback else input_size

        self.W = nn.Parameter(torch.empty(fan_in, N))
        self.bias = nn.Parameter(torch.zeros(N))
        nn.init.xavier_uniform_(self.W)

        # Tau — optionally trainable (stored in unconstrained space, applied
        # through softplus so it stays positive).
        self._fix_tau = cfg.fix_tau
        if cfg.fix_tau:
            self.register_buffer("tau", torch.tensor(cfg.tau))
        else:
            # Raw parameter; use softplus(tau_raw) at runtime.
            self.tau_raw = nn.Parameter(torch.tensor(math.log(math.exp(cfg.tau) - 1.0)))

        # Optional mask
        if W_in_mask is not None:
            self.register_buffer("W_in_mask", W_in_mask.view(1, -1))
        else:
            self.W_in_mask: Optional[Tensor] = None

    @property
    def state_size(self) -> int:
        return self.num_units

    def _get_tau(self) -> Tensor:
        if self._fix_tau:
            return self.tau
        return F.softplus(self.tau_raw)

    def forward(self, inputs: Tensor, state: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            inputs: (batch, input_size)
            state:  (batch, num_units)

        Returns:
            output:    (batch, num_units)
            new_state: (batch, num_units)
        """
        tau = self._get_tau()

        if not self.global_feedback:
            # Pre-compute input projection once (state-independent).
            input_f_prime = torch.tanh(inputs @ self.W + self.bias)
            if self.W_in_mask is not None:
                input_f_prime = input_f_prime * self.W_in_mask

        for _ in range(self.unfolds):
            if self.global_feedback:
                fused = torch.cat([inputs, state], dim=-1)
                input_f_prime = torch.tanh(fused @ self.W + self.bias)
                if self.W_in_mask is not None:
                    input_f_prime = input_f_prime * self.W_in_mask

            f_prime = -state / tau + input_f_prime
            state = state + self.delta_t * f_prime

            if self.cell_clip > 0:
                state = state.clamp(-self.cell_clip, self.cell_clip)

        return state, state

    def init_state(self, batch_size: int, device: torch.device | None = None) -> Tensor:
        return torch.zeros(batch_size, self.num_units, device=device)


# ======================================================================
# NODECell
# ======================================================================

class NODECell(nn.Module):
    """Neural ODE cell (RK4 integration).

    Always uses global feedback (input + state concatenated).

    Parameters
    ----------
    input_size : int
        Feature dimension of each input vector.
    config : NODEConfig, optional
        Hyper-parameters.
    W_in_mask : Tensor | None
        Optional ``(num_units,)`` binary mask.
    """

    def __init__(
        self,
        input_size: int,
        config: NODEConfig | None = None,
        W_in_mask: Optional[Tensor] = None,
    ) -> None:
        super().__init__()
        cfg = config or NODEConfig()
        self.num_units = cfg.num_units
        self.cell_clip = cfg.cell_clip
        self.unfolds = cfg.unfolds
        self.h = cfg.h

        N = cfg.num_units
        fan_in = input_size + N

        self.W = nn.Parameter(torch.empty(fan_in, N))
        self.bias = nn.Parameter(torch.zeros(N))
        nn.init.xavier_uniform_(self.W)

        self._fix_tau = cfg.fix_tau
        if cfg.fix_tau:
            self.register_buffer("tau", torch.tensor(cfg.tau))
        else:
            self.tau_raw = nn.Parameter(torch.tensor(math.log(math.exp(cfg.tau) - 1.0)))

        if W_in_mask is not None:
            self.register_buffer("W_in_mask", W_in_mask.view(1, -1))
        else:
            self.W_in_mask: Optional[Tensor] = None

    @property
    def state_size(self) -> int:
        return self.num_units

    def _get_tau(self) -> Tensor:
        if self._fix_tau:
            return self.tau
        return F.softplus(self.tau_raw)

    def _f_prime(self, inputs: Tensor, state: Tensor) -> Tensor:
        fused = torch.cat([inputs, state], dim=-1)
        out = torch.tanh(fused @ self.W + self.bias)
        if self.W_in_mask is not None:
            out = out * self.W_in_mask
        return out

    def forward(self, inputs: Tensor, state: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            inputs: (batch, input_size)
            state:  (batch, num_units)

        Returns:
            output:    (batch, num_units)
            new_state: (batch, num_units)
        """
        h = self.h
        for _ in range(self.unfolds):
            k1 = h * self._f_prime(inputs, state)
            k2 = h * self._f_prime(inputs, state + 0.5 * k1)
            k3 = h * self._f_prime(inputs, state + 0.5 * k2)
            k4 = h * self._f_prime(inputs, state + k3)
            state = state + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

            if self.cell_clip > 0:
                state = state.clamp(-self.cell_clip, self.cell_clip)

        return state, state

    def init_state(self, batch_size: int, device: torch.device | None = None) -> Tensor:
        return torch.zeros(batch_size, self.num_units, device=device)


# ======================================================================
# CTGRUCell
# ======================================================================

class CTGRUCell(nn.Module):
    """Multi-timescale Continuous-Time GRU cell.

    Maintains *M* parallel timescales per neuron.  The hidden state has
    shape ``(batch, num_units * M)``; the output is the collapsed sum
    over timescales ``(batch, num_units)``.

    Reference: https://arxiv.org/abs/1710.04110

    Parameters
    ----------
    input_size : int
        Feature dimension of each input vector.
    config : CTGRUConfig, optional
        Hyper-parameters.
    W_in_mask : Tensor | None
        Optional ``(num_units,)`` binary mask applied to the candidate.
    """

    def __init__(
        self,
        input_size: int,
        config: CTGRUConfig | None = None,
        W_in_mask: Optional[Tensor] = None,
    ) -> None:
        super().__init__()
        cfg = config or CTGRUConfig()
        self.num_units = cfg.num_units
        self.M = cfg.M

        N = cfg.num_units
        M = cfg.M
        fan_in = input_size + N

        # Gate weights
        self.W_z = nn.Parameter(torch.empty(fan_in, N * M))
        self.b_z = nn.Parameter(torch.zeros(N * M))
        self.W_r = nn.Parameter(torch.empty(fan_in, N * M))
        self.b_r = nn.Parameter(torch.zeros(N * M))
        self.W_h = nn.Parameter(torch.empty(fan_in, N * M))
        self.b_h = nn.Parameter(torch.zeros(N * M))

        nn.init.xavier_uniform_(self.W_z)
        nn.init.xavier_uniform_(self.W_r)
        nn.init.xavier_uniform_(self.W_h)

        # Pre-compute log-timescale table as a buffer (not trainable).
        # ln_tau[i] = log(tau_base * (10^0.5)^i)
        ln_tau = torch.zeros(M)
        tau = cfg.tau_base
        for i in range(M):
            ln_tau[i] = math.log(tau)
            tau *= 10.0 ** 0.5
        self.register_buffer("ln_tau_table", ln_tau)  # (M,)

        # Softmax weight: faster timescales (smaller tau) get more weight.
        # alpha = softmax(-ln_tau)   — computed once and cached.
        self.register_buffer("alpha", F.softmax(-ln_tau, dim=0))  # (M,)

        # Optional mask
        if W_in_mask is not None:
            self.register_buffer("W_in_mask", W_in_mask.view(1, -1))
        else:
            self.W_in_mask: Optional[Tensor] = None

    @property
    def state_size(self) -> int:
        return self.num_units * self.M

    @property
    def output_size(self) -> int:
        return self.num_units

    def forward(self, inputs: Tensor, state: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            inputs: (batch, input_size)
            state:  (batch, num_units * M)

        Returns:
            output:    (batch, num_units)
            new_state: (batch, num_units * M)
        """
        B = inputs.size(0)
        N = self.num_units
        M = self.M

        h_state = state.view(B, N, M)        # (B, N, M)
        h_sum = h_state.sum(dim=2)            # (B, N) — collapsed state

        fused = torch.cat([inputs, h_sum], dim=-1)  # (B, fan_in)

        # Update gate z
        z = torch.sigmoid(fused @ self.W_z + self.b_z).view(B, N, M)

        # Reset gate r
        r = torch.sigmoid(fused @ self.W_r + self.b_r).view(B, N, M)

        # Reset-gated state collapse
        r_h = (r * h_state).sum(dim=2)  # (B, N)

        # Candidate
        fused_r = torch.cat([inputs, r_h], dim=-1)  # (B, fan_in)
        h_hat = torch.tanh(fused_r @ self.W_h + self.b_h)  # (B, N*M)
        if self.W_in_mask is not None:
            # Mask is (1, N); tile across M timescales: (1, N) -> (1, N*M)
            mask_expanded = self.W_in_mask.repeat(1, M)  # (1, N*M)
            h_hat = h_hat * mask_expanded
        h_hat = h_hat.view(B, N, M)

        # Timescale-weighted update
        # alpha: (M,) broadcast to (1, 1, M)
        h_new = (1.0 - z) * h_state + z * self.alpha * h_hat  # (B, N, M)

        output = h_new.sum(dim=2)                 # (B, N)
        new_state = h_new.view(B, N * M)          # (B, N*M)
        return output, new_state

    def init_state(self, batch_size: int, device: torch.device | None = None) -> Tensor:
        return torch.zeros(batch_size, self.num_units * self.M, device=device)
