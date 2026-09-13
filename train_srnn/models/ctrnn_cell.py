"""Continuous-time RNN baselines: CTRNN, neural ODE, and CT-GRU.

Ported to PyTorch from the TensorFlow 1.x implementation in
https://github.com/raminmh/liquid_time_constant_networks
(experiments_with_ltcs/ctrnn_model.py), Apache License 2.0. Copyright (c)
the original authors. Modifications copyright (c) 2026 Thomas Richner:
dataclass configs, the neural ODE as a CTRNN subclass sharing its
parameters, shared ODE helpers, and the RNNCell interface.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from train_srnn.models.base import RNNCell
from train_srnn.models.ode import euler_step, rk4_step


@dataclass
class CTRNNConfig:
    num_units: int = 32
    solver: str = "euler"          # euler | rk4
    global_feedback: bool = True   # feed the state back through the input projection
    cell_clip: float = 0.0         # clamp |state|; 0 disables
    unfolds: int = 6               # integrator steps per call
    h: float = 0.1                 # integrator step
    fix_tau: bool = True
    tau: float = 1.0


@dataclass
class CTGRUConfig:
    num_units: int = 32
    M: int = 8                     # parallel timescales per unit
    tau_base: float = 1.0
    cell_clip: float = -1.0        # negative disables


class CTRNNCell(RNNCell):
    """``tau dv/dt = -v + tanh(W [x, v] + b)``, integrated with Euler or RK4."""

    SOLVERS = ("euler", "rk4")

    def __init__(self, input_size: int, config: CTRNNConfig | None = None,
                 W_in_mask: Optional[Tensor] = None) -> None:
        cfg = config or CTRNNConfig()
        if cfg.solver not in self.SOLVERS:
            raise ValueError(f"Unknown CTRNN solver {cfg.solver!r}; expected one of {self.SOLVERS}")
        super().__init__(input_size, cfg.num_units, state_size=cfg.num_units,
                         W_in_mask=W_in_mask, mask_shape=(1, -1))
        self.config = cfg
        N = cfg.num_units
        fan_in = input_size + N if cfg.global_feedback else input_size
        self.W = nn.Parameter(torch.empty(fan_in, N))
        self.bias = nn.Parameter(torch.zeros(N))
        nn.init.xavier_uniform_(self.W)
        if cfg.fix_tau:
            self.register_buffer("tau", torch.tensor(cfg.tau))
        else:
            self.tau_raw = nn.Parameter(torch.tensor(math.log(math.exp(cfg.tau) - 1.0)))

    def _tau(self) -> Tensor:
        return self.tau if self.config.fix_tau else F.softplus(self.tau_raw)

    def _drive(self, inputs: Tensor, state: Tensor) -> Tensor:
        fused = torch.cat([inputs, state], dim=-1) if self.config.global_feedback else inputs
        out = torch.tanh(fused @ self.W + self.bias)
        return out * self.W_in_mask if self.W_in_mask is not None else out

    def _dv_dt(self, inputs: Tensor, state: Tensor, drive: Optional[Tensor]) -> Tensor:
        drive = self._drive(inputs, state) if drive is None else drive
        return -state / self._tau() + drive

    def forward(self, inputs: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
        cfg = self.config
        # Without feedback the drive does not depend on the state: compute it once.
        drive = None if cfg.global_feedback else self._drive(inputs, state)
        step = euler_step if cfg.solver == "euler" else rk4_step
        f = lambda v: self._dv_dt(inputs, v, drive)  # noqa: E731
        for _ in range(cfg.unfolds):
            state = step(f, state, cfg.h)
            if cfg.cell_clip > 0:
                state = state.clamp(-cfg.cell_clip, cfg.cell_clip)
        return state, state

    def init_state(self, batch_size: int, device: torch.device | None = None) -> Tensor:
        return torch.zeros(batch_size, self.num_units, device=device)


class NODECell(CTRNNCell):
    """Neural ODE as in the upstream code: RK4 on ``dv/dt = tanh(W [x, v] + b)`` with no leak."""

    def __init__(self, input_size: int, config: CTRNNConfig | None = None,
                 W_in_mask: Optional[Tensor] = None) -> None:
        cfg = replace(config or CTRNNConfig(), solver="rk4", global_feedback=True)
        super().__init__(input_size, cfg, W_in_mask=W_in_mask)

    def _dv_dt(self, inputs: Tensor, state: Tensor, drive: Optional[Tensor]) -> Tensor:
        return self._drive(inputs, state)


class CTGRUCell(RNNCell):
    """Continuous-time GRU with ``M`` parallel timescales per unit (Mozer et al. 2017).

    State is ``(B, N * M)``; the output is the sum over timescales. The decay
    factor ``exp(-1 / ln tau)`` follows the upstream implementation verbatim.
    """

    def __init__(self, input_size: int, config: CTGRUConfig | None = None,
                 W_in_mask: Optional[Tensor] = None) -> None:
        cfg = config or CTGRUConfig()
        super().__init__(input_size, cfg.num_units, state_size=cfg.num_units * cfg.M,
                         W_in_mask=W_in_mask, mask_shape=(1, -1))
        self.config = cfg
        N, M = cfg.num_units, cfg.M
        fan_in = input_size + N
        self.tau_r_dense = nn.Linear(fan_in, N * M)
        self.tau_s_dense = nn.Linear(fan_in, N * M)
        self.signal_dense = nn.Linear(fan_in, N)
        ln_tau, tau = torch.zeros(M), cfg.tau_base
        for i in range(M):                       # tau_base * sqrt(10)^i
            ln_tau[i] = math.log(tau)
            tau *= 10.0 ** 0.5
        self.register_buffer("ln_tau_table", ln_tau)
        self.register_buffer("exp_decay", torch.exp(-1.0 / ln_tau))

    @property
    def M(self) -> int:
        return self.config.M

    def forward(self, inputs: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
        B, N, M = inputs.size(0), self.num_units, self.M
        h_hat = state.view(B, N, M)
        fused = torch.cat([inputs, h_hat.sum(dim=2)], dim=-1)

        ln_tau_r = self.tau_r_dense(fused).view(B, N, M)
        rki = F.softmax(-(ln_tau_r - self.ln_tau_table).square(), dim=2)
        reset_value = torch.cat([inputs, (rki * h_hat).sum(dim=2)], dim=-1)
        qk = torch.tanh(self.signal_dense(reset_value))
        if self.W_in_mask is not None:
            qk = qk * self.W_in_mask
        qk = qk.unsqueeze(2)

        ln_tau_s = self.tau_s_dense(fused).view(B, N, M)
        ski = F.softmax(-(ln_tau_s - self.ln_tau_table).square(), dim=2)
        h_hat_next = ((1.0 - ski) * h_hat + ski * qk) * self.exp_decay
        if self.config.cell_clip > 0:
            h_hat_next = h_hat_next.clamp(-self.config.cell_clip, self.config.cell_clip)
        return h_hat_next.sum(dim=2), h_hat_next.view(B, N * M)

    def init_state(self, batch_size: int, device: torch.device | None = None) -> Tensor:
        return torch.zeros(batch_size, self.state_size, device=device)
