"""Liquid time-constant cell (Hasani et al., AAAI 2021).

Ported to PyTorch from the TensorFlow 1.x implementation in
https://github.com/raminmh/liquid_time_constant_networks
(experiments_with_ltcs/ltc_model.py), Apache License 2.0. Copyright (c)
the original authors. Modifications copyright (c) 2026 Thomas Richner:
dataclass config, configurable step size for the explicit solvers, shared
ODE helpers, and the RNNCell interface.

Each neuron has a trainable leak conductance, capacitance, and resting
potential; synapses are gated by a sigmoid with learned centre and width.
Solvers: semi-implicit (default, unconditionally stable), explicit Euler,
and RK4.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from train_srnn.models.base import RNNCell
from train_srnn.models.ode import euler_step, rk4_step


@dataclass
class LTCConfig:
    num_units: int = 32
    solver: str = "semi_implicit"  # semi_implicit | explicit | rk4
    ode_unfolds: int = 6
    h: float = 0.1                 # step for the explicit solvers (upstream constant)
    erev_init_factor: float = 1.0
    w_init_min: float = 0.01
    w_init_max: float = 1.0
    gleak_init_min: float = 0.001
    gleak_init_max: float = 1.0
    cm_init_min: float = 0.4
    cm_init_max: float = 0.6
    fix_vleak: bool = False
    fix_gleak: bool = False
    fix_cm: bool = False


def _sigmoid(v_pre: Tensor, mu: Tensor, sigma: Tensor) -> Tensor:
    """Synaptic gate: ``(B, D_in)`` presynaptic voltage -> ``(B, D_in, N)``."""
    return torch.sigmoid(sigma * (v_pre.unsqueeze(-1) - mu))


def _erev_init(rows: int, cols: int, factor: float) -> Tensor:
    """Random +/- reversal potentials."""
    return (2 * torch.randint(0, 2, (rows, cols)).float() - 1) * factor


def _uniform_or_const(n: int, lo: float, hi: float) -> Tensor:
    return torch.empty(n).uniform_(lo, hi) if hi > lo else torch.full((n,), lo)


class LTCCell(RNNCell):
    SOLVERS = ("semi_implicit", "explicit", "rk4")

    def __init__(self, input_size: int, config: LTCConfig | None = None,
                 W_in_mask: Optional[Tensor] = None) -> None:
        cfg = config or LTCConfig()
        if cfg.solver not in self.SOLVERS:
            raise ValueError(f"Unknown LTC solver {cfg.solver!r}; expected one of {self.SOLVERS}")
        super().__init__(input_size, cfg.num_units, state_size=cfg.num_units,
                         W_in_mask=W_in_mask, mask_shape=(1, 1, -1))
        self.config = cfg
        N, I = cfg.num_units, input_size

        self.sensory_mu = nn.Parameter(torch.empty(I, N).uniform_(0.3, 0.8))
        self.sensory_sigma = nn.Parameter(torch.empty(I, N).uniform_(3.0, 8.0))
        self.sensory_W = nn.Parameter(torch.empty(I, N).uniform_(cfg.w_init_min, cfg.w_init_max))
        self.sensory_erev = nn.Parameter(_erev_init(I, N, cfg.erev_init_factor))

        self.mu = nn.Parameter(torch.empty(N, N).uniform_(0.3, 0.8))
        self.sigma = nn.Parameter(torch.empty(N, N).uniform_(3.0, 8.0))
        self.W = nn.Parameter(torch.empty(N, N).uniform_(cfg.w_init_min, cfg.w_init_max))
        self.erev = nn.Parameter(_erev_init(N, N, cfg.erev_init_factor))

        self.vleak = nn.Parameter(torch.empty(N).uniform_(-0.2, 0.2), requires_grad=not cfg.fix_vleak)
        self.gleak = nn.Parameter(_uniform_or_const(N, cfg.gleak_init_min, cfg.gleak_init_max),
                                  requires_grad=not cfg.fix_gleak)
        self.cm_t = nn.Parameter(_uniform_or_const(N, cfg.cm_init_min, cfg.cm_init_max),
                                 requires_grad=not cfg.fix_cm)

    @torch.no_grad()
    def constrain_parameters(self) -> None:
        self.cm_t.clamp_(1e-6, 1000.0)
        self.gleak.clamp_(1e-5, 1000.0)
        self.W.clamp_(1e-5, 1000.0)
        self.sensory_W.clamp_(1e-5, 1000.0)

    def _sensory_activation(self, inputs: Tensor) -> Tensor:
        act = self.sensory_W * _sigmoid(inputs, self.sensory_mu, self.sensory_sigma)
        return act * self.W_in_mask if self.W_in_mask is not None else act

    def _step_semi_implicit(self, inputs: Tensor, v: Tensor) -> Tensor:
        sensory = self._sensory_activation(inputs)
        w_num_sensory = (sensory * self.sensory_erev).sum(dim=1)
        w_den_sensory = sensory.sum(dim=1)
        for _ in range(self.config.ode_unfolds):
            w_act = self.W * _sigmoid(v, self.mu, self.sigma)
            w_num = (w_act * self.erev).sum(dim=1) + w_num_sensory
            w_den = w_act.sum(dim=1) + w_den_sensory
            v = (self.cm_t * v + self.gleak * self.vleak + w_num) / (self.cm_t + self.gleak + w_den)
        return v

    def _dv_dt(self, inputs: Tensor, v: Tensor) -> Tensor:
        sensory = self._sensory_activation(inputs)
        w_act = self.W * _sigmoid(v, self.mu, self.sigma)
        sum_in = ((self.sensory_erev * sensory).sum(dim=1) - v * sensory.sum(dim=1)
                  + (self.erev * w_act).sum(dim=1) - v * w_act.sum(dim=1))
        return (1.0 / self.cm_t) * (self.gleak * (self.vleak - v) + sum_in)

    def forward(self, inputs: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
        cfg = self.config
        if cfg.solver == "semi_implicit":
            v = self._step_semi_implicit(inputs, state)
        else:
            step = euler_step if cfg.solver == "explicit" else rk4_step
            f = lambda y: self._dv_dt(inputs, y)  # noqa: E731
            v = state
            for _ in range(cfg.ode_unfolds):
                v = step(f, v, cfg.h)
        return v, v

    def init_state(self, batch_size: int, device: torch.device | None = None) -> Tensor:
        return torch.zeros(batch_size, self.num_units, device=device)
