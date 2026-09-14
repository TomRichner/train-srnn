"""SRNN: a continuous-time rate network with Dale's law, spike-frequency
adaptation (SFA), and short-term synaptic depression (STD).

One ``SRNNCell`` runs K variants of the model side by side. Every parameter
carries a leading K axis and the ablations (no adaptation, E-only, no
Dale's law, frozen recurrent weights, per-neuron parameters, ...) are
multiplicative masks, so the forward pass has no Python branching on the
variant and compiles to one batched graph. The flat state of variant k is

    [ a_E (n_E x n_a_E) | a_I (n_I x n_a_I) | b_E (n_E) | b_I (n_I) | x (N) ]

zero-padded to the widest variant in the batch. Positive parameters are
stored in inverse-softplus space (``isp_*``); per-neuron vectors are paired
with a per-variant scalar or log-gain so that variants without per-neuron
adaptation train only the shared value. See docs/model.md for the equations.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from train_srnn.models.base import RNNCell
from train_srnn.models.ode import euler_step, rk4_step


def inv_softplus(x: float) -> float:
    return math.log(math.expm1(x))


def piecewise_sigmoid(x: Tensor, S_a: float = 0.9, S_c: float = 0.0) -> Tensor:
    """Activation in [0, 1]: linear over a fraction ``S_a`` of the range, quadratic edges."""
    a = S_a / 2.0
    c = S_c
    k = 0.5 / (1.0 - 2.0 * a) if abs(1.0 - 2.0 * a) > 1e-8 else 0.0
    x1, x2, x3, x4 = c + a - 1.0, c - a, c + a, c + 1.0 - a
    quad_rise = k * (x - x1) * (x - x1)
    linear = (x - c) + 0.5
    quad_cap = 1.0 - k * (x - x4) * (x - x4)
    out = torch.where(x < x1, torch.zeros_like(x), quad_rise)
    out = torch.where(x >= x2, linear, out)
    out = torch.where(x > x3, quad_cap, out)
    out = torch.where(x > x4, torch.ones_like(x), out)
    return out


@dataclass
class SRNNConfig:
    """One variant. ``n_a_*`` counts SFA timescales (0 = off); ``n_b_*`` is STD on/off."""

    num_units: int = 32
    dales: bool = True
    n_a_E: int = 3
    n_a_I: int = 3
    n_b_E: int = 1
    n_b_I: int = 1
    per_neuron: bool = False       # train per-neuron adaptation parameters
    echo: bool = False             # frozen recurrent weights (reservoir)
    skip: bool = False             # y = readout(state) + x; applied by SequenceModel
    solver: str = "semi_implicit"  # semi_implicit | explicit | rk4
    h: float = 0.02
    ode_unfolds: int = 1
    readout: str = "synaptic"      # synaptic | rate | dendritic
    tau_global_init: float = 1.0
    tau_a_lo_init: float = 0.25    # fastest SFA timescale (s)
    tau_a_hi_init: float = 4.0     # slowest SFA timescale (s)
    std_zero_floor: bool = True    # rescale b so the synaptic gain reaches 0 at saturation

    @property
    def n_E(self) -> int:
        return self.num_units // 2

    @property
    def n_I(self) -> int:
        return self.num_units - self.n_E

    @property
    def state_size(self) -> int:
        return (self.n_E * self.n_a_E + self.n_I * self.n_a_I
                + self.n_E * self.n_b_E + self.n_I * self.n_b_I + self.num_units)


# Logical parameter groups for freezing; every attribute must exist on the cell.
FREEZE_GROUPS: dict[str, tuple[str, ...]] = {
    "a_0": ("a_0_vec", "a_0_scalar"),
    "W_raw": ("W_raw",), "W_in": ("W_in",),
    "W_raw_gain": ("W_raw_gain",), "W_in_gain": ("W_in_gain",),
    "tau_global": ("isp_tau_global",),
    "tau_d": ("isp_tau_d_vec", "log_tau_d_gain"),
    "tau_a_E": ("isp_tau_a_E_vec", "log_tau_a_E_gain"),
    "c_E": ("isp_c_E_vec", "log_c_E_gain"),
    "c_0_E": ("c_0_E_vec", "c_0_E_scalar"),
    "tau_a_I": ("isp_tau_a_I_vec", "log_tau_a_I_gain"),
    "c_I": ("isp_c_I_vec", "log_c_I_gain"),
    "c_0_I": ("c_0_I_vec", "c_0_I_scalar"),
    "tau_b_rec_E": ("isp_tau_b_rec_E_vec", "log_tau_b_rec_E_gain"),
    "tau_b_rel_E": ("isp_tau_b_rel_E_vec", "log_tau_b_rel_E_gain"),
    "tau_b_rec_I": ("isp_tau_b_rec_I_vec", "log_tau_b_rec_I_gain"),
    "tau_b_rel_I": ("isp_tau_b_rel_I_vec", "log_tau_b_rel_I_gain"),
}


class SRNNCell(RNNCell):
    SOLVERS = ("semi_implicit", "explicit", "rk4")
    READOUTS = {"synaptic": 0, "rate": 1, "dendritic": 2}

    def __init__(self, configs: list[SRNNConfig], input_size: int,
                 rmt_exports: list[dict], W_in_mask: Optional[Tensor] = None):
        if not configs:
            raise ValueError("SRNNCell needs at least one config")
        if len(rmt_exports) != len(configs):
            raise ValueError("Provide one rmt_export per config")
        first = configs[0]
        for c in configs:
            if (c.num_units, c.solver, c.h, c.ode_unfolds) != (
                    first.num_units, first.solver, first.h, first.ode_unfolds):
                raise ValueError("All variants must share num_units, solver, h, and ode_unfolds")
        if first.solver not in self.SOLVERS:
            raise ValueError(f"Unknown solver {first.solver!r}; expected one of {self.SOLVERS}")
        for c in configs:
            if c.readout not in self.READOUTS:
                raise ValueError(f"Unknown readout {c.readout!r}; expected one of {list(self.READOUTS)}")

        N = first.num_units
        n_E, n_I = N // 2, N - N // 2
        self._max = {k: max(getattr(c, k) for c in configs) for k in ("n_a_E", "n_a_I", "n_b_E", "n_b_I")}
        state_size = (n_E * self._max["n_a_E"] + n_I * self._max["n_a_I"]
                      + n_E * self._max["n_b_E"] + n_I * self._max["n_b_I"] + N)
        super().__init__(input_size, N, state_size=state_size, dt=first.h,
                         W_in_mask=W_in_mask, mask_shape=(1, -1, 1))
        self.K, self.N, self.n_E, self.n_I = len(configs), N, n_E, n_I
        self.configs = configs
        self.solver, self.h, self.ode_unfolds = first.solver, first.h, first.ode_unfolds
        self.variant_names: list[str] = [f"variant_{k}" for k in range(self.K)]

        self._init_recurrent(rmt_exports)
        self.a_0_vec = nn.Parameter(torch.full((self.K, N), 0.35))
        self.a_0_scalar = nn.Parameter(torch.zeros(self.K))
        self.isp_tau_global = nn.Parameter(torch.tensor([inv_softplus(c.tau_global_init) for c in configs]))
        self.isp_tau_d_vec = nn.Parameter(torch.full((self.K, N), inv_softplus(0.1)))
        self.log_tau_d_gain = nn.Parameter(torch.zeros(self.K))
        for side, n in (("E", n_E), ("I", n_I)):
            self._init_sfa(side, n)
        for side, n in (("E", n_E), ("I", n_I)):
            self._init_std(side, n)
        self._init_masks()

    # -- construction ---------------------------------------------------------

    @property
    def max_n_a_E(self) -> int:
        return self._max["n_a_E"]

    @property
    def max_n_a_I(self) -> int:
        return self._max["n_a_I"]

    @property
    def max_n_b_E(self) -> int:
        return self._max["n_b_E"]

    @property
    def max_n_b_I(self) -> int:
        return self._max["n_b_I"]

    def _init_recurrent(self, rmt_exports: list[dict]) -> None:
        self.W_raw = nn.Parameter(torch.stack([e["W_init"].clone() for e in rmt_exports]))
        self.register_buffer("sparsity_masks", torch.stack([e["sparsity_mask"].clone() for e in rmt_exports]))
        self.register_buffer("dales_signs", torch.stack([e["dales_sign"].clone() for e in rmt_exports]))
        self.W_in = nn.Parameter(torch.randn(self.K, self.N, self.input_size) * 0.1)
        # Per-variant scalar gains on the whole recurrent / input matrix. Init 1
        # is the identity; they stay trainable for echo variants, whose W_raw
        # structure is frozen but whose overall gain may still adapt.
        self.W_raw_gain = nn.Parameter(torch.ones(self.K))
        self.W_in_gain = nn.Parameter(torch.ones(self.K))

    def _init_sfa(self, side: str, n: int) -> None:
        """SFA parameters for one population: per-neuron vectors plus per-variant gains."""
        A = self._max[f"n_a_{side}"]
        if A == 0:
            for name in ("isp_tau_a_{}_vec", "log_tau_a_{}_gain", "isp_c_{}_vec",
                         "log_c_{}_gain", "c_0_{}_vec", "c_0_{}_scalar"):
                setattr(self, name.format(side), None)
            return
        isp_tau = torch.zeros(self.K, n, A)
        for k, c in enumerate(self.configs):
            n_a = getattr(c, f"n_a_{side}")
            if n_a == 1:
                isp_tau[k, :, 0] = inv_softplus(1.0)
            elif n_a >= 2:   # timescales spread between tau_a_lo and tau_a_hi
                lo, hi = inv_softplus(c.tau_a_lo_init), inv_softplus(c.tau_a_hi_init)
                for a in range(n_a):
                    isp_tau[k, :, a] = lo + (hi - lo) * a / (n_a - 1)
        setattr(self, f"isp_tau_a_{side}_vec", nn.Parameter(isp_tau))
        setattr(self, f"log_tau_a_{side}_gain", nn.Parameter(torch.zeros(self.K)))
        setattr(self, f"isp_c_{side}_vec", nn.Parameter(torch.full((self.K, n, A), inv_softplus(0.05))))
        setattr(self, f"log_c_{side}_gain", nn.Parameter(torch.zeros(self.K)))
        setattr(self, f"c_0_{side}_vec", nn.Parameter(torch.zeros(self.K, n, A)))
        setattr(self, f"c_0_{side}_scalar", nn.Parameter(torch.zeros(self.K)))

    def _init_std(self, side: str, n: int) -> None:
        if self._max[f"n_b_{side}"] == 0:
            for name in ("isp_tau_b_rec_{}_vec", "log_tau_b_rec_{}_gain",
                         "isp_tau_b_rel_{}_vec", "log_tau_b_rel_{}_gain"):
                setattr(self, name.format(side), None)
            return
        setattr(self, f"isp_tau_b_rec_{side}_vec", nn.Parameter(torch.full((self.K, n), inv_softplus(1.0))))
        setattr(self, f"log_tau_b_rec_{side}_gain", nn.Parameter(torch.zeros(self.K)))
        setattr(self, f"isp_tau_b_rel_{side}_vec", nn.Parameter(torch.full((self.K, n), inv_softplus(0.25))))
        setattr(self, f"log_tau_b_rel_{side}_gain", nn.Parameter(torch.zeros(self.K)))

    def _init_masks(self) -> None:
        cfgs, K = self.configs, self.K
        flag = lambda attr: torch.tensor([float(getattr(c, attr)) for c in cfgs])  # noqa: E731
        self.register_buffer("dales_mask", flag("dales").reshape(K, 1, 1))
        self.register_buffer("echo_flags", flag("echo").reshape(K, 1, 1))
        self.register_buffer("skip_flags", flag("skip"))
        self.register_buffer("per_neuron_mask", flag("per_neuron"))
        for side in ("E", "I"):
            A = self._max[f"n_a_{side}"]
            mask = torch.zeros(K, 1, max(A, 1))
            for k, c in enumerate(cfgs):
                mask[k, :, :getattr(c, f"n_a_{side}")] = 1.0
            self.register_buffer(f"sfa_{side}_mask", mask)
            self.register_buffer(f"std_{side}_mask",
                                 torch.tensor([float(getattr(c, f"n_b_{side}") > 0) for c in cfgs]).reshape(K, 1))
        # Fully determined by the configs, so not part of the state_dict.
        self.register_buffer("std_zero_floor_mask", flag("std_zero_floor").reshape(K, 1, 1), persistent=False)
        self.register_buffer("readout_ids", torch.tensor([self.READOUTS[c.readout] for c in cfgs]))

    # -- linked parameters ----------------------------------------------------

    def _linked(self, vec: Tensor) -> Tensor:
        """Detach ``vec`` for variants without per-neuron parameters.

        The value is unchanged; only the gradient is cut, so those variants
        train the paired scalar or gain alone and their per-neuron entries
        stay identical to their initialisation.
        """
        m = self.per_neuron_mask.view(self.K, *([1] * (vec.dim() - 1)))
        return m * vec + (1.0 - m) * vec.detach()

    def _effective_W(self) -> Tensor:
        """``(K, N, N)``: signed softplus magnitudes under Dale's law, sparsity, gain."""
        W_raw = self.echo_flags * self.W_raw.detach() + (1.0 - self.echo_flags) * self.W_raw
        signs = self.dales_signs.unsqueeze(1)
        W = (self.dales_mask * signs * F.softplus(W_raw) + (1.0 - self.dales_mask) * W_raw) * self.sparsity_masks
        return self.W_raw_gain.view(self.K, 1, 1) * W

    def hoist(self) -> Tensor:
        return self._effective_W()

    def skip_mask(self) -> Tensor:
        return self.skip_flags

    def _tau_global(self) -> Tensor:
        return F.softplus(self.isp_tau_global)

    def _tau_d(self) -> Tensor:
        gain = torch.exp(self.log_tau_d_gain).view(self.K, 1)
        return self._tau_global().unsqueeze(-1) * gain * F.softplus(self._linked(self.isp_tau_d_vec))

    def _tau_a(self, side: str) -> Tensor:
        gain = torch.exp(getattr(self, f"log_tau_a_{side}_gain")).view(self.K, 1, 1)
        vec = self._linked(getattr(self, f"isp_tau_a_{side}_vec"))
        return self._tau_global().reshape(self.K, 1, 1) * gain * F.softplus(vec)

    def _tau_b(self, side: str, kind: str) -> Tensor:
        gain = torch.exp(getattr(self, f"log_tau_b_{kind}_{side}_gain")).view(self.K, 1)
        vec = self._linked(getattr(self, f"isp_tau_b_{kind}_{side}_vec"))
        return self._tau_global().unsqueeze(-1) * gain * F.softplus(vec)

    def _a_0(self) -> Tensor:
        return self._linked(self.a_0_vec) + self.a_0_scalar.view(self.K, 1)

    def _c(self, side: str) -> Tensor:
        gain = torch.exp(getattr(self, f"log_c_{side}_gain")).view(self.K, 1, 1)
        return gain * F.softplus(self._linked(getattr(self, f"isp_c_{side}_vec")))

    def _c_0(self, side: str) -> Tensor:
        vec = self._linked(getattr(self, f"c_0_{side}_vec"))
        return vec + getattr(self, f"c_0_{side}_scalar").view(self.K, 1, 1)

    def effective_params(self) -> dict[str, Tensor]:
        """Post-transform values: what actually enters the equations."""
        out = {"W": self._effective_W(), "tau_global": self._tau_global(),
               "tau_d": self._tau_d(), "a_0": self._a_0()}
        for side in ("E", "I"):
            if self._max[f"n_a_{side}"] > 0:
                out[f"tau_a_{side}"] = self._tau_a(side)
                out[f"c_{side}"] = self._c(side)
                out[f"c_0_{side}"] = self._c_0(side)
            if self._max[f"n_b_{side}"] > 0:
                out[f"tau_b_rec_{side}"] = self._tau_b(side, "rec")
                out[f"tau_b_rel_{side}"] = self._tau_b(side, "rel")
        return {k: v.detach() for k, v in out.items()}

    # -- state ----------------------------------------------------------------

    def unpack_state(self, state: Tensor):
        K, B = state.shape[0], state.shape[1]
        n_E, n_I, N = self.n_E, self.n_I, self.N
        A_E, A_I = self._max["n_a_E"], self._max["n_a_I"]
        idx = 0
        if A_E > 0:
            a_E = state[:, :, idx:idx + n_E * A_E].reshape(K, B, n_E, A_E)
            idx += n_E * A_E
        else:
            a_E = torch.zeros(K, B, n_E, 1, device=state.device)
        if A_I > 0:
            a_I = state[:, :, idx:idx + n_I * A_I].reshape(K, B, n_I, A_I)
            idx += n_I * A_I
        else:
            a_I = torch.zeros(K, B, n_I, 1, device=state.device)
        if self._max["n_b_E"] > 0:
            b_E = state[:, :, idx:idx + n_E]
            idx += n_E
        else:
            b_E = torch.ones(K, B, n_E, device=state.device)
        if self._max["n_b_I"] > 0:
            b_I = state[:, :, idx:idx + n_I]
            idx += n_I
        else:
            b_I = torch.ones(K, B, n_I, device=state.device)
        x = state[:, :, idx:idx + N]
        return a_E, a_I, b_E, b_I, x

    def pack_state(self, a_E, a_I, b_E, b_I, x) -> Tensor:
        parts = []
        if self._max["n_a_E"] > 0:
            parts.append(a_E.flatten(start_dim=2))
        if self._max["n_a_I"] > 0:
            parts.append(a_I.flatten(start_dim=2))
        if self._max["n_b_E"] > 0:
            parts.append(b_E)
        if self._max["n_b_I"] > 0:
            parts.append(b_I)
        parts.append(x)
        return torch.cat(parts, dim=-1)

    def init_state(self, batch_size: int, device: torch.device | None = None) -> Tensor:
        """Adaptation at rest (a = 0), synapses fully available (b = 1), x = 0.1 * randn."""
        device = self.W_raw.device if device is None else device
        state = torch.zeros(self.K, batch_size, self.state_size, device=device)
        a_E, a_I, b_E, b_I, x = self.unpack_state(state)
        return self.pack_state(a_E, a_I, torch.ones_like(b_E), torch.ones_like(b_I),
                               0.1 * torch.randn_like(x))

    # -- dynamics -------------------------------------------------------------

    def _input_drive(self, inputs: Tensor) -> Tensor:
        """``u = W_in x`` for every variant: ``(K, B, N)``."""
        if inputs.dim() == 2:
            inputs = inputs.unsqueeze(0).expand(self.K, -1, -1)
        K, B, D = inputs.shape
        W_in = self.W_in if self.W_in_mask is None else self.W_in * self.W_in_mask
        W_in = self.W_in_gain.view(self.K, 1, 1) * W_in
        W_in = W_in.unsqueeze(1).expand(K, B, self.N, D).reshape(K * B, self.N, D)
        return torch.bmm(W_in, inputs.reshape(K * B, D, 1)).squeeze(-1).reshape(K, B, self.N)

    def _recurrent_drive(self, br: Tensor, W_eff: Tensor) -> Tensor:
        """``W br`` for every variant and batch row: ``(K, B, N)``."""
        K, B, N = br.shape
        W = W_eff.unsqueeze(1).expand(K, B, N, N).reshape(K * B, N, N)
        return torch.bmm(W, br.reshape(K * B, N, 1)).squeeze(-1).reshape(K, B, N)

    def _b_full(self, b_E: Tensor, b_I: Tensor) -> Tensor:
        """``(K, B, N)`` synaptic gain: optionally rescaled to [0, 1], ones where STD is off."""
        flag = self.std_zero_floor_mask
        used = {}
        for side, b in (("E", b_E), ("I", b_I)):
            if self._max[f"n_b_{side}"] > 0:
                tau_rec = self._tau_b(side, "rec").unsqueeze(1)
                tau_rel = self._tau_b(side, "rel").unsqueeze(1)
                b_min = tau_rel / (tau_rec + tau_rel)      # asymptote of b at r = 1
                b = b * (1.0 - flag) + ((b - b_min) / (1.0 - b_min)) * flag
            mask = getattr(self, f"std_{side}_mask").unsqueeze(1)
            used[side] = b * mask + (1.0 - mask)
        return torch.cat([used["E"], used["I"]], dim=-1)

    def _drive(self, x, a_E, a_I, b_E, b_I, W_eff: Optional[Tensor]):
        """Firing rate and synaptic output from the state.

        Returns ``(x_eff, r, b_full, Wbr)``; ``Wbr`` is None without ``W_eff``.
        """
        n_E = self.n_E
        x_eff = x.clone()
        if self._max["n_a_E"] > 0:
            contrib = ((self._c("E") * self.sfa_E_mask).unsqueeze(1) * a_E).sum(-1)
            x_eff = torch.cat([x[:, :, :n_E] - contrib, x_eff[:, :, n_E:]], dim=-1)
        if self._max["n_a_I"] > 0:
            contrib = ((self._c("I") * self.sfa_I_mask).unsqueeze(1) * a_I).sum(-1)
            x_eff = torch.cat([x_eff[:, :, :n_E], x_eff[:, :, n_E:] - contrib], dim=-1)
        r = piecewise_sigmoid(x_eff - self._a_0().unsqueeze(1))
        b_full = self._b_full(b_E, b_I)
        Wbr = None if W_eff is None else self._recurrent_drive(b_full * r, W_eff)
        return x_eff, r, b_full, Wbr

    def _split(self, r: Tensor, side: str) -> Tensor:
        return r[:, :, :self.n_E] if side == "E" else r[:, :, self.n_E:]

    def _step_semi_implicit(self, dt, x, a_E, a_I, b_E, b_I, u, W_eff):
        """Linearly implicit Euler: each variable relaxes to its target with its own alpha = dt / tau."""
        _, r, b_full, Wbr = self._drive(x, a_E, a_I, b_E, b_I, W_eff)
        alpha_x = dt / self._tau_d().unsqueeze(1)
        x_new = (x + alpha_x * (u + Wbr)) / (1.0 + alpha_x)
        a_new = {"E": a_E, "I": a_I}
        for side, a in (("E", a_E), ("I", a_I)):
            if self._max[f"n_a_{side}"] > 0:
                alpha = (dt / self._tau_a(side)).unsqueeze(1)
                target = self._c_0(side).unsqueeze(1) + self._split(r, side).unsqueeze(-1)
                mask = getattr(self, f"sfa_{side}_mask").unsqueeze(1)
                a_new[side] = a * (1.0 - mask) + ((a + alpha * target) / (1.0 + alpha)) * mask
        b_new = {"E": b_E, "I": b_I}
        for side, b in (("E", b_E), ("I", b_I)):
            if self._max[f"n_b_{side}"] > 0:
                tau_rec = self._tau_b(side, "rec").unsqueeze(1)
                tau_rel = self._tau_b(side, "rel").unsqueeze(1)
                r_side = self._split(r, side)
                updated = ((b + dt / tau_rec) / (1.0 + dt * (1.0 / tau_rec + r_side / tau_rel))).clamp(0.0, 1.0)
                mask = getattr(self, f"std_{side}_mask").unsqueeze(1)
                b_new[side] = b * (1.0 - mask) + updated * mask
        return x_new, a_new["E"], a_new["I"], b_new["E"], b_new["I"], r, b_full

    def _rhs(self, x, a_E, a_I, b_E, b_I, u, W_eff):
        """Time derivatives ``(dx, da_E, da_I, db_E, db_I)`` plus ``(r, b_full)``."""
        _, r, b_full, Wbr = self._drive(x, a_E, a_I, b_E, b_I, W_eff)
        dx = (-x + u + Wbr) / self._tau_d().unsqueeze(1)
        da = {"E": torch.zeros_like(a_E), "I": torch.zeros_like(a_I)}
        for side, a in (("E", a_E), ("I", a_I)):
            if self._max[f"n_a_{side}"] > 0:
                target = self._c_0(side).unsqueeze(1) + self._split(r, side).unsqueeze(-1)
                da[side] = ((-a + target) / self._tau_a(side).unsqueeze(1)) * getattr(self, f"sfa_{side}_mask").unsqueeze(1)
        db = {"E": torch.zeros_like(b_E), "I": torch.zeros_like(b_I)}
        for side, b in (("E", b_E), ("I", b_I)):
            if self._max[f"n_b_{side}"] > 0:
                tau_rec = self._tau_b(side, "rec").unsqueeze(1)
                tau_rel = self._tau_b(side, "rel").unsqueeze(1)
                db[side] = ((1.0 - b) / tau_rec - self._split(r, side) * b / tau_rel) * getattr(self, f"std_{side}_mask").unsqueeze(1)
        return (dx, da["E"], da["I"], db["E"], db["I"]), r, b_full

    def _step_explicit_family(self, step, dt, x, a_E, a_I, b_E, b_I, u, W_eff):
        first = {}

        def f(y):
            derivs, r, b_full = self._rhs(*y, u, W_eff)
            first.setdefault("r", r)
            first.setdefault("b_full", b_full)
            return derivs

        x, a_E, a_I, b_E, b_I = step(f, (x, a_E, a_I, b_E, b_I), dt)
        # r and b_full evaluated at the pre-step state feed the readout.
        return x, a_E, a_I, b_E.clamp(0.0, 1.0), b_I.clamp(0.0, 1.0), first["r"], first["b_full"]

    def forward(self, inputs: Tensor, state: Tensor, W_eff: Optional[Tensor] = None):
        """One step of ``h`` seconds for every variant.

        Args:
            inputs: ``(B, I)`` shared by all variants, or ``(K, B, I)``.
            state: ``(K, B, state_size)``.
            W_eff: effective recurrent weight from :meth:`hoist`, when the
                caller already computed it for this pass.
        Returns:
            output ``(K, B, N)`` per the variant's readout, and the new state.
        """
        a_E, a_I, b_E, b_I, x = self.unpack_state(state)
        if W_eff is None:
            W_eff = self._effective_W()
        u = self._input_drive(inputs)
        dt = self.h / self.ode_unfolds
        for _ in range(self.ode_unfolds):
            if self.solver == "semi_implicit":
                x, a_E, a_I, b_E, b_I, r, b_full = self._step_semi_implicit(dt, x, a_E, a_I, b_E, b_I, u, W_eff)
            else:
                step = euler_step if self.solver == "explicit" else rk4_step
                x, a_E, a_I, b_E, b_I, r, b_full = self._step_explicit_family(step, dt, x, a_E, a_I, b_E, b_I, u, W_eff)
        is_synaptic = (self.readout_ids == 0).float().reshape(self.K, 1, 1)
        is_rate = (self.readout_ids == 1).float().reshape(self.K, 1, 1)
        is_dendritic = (self.readout_ids == 2).float().reshape(self.K, 1, 1)
        output = is_synaptic * (b_full * r) + is_rate * r + is_dendritic * x
        return output, self.pack_state(a_E, a_I, b_E, b_I, x)

    # -- introspection ----------------------------------------------------------

    def get_diagnostics(self, state: Tensor, inputs: Optional[Tensor] = None) -> dict:
        """Solver-internal quantities for a packed state (no parameters change)."""
        a_E, a_I, b_E, b_I, x = self.unpack_state(state)
        x_eff, r, b_full, _ = self._drive(x, a_E, a_I, b_E, b_I, None)
        out = {"x": x, "x_eff": x_eff, "r": r, "b_E": b_E, "b_I": b_I,
               "b_full": b_full, "a_E": a_E, "a_I": a_I, "br": b_full * r}
        if inputs is not None:
            out["u"] = self._input_drive(inputs)
        return out

    def freeze(self, groups: list[str]) -> list[str]:
        """Pin logical parameter groups (see ``FREEZE_GROUPS``); returns the attributes frozen."""
        frozen = []
        for g in groups:
            if g not in FREEZE_GROUPS:
                raise ValueError(f"Unknown freeze group {g!r}. Valid: {sorted(FREEZE_GROUPS)}")
            hits = [a for a in FREEZE_GROUPS[g] if isinstance(getattr(self, a, None), nn.Parameter)]
            if not hits:
                raise ValueError(f"freeze group {g!r} has no parameters on this cell (ablated away)")
            for a in hits:
                getattr(self, a).requires_grad_(False)
                frozen.append(a)
        return frozen
