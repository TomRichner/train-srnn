# `BatchedSRNNCell` — Full Equations, Parameters, Transforms

Faithful description of every equation, parameter, transform, init, and
ablation mask in
`train_srnn/models/srnn_cell.py::BatchedSRNNCell`. All line numbers in
this document refer to that file.

The cell runs $K$ SRNN ablation variants in parallel via `torch.bmm`.
A leading $K$ axis appears on every parameter and state tensor; the
single-cell `SRNNCell` (lines 172–799) drops that axis but otherwise
performs the same math (with two minor differences flagged in
`KnownIssues.md` §1).

---

## 0. Reading guide & notation

- $K$ — number of stacked ablation variants (`self.K`).
- $B$ — batch size, $T$ — sequence length, $N$ — `num_units`.
- $n_E = \lfloor N/2 \rfloor$, $n_I = N - n_E$ — E / I partition sizes.
- $D$ — `input_size`, $O$ — readout output size.
- $h$ — wall-clock step (`cfg.h`, default 0.02 s).
- $u_f$ — `cfg.ode_unfolds` (default 1).
- $\Delta t = h / u_f$ — solver substep used inside one `forward` call
  (line 1722).
- $n_{aE}^{\max}$, $n_{aI}^{\max}$ — max SFA tier counts across the $K$
  configs (`self.max_n_a_E`, `self.max_n_a_I`); inactive tiers are
  zero-padded then masked.
- $n_{bE}^{\max}, n_{bI}^{\max} \in \{0, 1\}$ analogously for STD.

Each equation block below is given **twice**:

1. **Code form** — using the literal Python attribute names so you can
   grep the source.
2. **LaTeX form** — using single-letter symbols suitable for a paper.

A symbol cheatsheet is at the end (§ 13).

---

## 1. Transform reference

Two parameter-positivity transforms are used:

| transform | code | $f(\theta)$ | inverse used at init | init helper (lines 26–37) |
|---|---|---|---|---|
| **softplus** | `F.softplus` | $\log(1 + e^\theta)$ | $f^{-1}(y) = \log(e^y - 1)$ | `inv_softplus(y)` (scalar), `_softplus_inv(y)` (tensor) |
| **exp** | `torch.exp` | $e^\theta$ | $f^{-1}(y) = \log y$ | plain `math.log` (init at $\log 1 = 0$) |

**Naming convention.** Parameters are prefixed by the transform that
makes them positive at use-time:

- `isp_…` — stored in **inverse-softplus space**, transformed via
  `F.softplus` at use-time. (`isp` for "inverse softplus.")
- `log_…_gain` — stored in **true log space**, transformed via
  `torch.exp` at use-time. Exclusively the per-variant scalar gains.

Identifying the transform is now a pure prefix lookup. (Earlier the
inverse-softplus params were also prefixed `log_…`, which collided
visually with the genuine log-space gains; that was renamed in this
doc's companion refactor.)

Concretely, for any time constant $\tau \in \{\tau_d, \tau_a^E,
\tau_a^I, \tau_b^{\text{rec},E}, \tau_b^{\text{rel},E},
\tau_b^{\text{rec},I}, \tau_b^{\text{rel},I}\}$:

$$
\tau \;=\; \tau_{\text{global}} \cdot \underbrace{\exp(\theta_{\text{gain}})}_{\text{exp on } \log\!\!-\!\text{gain}} \cdot \underbrace{\mathrm{softplus}(\theta_{\text{vec}})}_{\text{softplus on inverse-softplus init}}
$$

i.e. each $\tau$ is built from **three** factors with **two different
positivity transforms** stacked. They are mathematically equivalent in
the sense that both maps $\mathbb{R} \to \mathbb{R}_{>0}$, but they
have different gradient curvatures (softplus saturates linearly for
large positive inputs; exp grows multiplicatively). This asymmetry is
what looked sloppy at the documentation pass and is explicitly called
out here.

**Identity (no positivity transform).** A handful of parameters are
stored in their natural space and can take any sign:

| param | shape | init | semantics |
|---|---|---|---|
| `W_raw_gain` | $(K,)$ | $1.0$ | recurrent gain (ESN spectral-radius knob); free real |
| `W_in_gain` | $(K,)$ | $1.0$ | input gain; free real |
| `a_0_scalar` | $(K,)$ | $0.0$ | shared additive threshold offset |
| `c_0_E_scalar`, `c_0_I_scalar` | $(K,)$ | $0.0$ | shared additive SFA-offset |
| `c_0_E_vec`, `c_0_I_vec` | $(K, n_*, n_a^{\max})$ | $0.0$ | per-neuron SFA offsets |
| `a_0_vec` | $(K, N)$ | $0.35$ | per-neuron threshold |

Of these, `W_raw_gain` and `W_in_gain` initialise to $+1$ (multiplicative
identity) but are *not* in log space — Adam can drive them negative,
which would invert the sign of the recurrent / input drive. This is
intentional in the current design (it lets the optimizer flip overall
gain polarity if needed) but is a candidate for the cleanup.

**The threshold split.** The effective threshold is the sum of a vec
and a scalar:

```
a_0  =  a_0_vec  +  a_0_scalar          # (K, N) + (K,) -> (K, N)
```

When `per_neuron=False`, `a_0_vec`'s gradient is zeroed by a hook
(line 1034) so only `a_0_scalar` moves and the per-neuron values stay
locked at their identical init (0.35). When `per_neuron=True`, both
train: the scalar captures the shared direction and the vec captures
per-neuron deviations.

The same scalar+vec split applies to every per-neuron SFA / STD / dendritic
parameter (`isp_tau_d`, `isp_tau_a_E`, `isp_c_E`, `c_0_E`,
`isp_tau_b_rec_E`, `isp_tau_b_rel_E`, and the I-side mirrors). The vec
component is gradient-masked off in `per_neuron=False` variants.

---

## 2. Configuration & ablation switches

Each of the $K$ stacked variants is described by an `SRNNConfig`
dataclass (lines 82–119). The fields and their defaults:

| field | default | meaning |
|---|---|---|
| `num_units` | 32 | $N$ |
| `dales` | `True` | apply Dale's softplus + sign |
| `n_a_E`, `n_a_I` | 3, 3 | SFA tier counts (0 disables) |
| `n_b_E`, `n_b_I` | 1, 1 | STD presence (0 disables, 1 enables) |
| `per_neuron` | `False` | unmask per-neuron vec gradients |
| `echo` | `False` | freeze `W_raw` (reservoir mode) |
| `skip` | `False` | post-readout residual `y += x_in` (autoregressive only) |
| `solver` | `"semi_implicit"` | one of `semi_implicit`, `explicit`, `rk4`, `exponential` |
| `h` | 0.02 | wall-clock step (s) |
| `ode_unfolds` | 1 | sub-stepping inside `forward` |
| `readout` | `"synaptic"` | one of `synaptic`, `rate`, `dendritic` |
| `tau_global_init` | 1.0 | initial value of $\tau_{\text{global}}$ |
| `std_zero_floor` | `True` | rescale $b \to (b-b_{\min})/(1-b_{\min})$ at readout |

`solver`, `h`, `ode_unfolds` must be identical across the $K$ stacked
configs (asserted at lines 1101–1111). All other fields can vary
across $K$ and are converted into multiplicative masks (§ 9).

A registry of preset configs lives in `SRNN_PRESETS` (lines 126–165);
new ablations are typically added by editing that dict and the matching
YAML in `conf/model/srnn_*.yaml`.

---

## 3. State layout & initialization

`init_state` (lines 1304–1320) returns a packed flat tensor of shape
$(K, B, S_{\max})$ where

$$
S_{\max} \;=\; n_E\, n_{aE}^{\max} \;+\; n_I\, n_{aI}^{\max} \;+\; n_E\, n_{bE}^{\max} \;+\; n_I\, n_{bI}^{\max} \;+\; N
$$

with packing order $[a_E\,|\,a_I\,|\,b_E\,|\,b_I\,|\,x]$
(`unpack_state` at lines 1236–1279, `pack_state` at lines 1281–1293).

Initial values (hard-coded, matches MATLAB `SRNNModel2.initialize_state`):

| state | shape | init |
|---|---|---|
| `a_E` | $(K, B, n_E, n_{aE}^{\max})$ | $0$ |
| `a_I` | $(K, B, n_I, n_{aI}^{\max})$ | $0$ |
| `b_E` | $(K, B, n_E)$ | $1$ (synapses fully available) |
| `b_I` | $(K, B, n_I)$ | $1$ |
| `x` | $(K, B, N)$ | $0.1 \cdot \mathcal{N}(0,1)$ |

Inactive tiers (e.g. `n_a_E=0`) reserve no slots in the packed tensor;
`unpack_state` returns synthetic zero / one tensors of the right shape
so downstream code can branch-freely (lines 1252, 1260, 1267, 1274).

---

## 4. Parameters — code-form effective expressions

Below, each effective quantity appears once with its **code-name
formula** (literal attribute names), its **defining line(s)**, the
**stored shape**, the **initialization**, and notes on overrides /
gradient masks. The companion LaTeX symbol is in column 1 to make the
table searchable.

### 4.1 Recurrent weight $W^{\text{eff}}$ — line 1117

Code form (line 1129–1131):

```
W_eff = W_raw_gain * ( dales_mask * dales_signs * softplus(W_raw)
                     + (1 - dales_mask) * W_raw ) * sparsity_masks
```

| component | shape | init | source / override |
|---|---|---|---|
| `W_raw` | $(K, N, N)$ | from `rmt_export["W_init"]` | RMTMatrix builder (factory.py); not user-tweakable per-config |
| `W_raw_gain` | $(K,)$ | $1.0$ (line 877) | always trainable, hard-coded init |
| `sparsity_masks` | $(K, N, N)$ buffer | from `rmt_export["sparsity_mask"]` | RMTMatrix |
| `dales_signs` | $(K, N)$ buffer | from `rmt_export["dales_sign"]` | RMTMatrix; column-broadcast |
| `dales_mask` | $(K, 1, 1)$ buffer | `float(c.dales)` per-variant (line 991) | from `cfg.dales` |
| `echo_flags` | $(K, 1, 1)$ buffer | `float(c.echo)` (line 995) | from `cfg.echo`; gradient hook on `W_raw` (line 1008) zeros gradient for echo variants — `W_raw_gain` still trains |

**Echo gradient masking** (lines 1004–1008):

```python
echo_grad_mask = 1.0 - echo_flags          # (K, 1, 1): 0 for echo, 1 for non-echo
self.W_raw.register_hook(lambda grad: grad * self._echo_grad_mask)
```

Echo variants therefore have a frozen recurrent **structure** but a
trainable per-variant scalar **gain** (`W_raw_gain`).

### 4.2 Input weight $W^{\text{in,eff}}$ — line 1423

Code form (line 1440–1443):

```
W_in_eff = W_in_gain * (W_in * W_in_mask)        # if W_in_mask provided
         = W_in_gain *  W_in                      # otherwise
```

| component | shape | init | source / override |
|---|---|---|---|
| `W_in` | $(K, N, D)$ | $0.1 \cdot \mathcal{N}(0,1)$ (line 864) | hard-coded scale |
| `W_in_gain` | $(K,)$ | $1.0$ (line 878) | always trainable |
| `W_in_mask` | $(1, N, 1)$ buffer | from external `W_in_mask` arg | typically the input-neuron partition (~25 % of units); built by `factory.build_batched_model` |

The drive is then $u = W^{\text{in,eff}} \cdot u_{\text{ext}}$, computed
batched as $\texttt{u} = \texttt{inputs} \,@\, W^{\text{in,eff}\top}$
(line 1447).

### 4.3 Threshold $a_0$ — line 1212

Code form:

```
a_0 = a_0_vec + a_0_scalar          # (K, N) + (K,1) broadcast
```

| component | shape | init (hard-coded) | gradient mask |
|---|---|---|---|
| `a_0_vec` | $(K, N)$ | $0.35$ (line 885) | zeroed when `per_neuron=False` (line 1034) |
| `a_0_scalar` | $(K,)$ | $0.0$ (line 886) | always trains |

### 4.4 Global timescale $\tau_{\text{global}}$ — line 1135

Code form:

```
tau_global = softplus(isp_tau_global)        # (K,)
```

| component | shape | init | override? |
|---|---|---|---|
| `isp_tau_global` | $(K,)$ | `inv_softplus(c.tau_global_init)`, default `inv_softplus(1.0)` (lines 889–891) | per-variant via `SRNNConfig.tau_global_init` |

### 4.5 Dendritic time constant $\tau_d$ — line 1139

Code form (3-factor decomposition):

```
tau_d = tau_global.unsqueeze(-1)             # (K, 1)
      * exp(log_tau_d_gain).view(K, 1)       # (K, 1)
      * softplus(isp_tau_d_vec)              # (K, N)
```

| component | shape | init | gradient mask |
|---|---|---|---|
| `isp_tau_d_vec` | $(K, N)$ | `inv_softplus(0.1)` (line 896) → effective $\tau_d \approx 0.1$ s base | zeroed when `per_neuron=False` (line 1036) |
| `log_tau_d_gain` | $(K,)$ | $0.0$ → $\exp = 1$ (line 897) | always trains |

Note the asymmetry: `isp_tau_d_vec` uses **softplus**, `log_tau_d_gain`
uses **exp**. Same pattern repeats for every other $\tau$ below.

### 4.6 SFA time constants $\tau^E_a$, $\tau^I_a$ — lines 1144, 1149

Code form (E side; I side mirrors with $n_E \to n_I$):

```
tau_a_E = tau_global.reshape(K, 1, 1)
        * exp(log_tau_a_E_gain).view(K, 1, 1)         # (K, 1, 1)
        * softplus(isp_tau_a_E_vec)                    # (K, n_E, max_n_a_E)
```

| component | shape | init | gradient mask |
|---|---|---|---|
| `isp_tau_a_E_vec` | $(K, n_E, n_{aE}^{\max})$ | per-variant: `n_a_E=0` → all 0 (masked); `n_a_E=1` → `inv_softplus(1.0)` in slot 0; `n_a_E≥2` → **linear interpolation in inverse-softplus space** between `inv_softplus(0.25)` and `inv_softplus(10.0)` across the active tiers (lines 904–914) | zeroed when `per_neuron=False` (line 1039) |
| `log_tau_a_E_gain` | $(K,)$ | $0.0$ (line 916) | always trains |

The interpolation in inverse-softplus space differs from the
single-cell `SRNNCell` (lines 400–411) which interpolates in **log
space** at runtime — this is documented as the multi-tier difference
between the two cells and is the math source of `KnownIssues.md` §1
column 2.

### 4.7 SFA coupling $c_E$, $c_I$ — line 1216

Code form (E):

```
c_E = exp(log_c_E_gain).view(K, 1, 1)        # (K, 1, 1)
    * softplus(isp_c_E_vec)                   # (K, n_E, max_n_a_E)
```

| component | shape | init | gradient mask |
|---|---|---|---|
| `isp_c_E_vec` | $(K, n_E, n_{aE}^{\max})$ | `inv_softplus(0.05)` (line 902) | zeroed when `per_neuron=False` (line 1040) |
| `log_c_E_gain` | $(K,)$ | $0.0$ (line 918) | always trains |

**Note: $c_E$ is NOT scaled by $\tau_{\text{global}}$** — this is a
deliberate departure from the $\tau$ family. $c$ is a coupling, not a
timescale.

### 4.8 SFA offset $c^{0}_E$, $c^{0}_I$ — line 1226

Code form:

```
c_0_E = c_0_E_vec + c_0_E_scalar.view(K, 1, 1)   # (K, n_E, max_n_a_E)
```

| component | shape | init (hard-coded) | gradient mask |
|---|---|---|---|
| `c_0_E_vec` | $(K, n_E, n_{aE}^{\max})$ | $0$ (line 919) | zeroed when `per_neuron=False` (line 1041) |
| `c_0_E_scalar` | $(K,)$ | $0$ (line 920) | always trains |

No positivity transform — offsets can go negative.

### 4.9 STD time constants $\tau^{\text{rec}}_b$, $\tau^{\text{rel}}_b$ — lines 1154–1172

Code form (E side, recovery; relaxation and I side mirror):

```
tau_b_rec_E = tau_global.unsqueeze(-1)              # (K, 1)
            * exp(log_tau_b_rec_E_gain).view(K, 1)  # (K, 1)
            * softplus(isp_tau_b_rec_E_vec)         # (K, n_E)
```

| component | shape | init | gradient mask |
|---|---|---|---|
| `isp_tau_b_rec_E_vec` | $(K, n_E)$ | `inv_softplus(1.0)` (line 960) | zeroed when `per_neuron=False` (line 1049) |
| `log_tau_b_rec_E_gain` | $(K,)$ | $0.0$ (line 962) | always trains |
| `isp_tau_b_rel_E_vec` | $(K, n_E)$ | `inv_softplus(0.25)` (line 964) | zeroed when `per_neuron=False` (line 1050) |
| `log_tau_b_rel_E_gain` | $(K,)$ | $0.0$ (line 966) | always trains |

I-side analogues (`*_I_vec`, `*_I_gain`) at lines 974–987 with the
same hard-coded inits.

### 4.10 Per-variant $b$-zero-floor flag — lines 1087–1090

```
std_zero_floor_mask = float(c.std_zero_floor)      # (K, 1, 1) buffer (non-persistent)
```

Per-variant $0$/$1$ gating; non-persistent so old checkpoints load
cleanly. Used to rescale $b$ via $b \mapsto (b - b_{\min})/(1 - b_{\min})$
at readout (§ 5.3), with $b_{\min} = \tau^{\text{rel}}/(\tau^{\text{rec}} + \tau^{\text{rel}})$
the steady-state value at $r = 1$.

---

## 5. Forward dynamics — equations in code-form variable names

This section is the literal RHS implemented by `_compute_b_full`,
`_batched_compute_rhs` and `_batched_step_semi_implicit`. The next
section (§ 6) restates the same equations in single-letter LaTeX.

### 5.1 Effective potential (SFA subtraction) — lines 1457–1472

```
sfa_E_contrib[i] = sum_k  c_E_masked[i, k] * a_E[i, k]      # (K, B, n_E)
sfa_I_contrib[j] = sum_k  c_I_masked[j, k] * a_I[j, k]      # (K, B, n_I)

x_eff[:, :n_E] = x[:, :n_E] - sfa_E_contrib
x_eff[:, n_E:] = x[:, n_E:] - sfa_I_contrib
```

where `c_E_masked = c_E * sfa_E_mask` (and same for I) zeroes
inactive SFA tiers.

### 5.2 Firing rate — line 1475

```
r = piecewise_sigmoid( x_eff - a_0.unsqueeze(1) )            # (K, B, N)
```

`piecewise_sigmoid` (lines 40–75) is a $C^1$ approximation to a
sigmoid: it is $0$ for $x < x_1$, then a quadratic rise on
$[x_1, x_2]$, then linear on $[x_2, x_3]$ with slope $1$, then a
quadratic cap on $[x_3, x_4]$, then $1$ for $x > x_4$, with
$x_{1..4} = c \pm a, \, c \pm 1 \mp a$ where $a = S_a / 2$, $c = S_c$,
and $k = 0.5/(1 - 2a)$. Defaults: $S_a = 0.9$, $S_c = 0$.

### 5.3 STD-rescaled synaptic gain $b_{\text{full}}$ — lines 1174–1208

For the E side (I mirrors):

```
tau_rec_E    = self._tau_b_rec_E().unsqueeze(1)               # (K, 1, n_E)
tau_rel_E    = self._tau_b_rel_E().unsqueeze(1)
b_min_E      = tau_rel_E / (tau_rec_E + tau_rel_E)            # steady state at r=1
b_E_rescaled = (b_E - b_min_E) / (1 - b_min_E)                # zero-floor map
b_E_used     = b_E * (1 - flag)  +  b_E_rescaled * flag       # blend by std_zero_floor_mask
b_full_E     = b_E_used * std_E_mask + (1 - std_E_mask)       # 1 when STD inactive
```

The full $(K, B, N)$ vector is `b_full = cat([b_full_E, b_full_I], -1)`.

### 5.4 Recurrent drive — lines 1397–1419

```
br = b_full * r                                                # (K, B, N)
Wbr = bmm(W_eff, br_col).squeeze(-1)                           # (K, B, N)
```

(Implementation expands `W_eff` to $(KB, N, N)$, reshapes `br` to
$(KB, N, 1)$, runs `torch.bmm`, reshapes back.)

### 5.5 Dendritic ODE — line 1574 (RHS) / line 1488 (semi-implicit)

```
dx_dt = (-x + u + Wbr) / tau_d                                  # explicit / RK4 RHS
```

Semi-implicit closed form (Backward-Euler on the linear part):

```
alpha_x = dt / tau_d                                            # (K, 1, N)
x_new   = (x + alpha_x * (u + Wbr)) / (1 + alpha_x)
```

Exponential Euler (line 1663):

```
decay_x = exp(-dt / tau_d)
x_new   = x * decay_x + (1 - decay_x) * (u + Wbr)
```

### 5.6 SFA ODE — lines 1497, 1581 (E; I mirrors)

```
da_E_dt[i, k] = ( -a_E[i, k] + c_0_E[i, k] + r_E[i] ) / tau_a_E[i, k]
                * sfa_E_mask[k]
```

Semi-implicit:

```
alpha_a_E = dt / tau_a_E                                        # (K, n_E, max_n_a_E)
a_E_updated = (a_E + alpha_a_E * (c_0_E + r_E)) / (1 + alpha_a_E)
a_E_new     = a_E * (1 - sfa_E_mask) + a_E_updated * sfa_E_mask
```

(The mask blend keeps unused tiers at their padded value of 0 so that
the gradient never flows through inactive tiers.)

### 5.7 STD ODE — lines 1521, 1598 (E; I mirrors)

```
db_E_dt[i] = (1 - b_E[i]) / tau_b_rec_E[i]
           - r_E[i] * b_E[i] / tau_b_rel_E[i]
db_E_dt    *= std_E_mask
```

Semi-implicit (Backward-Euler on both linear parts; the bilinear
$b \cdot r$ term is treated implicitly in $b$):

```
b_E_updated = (b_E + dt / tau_b_rec_E)
            / (1 + dt * (1 / tau_b_rec_E + r_E / tau_b_rel_E))
b_E_updated = b_E_updated.clamp(0, 1)
b_E_new     = b_E * (1 - std_E_mask) + b_E_updated * std_E_mask
```

The `clamp(0, 1)` enforces the physical range of an
available-resource fraction; without it, large explicit-Euler steps
can briefly leave the simplex.

### 5.8 Readout & skip — line 1750 (cell), `sequence_model.py:250` (skip)

The cell emits one of three pre-readout signals according to per-variant
`readout_ids`:

```
out_synaptic  = b_last * r_last
out_rate      = r_last
out_dendritic = x

is_synaptic   = (readout_ids == 0).float().reshape(K, 1, 1)
is_rate       = (readout_ids == 1).float().reshape(K, 1, 1)
is_dendritic  = (readout_ids == 2).float().reshape(K, 1, 1)

output = is_synaptic * out_synaptic + is_rate * out_rate + is_dendritic * out_dendritic
```

In `SequenceModel`, this `output` is multiplied by the optional
`output_mask`, projected through a $(K, O, N)$ readout head, then
optionally augmented with a per-variant skip residual:

```
logits = einsum("k...e,koe->k...o", out_t, readout_weight) + readout_bias
if any_skip:
    logits = logits + skip_flags.view(K, 1, 1) * x_in_t       # autoregressive only
```

`skip_flags` is the $(K,)$ buffer registered at line 1001;
non-skip variants contribute $0$. The skip residual is in OUTPUT
space (post-readout), not state space.

---

## 6. The same equations in LaTeX symbols

Drop the $K$ axis (one variant). Let

- $i \in \{1, \dots, n_E\}$ index excitatory neurons (E), and use
  $\bar i \in \{n_E+1, \dots, N\}$ for inhibitory (I). Where convenient
  we treat all neurons uniformly with $i \in \{1, \dots, N\}$ and let
  the side-specific quantities be selected by membership.
- $k \in \{1, \dots, n_a^{(i)}\}$ index SFA tiers, where
  $n_a^{(i)} = n_{aE}$ if $i \le n_E$, else $n_{aI}$.

### 6.1 Effective inputs

$$
x^{\text{eff}}_i \;=\; x_i \;-\; \sum_{k=1}^{n_a^{(i)}} c_{i,k}\, a_{i,k}
$$

$$
r_i \;=\; \phi\!\left( x^{\text{eff}}_i \;-\; a_{0,i} \right),
\qquad \phi \;=\; \texttt{piecewise\_sigmoid}
$$

$$
\tilde b_i \;=\;
\begin{cases}
\dfrac{b_i - b_{\min, i}}{1 - b_{\min, i}}, & \text{if } \texttt{std\_zero\_floor} \\[6pt]
b_i, & \text{otherwise}
\end{cases},
\qquad b_{\min, i} \;=\; \frac{\tau^{\text{rel}}_i}{\tau^{\text{rec}}_i + \tau^{\text{rel}}_i}
$$

$$
u \;=\; W^{\text{in,eff}} \, u_{\text{ext}}, \qquad
W^{\text{in,eff}} \;=\; g^{\text{in}}\, (W^{\text{in}} \odot M^{\text{in}})
$$

### 6.2 Recurrent weight

$$
W^{\text{eff}}_{ij} \;=\; g^{W} \cdot \big[\, d \cdot s_j \cdot \mathrm{softplus}(W^{\text{raw}}_{ij}) \;+\; (1-d)\, W^{\text{raw}}_{ij}\,\big] \cdot M^{\text{sp}}_{ij}
$$

with $d \in \{0, 1\}$ the per-variant Dale's flag, $s_j \in \{+1, -1\}$
the sign of column $j$, $M^{\text{sp}}$ the sparsity mask, and $g^W$
the trainable scalar `W_raw_gain`.

### 6.3 ODE system

$$
\boxed{\;\;
\tau_{d, i}\, \dot x_i \;=\; -\, x_i \;+\; u_i \;+\; \sum_{j=1}^{N} W^{\text{eff}}_{ij}\, \tilde b_j \, r_j
\;\;}
\qquad\text{(line 1574)}
$$

$$
\boxed{\;\;
\tau_{a, (i,k)}\, \dot a_{i,k} \;=\; -\, a_{i,k} \;+\; c^{0}_{(i,k)} \;+\; r_i
\;\;}
\qquad\text{(lines 1577–1591)}
$$

$$
\boxed{\;\;
\dot b_i \;=\; \frac{1 - b_i}{\tau^{\text{rec}}_i} \;-\; \frac{b_i\, r_i}{\tau^{\text{rel}}_i}
\;\;}
\qquad\text{(lines 1594–1608)}
$$

(SFA and STD equations are multiplied by their respective binary
masks `sfa_*_mask`, `std_*_mask`, which gate the entire RHS to $0$
for inactive tiers / sides.)

### 6.4 Time-constant decompositions

For each $\tau \in \{\tau_d, \tau_a^E, \tau_a^I, \tau_b^{\text{rec},E}, \tau_b^{\text{rel},E}, \tau_b^{\text{rec},I}, \tau_b^{\text{rel},I}\}$:

$$
\tau_{*,i} \;=\; \underbrace{\mathrm{softplus}(\theta^{\tau,\text{global}})}_{\tau_{\text{global}}}
\;\cdot\; \underbrace{\exp(\theta^{\tau,\text{gain}})}_{g^\tau}
\;\cdot\; \underbrace{\mathrm{softplus}(\theta^{\tau,\text{vec}}_i)}_{\text{per-neuron}}
$$

For SFA coupling $c$ (E and I), the formula is the same minus the
$\tau_{\text{global}}$ factor:

$$
c_{(i,k)} \;=\; \exp(\theta^{c,\text{gain}}) \cdot \mathrm{softplus}(\theta^{c,\text{vec}}_{(i,k)})
$$

For all additive parameters $\theta \in \{a_0, c^{0}_E, c^{0}_I\}$:

$$
\theta_i \;=\; \theta^{\text{vec}}_i \;+\; \theta^{\text{scalar}}
$$

with no positivity transform. Per-neuron splits are enabled by zeroing
the gradient on $\theta^{\text{vec}}$ when `per_neuron=False`, leaving
the optimizer to move only $\theta^{\text{scalar}}$ — i.e. the
"per-neuron" term collapses to "shared across neurons."

### 6.5 Solvers

For brevity write the dendritic ODE generically as
$\tau \dot x = -x + F(x, t)$ with $F(x, t) = u + W \tilde b r$ taken
**explicit** in $x$ inside one substep.

- **Explicit Euler** (line 1612):
  $x \leftarrow x + \Delta t\, (-x + F)/\tau$.
- **Semi-implicit / linearly-implicit Euler** (line 1488): treat the
  $-x/\tau$ term implicitly,
  $x \leftarrow (x + \alpha\, F)/(1 + \alpha)$ with $\alpha = \Delta t / \tau$.
- **RK4** (line 1626): four evaluations of `_batched_compute_rhs`,
  classical weights $\tfrac{\Delta t}{6}(k_1 + 2k_2 + 2k_3 + k_4)$.
- **Exponential Euler** (line 1654):
  $x \leftarrow x\, e^{-\Delta t/\tau} + (1 - e^{-\Delta t/\tau})\, F$.

The same template is applied analogously to the SFA equation
($-a/\tau$ implicit, $r/\tau$ explicit). The STD equation uses the
*bilinear-implicit* form $b_{\text{new}} = (b + \Delta t/\tau^{\text{rec}}) /
(1 + \Delta t \cdot (1/\tau^{\text{rec}} + r/\tau^{\text{rel}}))$ in the
semi-implicit solver, and explicit Euler in the exponential solver.

---

## 7. Ablation masks — full listing

All masks are buffers (not parameters); they are derived from the
per-variant `SRNNConfig` at `__init__` time and are constant across
training. Multiplicative masks live in $\{0, 1\}$.

| mask | shape | source field | effect |
|---|---|---|---|
| `dales_mask` | $(K, 1, 1)$ | `cfg.dales` | gates the softplus + sign branch in `_effective_W` (line 1129) |
| `dales_signs` | $(K, N)$ | `rmt_export["dales_sign"]` | column sign $\pm 1$; multiplied into $W^{\text{eff}}$ when Dale's is on |
| `sparsity_masks` | $(K, N, N)$ | `rmt_export["sparsity_mask"]` | structural zeros in $W^{\text{eff}}$ |
| `echo_flags` | $(K, 1, 1)$ | `cfg.echo` | source for `_echo_grad_mask = 1 - echo_flags`, applied as a backward hook on `W_raw` (line 1008) |
| `_echo_grad_mask` | $(K, 1, 1)$ | derived | zeros gradient on `W_raw` for echo variants; `W_raw_gain` is unaffected |
| `skip_flags` | $(K,)$ | `cfg.skip` | adds `flag * x_in_t` to logits in the readout (sequence_model.py:257) |
| `sfa_E_mask`, `sfa_I_mask` | $(K, 1, n_a^{\max})$ | `cfg.n_a_E`, `cfg.n_a_I` | per-tier $0/1$; zeroes inactive SFA tier contributions to $x^{\text{eff}}$ and to $\dot a$ |
| `std_E_mask`, `std_I_mask` | $(K, 1)$ | `cfg.n_b_E > 0`, `cfg.n_b_I > 0` | gates $b_{\text{full}}$ to $1$ and $\dot b$ to $0$ when STD off |
| `std_zero_floor_mask` | $(K, 1, 1)$, non-persistent | `cfg.std_zero_floor` | blends raw $b$ vs. $(b - b_{\min})/(1 - b_{\min})$ in `_compute_b_full` |
| `readout_ids` | $(K,)$ long | `cfg.readout` | integer code 0 / 1 / 2 ; selects `b·r` vs. `r` vs. `x` at output |
| `_a_0_vec_mask` | $(K, 1)$ | `cfg.per_neuron` | gradient hook on `a_0_vec` (line 1034) |
| `_isp_tau_d_vec_mask` | $(K, 1)$ | `cfg.per_neuron` | gradient hook on `isp_tau_d_vec` (line 1036) |
| `_isp_tau_a_E_vec_mask` | $(K, 1, 1)$ | `cfg.per_neuron` | gradient hook on `isp_tau_a_E_vec` (line 1039) |
| `_isp_c_E_vec_mask` | $(K, 1, 1)$ | `cfg.per_neuron` | gradient hook on `isp_c_E_vec` (line 1040) |
| `_c_0_E_vec_mask` | $(K, 1, 1)$ | `cfg.per_neuron` | gradient hook on `c_0_E_vec` (line 1041) |
| `_isp_tau_a_I_vec_mask` etc. | $(K, 1, 1)$ | `cfg.per_neuron` | I-side analogues, lines 1043–1046 |
| `_isp_tau_b_rec_E_vec_mask`, `_isp_tau_b_rel_E_vec_mask` | $(K, 1)$ | `cfg.per_neuron` | gradient hooks (lines 1049–1050) |
| `_isp_tau_b_rec_I_vec_mask`, `_isp_tau_b_rel_I_vec_mask` | $(K, 1)$ | `cfg.per_neuron` | gradient hooks (lines 1053–1054) |
| `W_in_mask` | $(1, N, 1)$ | external arg from `factory.build_batched_model` | zeros $W^{\text{in}}$ rows for non-input neurons (the ~25 % input partition) |

### 7.1 Per-neuron linking (the $\theta = \theta^{\text{vec}} + \theta^{\text{scalar}}$ pattern)

The `per_neuron` flag controls **only** whether the
`*_vec` component receives gradient updates. The scalar / gain
component **always** trains. As a result:

| `per_neuron` | scalar / gain trains | vec trains | semantics |
|---|---|---|---|
| `False` | yes | **no** (gradient masked to 0) | shared per-variant value; vec stays at hard-coded init |
| `True` | yes | yes | scalar is shared direction; vec is per-neuron deviation |

This is the "linking" architecture documented in `KnownIssues.md` §1.
It applies uniformly to every per-neuron parameter:
`a_0`, `isp_tau_d`, `isp_tau_a_E/I`, `isp_c_E/I`, `c_0_E/I`,
`isp_tau_b_rec_E/I`, `isp_tau_b_rel_E/I` (i.e. 11 parameter families).

---

## 8. Initialization summary table

Every initial value used at construction time, with its source. "Hard-coded"
means the value is a Python literal in `srnn_cell.py` and is **not**
overrideable through `SRNNConfig` or YAML. Overrides can still be made
by editing the file or by post-construction parameter assignment, but
not via the standard config path.

| parameter | init expression | hard-coded? | line |
|---|---|---|---|
| `W_raw` | `rmt_export["W_init"]` | no — comes from RMTMatrix builder in factory | 859 |
| `W_in` | `0.1 * randn(K, N, D)` | yes — scale `0.1` | 864 |
| `W_raw_gain` | `ones(K)` | yes | 877 |
| `W_in_gain` | `ones(K)` | yes | 878 |
| `a_0_vec` | `full((K, N), 0.35)` | yes | 885 |
| `a_0_scalar` | `zeros(K)` | yes | 886 |
| `isp_tau_global` | `inv_softplus(c.tau_global_init)` | overrideable (`cfg.tau_global_init`) | 889–891 |
| `isp_tau_d_vec` | `inv_softplus(0.1)` | yes | 896 |
| `log_tau_d_gain` | `zeros(K)` | yes | 897 |
| `isp_tau_a_E_vec` | per-tier; `n_a_E=1` → `inv_softplus(1.0)`; `n_a_E≥2` → linear interp in inverse-softplus space between `inv_softplus(0.25)` and `inv_softplus(10.0)` | yes | 904–914 |
| `log_tau_a_E_gain` | `zeros(K)` | yes | 916 |
| `isp_c_E_vec` | `inv_softplus(0.05)` | yes | 902 |
| `log_c_E_gain` | `zeros(K)` | yes | 918 |
| `c_0_E_vec`, `c_0_E_scalar` | $0$ | yes | 919–920 |
| `isp_tau_a_I_*`, `isp_c_I_*`, `c_0_I_*` | I-side mirrors of E above | yes | 930–948 |
| `isp_tau_b_rec_E_vec` | `inv_softplus(1.0)` | yes | 960 |
| `log_tau_b_rec_E_gain` | `zeros(K)` | yes | 962 |
| `isp_tau_b_rel_E_vec` | `inv_softplus(0.25)` | yes | 964 |
| `log_tau_b_rel_E_gain` | `zeros(K)` | yes | 966 |
| `isp_tau_b_*_I_*` | I-side mirrors | yes | 974–982 |
| state `a_E`, `a_I` | $0$ | yes (matches MATLAB) | init_state |
| state `b_E`, `b_I` | $1$ | yes | init_state |
| state `x` | $0.1 \cdot \mathcal{N}(0, 1)$ | yes | init_state |

**About `freeze_params`.** A recent CLI knob (commit 34600c5) lets the
user pin selected parameters at their init values for the whole run.
This is implemented at the *training-loop / parameter-group* level, not
inside the cell — `BatchedSRNNCell` itself is unaware of it and would
behave identically with or without the knob.

---

## 9. Effective values vs stored values — quick reference

When inspecting a checkpoint by hand, remember:

```
stored                        effective in the ODE
------------------------------------------------------------------
isp_tau_global         ->     softplus(.)
isp_tau_*_vec          ->     softplus(.)          (so 0 -> ~0.693 s baseline)
log_tau_*_gain         ->     exp(.)                (so 0 -> 1.0 multiplier)
isp_c_*_vec            ->     softplus(.)
log_c_*_gain           ->     exp(.)
c_0_*_vec, c_0_*_scalar ->    identity (additive)
a_0_vec, a_0_scalar    ->     identity (additive)
W_raw_gain, W_in_gain  ->     identity (multiplicative; can go negative)
W_raw (Dale's on)      ->     dales_sign · softplus(.)
W_raw (Dale's off)     ->     identity
```

The plot helpers in `scripts/plots/` (e.g. `plot_all_taus.py`,
`weight_evolution.png`) and the param tables produced by
`scripts/postprocess.py` always report **effective** values to avoid
the mental gymnastics above.

---

## 10. Inverse transforms (init helpers)

Two helpers in lines 26–37:

```
inv_softplus(x: float)   -> log(expm1(x))                      # scalar, used at __init__
_softplus_inv(x: Tensor) -> torch.log(torch.expm1(x))          # tensor
```

Both require $x > 0$. Numerically stable for large $x$ (where
$\mathrm{softplus}(y) \approx y$), they reduce to $\log(e^x - 1)$.

There is no helper for `exp` because every log-gain is initialised to
$\log 1 = 0$ directly.

---

## 11. Side-by-side: stored parameter ↔ effective ODE coefficient

A condensed all-in-one cheatsheet of every quantity that appears on
the RHS of the ODE.

| ODE symbol | stored params (code) | line | effective expression (code) |
|---|---|---|---|
| $x_i$ | `state["x"]` | 1277 | — (state) |
| $a^E_{i,k}$ | `state["a_E"]` | 1249 | — (state) |
| $a^I_{j,k}$ | `state["a_I"]` | 1257 | — (state) |
| $b^E_i$ | `state["b_E"]` | 1264 | — (state) |
| $b^I_j$ | `state["b_I"]` | 1271 | — (state) |
| $u_i$ | `W_in`, `W_in_gain`, `W_in_mask` | 1423–1447 | `W_in_gain * (W_in * W_in_mask) @ inputs.T` |
| $W^{\text{eff}}_{ij}$ | `W_raw`, `W_raw_gain`, `dales_signs`, `dales_mask`, `sparsity_masks` | 1117–1131 | see § 4.1 |
| $a_{0,i}$ | `a_0_vec`, `a_0_scalar` | 1212 | `a_0_vec + a_0_scalar.view(K,1)` |
| $\tau_{\text{global}}$ | `isp_tau_global` | 1135 | `softplus(isp_tau_global)` |
| $\tau_{d,i}$ | `log_tau_d_gain`, `isp_tau_d_vec`, plus $\tau_{\text{global}}$ | 1139 | `tau_global * exp(log_tau_d_gain) * softplus(isp_tau_d_vec)` |
| $\tau_{a,(i,k)}^E$ | `log_tau_a_E_gain`, `isp_tau_a_E_vec`, plus $\tau_{\text{global}}$ | 1144 | `tau_global * exp(log_tau_a_E_gain) * softplus(isp_tau_a_E_vec)` |
| $\tau_{a,(j,k)}^I$ | `log_tau_a_I_gain`, `isp_tau_a_I_vec`, plus $\tau_{\text{global}}$ | 1149 | analogous |
| $c^E_{(i,k)}$ | `log_c_E_gain`, `isp_c_E_vec` | 1216 | `exp(log_c_E_gain) * softplus(isp_c_E_vec)` |
| $c^I_{(j,k)}$ | `log_c_I_gain`, `isp_c_I_vec` | 1221 | analogous |
| $c^{0,E}_{(i,k)}$ | `c_0_E_vec`, `c_0_E_scalar` | 1226 | `c_0_E_vec + c_0_E_scalar.view(K,1,1)` |
| $c^{0,I}_{(j,k)}$ | `c_0_I_vec`, `c_0_I_scalar` | 1230 | analogous |
| $\tau^{\text{rec},E}_{b, i}$ | `log_tau_b_rec_E_gain`, `isp_tau_b_rec_E_vec`, plus $\tau_{\text{global}}$ | 1154 | `tau_global * exp(.) * softplus(.)` |
| $\tau^{\text{rel},E}_{b, i}$ | `log_tau_b_rel_E_gain`, `isp_tau_b_rel_E_vec`, plus $\tau_{\text{global}}$ | 1159 | analogous |
| $\tau^{\text{rec},I}_{b, j}$ | `log_tau_b_rec_I_gain`, `isp_tau_b_rec_I_vec`, plus $\tau_{\text{global}}$ | 1164 | analogous |
| $\tau^{\text{rel},I}_{b, j}$ | `log_tau_b_rel_I_gain`, `isp_tau_b_rel_I_vec`, plus $\tau_{\text{global}}$ | 1169 | analogous |
| $b_{\min, i}$ | (derived) | 1190, 1200 | `tau_b_rel / (tau_b_rec + tau_b_rel)` |
| $\tilde b_i$ | $b_i$ + `std_zero_floor_mask` + `std_E_mask`/`std_I_mask` | 1174–1208 | see § 5.3 |
| $\phi(\cdot)$ | (no params) | 40–75 | piecewise-linear with quadratic edges |

---

## 12. Inconsistencies worth resolving

Surfaced by writing this doc. Recording them here so the cleanup pass
has a checklist:

1. **Mixed positivity transforms inside one $\tau$.** `softplus` for
   `isp_*_vec`, `exp` for `log_*_gain`. Picking one would simplify
   reasoning and gradient analysis. Note `exp` was likely chosen for
   the gain parameters because it gives true multiplicative identity at
   $\theta = 0$, while $\mathrm{softplus}(0) = \log 2 \approx 0.693$ would offset
   the baseline.
2. ~~**`log_*_vec` is misnamed.**~~ **Resolved.** Inverse-softplus
   parameters are now prefixed `isp_*` (e.g. `isp_tau_d_vec`,
   `isp_tau_a_E_vec`). Genuine log-space scalars keep `log_*_gain`.
   The transform is now a pure prefix lookup.
3. **`W_raw_gain`, `W_in_gain` are unconstrained.** They initialise to
   $+1$ but can be driven to any sign by the optimizer. Decide whether
   to (a) leave free, (b) constrain to $\mathbb{R}_{>0}$ via
   `softplus`, or (c) parameterise in log space. Same call as for the
   timescale gains.
4. **`c_E` skips the $\tau_{\text{global}}$ factor** but otherwise uses
   the same gain × vec template as the time constants. Document or
   re-evaluate.
5. **Multi-tier SFA init disagrees between `SRNNCell` and
   `BatchedSRNNCell`.** Single-cell interpolates in **log** space
   at runtime (line 411); batched interpolates in **inverse-softplus**
   space at init (line 913). See `KnownIssues.md` §1.

---

## 13. Symbol cheatsheet

| LaTeX | code | meaning | shape (one variant) |
|---|---|---|---|
| $x_i$ | `x` | dendritic potential | $(B, N)$ |
| $a_{i,k}$ | `a_E[i,k]`, `a_I[j,k]` | SFA state, tier $k$ | $(B, n_*, n_a^{\max})$ |
| $b_i$ | `b_E[i]`, `b_I[j]` | available-resource fraction (STD) | $(B, n_*)$ |
| $r_i$ | `r` | firing rate | $(B, N)$ |
| $\tilde b_i$ | `b_full` | effective synaptic gain | $(B, N)$ |
| $u_i$ | `u` | external drive | $(B, N)$ |
| $W^{\text{eff}}_{ij}$ | `W_eff` | effective recurrent weight | $(N, N)$ |
| $W^{\text{in,eff}}$ | `W_in_eff` | effective input weight | $(N, D)$ |
| $a_{0,i}$ | `_a_0()` | threshold | $(N,)$ |
| $\tau_d$ | `_tau_d()` | dendritic time constant | $(N,)$ |
| $\tau_{a,(i,k)}$ | `_tau_a_E()`, `_tau_a_I()` | SFA time constant | $(n_*, n_a^{\max})$ |
| $c_{(i,k)}$ | `_c_E()`, `_c_I()` | SFA coupling | $(n_*, n_a^{\max})$ |
| $c^{0}_{(i,k)}$ | `_c_0_E()`, `_c_0_I()` | SFA offset | $(n_*, n_a^{\max})$ |
| $\tau^{\text{rec}}_b, \tau^{\text{rel}}_b$ | `_tau_b_rec_E/I()`, `_tau_b_rel_E/I()` | STD time constants | $(n_*,)$ |
| $b_{\min}$ | (derived in `_compute_b_full`) | STD zero-floor | $(n_*,)$ |
| $g^W$ | `W_raw_gain` | recurrent gain | $()$ |
| $g^{\text{in}}$ | `W_in_gain` | input gain | $()$ |
| $\tau_{\text{global}}$ | `_tau_global()` | global timescale multiplier | $()$ |
| $d$ | `dales_mask` | Dale's flag | $()$ |
| $s_j$ | `dales_signs` | column sign $\pm 1$ | $(N,)$ |
| $M^{\text{sp}}$ | `sparsity_masks` | sparsity mask | $(N, N)$ |
| $M^{\text{in}}$ | `W_in_mask` | input-neuron partition mask | $(N, 1)$ |
| $\phi$ | `piecewise_sigmoid` | bounded $[0,1]$ activation | scalar fn |
| $\Delta t$ | `dt = h / ode_unfolds` | solver substep | scalar |
