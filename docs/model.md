# SRNN model specification

Complete mathematical specification of the SRNN as implemented by
`SRNNCell` (`train_srnn/models/srnn_cell.py`), wrapped by `SequenceModel`
(`train_srnn/models/sequence_model.py`), with recurrent weights from
`RMTMatrix` (`train_srnn/models/rmt_matrix.py`). Every formula uses the
exact transform applied in code; parameter names in the math match the keys
in the saved `model_state_dict`. Where this document and the code disagree,
the code is authoritative.

The SRNN is a continuous-time rate network with Dale's law, spike-frequency
adaptation (SFA) and short-term synaptic depression (STD). One `SRNNCell`
runs $K$ variants of the model side by side: every parameter and state
tensor carries a leading $K$ axis, and the ablations (no adaptation, E-side
only, no Dale's law, frozen recurrent weights, per-neuron parameters, skip
residual) are multiplicative masks, so the forward pass has no branching on
the variant. All $K$ variants share `num_units`, `solver`, `h` and
`ode_unfolds`.

**Notation.** $K$ = number of variants, $B$ = batch, $T$ = sequence length,
$N$ = `num_units`, $n_E = \lfloor N/2 \rfloor$, $n_I = N - n_E$,
$D$ = `input_size`, $O$ = `output_size`, $E$ = number of readout neurons after
I/O masking, $h$ = seconds of simulated time per cell call,
$n_{\text{ode}}$ = `ode_unfolds`, $\Delta t = h / n_{\text{ode}}$.
$n_{aE}^{\max}$, $n_{aI}^{\max}$ are the maximum SFA tier counts over the $K$
variants; $n_{bE}^{\max}, n_{bI}^{\max} \in \{0, 1\}$ likewise for STD.
$\mathrm{softplus}(\theta) = \log(1 + e^\theta)$ and
$\mathrm{softplus}^{-1}(y) = \log(e^y - 1)$ (`inv_softplus`).

---

## 1. State

The cell state of variant $k$ is one flat vector per batch element, packed as
$[\,a_E\,|\,a_I\,|\,b_E\,|\,b_I\,|\,x\,]$ and zero-padded to the widest
variant in the batch:

$$
S = n_E\, n_{aE}^{\max} + n_I\, n_{aI}^{\max} + n_E\, n_{bE}^{\max} + n_I\, n_{bI}^{\max} + N .
$$

| symbol | meaning | shape | range |
|---|---|---|---|
| $x$ | dendritic potential | $(K, B, N)$ | $\mathbb R$ |
| $a_E$ | SFA on E neurons, one column per timescale tier | $(K, B, n_E, n_{aE}^{\max})$ | $[0, 1]$ |
| $a_I$ | SFA on I neurons | $(K, B, n_I, n_{aI}^{\max})$ | $[0, 1]$ |
| $b_E$ | STD on E neurons, fraction of synaptic resource available | $(K, B, n_E)$ | $[0, 1]$ |
| $b_I$ | STD on I neurons | $(K, B, n_I)$ | $[0, 1]$ |

A block whose $\max$ count is $0$ reserves no slots; `unpack_state` then
substitutes zeros (for $a$) or ones (for $b$) of the right shape so downstream
code is branch-free. Tiers or sides that are inactive for a particular variant
but present in the packed layout are held at their padded value by the masks of
§3 and never receive gradient.

Quantities recomputed at every substep, all $(K, B, N)$: the drive
$u = W^{\text{in,eff}} x_{\text{in}}$, the potential after SFA subtraction
$x^{\text{eff}}$, the rate $r = \phi(x^{\text{eff}} - a_0)$, the effective synaptic
gain $\tilde b$ (`b_full`) and the synaptic output $\tilde b \odot r$ (`br`).
`get_diagnostics(state, inputs)` returns all of these for a packed state
without modifying anything.

**Reference initial state.** `init_state` returns $a = 0$, $b = 1$,
$x = 0.1 \cdot \mathcal N(0, 1)$. This is the starting point of the burn-in
(§8.1); the state that actually begins each training window is the trainable
initial condition `ic.ic` described there.

---

## 2. Learnable parameters (as stored)

All keys below live under `cell.` in the state dict unless marked otherwise.
Positive quantities are stored in inverse-softplus space (`isp_*`) and passed
through `softplus` at use time; per-variant scalar gains are stored in log
space (`log_*_gain`) and passed through `exp`. The two transforms differ:
$\exp(0) = 1$ is the multiplicative identity, while
$\mathrm{softplus}(0) = \log 2$, and softplus is asymptotically linear where
exp is multiplicative.

Every per-neuron vector is paired with a per-variant scalar (additive
parameters) or log-gain (multiplicative parameters). Variants with
`per_neuron=False` train only the paired scalar or gain; their vector stays at
its uniform initial value (§4.1).

### 2.1 Recurrent and input weights

| parameter | shape | init |
|---|---|---|
| `W_raw` | $(K, N, N)$ | from `RMTMatrix.export_for_srnn` (§9) |
| `W_in` | $(K, N, D)$ | $\mathcal N(0, 0.1^2)$ |
| `W_raw_gain` | $(K,)$ | $1$ |
| `W_in_gain` | $(K,)$ | $1$ |

`W_raw_gain` and `W_in_gain` are free reals; nothing prevents the optimizer
from driving them negative. They remain trainable for echo variants.

### 2.2 Threshold

| parameter | shape | init |
|---|---|---|
| `a_0_vec` | $(K, N)$ | $0.35$ |
| `a_0_scalar` | $(K,)$ | $0$ |

### 2.3 Time constants

| parameter | shape | init |
|---|---|---|
| `isp_tau_global` | $(K,)$ | $\mathrm{softplus}^{-1}(\tau_g^{\text{init}})$, `tau_global_init` default $1$ s |
| `isp_tau_d_vec` | $(K, N)$ | $\mathrm{softplus}^{-1}(0.1)$ |
| `log_tau_d_gain` | $(K,)$ | $0$ |

### 2.4 SFA parameters, E side (allocated iff $n_{aE}^{\max} > 0$)

| parameter | shape | init |
|---|---|---|
| `isp_tau_a_E_vec` | $(K, n_E, n_{aE}^{\max})$ | see below |
| `log_tau_a_E_gain` | $(K,)$ | $0$ |
| `isp_c_E_vec` | $(K, n_E, n_{aE}^{\max})$ | $\mathrm{softplus}^{-1}(0.05)$ |
| `log_c_E_gain` | $(K,)$ | $0$ |
| `c_0_E_vec` | $(K, n_E, n_{aE}^{\max})$ | $0$ |
| `c_0_E_scalar` | $(K,)$ | $0$ |

SFA time-constant init for variant $k$ with $n_{aE}$ active tiers:

- $n_{aE} = 1$: tier $0$ at $\mathrm{softplus}^{-1}(1.0)$ (1 s).
- $n_{aE} \ge 2$: tiers spread **linearly in inverse-softplus space** between
  $\ell_{\text{lo}} = \mathrm{softplus}^{-1}(\tau_a^{\text{lo}})$ and
  $\ell_{\text{hi}} = \mathrm{softplus}^{-1}(\tau_a^{\text{hi}})$,

$$
\texttt{isp\_tau\_a\_E\_vec}[k, :, j] = \ell_{\text{lo}} + (\ell_{\text{hi}} - \ell_{\text{lo}})\,\frac{j}{n_{aE} - 1},
\qquad j = 0, \dots, n_{aE} - 1,
$$

  with `tau_a_lo_init` $= 0.25$ s and `tau_a_hi_init` $= 4$ s by default. The
  effective tiers are therefore neither linearly nor log spaced in seconds:
  for the default three tiers they sit at $0.25$, $\approx 1.59$ and $4$ s.
- Inactive tiers ($j \ge n_{aE}$) are stored as $0$ and masked.

### 2.5 SFA parameters, I side

Same as §2.4 with `_E_` replaced by `_I_` and $n_E \to n_I$; allocated iff
$n_{aI}^{\max} > 0$.

### 2.6 STD parameters, E side (allocated iff $n_{bE}^{\max} > 0$)

| parameter | shape | init |
|---|---|---|
| `isp_tau_b_rec_E_vec` | $(K, n_E)$ | $\mathrm{softplus}^{-1}(1.0)$ |
| `log_tau_b_rec_E_gain` | $(K,)$ | $0$ |
| `isp_tau_b_rel_E_vec` | $(K, n_E)$ | $\mathrm{softplus}^{-1}(0.25)$ |
| `log_tau_b_rel_E_gain` | $(K,)$ | $0$ |

### 2.7 STD parameters, I side

Same as §2.6 with `_E_` replaced by `_I_`; allocated iff $n_{bI}^{\max} > 0$.

### 2.8 Readout and initial condition (on `SequenceModel`)

| parameter | shape | init |
|---|---|---|
| `readout_weight` | $(K, O, E)$ | Kaiming-uniform per slice $k$ (`nn.Linear` default, $a = \sqrt 5$) |
| `readout_bias` | $(K, 1, O)$ | $0$ |
| `W_out_gain` | $(K,)$ | $1$; scales `readout_weight` only, not the bias |
| `ic.ic` | $(K, S)$ | $0$; overwritten by burn-in (§8.1) |

Values given as literals above (`W_in` scale, `a_0_vec`, `isp_tau_d_vec`, `isp_c_*_vec`,
`isp_tau_b_*_vec`, all gains and scalars) are fixed in `SRNNCell` and cannot be
changed through the config; `W_raw`, `isp_tau_global` and `isp_tau_a_*_vec` are
set from `SRNNModelConfig` (§10).

---

## 3. Buffers (non-trainable)

All are set once at construction from the per-variant `SRNNConfig` or the RMT
export and are constant during training.

| buffer | shape | values | role |
|---|---|---|---|
| `sparsity_masks` | $(K, N, N)$ | $\{0, 1\}$ | structural zeros of $W^{\text{eff}}$ |
| `dales_signs` | $(K, N)$ | $\pm 1$ | column sign: $+1$ for E source, $-1$ for I source |
| `dales_mask` | $(K, 1, 1)$ | $\{0, 1\}$ | Dale's law enforced for variant $k$ |
| `echo_flags` | $(K, 1, 1)$ | $\{0, 1\}$ | `W_raw` frozen (reservoir) |
| `skip_flags` | $(K,)$ | $\{0, 1\}$ | output residual (§7.3) |
| `per_neuron_mask` | $(K,)$ | $\{0, 1\}$ | per-neuron vectors trainable (§4.1) |
| `sfa_E_mask`, `sfa_I_mask` | $(K, 1, \max(n_a^{\max}, 1))$ | $\{0, 1\}$ | active SFA tiers of variant $k$ |
| `std_E_mask`, `std_I_mask` | $(K, 1)$ | $\{0, 1\}$ | STD active for variant $k$ |
| `std_zero_floor_mask` | $(K, 1, 1)$ | $\{0, 1\}$ | zero-floor rescale of $b$ (§5.3); non-persistent |
| `readout_ids` | $(K,)$ | $\{0, 1, 2\}$ | synaptic / rate / dendritic cell output (§6) |
| `W_in_mask` | $(1, N, 1)$ | $\{0, 1\}$ | input-neuron partition, rows of $W^{\text{in}}$ |
| `input_mask`, `output_mask` (on `SequenceModel`) | $(N,)$ | $\{0, 1\}$ | neuron partition; `output_mask` selects the $E$ readout neurons |

**Neuron partition.** `generate_neuron_partition(N, seed)` permutes the
neuron indices with the run seed and assigns the first $\max(1, \lfloor N/4
\rfloor)$ to the input group and the last $\max(1, \lfloor N/4 \rfloor)$ to the
output group; the remainder are interneurons. The partition is shared across
the $K$ variants and is independent of the E/I split.

---

## 4. Effective parameters

These are the values that enter the equations and the values reported in the
per-variant parameter tables. `effective_params()` returns them by name
(`W`, `tau_global`, `tau_d`, `a_0`, `tau_a_E`, `c_E`, `c_0_E`,
`tau_b_rec_E`, `tau_b_rel_E`, and the I-side equivalents).

### 4.1 Per-neuron linking

For every per-neuron vector $v$ the cell uses `_linked(v)`,

$$
v' = m_k\, v + (1 - m_k)\, \mathrm{sg}(v), \qquad m_k = \texttt{per\_neuron\_mask}[k],
$$

where $\mathrm{sg}$ is the stop-gradient (`detach`). The value is unchanged;
for variants with `per_neuron=False` the gradient to the vector is cut, so
only the paired scalar or gain moves and the vector stays uniform across
neurons. For `per_neuron=True` both train: the scalar or gain carries the
shared direction, the vector the per-neuron deviation. This applies to
`a_0_vec`, `isp_tau_d_vec`, `isp_tau_a_*_vec`, `isp_c_*_vec`, `c_0_*_vec`,
`isp_tau_b_rec_*_vec` and `isp_tau_b_rel_*_vec`.

### 4.2 Recurrent weight

With $d_k = \texttt{dales\_mask}[k]$, $s_j = \texttt{dales\_signs}[k, j]$,
$M_{ij} = \texttt{sparsity\_masks}[k, i, j]$, $g_W = \texttt{W\_raw\_gain}[k]$
and $e_k = \texttt{echo\_flags}[k]$:

$$
\hat W^{\text{raw}} = e_k\, \mathrm{sg}(W^{\text{raw}}) + (1 - e_k)\, W^{\text{raw}},
$$

$$
W^{\text{eff}}_{ij} = g_W\, M_{ij} \Big[ d_k\, s_j\, \mathrm{softplus}\big(\hat W^{\text{raw}}_{ij}\big) + (1 - d_k)\, \hat W^{\text{raw}}_{ij} \Big].
$$

Index $j$ is the presynaptic (source) neuron. Under Dale's law every column
has one sign: $W^{\text{eff}}_{:,j} \ge 0$ for E sources and $\le 0$ for I
sources, with magnitudes $\mathrm{softplus}(W^{\text{raw}})$. Without Dale's
law $W^{\text{raw}}$ is used directly and may take either sign. The sparsity
mask and the gain apply in both cases. For echo variants the stop-gradient
freezes the recurrent structure at its initial value while $g_W$ still trains,
leaving a single learnable spectral-radius knob. $W^{\text{eff}}$ is computed
once per forward pass (`hoist`) and passed into every cell call.

### 4.3 Input weight

$$
W^{\text{in,eff}} = g_{\text{in}}\, \big(W^{\text{in}} \odot M^{\text{in}}\big),
\qquad M^{\text{in}}_{i,:} = \texttt{W\_in\_mask}[i],
\qquad g_{\text{in}} = \texttt{W\_in\_gain}[k],
$$

so only rows belonging to the input partition are non-zero.

### 4.4 Time constants

$$
\tau_g = \mathrm{softplus}(\texttt{isp\_tau\_global}),
$$

$$
\tau_d(i) = \tau_g \cdot \exp(\texttt{log\_tau\_d\_gain}) \cdot \mathrm{softplus}(\texttt{isp\_tau\_d\_vec}[i]),
$$

$$
\tau_{a,E}(i, j) = \tau_g \cdot \exp(\texttt{log\_tau\_a\_E\_gain}) \cdot \mathrm{softplus}(\texttt{isp\_tau\_a\_E\_vec}[i, j]),
$$

$$
\tau_{b,\text{rec},E}(i) = \tau_g \cdot \exp(\texttt{log\_tau\_b\_rec\_E\_gain}) \cdot \mathrm{softplus}(\texttt{isp\_tau\_b\_rec\_E\_vec}[i]),
$$

and identically for $\tau_{a,I}$, $\tau_{b,\text{rel},E}$,
$\tau_{b,\text{rec},I}$, $\tau_{b,\text{rel},I}$. Every time constant is the
product of three factors: the shared $\tau_g$, a per-variant class gain, and a
per-neuron (per-tier) base. Substituting $t' = t / \tau_g$ removes $\tau_g$
from every equation of §5, so $\tau_g$ is a uniform stretch of the time axis
and the class gains set the relative drift between $\tau_d$, $\tau_a$ and
$\tau_b$.

### 4.5 SFA coupling and offset

$$
c_E(i, j) = \exp(\texttt{log\_c\_E\_gain}) \cdot \mathrm{softplus}(\texttt{isp\_c\_E\_vec}[i, j]),
\qquad
c_{0,E}(i, j) = \texttt{c\_0\_E\_vec}[i, j] + \texttt{c\_0\_E\_scalar},
$$

and likewise for the I side. $c$ is a coupling, not a timescale, and carries
no $\tau_g$ factor. $c_0$ has no positivity transform.

### 4.6 Threshold

$$
a_0(i) = \texttt{a\_0\_vec}[i] + \texttt{a\_0\_scalar}.
$$

No positivity transform. Note that $a_0$ and $c_0$ share a flat direction:
raising $a_0$ by $\delta$ and $c_0$ by $\delta / c$ for a single-tier SFA at
equilibrium leaves $r$ unchanged.

### 4.7 Stored versus effective, quick reference

```
stored                              effective
--------------------------------------------------------------------
isp_tau_global                ->    softplus(.)
isp_tau_*_vec, isp_c_*_vec    ->    softplus(.)          (0 -> log 2)
log_tau_*_gain, log_c_*_gain  ->    exp(.)               (0 -> 1)
a_0_vec + a_0_scalar          ->    identity (additive)
c_0_*_vec + c_0_*_scalar      ->    identity (additive)
W_raw_gain, W_in_gain,
  W_out_gain                  ->    identity (multiplicative, unconstrained)
W_raw, dales on               ->    dales_signs * softplus(.) * sparsity_masks
W_raw, dales off              ->    identity * sparsity_masks
```

---

## 5. Continuous-time dynamics

For one variant, neuron $i \in \{1, \dots, N\}$ with $i \le n_E$ excitatory,
and SFA tier $j$:

$$
\boxed{\;\tau_{d,i}\, \dot x_i = -x_i + u_i + \sum_{m=1}^{N} W^{\text{eff}}_{im}\, \tilde b_m\, r_m\;}
$$

$$
\boxed{\;\tau_{a,(i,j)}\, \dot a_{i,j} = -a_{i,j} + c_{0,(i,j)} + r_i\;}
$$

$$
\boxed{\;\dot b_i = \frac{1 - b_i}{\tau_{\text{rec},i}} - \frac{r_i\, b_i}{\tau_{\text{rel},i}}\;}
$$

with

$$
u = W^{\text{in,eff}}\, x_{\text{in}}, \qquad
x^{\text{eff}}_i = x_i - \sum_j c_{i,j}\, a_{i,j}, \qquad
r_i = \phi\big(x^{\text{eff}}_i - a_{0,i}\big),
$$

where $\phi$ is the piecewise sigmoid of §5.4 and $\tilde b$ is the effective
synaptic gain of §5.3. The E/I side of $c$, $c_0$, $\tau_a$, $\tau_{\text{rec}}$
and $\tau_{\text{rel}}$ is selected by the side of neuron $i$.

### 5.1 SFA masking

In code the SFA contribution and the SFA update are multiplied by
`sfa_E_mask` / `sfa_I_mask`:

$$
x^{\text{eff}}_E = x_E - \sum_{j=0}^{n_{aE}^{\max}-1} \big(c_E \odot \texttt{sfa\_E\_mask}\big)_{:,j} \odot a_{E,:,j},
$$

and $\dot a_{E,j}$ is multiplied by $\texttt{sfa\_E\_mask}[k, 0, j]$, so
inactive tiers contribute nothing and stay at their padded value. With
$n_{aE} = 0$ the SFA term vanishes for E neurons entirely.

### 5.2 STD masking

$\dot b_E$ is multiplied by `std_E_mask`, and in the effective gain (§5.3)
$b$ is replaced by $1$ where STD is off, so a variant with $n_{bE} = 0$ has
$\tilde b_E \equiv 1$.

### 5.3 Effective synaptic gain (`b_full`)

The steady state of $\dot b = 0$ at saturating rate $r = 1$ is
$b_{\min} = \tau_{\text{rel}} / (\tau_{\text{rec}} + \tau_{\text{rel}})$, so
without rescaling a fully driven synapse still transmits a fraction
$b_{\min}$ of its rate. With `std_zero_floor=True` (default) the gain is
rescaled so that this floor maps to zero:

$$
\tilde b_i = z_k\, \frac{b_i - b_{\min,i}}{1 - b_{\min,i}} + (1 - z_k)\, b_i,
\qquad z_k = \texttt{std\_zero\_floor\_mask}[k],
$$

then masked,

$$
\tilde b_E = \tilde b_E \cdot \texttt{std\_E\_mask} + (1 - \texttt{std\_E\_mask}),
$$

and concatenated over sides. $b_{\min}$ uses the current effective time
constants, so it changes as $\tau_{\text{rec}}$ and $\tau_{\text{rel}}$ train.
The ODE for $b$ integrates the raw $b$; only the synaptic output
$\tilde b \odot r$ and the synaptic readout use $\tilde b$. At initialisation
(default taus) $b_{\min} = 0.2$.

### 5.4 Activation

`piecewise_sigmoid(x; S_a, S_c)` with defaults $S_a = 0.9$, $S_c = 0$. Let
$a = S_a / 2$, $c = S_c$, $\kappa = 0.5 / (1 - 2a)$ (or $0$ if
$|1 - 2a| < 10^{-8}$), and knots

$$
x_1 = c + a - 1, \quad x_2 = c - a, \quad x_3 = c + a, \quad x_4 = c + 1 - a .
$$

$$
\phi(z) = \begin{cases}
0, & z < x_1 \\
\kappa\,(z - x_1)^2, & x_1 \le z < x_2 \\
(z - c) + \tfrac12, & x_2 \le z \le x_3 \\
1 - \kappa\,(z - x_4)^2, & x_3 < z \le x_4 \\
1, & z > x_4
\end{cases}
$$

$\phi$ maps $\mathbb R \to [0, 1]$, is linear with unit slope over a central
region of width $S_a$, joins $0$ and $1$ through quadratic shoulders, and is
$C^1$. With $S_c = 0$, $\phi(0) = \tfrac12$; the threshold enters as the
argument $x^{\text{eff}} - a_0$.

---

## 6. Solvers

One cell call advances $h$ seconds in $n_{\text{ode}}$ substeps of
$\Delta t = h / n_{\text{ode}}$. Three discretisations share the right-hand
side of §5 and differ only in the update. In every solver $u$ is computed once
per call, and $r$ and $\tilde b$ that feed the cell output (§6.4) are those
evaluated at the state entering the final substep.

### 6.1 Semi-implicit (linearly implicit Euler), default

Each linear relaxation $\tau \dot y = -y + S$ is stepped with the decay
implicit and the source explicit, $y^+ = (y + \alpha S) / (1 + \alpha)$,
$\alpha = \Delta t / \tau$. Evaluate $r$, $\tilde b$, $W^{\text{eff}}(\tilde b
\odot r)$ at the current state, then

$$
x^+ = \frac{x + \alpha_x\,\big(u + W^{\text{eff}}(\tilde b \odot r)\big)}{1 + \alpha_x},
\qquad \alpha_x = \Delta t / \tau_d ,
$$

$$
a^+_{i,j} = \frac{a_{i,j} + \alpha_{a,(i,j)}\,(c_{0,(i,j)} + r_i)}{1 + \alpha_{a,(i,j)}},
\qquad \alpha_{a,(i,j)} = \Delta t / \tau_{a,(i,j)} ,
$$

$$
b^+_i = \frac{b_i + \Delta t / \tau_{\text{rec},i}}{1 + \Delta t\,\big(1/\tau_{\text{rec},i} + r_i / \tau_{\text{rel},i}\big)},
\qquad b^+_i \leftarrow \mathrm{clip}(b^+_i, 0, 1).
$$

The $b$ update treats both the recovery term and the bilinear $r\,b$ term
implicitly in $b$. Inactive SFA tiers and STD sides are blended back to their
previous values by the masks.

### 6.2 Explicit Euler (`solver=explicit`)

`euler_step` from `train_srnn/models/ode.py` on the joint state
$y = (x, a_E, a_I, b_E, b_I)$: $y^+ = y + \Delta t\, f(y)$, with $f$ the masked
right-hand side (`_rhs`) and $b$ clipped to $[0, 1]$ afterwards.

### 6.3 Classical RK4 (`solver=rk4`)

`rk4_step`: $k_1 = \Delta t\, f(y)$, $k_2 = \Delta t\, f(y + k_1/2)$,
$k_3 = \Delta t\, f(y + k_2/2)$, $k_4 = \Delta t\, f(y + k_3)$,
$y^+ = y + (k_1 + 2k_2 + 2k_3 + k_4)/6$, then $b$ clipped to $[0, 1]$.

### 6.4 Cell output

After the last substep the cell emits one of three views of the state,
selected per variant by `readout_ids`:

| `readout` | id | output $o_t$ |
|---|---|---|
| `synaptic` (default) | 0 | $\tilde b \odot r$ |
| `rate` | 1 | $r$ |
| `dendritic` | 2 | $x^+$ |

$$
o_t = \mathbb 1[\text{id} = 0]\,\tilde b \odot r + \mathbb 1[\text{id} = 1]\, r + \mathbb 1[\text{id} = 2]\, x^+ ,
$$

shape $(K, B, N)$. `synaptic` and `rate` use $r$, $\tilde b$ from the state
entering the last substep; `dendritic` uses the updated $x$.

---

## 7. SequenceModel head

### 7.1 Unrolling

Given input $X \in \mathbb R^{B \times T \times D}$ and initial state
$s_0 = \texttt{ic}(B) \in \mathbb R^{K \times B \times S}$, `unroll` iterates
$(o_t, s_t) = \text{cell}(X_{:,t,:},\, s_{t-1},\, W^{\text{eff}})$ for one
contiguous segment and writes $o_t$ into a preallocated time buffer, returning
$\mathbf O \in \mathbb R^{K \times B \times T \times N}$. `forward` splits the
sequence into segments:

1. Steps $t < $ `bptt_start_idx` run under `torch.no_grad()`; the state is
   detached at the boundary.
2. The remaining steps run in segments of `grad_checkpoint_segment_len` (when
   `grad_checkpoint=True`, each segment wrapped in `torch.utils.checkpoint`)
   or `bptt_chunk_len` otherwise; the state is detached every
   `bptt_chunk_len` steps (truncated BPTT).
3. Segment outputs are concatenated along time.

### 7.2 Time-step selection and readout

`readout_idx` picks one step (`int`, `None` meaning $-1$) or a range
(`slice`) from $\mathbf O$, giving $\mathbf o \in \mathbb R^{K \times B \times
N}$ or $\mathbb R^{K \times B \times T' \times N}$. `apply_readout` then:

1. masks and selects the readout neurons,
   $\mathbf o' = \mathbf o[\dots, \texttt{output\_mask} = 1] \in \mathbb R^{\dots \times E}$;
2. applies the per-variant linear head,

$$
\mathbf y = \big(g_{\text{out}}\, W^{\text{ro}}\big)\, \mathbf o' + b^{\text{ro}},
\qquad W^{\text{ro}} \in \mathbb R^{K \times O \times E},\;
b^{\text{ro}} \in \mathbb R^{K \times 1 \times O},\;
g_{\text{out}} = \texttt{W\_out\_gain}[k],
$$

   implemented as `einsum("k...e,koe->k...o")`.

### 7.3 Skip residual

For variants with $\texttt{skip\_flags}[k] = 1$ (requires $D = O$):

$$
\mathbf y \leftarrow \mathbf y + \texttt{skip\_flags}[k]\; X_{:, \texttt{readout\_idx}, :} .
$$

The network then learns the residual between the current sample and the next,
the natural parameterisation for one-step-ahead autoregressive forecasting.
The residual is added in output space after the gain and bias.

### 7.4 Closed loop (variable teacher forcing)

With an `alpha_schedule` $\in [0, 1]^{T \times C}$ the input at each step is
the per-channel blend

$$
x_t^{\text{in}} = (1 - \alpha_t) \odot X_{:,t,:} + \alpha_t \odot \mathbf y_{t-1},
\qquad \mathbf y_{-1} = 0,
$$

and the readout (§7.2 to §7.3, with the skip residual taken from the blended
input) is applied at every step because $\mathbf y_{t-1}$ is needed for the
next input. Requires $D = O$. Evaluation always runs open loop
($\alpha = 0$).

---

## 8. Forward pipeline

For one minibatch:

1. **Initial state.** $s_0 = \texttt{ic.ic}$ expanded over the batch.
2. **Hoist.** $W^{\text{eff}}$ from §4.2, once.
3. **Warm-up (no grad).** Steps $t < $ `bptt_start_idx`.
4. **Gradient region.** For each step: $u_t = W^{\text{in,eff}} x_t^{\text{in}}$;
   $n_{\text{ode}}$ solver substeps (§6); cell output $o_t$ (§6.4).
5. **Select** $\mathbf o = \mathbf O[\dots, \texttt{readout\_idx}, :]$.
6. **Mask** to the $E$ readout neurons.
7. **Head** $\mathbf y = g_{\text{out}} W^{\text{ro}} \mathbf o' + b^{\text{ro}}$.
8. **Skip** $\mathbf y \leftarrow \mathbf y + \texttt{skip\_flags}[k]\, X_{:, \texttt{readout\_idx}, :}$.

### 8.1 Initial condition and burn-in

`TrainableIC` holds `ic.ic` $\in \mathbb R^{K \times S}$, initialised to
zeros. When `burn_in > 0` (default $10$ s), `compute_burn_in` starts from
`init_state` ($a = 0$, $b = 1$, $x = 0.1\,\mathcal N(0,1)$), runs the cell
with zero input for $\lfloor \texttt{burn\_in} / h \rfloor$ steps and copies
the settled state into `ic.ic`. With `freeze_ic_after_burnin=True` (default)
the IC is then frozen; the continuous trainer always freezes it. With
`burn_in_every` $= n > 0$ the burn-in is re-run every $n$ epochs (windowed
trainer only). If `burn_in = 0` the IC stays at zeros, including $b = 0$.

### 8.2 Freezing parameter groups

`freeze_params` pins logical groups at their initial values by setting
`requires_grad=False`. Groups (`FREEZE_GROUPS`): `a_0`, `W_raw`, `W_in`,
`W_raw_gain`, `W_in_gain`, `tau_global`, `tau_d`, `tau_a_E`, `c_E`, `c_0_E`,
`tau_a_I`, `c_I`, `c_0_I`, `tau_b_rec_E`, `tau_b_rel_E`, `tau_b_rec_I`,
`tau_b_rel_I`, each covering its vector and scalar or gain; plus
`W_out_gain` on `SequenceModel`. Freezing applies to all $K$ variants at once.

---

## 9. Recurrent weight construction (`RMTMatrix`)

`W_raw`, `sparsity_masks` and `dales_signs` come from `RMTMatrix`, the
author's implementation of the sparse E/I random-matrix construction of
Harris et al. (2023). The factory calls it with `n = N`,
`density` (`rmt.density`, default $1/3$), `level_of_chaos`
(`rmt.level_of_chaos`, default $1$) and the variant's seed; all other
arguments take their defaults ($f = 0.5$, `E_W` $= 0$, `zrs_mode="none"`,
`rescale_by_abscissa=False`).

With $\alpha = \texttt{indegree} / N$, $\texttt{indegree} =
\mathrm{round}(\texttt{density} \cdot N)$, $n_E^{\text{rmt}} = \mathrm{round}(f N)$:

$$
F = \frac{1}{\sqrt{N \alpha (2 - \alpha)}}, \qquad
\tilde\mu_E = 3F, \quad \tilde\mu_I = -4F, \quad \tilde\sigma_E = \tilde\sigma_I = F .
$$

Draw $A_{ij} \sim \mathcal N(0, 1)$ and $S_{ij} \sim \mathrm{Bernoulli}(\alpha)$
(all ones if $\alpha = 1$). With $D = \mathrm{diag}(\tilde\sigma)$ (E entries
then I entries), $v_j = \tilde\mu_{E \text{ or } I}$ by column and
$\mathbf u = \mathbf 1_N$:

$$
W = \texttt{level\_of\_chaos} \cdot S \odot \big(A D + \mathbf u\, \mathbf v^{\top}\big).
$$

The sparse population statistics
$\mu_{s\bullet} = \alpha\, \tilde\mu_\bullet$,
$\sigma^2_{s\bullet} = \alpha(1 - \alpha)\tilde\mu_\bullet^2 + \alpha\tilde\sigma_\bullet^2$
give the predicted bulk radius and outlier

$$
R = \texttt{level\_of\_chaos} \sqrt{N\big(f\sigma^2_{sE} + (1-f)\sigma^2_{sI}\big)}, \qquad
\lambda_O = N\big(f\mu_{sE} + (1-f)\mu_{sI}\big),
$$

which `plot_spectrum` overlays on the empirical eigenvalues. `zrs_mode`
selects an optional zero-row-sum correction (`ZRS`, `SZRS`, `Partial_SZRS`)
and `rescale_by_abscissa` normalises the spectral abscissa to
`level_of_chaos`; neither is used by the factory.

`export_for_srnn(dales)` returns `sparsity_mask` $= S$, `dales_sign`
($+1$ for the first $n_E^{\text{rmt}}$ columns, $-1$ after) and

$$
\texttt{W\_init} = \begin{cases}
\mathrm{softplus}^{-1}\big(\max(|W|, 10^{-7})\big), & \texttt{dales} = \text{true} \\
W, & \texttt{dales} = \text{false}
\end{cases}
$$

so that §4.2 reproduces $W$ exactly at initialisation in both cases (up to the
$10^{-7}$ floor). Variants that share a seed share one `RMTMatrix`, so
comparisons across variants at the same seed are paired on connectivity.

---

## 10. Configuration and ablation knobs

`SRNNModelConfig` (`train_srnn/config.py`) holds the shared settings; the
variant names in `model.variants` override the per-variant flags marked
"per variant" below, and `model.variant_seeds` crosses the names with
recurrent-matrix seeds. `SRNNConfig` is the per-variant dataclass the cell
receives.

| field | default | per variant | effect |
|---|---|---|---|
| `num_units` | 300 | no | $N$ |
| `dales` | `true` | yes | §4.2 branch: signed softplus magnitudes vs raw $W^{\text{raw}}$ |
| `n_a_E`, `n_a_I` | 3, 3 | yes | SFA tier count per side; $0$ removes the SFA term (§5.1) |
| `n_b_E`, `n_b_I` | 1, 1 | yes | STD per side; $0$ sets $\tilde b \equiv 1$ (§5.2) |
| `per_neuron` | `false` | yes | per-neuron vectors trainable (§4.1) |
| `echo` | `false` | yes | $W^{\text{raw}}$ frozen, $g_W$ trainable (§4.2) |
| `skip` | `false` | yes | output residual (§7.3); needs $D = O$ |
| `solver` | `semi_implicit` | no | `semi_implicit`, `explicit`, `rk4` (§6) |
| `h` | `task.h` | no | seconds per cell call |
| `ode_unfolds` | `task.ode_unfolds` | no | substeps per call |
| `readout` | `synaptic` | no | cell output (§6.4) |
| `tau_global_init` | 1.0 | no | init of $\tau_g$ (s) |
| `tau_a_lo_init` | 0.25 | no | fastest SFA tier at init (s), $n_a \ge 2$ |
| `tau_a_hi_init` | 4.0 | no | slowest SFA tier at init (s), $n_a \ge 2$ |
| `std_zero_floor` | `true` | no | zero-floor rescale of $b$ (§5.3) |
| `rmt.density` | 1/3 | no | connection probability $\alpha$ (§9) |
| `rmt.level_of_chaos` | 1.0 | no | global scale of $W$ (§9) |

### 10.1 Variants

A variant name is `srnn` followed by tokens (`no-adapt`, `sfa-only`,
`std-only`, `e-only`, `no-dales`, `per-neuron`, `echo`, `skip`, and the
aliases `sfa-e-only`, `std-e-only`) and an optional `-seed<n>` suffix.
Each token overrides the per-variant flags above relative to the shared
config; names are canonicalised to a fixed token order. The grammar and the
meaning of every token are specified in `docs/variants.md`.

---

## 11. Loss and training

The trainer computes the per-network loss between $\mathbf y_k$ at
`readout_idx` and the matching target slice: `nn.CrossEntropyLoss` for
classification, `nn.MSELoss` for regression (a trailing singleton output axis
is squeezed). The $K$ losses are summed for the backward pass; since the
variants share no parameters, each variant's gradient is exactly the gradient
of its own loss. Optimiser is Adam with `WarmupHoldCosineSchedule` stepped per
optimizer step: linear warm-up over
$\min(\texttt{warmup\_epochs} \cdot \text{steps/epoch} / \text{total}, 0.2)$ of
the run from $10^{-8}$ to `lr`, then hold; with `cosine_decay=true` a cosine
tail from 70 % of the run down to `lr`$/20$. Gradients are clipped to
`grad_clip` per variant (`clip_grad_norm_per_variant`), so one variant's
large gradient does not scale the others. `constrain_parameters` is a no-op
for the SRNN. Mixed precision (`amp=bf16`, CUDA only) wraps forward and loss;
optimizer state stays fp32.

---

## 12. Symbol to parameter map

| math | code (`cell.*` unless noted) | shape | trainable |
|---|---|---|---|
| $W^{\text{raw}}$ | `W_raw` | $(K, N, N)$ | yes; stop-gradient iff echo |
| $g_W$ | `W_raw_gain` | $(K,)$ | yes |
| $W^{\text{in}}$ | `W_in` | $(K, N, D)$ | yes |
| $g_{\text{in}}$ | `W_in_gain` | $(K,)$ | yes |
| $g_{\text{out}}$ | `W_out_gain` (`SequenceModel`) | $(K,)$ | yes |
| $a_0$ | `a_0_vec` $+$ `a_0_scalar` | $(K, N)$, $(K,)$ | vec iff per_neuron; scalar yes |
| $\tau_g$ | $\mathrm{softplus}(\texttt{isp\_tau\_global})$ | $(K,)$ | yes |
| $\tau_d$ | $\tau_g \exp(\texttt{log\_tau\_d\_gain})\,\mathrm{softplus}(\texttt{isp\_tau\_d\_vec})$ | $(K, N)$ | gain yes; vec iff per_neuron |
| $\tau_{a,E}^{(j)}$ | $\tau_g \exp(\texttt{log\_tau\_a\_E\_gain})\,\mathrm{softplus}(\texttt{isp\_tau\_a\_E\_vec}_{:,:,j})$ | $(K, n_E, n_{aE}^{\max})$ | as above; tier active iff `sfa_E_mask[k,0,j]=1` |
| $c_E$ | $\exp(\texttt{log\_c\_E\_gain})\,\mathrm{softplus}(\texttt{isp\_c\_E\_vec})$ | $(K, n_E, n_{aE}^{\max})$ | as above |
| $c_{0,E}$ | `c_0_E_vec` $+$ `c_0_E_scalar` | same, $(K,)$ | vec iff per_neuron; scalar yes |
| $\tau_{b,\text{rec},E}$ | $\tau_g \exp(\texttt{log\_tau\_b\_rec\_E\_gain})\,\mathrm{softplus}(\texttt{isp\_tau\_b\_rec\_E\_vec})$ | $(K, n_E)$ | as above |
| $\tau_{b,\text{rel},E}$ | analogous with `rel` | $(K, n_E)$ | as above |
| $b_{\min}$ | $\tau_{\text{rel}} / (\tau_{\text{rec}} + \tau_{\text{rel}})$ | $(K, n_\bullet)$ | derived |
| $\tilde b$ | `b_full` | $(K, B, N)$ | derived |
| $\phi$ | `piecewise_sigmoid` | | none |
| $d_k, s_j, M$ | `dales_mask`, `dales_signs`, `sparsity_masks` | $(K,1,1)$, $(K,N)$, $(K,N,N)$ | buffers |
| $e_k, m_k$ | `echo_flags`, `per_neuron_mask` | $(K,1,1)$, $(K,)$ | buffers |
| $M^{\text{in}}$ | `W_in_mask` | $(1, N, 1)$ | buffer |
| $W^{\text{ro}}, b^{\text{ro}}$ | `readout_weight`, `readout_bias` (`SequenceModel`) | $(K, O, E)$, $(K, 1, O)$ | yes |
| $s_0$ | `ic.ic` (`SequenceModel`) | $(K, S)$ | yes unless frozen after burn-in |

Replace `_E` with `_I` for the inhibitory side; the structure is identical.

---

## References

Harris, I. D., Meffin, H., Burkitt, A. N., & Peterson, A. D. H. (2023).
Effect of sparsity on network stability in random neural networks obeying
Dale's law. *Physical Review Research*, 5(4), 043132.
https://doi.org/10.1103/PhysRevResearch.5.043132
