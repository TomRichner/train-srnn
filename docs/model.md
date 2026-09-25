# SRNN model version 2

`SRNNCell` implements the deterministic two-population specialization of the
manuscript model. Its scientific reference is the sibling MATLAB repository's
[authoritative equations](../../FractionalReservoir/docs/EquationsParametersDocs/Equations_stability_paper.md),
instantiated by `sfaEI_mu7_fast_config.m` and the generated
`figs/sfaEI_mu7_fast/doc_tables/equation_table.md`. Those links require the
sibling checkout. The [experiment protocol](matlab_alignment.md) specifies the
three-condition comparison and its verification.

This document describes the learned-model implementation and intentional
choices relative to MATLAB. There is no Wiener process, facilitation, or
arbitrary cell-type/route generality. A presynaptic neuron has one set of STD
states shared across its outgoing connections. MATLAB route states can be
collapsed this way only when their parameters and initial states agree.

## Dynamics and tensors

Let V denote the number of batched variants, B the batch, N the neuron count,
and A/M the active SFA/STD timescale counts for a neuron. Code uses `K` for V;
the manuscript's adaptation count K is a different quantity. N defaults to
500, with `n_E=floor(N/2)` and `n_I=N-n_E`.

The implemented deterministic equations are

\[
r_i=\phi\left(x_i-a_{0i}-\frac{c_i}{A_i}\sum_{k=1}^{A_i}a_{ik}\right),
\quad \theta_i=r_i\prod_{m=1}^{M_i}b_{im},
\]
\[
\dot x_i=\frac{-x_i+(W_{\mathrm{in}}u)_i+\sum_jW_{ij}\theta_j}{\tau_{di}},
\quad \dot a_{ik}=\frac{r_i-a_{ik}}{\tau_{aik}},
\quad \dot b_{im}=\frac{1-b_{im}}{\tau^{rec}_{bim}}-
                 \frac{r_i b_{im}}{\tau^{rel}_{bim}}.
\]

When A=0 the SFA term is zero; when M=0 the depression product is one.
Both plasticity mechanisms are driven by raw rate. The piecewise quadratic/
linear activation `piecewise_sigmoid` has range [0,1], central slope one,
and `S_a=0.8`. The setpoint `a_0` is subtracted inside this activation.
The budget c is a total feedback budget, independent of the number of SFA
timescales. There is no SFA-state offset, zero-floor depression remapping,
or global time-constant multiplier.

The flat state `(V,B,S)` packs `[a_E | a_I | b_E | b_I | x]`:

| Block | Unpacked shape |
|---|---|
| `a_E`, `a_I` | `(V,B,n_side,A_side_max)` |
| `b_E`, `b_I` | `(V,B,n_side,M_side_max)` |
| `x` | `(V,B,N)` |

Maxima run over all variants. Packing uses PyTorch row-major order with
neuron then timescale; MATLAB column-major vectors require explicit
translation. A globally absent block occupies no packed slots; unpacking
supplies a singleton neutral block. Masked SFA entries contribute zero,
and masked STD entries contribute one to the product. Masks hold inactive
state derivatives at zero. State size is
`n_E*A_E_max + n_I*A_I_max + n_E*M_E_max + n_I*M_I_max + N`.
The reference initialization is `a=0`, `b=1`, and seeded `x=0.1*N(0,1)`.

## Initial parameters and learning

All parameters carry a leading variant axis. The primary protocol uses
shared population gains with fixed heterogeneous neuronal vectors;
`per-neuron` also learns those vectors. Detachment preserves the realized
heterogeneity when per-neuron learning is disabled; it does not replace the
vectors with their population means.

| Quantity | Stored parameters and transform | Initial value |
|---|---|---|
| Dendritic time | `exp(log_tau_d_gain) * softplus(isp_tau_d_vec)` | 0.1 s |
| SFA times | `exp(log_tau_a_SIDE_gain) * softplus(isp_tau_a_SIDE_vec)` | ladder below |
| SFA budget | `exp(log_c_SIDE_gain) * softplus(isp_c_SIDE_vec)` | 0.5 |
| STD times | `exp(log_tau_b_KIND_SIDE_gain) * softplus(isp_tau_b_KIND_SIDE_vec)` | pairs below |
| Setpoint | `a_0_vec + a_0_scalar` | `0.2 + 0.1*N(0,1)` |

`SIDE` is E or I; `KIND` is rec or rel. Scalar gains have shape `(V,)`.
SFA and STD times have shape `(V,n_side,A_max)` and `(V,n_side,M_max)`;
c has shape `(V,n_side)` without a timescale axis. Dendritic time and
setpoint vectors have shape `(V,N)`. Gains start at one and the additive
setpoint scalar at zero. Positive values are transformed to inverse-softplus
space only after initialization in physical units.

SFA nominal times are logarithmically spaced from 0.25 to 10 s:
0.25, sqrt(2.5), and 10 s for A=3. Gaussian endpoint offsets with standard
deviation 0.25 are added in natural-log time and interpolated across the
ladder. The endpoint rule enforces at least one quarter of the nominal
log-span. The A=1 convention uses the fast nominal time, 0.25 s, with the
second endpoint draw, matching MATLAB. These are heterogeneous ladders;
nominal values alone do not describe each realized neuron.

STD recovery/release pairs are 2/0.25 s and 4/0.5 s. M=1 uses the first
pair and M=2 uses both. By default STD strengths are not matched across M=1
and M=2: with both pairs at rho = tau_rel/tau_rec = 0.125, the steady-state
product is `(1 + r/rho)^-2`, the square of one factor.

Optional per-variant matching at the reference rate `r0 = model.std_match_rate`
(default 0.25, the occupied median rate of FractionalReservoir's decision note
`docs/notes/STD_strength_matching_2026-09-13.md`), via tokens in
[variants.md](variants.md): `std-scale` multiplies the presynaptic recurrent
weights of the M-timescale variant by `theta_1(r0)/theta_M(r0)` (MATLAB's route
scale; the synaptic readout is unscaled, as in MATLAB's `plot_data`);
`std-usage` lengthens its release times to a common usage ratio with
`(1 + r0/rho_u)^M = 1 + r0/rho_1`; `std-strong` instead shortens the single
release time so one factor equals the two-pair product; and `std-geo` replaces
the product by the geometric mean `(prod_m b_m)^(1/M)`, which equals one factor
at every constant rate when all pairs share rho (the analog of SFA's `c/K`
normalization). `w-matched` starts the recurrent gain at `1/(1 + r0/rho_1)`.

## Connectivity, input, and readout

The manuscript initialization has density 0.2, equal E/I populations,
nonzero weight means +7F/-7F, standard deviations 1.5F, and gain one.
By default F is fixed at the reference N=500 and indegree=100:
`F=1/sqrt(N_ref * alpha_ref * (2-alpha_ref))`. The typed RMT configuration
exposes these statistics and `F_tracks_network` for explicit alternatives.
These draws specify an initial matrix, not a spectral constraint maintained
through learning.

Initialization and optimization have separate Dale controls:
`model.rmt.dales_init=true` corrects the initial signs. `dales=true` then
uses column signs times `softplus(W_raw)` times the sparsity mask.
`no-dales` uses `W_raw` directly with the same mask, starting from the same
sign-corrected weights but permitting sign changes during optimization.
Both forms multiply by `exp(log_W_raw_gain)`, which stays positive.
`echo` detaches recurrent entries but still learns this gain.

Input weights `(V,N,D)` start as independent `N(0,0.1^2)` draws, are masked
to input neurons, and multiply by a free real `W_in_gain`. The output is a
linear map from the selected output neurons, with a free real `W_out_gain`
and additive bias. The optional skip adds the current input after readout
and requires matching input/output feature counts. Input/output gains are
not subject to Dale's law.

By default the cell exposes synaptic output theta to the readout. `rate`
and `dendritic` readout options expose raw rate and x. Every solver computes
these values from the completed state, avoiding a one-substep output lag.
Seed pairing covers the recurrent matrix, input weights, readout weights,
setpoints, endpoint jitter draws, and initial dendritic state, independently
of variant order or total seed count. Parameters and optimizer moments remain
independent after initialization; burned-in states may differ by condition.

## Integration and training state

The default solver is deterministic SRA1, exactly the zero-noise MATLAB
update for the joint state y:

\[
k_1=f(y),\quad k_2=f(y+3\Delta t k_1/4),\quad
 y_{new}=y+\Delta t(k_1+2k_2)/3.
\]

For 100-Hz observations, `h=0.01` and `ode_unfolds=4` give 2.5-ms internal
steps. Input is held fixed through the substeps. SRA1 performs no state
clipping. Explicit Euler, RK4, and linearly implicit Euler remain available
and use the revised equations, but are not the manuscript comparison solver.
Positive learned time constants have no lower bound tied to the step size;
finite training and the final tau/step diagnostic must therefore be inspected.

`SequenceModel` supplies truncated BPTT and optional gradient checkpointing.
The continuous trainer retains the 24-reader ring state across chunks and
epochs. The primary experiment burns in for 10 seconds and freezes the
resulting initial condition. Burn-in is a finite simulation and need not
reach a fixed point. Evaluation starts each window from that initial state;
it does not inherit the training ring state. Skip and periodic test
evaluation remain supported. See [the protocol](matlab_alignment.md) for
the learning schedule and how test inspection is reported.

## Versioning and diagnostics

The state dictionary carries model version 2. Loading older SRNN states or
configs fails explicitly; reproduce archived runs at their recorded source
commit. Legacy checkpoints cannot be migrated by reshaping one STD axis:
equations, normalization, state layout, and initialization all changed.

`effective_params()` returns transformed recurrent weights, setpoints, time
constants, and budgets. Experiment runs save both initial and final effective
parameters, full trainer checkpoints, source/dataset hashes, and active versus
allocated parameter counts. Counts refer to optimized entries, not identifiable
degrees of freedom. Run the MATLAB derivative/step/trajectory checks and
step-refinement checks before using new results as manuscript evidence.
