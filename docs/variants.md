# SRNN variant names

A variant name is `srnn` followed by tokens and an optional `-seed<n>`
suffix. Names are lower-case and canonicalized, so token ordering does not
change the variant. All variants in a cell share network size, activation
shape, solver, observation step, and internal substep count.

| Token | Effect |
|---|---|
| `sfa1-std1` | One SFA and one STD timescale on both E and I |
| `sfa3-std2` | Three SFA and two STD timescales on both E and I |
| `no-adapt` | No SFA or STD on either population |
| `sfa-only` | STD off on both populations |
| `std-only` | SFA off on both populations |
| `e-only` | SFA and STD off on inhibitory neurons |
| `no-dales` | Permit recurrent sign changes during optimization |
| `per-neuron` | Learn neuronal parameter vectors as well as shared gains/offsets |
| `echo` | Detach recurrent weight entries; recurrent scalar gain still learns |
| `skip` | Add current input to the readout; requires matching feature counts |

`sfa-e-only` and `std-e-only` are aliases for `sfa-only-e-only` and
`std-only-e-only`. Explicit condition tokens are mutually exclusive;
combining `sfa1-std1` or `sfa3-std2` with adaptation-removal modifiers is
rejected. Options such as `skip`, `no-dales`, and `per-neuron` can accompany
a condition.

Bare `srnn` uses the shared configuration, defaulting to three SFA and two
STD timescales on both populations. Shared count overrides still support
other legacy ablations, but the explicit condition tokens pin their named
counts. Counts denote timescales, not neuron types.

Without `per-neuron`, initialized setpoint and timescale heterogeneity
remains fixed while shared gains or offsets learn. `no-dales` does not
change the default sign-corrected initialization: that is controlled
separately by `model.rmt.dales_init`. See [the model specification](model.md)
for transforms and initialization.

## Paired seeds

```text
model.variants=[srnn-no-adapt,srnn-sfa1-std1,srnn-sfa3-std2]
model.variant_seeds=[1,2,3]
```

This expands in variant-major order into nine independent networks in one
batched cell. Each seed pairs recurrent weights, input weights, readout
weights, setpoints, endpoint jitter draws, and initial dendritic draws
across conditions. Initialization does not depend on variant ordering or
the total seed count. The subsequent parameters and optimizer moments are
independent. Condition-specific burn-in can produce different settled
initial states despite the shared dendritic draw.

Without `variant_seeds`, unspecified variant seeds use the run-level seed;
explicit suffixes still select a seed. Expanded names are stored in
checkpoints and histories. The [experiment runner](matlab_alignment.md)
uses consecutive seeds and provides skip/Dale switches.

## Historical runs

Archived `srnn` results used the previous equations and usually one STD
state per neuron. The same name under version 2 now describes the revised
base configuration. Always interpret an archived name with its saved
configuration and source revision. Version 2 rejects incompatible old
checkpoints; do not label historical results as the new three-condition
comparison.
