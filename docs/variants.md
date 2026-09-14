# SRNN variant names

A variant name is `srnn` followed by tokens, each switching one ablation
relative to the shared model config, and an optional `-seed<n>` suffix that
selects the seed of the recurrent random matrix. Names are lower-case and
are canonicalised to the token order below, so `srnn-skip-no-dales` and
`srnn-no-dales-skip` are the same variant.

| token | effect |
|---|---|
| `no-adapt` | no spike-frequency adaptation and no synaptic depression (`n_a_E = n_a_I = n_b_E = n_b_I = 0`) |
| `sfa-only` | synaptic depression off (`n_b_E = n_b_I = 0`) |
| `std-only` | spike-frequency adaptation off (`n_a_E = n_a_I = 0`) |
| `e-only` | adaptation on excitatory neurons only (`n_a_I = n_b_I = 0`) |
| `no-dales` | unsigned recurrent weights; the softplus magnitude and sign mask are skipped |
| `per-neuron` | per-neuron adaptation parameters train; otherwise each population shares one value |
| `echo` | the recurrent weight structure is frozen (reservoir); its scalar gain still trains |
| `skip` | the readout adds the input, `y = readout(state) + x`; autoregressive tasks only |

`sfa-e-only` and `std-e-only` are accepted shorthands for `sfa-only-e-only`
and `std-only-e-only` and are printed that way.

Tokens modify the base flags in `SRNNModelConfig` (`model.n_a_E=3`,
`model.n_a_I=3`, `model.n_b_E=1`, `model.n_b_I=1`, `model.dales=true`, ...),
so the number of SFA timescales is a shared knob: `model.n_a_E=2
model.variants=[srnn,srnn-e-only]` runs both variants with two timescales.
Solver, step size, and unfolds are shared by every variant in a batch.

## Seeds

```
model.variants=[srnn-skip,srnn-no-adapt-skip] model.variant_seeds=[1,2,3]
```

expands, variant-major, to `srnn-skip-seed1, srnn-skip-seed2, srnn-skip-seed3,
srnn-no-adapt-skip-seed1, ...`. Variants at the same seed share one
recurrent matrix, so each pair is matched on connectivity; the input
weights and readouts are drawn independently per variant. Without
`variant_seeds`, every variant uses `seed` and the names carry no suffix.
The decorated names are stored in every checkpoint (`variant_names`) and
become the per-variant directories of the analysis scripts.

## Variants used in the archived runs

| name | SFA (E, I) | STD (E, I) | Dale | skip |
|---|---|---|---|---|
| `srnn` | 3, 3 | on, on | yes | no |
| `srnn-skip` | 3, 3 | on, on | yes | yes |
| `srnn-no-adapt` | 0, 0 | off, off | yes | no |
| `srnn-no-dales-skip` | 3, 3 | on, on | no | yes |
| `srnn-no-adapt-no-dales` | 0, 0 | off, off | no | no |
| `srnn-no-adapt-no-dales-skip` | 0, 0 | off, off | no | yes |

## Simulating variants

```
python scripts/simulate_srnn.py --variants srnn,srnn-no-adapt,srnn-e-only --N 128
python scripts/simulate_srnn.py --solvers semi_implicit,rk4
```

runs the named variants open loop under a step stimulus and plots synaptic
output, depression, and adaptation per variant into `$SRNN_CACHE_DIR/simulations/`.
