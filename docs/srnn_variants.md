# SRNN Ablation Variants

All variants are defined in `SRNN_PRESETS` in `train_srnn/models/srnn_cell.py` and as YAML configs in `conf/model/srnn_*.yaml`.

## Parameter Reference

| Parameter | Description |
|-----------|-------------|
| `n_a_E` | SFA timescales for excitatory neurons (0=off, 1=single, >=2=multi-timescale) |
| `n_a_I` | SFA timescales for inhibitory neurons |
| `n_b_E` | STD for excitatory neurons (0=off, 1=on) |
| `n_b_I` | STD for inhibitory neurons |
| `dales` | Dale's law constraint (E columns positive, I columns negative) |
| `echo` | Reservoir/echo-state mode (freeze recurrent weights W) |
| `per_neuron` | Per-neuron SFA/STD time constants vs per-population |
| `solver` | ODE solver (semi_implicit, explicit, rk4, exponential) |

Default values (from `SRNNConfig`): `n_a_E=3, n_a_I=3, n_b_E=1, n_b_I=1, dales=True, echo=False, per_neuron=False, solver=semi_implicit`.

## Variant Table

| Preset | n_a_E | n_a_I | n_b_E | n_b_I | dales | echo | per_neuron | solver | Description |
|--------|:-----:|:-----:|:-----:|:-----:|:-----:|:----:|:----------:|--------|-------------|
| `srnn` | 3 | 3 | 1 | 1 | Y | N | N | semi_implicit | Full model (all defaults) |
| `srnn-per-neuron` | 3 | 3 | 1 | 1 | Y | N | Y | semi_implicit | Per-neuron time constants |
| `srnn-echo` | 3 | 3 | 1 | 1 | Y | Y | N | semi_implicit | Reservoir mode (W frozen) |
| `srnn-no-adapt` | 0 | 0 | 0 | 0 | Y | N | N | semi_implicit | No SFA, no STD |
| `srnn-no-adapt-no-dales` | 0 | 0 | 0 | 0 | N | N | N | semi_implicit | No SFA, no STD, no Dale's law |
| `srnn-sfa-only` | 1 | 1 | 0 | 0 | Y | N | N | semi_implicit | SFA only (1 timescale each) |
| `srnn-std-only` | 0 | 0 | 1 | 1 | Y | N | N | semi_implicit | STD only |
| `srnn-E-only` | 3 | 0 | 1 | 0 | Y | N | N | semi_implicit | Adaptation on E neurons only |
| `srnn-e-only-echo` | 3 | 0 | 1 | 0 | Y | Y | N | semi_implicit | E-only + reservoir mode |
| `srnn-e-only-per-neuron` | 3 | 0 | 1 | 0 | Y | N | Y | semi_implicit | E-only + per-neuron time constants |
| `srnn-multi-sfa` | 2 | 2 | 1 | 1 | Y | N | N | semi_implicit | 2 SFA timescales (E and I) |
| `srnn-multi-sfa-E` | 2 | 0 | 1 | 0 | Y | N | N | semi_implicit | 2 SFA timescales (E only) + STD E |
| `srnn-no-dales` | 3 | 3 | 1 | 1 | N | N | N | semi_implicit | Full model without Dale's law |
| `srnn-explicit` | 3 | 3 | 1 | 1 | Y | N | N | explicit | Full model, forward Euler solver |
| `srnn-rk4` | 3 | 3 | 1 | 1 | Y | N | N | rk4 | Full model, RK4 solver |

## Batched Ablation Usage

Multiple variants can be run in parallel via `BatchedSRNNCell` using `torch.bmm`:

```bash
python train.py task=har batched_ablations='[srnn,srnn-no-adapt,srnn-E-only]'
```

All variants in a batch must share the same `solver`, `h`, and `ode_unfolds`. The solver-specific presets (`srnn-explicit`, `srnn-rk4`) cannot be mixed with the default `semi_implicit` variants in a single batch.

## Visualization

```bash
# Compare two variants in detail (6 panels each)
PYTHONPATH=. .venv/bin/python scripts/compare_ablations.py

# Survey all 13 non-solver variants (3 panels each, wrapped grid)
PYTHONPATH=. .venv/bin/python scripts/compare_all_ablations.py --N 300
```
