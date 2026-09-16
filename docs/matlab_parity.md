# Deterministic MATLAB–PyTorch SRNN parity

The revised PyTorch cell was compared in float64 with the production
`SRNNCellTypePairs.dynamics_fast` and `sde_fixed_step(..., 'sra1')` functions.
The exact named `sfaEI_mu7_fast` preset was built at 500 neurons, with noise
disabled, for all three conditions. No MATLAB production files were edited.

The fixtures import realized weights (including their rare sign exceptions),
setpoints, heterogeneous SFA ladders, adaptation budgets, STD parameters, and
initial states. Dale enforcement is disabled only for this exact-weight parity
comparison. Outgoing-route parameters and trajectories are asserted identical
before collapsing MATLAB's duplicate route states. MATLAB's column-major state
layout is explicitly converted to the PyTorch cell's row-major layout.

## Results

Activation, raw rate, synaptic output, all derivative blocks, and one SRA1 step
pass for every condition. Non-equilibrium probes additionally exercise nonzero
SFA and both non-unit STD states, rather than only resting adaptation states.

For the 500-neuron MTS model, maximum state errors over 40 seconds were
`4.87e-14` unforced and `7.90e-14` under a uniform 0.1-amplitude step from
10 to 30 seconds. Maximum rate error was below `1.0e-14`. These are well below
the prespecified `1e-7` trajectory tolerance. One-step errors were below
`1.81e-16`, passing `atol=1e-12`, `rtol=1e-10`.

A separate unforced run perturbed initial dendritic states with a unit-norm
uniform direction scaled by `1e-6`. Its canonical-state separation at 40 seconds
was `0.0007371` times its initial separation. This verifies finite-window
contraction for that perturbation and example; it is not a proof of global
stability or a largest-Lyapunov estimate.

Agreement between implementations does not remove time-discretization error.
Against MATLAB ode45 (`RelTol=1e-11`, `AbsTol=1e-13`) over the first second,
maximum absolute state errors for SRA1 were:

| Integration frequency | Maximum error |
|---|---:|
| 400 Hz | 0.0152380 |
| 800 Hz | 0.00380593 |
| 1600 Hz | 0.000955131 |

The roughly fourfold reduction on halving the step confirms second-order
convergence. The 400-Hz result is the approved integration resolution for the
training comparison; these errors should not be described as negligible merely
because cross-language parity is excellent.

## Reproduction

With the FractionalReservoir root selected in MATLAB MCP, run `setup_paths`,
add this repository's `scripts/matlab` directory temporarily, then call
`export_srnn_parity(output_dir)`. Remove that temporary path afterward.
The exporter writes standard SciPy-readable `.mat` files with the preset name
and MATLAB repository revision. It executes the production dynamics and
integrator, not a duplicated MATLAB equation implementation.

From the train-srnn root:

```bash
uv run python scripts/verify_matlab_parity.py /private/tmp/srnn-parity
SRNN_MATLAB_FIXTURES=/private/tmp/srnn-parity uv run pytest -q tests/test_matlab_parity.py
SRNN_MATLAB_FIXTURES=/private/tmp/srnn-parity uv run pytest -q -m slow tests/test_matlab_parity.py
```

The verifier writes `parity_report.json` and unforced/driven overlay PNGs beside
the fixtures. Generated arrays and figures stay outside the repository. Tests
skip explicitly when `SRNN_MATLAB_FIXTURES` is unset, so reference comparisons
cannot silently use an unrelated or missing fixture directory. The report was
produced with PyTorch 2.14.0 on CPU; training separately uses GPU/FP32.
