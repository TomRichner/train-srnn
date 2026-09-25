# STD strength matching, Dale, skip and timescales (2026-09-25)

30-epoch Modal runs on `cheetah100_warp_multi` (see [timewarp.md](timewarp.md)) unless
noted. Full reports, with activity plots and parameter evolution, are generated outside
the repository by `scripts/report_timewarp.py` in
`$SRNN_HOME/results/stdmatch-20260925/` and `$SRNN_HOME/results/dale-vs-nodales-20260925/`.

## Dale's law

Same five conditions x 15 seeds with and without Dale (`tw30-warpmulti-20260925`,
`tw30-warpmulti-nodales-20260925`). Without Dale every condition learns faster (final
geometric-mean test MSE: SFA1/STD1 0.145 vs 0.156, SFA3/STD2 0.281 vs 0.349, no
adaptation 0.544 vs 0.809); the STD-count penalty (1.96x vs 2.17x) and the null SFA-count
effect are unchanged.

## STD strength matching

Two STD timescales at the default pairs square the steady-state depression. The
matching tokens in [variants.md](../variants.md) equalize it at r_ref = 0.25. Final
geometric-mean test MSE (8 paired seeds, no Dale):

| Condition | default | skip | fast | fast, original data |
|---|---:|---:|---:|---:|
| No adaptation | 0.591 | 0.082 | 0.770 | 0.527 |
| No adaptation, `w-matched` | **0.120** | 0.045 | **0.064** | **0.051** |
| SFA1 / STD1 | 0.146 | **0.034** | 0.132 | 0.088 |
| SFA3 / STD1 | 0.148 | 0.035 | 0.141 | 0.090 |
| SFA3 / STD2 (unmatched) | 0.280 | 0.051 | 0.156 | 0.123 |
| SFA3 / STD2 `std-geo` | 0.151 | 0.036 | 0.139 | 0.087 |
| SFA1 / STD2 `std-geo` | 0.152 | 0.035 | 0.129 | 0.084 |
| SFA3 / STD2 `std-usage` | 0.211 | 0.038 | 0.140 | 0.088 |
| SFA3 / STD2 `std-scale` | 0.642 | 0.088 | 0.762 | 0.638 |
| SFA1 / STD1 `std-strong` | 0.224 | 0.037 | 0.160 | 0.124 |

"fast" = tau_d 0.025 s and every adaptation time x 0.25.

- The geometric mean removes the two-STD penalty (1.92x to 1.04x default; 0.99x on the
  original data with the fast preset), and the SFA count stays neutral. Usage matching
  helps less; the MATLAB route scale (x3 weights) and strengthening the single STD both
  hurt badly. Strong depression caps synaptic output near 1/32 and compresses its
  distribution; the geometric mean keeps the single-STD distribution.
- The unscaled no-adaptation network starts saturated. With its initial recurrent gain at
  the single-STD steady state (`w-matched`) it learns fastest without skip and is second
  with skip, so comparisons of adaptation against no adaptation need this control.
- The skip connection is the largest improvement; faster timescales help modestly and
  shrink the unmatched STD penalty. A single-neuron analysis
  (`scripts/neuron_transfer.py`) shows why: with tau_d = 0.1 s every neuron lags 57 deg
  at the 2.42 Hz gait, far above the adaptation band.
- A fractional-derivative mapping task (`scripts/make_fractional_dataset.py`,
  `tw30-frac-d05-20260925`) was not learned by any condition in 30 epochs (test MSE
  0.84-0.96 of target variance); inconclusive.

Caveats: 30 epochs; 8 seeds (smallest exact two-sided p 0.0078); matched at
initialization only. MATLAB now has the same geometric-mean option
(FractionalReservoir `a0c5cbb`, `synapse_config.<pre>.<post>.std.combine = 'geomean'`).
