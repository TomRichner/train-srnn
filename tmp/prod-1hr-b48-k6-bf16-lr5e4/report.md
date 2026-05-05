# Run: prod-1hr-b48-k6-bf16-lr5e4

## Log-log loss curves

![skip variants — log-log loss/metric](log_log_curves_skip.png){ width=95% }

![no-skip variants — log-log loss/metric](log_log_curves_no-skip.png){ width=95% }

\newpage

# srnn-e-only-per-neuron

![Tau evolution](srnn-e-only-per-neuron/tau_evolution.png){ width=95% }

![W\_EI evolution](srnn-e-only-per-neuron/W_EI_evolution.png){ width=95% }

\newpage

# srnn-e-only-per-neuron — parameter table

```text
# prod-1hr-b48-k6-bf16-lr5e4 — variant srnn-e-only-per-neuron (k=0)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                              +1      +0.36238           —           —     scalar
tau_d (s)                                 +0.1     +0.008386           0    0.003537        300
tau_a_E (s)                            +4.8777      +0.46351       3.996      0.3839        450
tau_b_rec_E (s)                             +1      +0.59595           0     0.07419        150
tau_b_rel_E (s)                          +0.25      +0.33863           0     0.07362        150
W_eff (E src, signed)                 +0.23267      +0.22179     0.07641     0.08375      15048
W_eff (I src, signed)                 -0.31062      -0.28856     0.07724     0.08292      14977
|W_eff| (E src)                       +0.23267      +0.22179     0.07641     0.08375      15048
|W_eff| (I src)                       +0.31062      +0.28856     0.07724     0.08292      14977
W_in_eff (input neurons)            -0.0003386   +0.00077924      0.1012     0.04677       6675
c_E (SFA coupling)                       +0.05      +0.14027           0     0.05316        450
c_0_E (SFA offset)                          +0       -0.8618           0       0.218        450
a_0 (threshold)                          +0.35       -0.4167           0      0.1519        300
readout_weight                     -2.8854e-05    -0.0042568     0.06604      0.2784       6675
readout_bias                                +0    +0.0055984           0      0.2357         89
ic.ic (per-variant init)              +0.10386      +0.10386      0.3828      0.3828        900
```

\newpage

# srnn-e-only-skip-per-neuron

![Tau evolution](srnn-e-only-skip-per-neuron/tau_evolution.png){ width=95% }

![W\_EI evolution](srnn-e-only-skip-per-neuron/W_EI_evolution.png){ width=95% }

\newpage

# srnn-e-only-skip-per-neuron — parameter table

```text
# prod-1hr-b48-k6-bf16-lr5e4 — variant srnn-e-only-skip-per-neuron (k=1)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                              +1      +0.41895           —           —     scalar
tau_d (s)                                 +0.1     +0.011451           0    0.004285        300
tau_a_E (s)                            +4.8777      +0.54239       3.996      0.4506        450
tau_b_rec_E (s)                             +1      +0.14509           0     0.02215        150
tau_b_rel_E (s)                          +0.25      +0.24571           0     0.04421        150
W_eff (E src, signed)                 +0.23267      +0.17774     0.07641     0.06325      15048
W_eff (I src, signed)                 -0.31062        -0.234     0.07724     0.06502      14977
|W_eff| (E src)                       +0.23267      +0.17774     0.07641     0.06325      15048
|W_eff| (I src)                       +0.31062        +0.234     0.07724     0.06502      14977
W_in_eff (input neurons)           -0.00084237   -0.00033092      0.1008     0.03457       6675
c_E (SFA coupling)                       +0.05      +0.15993           0     0.06398        450
c_0_E (SFA offset)                          +0      -0.37073           0      0.1716        450
a_0 (threshold)                          +0.35     -0.078787           0      0.1262        300
readout_weight                     +0.00078219    -0.0028954     0.06654      0.1552       6675
readout_bias                                +0      +0.12476           0      0.1602         89
ic.ic (per-variant init)              +0.10327      +0.10327      0.3829      0.3829        900
```

\newpage

# srnn-sfa-e-only-per-neuron

![Tau evolution](srnn-sfa-e-only-per-neuron/tau_evolution.png){ width=95% }

![W\_EI evolution](srnn-sfa-e-only-per-neuron/W_EI_evolution.png){ width=95% }

\newpage

# srnn-sfa-e-only-per-neuron — parameter table

```text
# prod-1hr-b48-k6-bf16-lr5e4 — variant srnn-sfa-e-only-per-neuron (k=2)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                              +1      +0.37433           —           —     scalar
tau_d (s)                                 +0.1    +0.0092731           0    0.003436        300
tau_a_E (s)                            +4.8777      +0.63029       3.996      0.5223        450
tau_b_rec_E (s)                             +1      +0.37433           0           0        150
tau_b_rel_E (s)                          +0.25     +0.093583           0           0        150
W_eff (E src, signed)                 +0.23267      +0.13908     0.07641     0.04858      15048
W_eff (I src, signed)                 -0.31062      -0.18443     0.07724     0.05026      14977
|W_eff| (E src)                       +0.23267      +0.13908     0.07641     0.04858      15048
|W_eff| (I src)                       +0.31062      +0.18443     0.07724     0.05026      14977
W_in_eff (input neurons)           -0.00015382    -0.0016668     0.09934      0.0522       6675
c_E (SFA coupling)                       +0.05     +0.094336           0     0.02552        450
c_0_E (SFA offset)                          +0      -0.34523           0      0.1735        450
a_0 (threshold)                          +0.35      +0.19378           0     0.09833        300
readout_weight                     -0.00098972    +0.0099345     0.06678      0.1995       6675
readout_bias                                +0     -0.074786           0      0.1139         89
ic.ic (per-variant init)              +0.13078      +0.13078      0.4172      0.4172        900
```

\newpage

# srnn-sfa-e-only-skip-per-neuron

![Tau evolution](srnn-sfa-e-only-skip-per-neuron/tau_evolution.png){ width=95% }

![W\_EI evolution](srnn-sfa-e-only-skip-per-neuron/W_EI_evolution.png){ width=95% }

\newpage

# srnn-sfa-e-only-skip-per-neuron — parameter table

```text
# prod-1hr-b48-k6-bf16-lr5e4 — variant srnn-sfa-e-only-skip-per-neuron (k=3)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                              +1      +0.58665           —           —     scalar
tau_d (s)                                 +0.1     +0.027612           0    0.003754        300
tau_a_E (s)                            +4.8777       +2.2891       3.996       1.882        450
tau_b_rec_E (s)                             +1      +0.58665           0           0        150
tau_b_rel_E (s)                          +0.25      +0.14666           0           0        150
W_eff (E src, signed)                 +0.23267      +0.22269     0.07641     0.07435      15048
W_eff (I src, signed)                 -0.31062      -0.29682     0.07724     0.07562      14977
|W_eff| (E src)                       +0.23267      +0.22269     0.07641     0.07435      15048
|W_eff| (I src)                       +0.31062      +0.29682     0.07724     0.07562      14977
W_in_eff (input neurons)            -0.0011621    -0.0024476     0.09962      0.0656       6675
c_E (SFA coupling)                       +0.05     +0.054701           0    0.008876        450
c_0_E (SFA offset)                          +0   -0.00043644           0      0.1036        450
a_0 (threshold)                          +0.35      +0.34929           0     0.08355        300
readout_weight                     +0.00016801    -0.0032955      0.0673     0.07105       6675
readout_bias                                +0     +0.045539           0     0.05694         89
ic.ic (per-variant init)              +0.09231      +0.09231      0.5294      0.5294        900
```

\newpage

# srnn-std-e-only-per-neuron

![Tau evolution](srnn-std-e-only-per-neuron/tau_evolution.png){ width=95% }

![W\_EI evolution](srnn-std-e-only-per-neuron/W_EI_evolution.png){ width=95% }

\newpage

# srnn-std-e-only-per-neuron — parameter table

```text
# prod-1hr-b48-k6-bf16-lr5e4 — variant srnn-std-e-only-per-neuron (k=4)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                              +1      +0.43615           —           —     scalar
tau_d (s)                                 +0.1     +0.012852           0    0.004471        300
tau_a_E (s)                           +0.69315      +0.30232           0           0        450
tau_b_rec_E (s)                             +1       +0.6102           0     0.08069        150
tau_b_rel_E (s)                          +0.25      +0.35639           0     0.05948        150
W_eff (E src, signed)                 +0.23267      +0.19455     0.07641     0.07003      15048
W_eff (I src, signed)                 -0.31062      -0.25558     0.07724     0.07187      14977
|W_eff| (E src)                       +0.23267      +0.19455     0.07641     0.07003      15048
|W_eff| (I src)                       +0.31062      +0.25558     0.07724     0.07187      14977
W_in_eff (input neurons)           +0.00019944   -0.00098555     0.09994     0.06439       6675
c_E (SFA coupling)                       +0.05         +0.05           0           0        450
c_0_E (SFA offset)                          +0            +0           0           0        450
a_0 (threshold)                          +0.35      -0.28501           0       0.129        300
readout_weight                     +0.00074661    +0.0058971     0.06657        0.23       6675
readout_bias                                +0     -0.090532           0      0.2547         89
ic.ic (per-variant init)             +0.090819     +0.090819      0.3835      0.3835        900
```

\newpage

# srnn-std-e-only-skip-per-neuron

![Tau evolution](srnn-std-e-only-skip-per-neuron/tau_evolution.png){ width=95% }

![W\_EI evolution](srnn-std-e-only-skip-per-neuron/W_EI_evolution.png){ width=95% }

\newpage

# srnn-std-e-only-skip-per-neuron — parameter table

```text
# prod-1hr-b48-k6-bf16-lr5e4 — variant srnn-std-e-only-skip-per-neuron (k=5)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                              +1      +0.41461           —           —     scalar
tau_d (s)                                 +0.1     +0.011418           0    0.004133        300
tau_a_E (s)                           +0.69315      +0.28739           0           0        450
tau_b_rec_E (s)                             +1      +0.14509           0      0.0241        150
tau_b_rel_E (s)                          +0.25       +0.2233           0     0.03794        150
W_eff (E src, signed)                 +0.23267      +0.18315     0.07641     0.06494      15048
W_eff (I src, signed)                 -0.31062      -0.24298     0.07724     0.06681      14977
|W_eff| (E src)                       +0.23267      +0.18315     0.07641     0.06494      15048
|W_eff| (I src)                       +0.31062      +0.24298     0.07724     0.06681      14977
W_in_eff (input neurons)           +0.00036119   +0.00034617     0.09956     0.03285       6675
c_E (SFA coupling)                       +0.05         +0.05           0           0        450
c_0_E (SFA offset)                          +0            +0           0           0        450
a_0 (threshold)                          +0.35    +0.0086861           0      0.1454        300
readout_weight                     +0.00055372    -0.0046403     0.06572      0.1426       6675
readout_bias                                +0    -0.0056223           0      0.1422         89
ic.ic (per-variant init)             +0.083543     +0.083543       0.384       0.384        900
```
