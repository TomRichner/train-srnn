# Run: overnight-cl250-resume120__concat

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
# overnight-cl250-resume120__concat — variant srnn-e-only-per-neuron (k=0)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                              +1      +0.52182           —           —     scalar
tau_d (s)                                 +0.1     +0.021479           0    0.007465        300
tau_a_E (s)                            +4.8777      +0.89414       3.996      0.7305        450
tau_b_rec_E (s)                             +1      +0.22725           0      0.0352        150
tau_b_rel_E (s)                          +0.25      +0.32289           0     0.05786        150
W_eff (E src, signed)                 +0.23267      +0.18151     0.07641     0.06566      15048
W_eff (I src, signed)                 -0.31062      -0.24028     0.07724     0.07004      14977
|W_eff| (E src)                       +0.23267      +0.18151     0.07641     0.06566      15048
|W_eff| (I src)                       +0.31062      +0.24028     0.07724     0.07004      14977
W_in_eff (input neurons)            -0.0003386   +0.00081433      0.1012     0.07879       6675
c_E (SFA coupling)                       +0.05     +0.082126           0     0.02424        450
c_0_E (SFA offset)                          +0      -0.36987           0      0.1679        450
a_0 (threshold)                          +0.35      +0.16904           0      0.1091        300
readout_weight                     -2.8854e-05   -0.00058804     0.06604      0.2205       6675
readout_bias                                +0   +0.00010319           0      0.2097         89
ic.ic (per-variant init)             +0.098875     -0.055855      0.3842      0.5219        900
```

\newpage

# srnn-e-only-skip-per-neuron

![Tau evolution](srnn-e-only-skip-per-neuron/tau_evolution.png){ width=95% }

![W\_EI evolution](srnn-e-only-skip-per-neuron/W_EI_evolution.png){ width=95% }

\newpage

# srnn-e-only-skip-per-neuron — parameter table

```text
# overnight-cl250-resume120__concat — variant srnn-e-only-skip-per-neuron (k=1)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                              +1      +0.48399           —           —     scalar
tau_d (s)                                 +0.1     +0.016801           0    0.005397        300
tau_a_E (s)                            +4.8777      +0.94549       3.996      0.7839        450
tau_b_rec_E (s)                             +1      +0.22835           0      0.0369        150
tau_b_rel_E (s)                          +0.25      +0.20273           0     0.03824        150
W_eff (E src, signed)                 +0.23267      +0.19911     0.07641     0.07014      15048
W_eff (I src, signed)                 -0.31062      -0.26301     0.07724     0.07369      14977
|W_eff| (E src)                       +0.23267      +0.19911     0.07641     0.07014      15048
|W_eff| (I src)                       +0.31062      +0.26301     0.07724     0.07369      14977
W_in_eff (input neurons)           -0.00084237   -0.00083735      0.1008     0.03921       6675
c_E (SFA coupling)                       +0.05     +0.099054           0     0.03138        450
c_0_E (SFA offset)                          +0       -0.2607           0      0.1696        450
a_0 (threshold)                          +0.35      +0.14095           0      0.1374        300
readout_weight                     +0.00078219    -0.0052303     0.06654       0.135       6675
readout_bias                                +0      +0.11501           0      0.1561         89
ic.ic (per-variant init)             +0.098961     -0.032454       0.387      0.5247        900
```

\newpage

# srnn-sfa-e-only-per-neuron

![Tau evolution](srnn-sfa-e-only-per-neuron/tau_evolution.png){ width=95% }

![W\_EI evolution](srnn-sfa-e-only-per-neuron/W_EI_evolution.png){ width=95% }

\newpage

# srnn-sfa-e-only-per-neuron — parameter table

```text
# overnight-cl250-resume120__concat — variant srnn-sfa-e-only-per-neuron (k=2)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                              +1       +0.6228           —           —     scalar
tau_d (s)                                 +0.1     +0.031946           0    0.008385        300
tau_a_E (s)                            +4.8777       +2.3462       3.996       1.912        450
tau_b_rec_E (s)                             +1       +0.6228           0           0        150
tau_b_rel_E (s)                          +0.25       +0.1557           0           0        150
W_eff (E src, signed)                 +0.23267      +0.22561     0.07641     0.07892      15048
W_eff (I src, signed)                 -0.31062      -0.29814     0.07724     0.08305      14977
|W_eff| (E src)                       +0.23267      +0.22561     0.07641     0.07892      15048
|W_eff| (I src)                       +0.31062      +0.29814     0.07724     0.08305      14977
W_in_eff (input neurons)           -0.00015382    -0.0018244     0.09934      0.1217       6675
c_E (SFA coupling)                       +0.05     +0.045337           0    0.005954        450
c_0_E (SFA offset)                          +0      -0.21542           0      0.1249        450
a_0 (threshold)                          +0.35        +0.366           0     0.08314        300
readout_weight                     -0.00098972    +0.0084713     0.06678      0.1704       6675
readout_bias                                +0      -0.10435           0      0.1289         89
ic.ic (per-variant init)              +0.13078    -0.0045598      0.4172      0.4608        900
```

\newpage

# srnn-sfa-e-only-skip-per-neuron

![Tau evolution](srnn-sfa-e-only-skip-per-neuron/tau_evolution.png){ width=95% }

![W\_EI evolution](srnn-sfa-e-only-skip-per-neuron/W_EI_evolution.png){ width=95% }

\newpage

# srnn-sfa-e-only-skip-per-neuron — parameter table

```text
# overnight-cl250-resume120__concat — variant srnn-sfa-e-only-skip-per-neuron (k=3)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                              +1      +0.68228           —           —     scalar
tau_d (s)                                 +0.1     +0.040003           0    0.004433        300
tau_a_E (s)                            +4.8777       +2.9413       3.996       2.417        450
tau_b_rec_E (s)                             +1      +0.68228           0           0        150
tau_b_rel_E (s)                          +0.25      +0.17057           0           0        150
W_eff (E src, signed)                 +0.23267      +0.22832     0.07641     0.07621      15048
W_eff (I src, signed)                 -0.31062      -0.30425     0.07724     0.07747      14977
|W_eff| (E src)                       +0.23267      +0.22832     0.07641     0.07621      15048
|W_eff| (I src)                       +0.31062      +0.30425     0.07724     0.07747      14977
W_in_eff (input neurons)            -0.0011621    -0.0030177     0.09962     0.07681       6675
c_E (SFA coupling)                       +0.05     +0.053708           0    0.007939        450
c_0_E (SFA offset)                          +0     +0.017061           0      0.1012        450
a_0 (threshold)                          +0.35      +0.35718           0      0.0842        300
readout_weight                     +0.00016801    -0.0028867      0.0673     0.06855       6675
readout_bias                                +0       +0.0371           0     0.04487         89
ic.ic (per-variant init)              +0.09231     +0.054494      0.5294      0.4889        900
```

\newpage

# srnn-std-e-only-per-neuron

![Tau evolution](srnn-std-e-only-per-neuron/tau_evolution.png){ width=95% }

![W\_EI evolution](srnn-std-e-only-per-neuron/W_EI_evolution.png){ width=95% }

\newpage

# srnn-std-e-only-per-neuron — parameter table

```text
# overnight-cl250-resume120__concat — variant srnn-std-e-only-per-neuron (k=4)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                              +1      +0.69512           —           —     scalar
tau_d (s)                                 +0.1     +0.039022           0    0.009972        300
tau_a_E (s)                           +0.69315      +0.48182           0           0        450
tau_b_rec_E (s)                             +1       +0.3318           0     0.03998        150
tau_b_rel_E (s)                          +0.25        +0.387           0     0.05346        150
W_eff (E src, signed)                 +0.23267      +0.25831     0.07641     0.09257      15048
W_eff (I src, signed)                 -0.31062      -0.34123     0.07724     0.09821      14977
|W_eff| (E src)                       +0.23267      +0.25831     0.07641     0.09257      15048
|W_eff| (I src)                       +0.31062      +0.34123     0.07724     0.09821      14977
W_in_eff (input neurons)           +0.00019944   -0.00098562     0.09994      0.1332       6675
c_E (SFA coupling)                       +0.05         +0.05           0           0        450
c_0_E (SFA offset)                          +0            +0           0           0        450
a_0 (threshold)                          +0.35      +0.28106           0      0.1036        300
readout_weight                     +0.00074661     +0.010267     0.06657      0.1858       6675
readout_bias                                +0     -0.022221           0       0.187         89
ic.ic (per-variant init)             +0.087739     +0.030821      0.3868      0.4926        900
```

\newpage

# srnn-std-e-only-skip-per-neuron

![Tau evolution](srnn-std-e-only-skip-per-neuron/tau_evolution.png){ width=95% }

![W\_EI evolution](srnn-std-e-only-skip-per-neuron/W_EI_evolution.png){ width=95% }

\newpage

# srnn-std-e-only-skip-per-neuron — parameter table

```text
# overnight-cl250-resume120__concat — variant srnn-std-e-only-skip-per-neuron (k=5)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                              +1      +0.48709           —           —     scalar
tau_d (s)                                 +0.1      +0.01728           0    0.005127        300
tau_a_E (s)                           +0.69315      +0.33763           0           0        450
tau_b_rec_E (s)                             +1      +0.25985           0     0.03932        150
tau_b_rel_E (s)                          +0.25      +0.17382           0     0.03227        150
W_eff (E src, signed)                 +0.23267       +0.2124     0.07641     0.07436      15048
W_eff (I src, signed)                 -0.31062      -0.28145     0.07724     0.07743      14977
|W_eff| (E src)                       +0.23267       +0.2124     0.07641     0.07436      15048
|W_eff| (I src)                       +0.31062      +0.28145     0.07724     0.07743      14977
W_in_eff (input neurons)           +0.00036119   +7.5388e-06     0.09956     0.03832       6675
c_E (SFA coupling)                       +0.05         +0.05           0           0        450
c_0_E (SFA offset)                          +0            +0           0           0        450
a_0 (threshold)                          +0.35      +0.20351           0      0.1406        300
readout_weight                     +0.00055372    -0.0079787     0.06572      0.1213       6675
readout_bias                                +0     +0.050723           0      0.1257         89
ic.ic (per-variant init)             +0.089141   -0.00010804      0.3857      0.4971        900
```
