# Run: overnight-cl250-resume120

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
# overnight-cl250-resume120 — variant srnn-e-only-per-neuron (k=0)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                        +0.75694      +0.52182           —           —     scalar
tau_d (s)                            +0.047285     +0.021479     0.00829    0.007465        300
tau_a_E (s)                            +2.5842      +0.89414       2.116      0.7305        450
tau_b_rec_E (s)                       +0.45844      +0.22725     0.04265      0.0352        150
tau_b_rel_E (s)                       +0.32092      +0.32289     0.03539     0.05786        150
W_eff (E src, signed)                 +0.24787      +0.18151     0.08534     0.06566      15048
W_eff (I src, signed)                 -0.32687      -0.24028     0.08695     0.07004      14977
|W_eff| (E src)                       +0.24787      +0.18151     0.08534     0.06566      15048
|W_eff| (I src)                       +0.32687      +0.24028     0.08695     0.07004      14977
W_in_eff (input neurons)            +0.0013842   +0.00081433      0.1342     0.07879       6675
c_E (SFA coupling)                    +0.05476     +0.082126    0.006898     0.02424        450
c_0_E (SFA offset)                    -0.27431      -0.36987     0.09773      0.1679        450
a_0 (threshold)                       +0.16777      +0.16904     0.08514      0.1091        300
readout_weight                      -0.0057749   -0.00058804      0.1456      0.2205       6675
readout_bias                        +0.0044741   +0.00010319      0.1548      0.2097         89
ic.ic (per-variant init)             -0.025816     -0.055855       0.504      0.5219        900
```

\newpage

# srnn-e-only-skip-per-neuron

![Tau evolution](srnn-e-only-skip-per-neuron/tau_evolution.png){ width=95% }

![W\_EI evolution](srnn-e-only-skip-per-neuron/W_EI_evolution.png){ width=95% }

\newpage

# srnn-e-only-skip-per-neuron — parameter table

```text
# overnight-cl250-resume120 — variant srnn-e-only-skip-per-neuron (k=1)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                        +0.71474      +0.48399           —           —     scalar
tau_d (s)                            +0.045103     +0.016801     0.00597    0.005397        300
tau_a_E (s)                            +2.9855      +0.94549       2.453      0.7839        450
tau_b_rec_E (s)                       +0.64798      +0.22835     0.02907      0.0369        150
tau_b_rel_E (s)                       +0.17806      +0.20273    0.008715     0.03824        150
W_eff (E src, signed)                 +0.22277      +0.19911     0.07386     0.07014      15048
W_eff (I src, signed)                 -0.29646      -0.26301     0.07555     0.07369      14977
|W_eff| (E src)                       +0.22277      +0.19911     0.07386     0.07014      15048
|W_eff| (I src)                       +0.29646      +0.26301     0.07555     0.07369      14977
W_in_eff (input neurons)             +0.001791   -0.00083735      0.0739     0.03921       6675
c_E (SFA coupling)                   +0.052935     +0.099054    0.003242     0.03138        450
c_0_E (SFA offset)                  -0.0012899       -0.2607     0.05942      0.1696        450
a_0 (threshold)                       +0.28407      +0.14095     0.07765      0.1374        300
readout_weight                       -0.013388    -0.0052303     0.07669       0.135       6675
readout_bias                         +0.075552      +0.11501     0.08734      0.1561         89
ic.ic (per-variant init)             +0.062319     -0.032454      0.4358      0.5247        900
```

\newpage

# srnn-sfa-e-only-per-neuron

![Tau evolution](srnn-sfa-e-only-per-neuron/tau_evolution.png){ width=95% }

![W\_EI evolution](srnn-sfa-e-only-per-neuron/W_EI_evolution.png){ width=95% }

\newpage

# srnn-sfa-e-only-per-neuron — parameter table

```text
# overnight-cl250-resume120 — variant srnn-sfa-e-only-per-neuron (k=2)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                        +0.77065       +0.6228           —           —     scalar
tau_d (s)                            +0.050201     +0.031946    0.006832    0.008385        300
tau_a_E (s)                            +3.2983       +2.3462       2.703       1.912        450
tau_b_rec_E (s)                       +0.77065       +0.6228           0           0        150
tau_b_rel_E (s)                       +0.19266       +0.1557           0           0        150
W_eff (E src, signed)                 +0.24106      +0.22561     0.08131     0.07892      15048
W_eff (I src, signed)                 -0.31873      -0.29814     0.08281     0.08305      14977
|W_eff| (E src)                       +0.24106      +0.22561     0.08131     0.07892      15048
|W_eff| (I src)                       +0.31873      +0.29814     0.08281     0.08305      14977
W_in_eff (input neurons)            -0.0020851    -0.0018244      0.1374      0.1217       6675
c_E (SFA coupling)                   +0.051482     +0.045337    0.003646    0.005954        450
c_0_E (SFA offset)                   -0.079732      -0.21542     0.06781      0.1249        450
a_0 (threshold)                       +0.34909        +0.366     0.05572     0.08314        300
readout_weight                      +0.0052418    +0.0084713      0.1206      0.1704       6675
readout_bias                         -0.026226      -0.10435     0.06046      0.1289         89
ic.ic (per-variant init)             +0.063944    -0.0045598      0.4472      0.4608        900
```

\newpage

# srnn-sfa-e-only-skip-per-neuron

![Tau evolution](srnn-sfa-e-only-skip-per-neuron/tau_evolution.png){ width=95% }

![W\_EI evolution](srnn-sfa-e-only-skip-per-neuron/W_EI_evolution.png){ width=95% }

\newpage

# srnn-sfa-e-only-skip-per-neuron — parameter table

```text
# overnight-cl250-resume120 — variant srnn-sfa-e-only-skip-per-neuron (k=3)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                          +1.016      +0.68228           —           —     scalar
tau_d (s)                             +0.10403     +0.040003    0.003474    0.004433        300
tau_a_E (s)                            +4.8605       +2.9413       3.986       2.417        450
tau_b_rec_E (s)                         +1.016      +0.68228           0           0        150
tau_b_rel_E (s)                         +0.254      +0.17057           0           0        150
W_eff (E src, signed)                 +0.22921      +0.22832     0.07535     0.07621      15048
W_eff (I src, signed)                 -0.30625      -0.30425     0.07637     0.07747      14977
|W_eff| (E src)                       +0.22921      +0.22832     0.07535     0.07621      15048
|W_eff| (I src)                       +0.30625      +0.30425     0.07637     0.07747      14977
W_in_eff (input neurons)            -0.0014667    -0.0030177     0.09388     0.07681       6675
c_E (SFA coupling)                   +0.051699     +0.053708    0.001434    0.007939        450
c_0_E (SFA offset)                   +0.036936     +0.017061     0.02567      0.1012        450
a_0 (threshold)                       +0.37426      +0.35718     0.03578      0.0842        300
readout_weight                      -0.0010738    -0.0028867     0.06088     0.06855       6675
readout_bias                        +0.0022042       +0.0371    0.003744     0.04487         89
ic.ic (per-variant init)               +0.1172     +0.054494      0.4144      0.4889        900
```

\newpage

# srnn-std-e-only-per-neuron

![Tau evolution](srnn-std-e-only-per-neuron/tau_evolution.png){ width=95% }

![W\_EI evolution](srnn-std-e-only-per-neuron/W_EI_evolution.png){ width=95% }

\newpage

# srnn-std-e-only-per-neuron — parameter table

```text
# overnight-cl250-resume120 — variant srnn-std-e-only-per-neuron (k=4)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                        +0.79317      +0.69512           —           —     scalar
tau_d (s)                            +0.052401     +0.039022    0.007879    0.009972        300
tau_a_E (s)                           +0.54978      +0.48182           0           0        450
tau_b_rec_E (s)                       +0.49281       +0.3318     0.03874     0.03998        150
tau_b_rel_E (s)                       +0.32451        +0.387     0.03178     0.05346        150
W_eff (E src, signed)                 +0.27677      +0.25831     0.09482     0.09257      15048
W_eff (I src, signed)                 -0.36449      -0.34123     0.09683     0.09821      14977
|W_eff| (E src)                       +0.27677      +0.25831     0.09482     0.09257      15048
|W_eff| (I src)                       +0.36449      +0.34123     0.09683     0.09821      14977
W_in_eff (input neurons)           -0.00091659   -0.00098562      0.1406      0.1332       6675
c_E (SFA coupling)                       +0.05         +0.05           0           0        450
c_0_E (SFA offset)                          +0            +0           0           0        450
a_0 (threshold)                       +0.22171      +0.28106     0.08034      0.1036        300
readout_weight                      +0.0096444     +0.010267      0.1413      0.1858       6675
readout_bias                         -0.022083     -0.022221      0.1547       0.187         89
ic.ic (per-variant init)            +0.0046818     +0.030821         0.5      0.4926        900
```

\newpage

# srnn-std-e-only-skip-per-neuron

![Tau evolution](srnn-std-e-only-skip-per-neuron/tau_evolution.png){ width=95% }

![W\_EI evolution](srnn-std-e-only-skip-per-neuron/W_EI_evolution.png){ width=95% }

\newpage

# srnn-std-e-only-skip-per-neuron — parameter table

```text
# overnight-cl250-resume120 — variant srnn-std-e-only-skip-per-neuron (k=5)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                        +0.73075      +0.48709           —           —     scalar
tau_d (s)                            +0.048476      +0.01728    0.004967    0.005127        300
tau_a_E (s)                           +0.50652      +0.33763           0           0        450
tau_b_rec_E (s)                       +0.68599      +0.25985     0.02816     0.03932        150
tau_b_rel_E (s)                       +0.18328      +0.17382    0.008936     0.03227        150
W_eff (E src, signed)                 +0.24529       +0.2124     0.08121     0.07436      15048
W_eff (I src, signed)                  -0.3268      -0.28145     0.08245     0.07743      14977
|W_eff| (E src)                       +0.24529       +0.2124     0.08121     0.07436      15048
|W_eff| (I src)                        +0.3268      +0.28145     0.08245     0.07743      14977
W_in_eff (input neurons)           -0.00069182   +7.5388e-06     0.08771     0.03832       6675
c_E (SFA coupling)                       +0.05         +0.05           0           0        450
c_0_E (SFA offset)                          +0            +0           0           0        450
a_0 (threshold)                        +0.3437      +0.20351     0.05943      0.1406        300
readout_weight                      -0.0083536    -0.0079787     0.06826      0.1213       6675
readout_bias                         +0.024059     +0.050723     0.04786      0.1257         89
ic.ic (per-variant init)             +0.073535   -0.00010804      0.4271      0.4971        900
```
