# Run: rk4-wout-300e

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
# rk4-wout-300e — variant srnn-e-only-per-neuron (k=0)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                              +1      +0.36117           —           —     scalar
tau_d (s)                                 +0.1    +0.0059435           0    0.003387        300
tau_a_E[j=0] (s)                         +0.25     +0.014364           0    0.006425        150
tau_a_E[j=1] (s)                       +4.3832      +0.53873           0     0.08523        150
tau_a_E[j=2] (s)                           +10       +1.2049           0     0.07775        150
tau_b_rec_E (s)                             +1       +2.4697           0      0.6851        150
tau_b_rel_E (s)                          +0.25      +0.46061           0      0.1399        150
W_eff (E src, signed)                 +0.23267      +0.28082     0.07641      0.1368      15048
W_eff (I src, signed)                 -0.31062      -0.34389     0.07724      0.1101      14977
|W_eff| (E src)                       +0.23267      +0.28082     0.07641      0.1368      15048
|W_eff| (I src)                       +0.31062      +0.34389     0.07724      0.1101      14977
W_in_eff (input neurons)            -0.0003386    +6.718e-05      0.1012     0.04455       6675
c_E[j=0] (SFA coupling)                  +0.05      +0.32175           0      0.2115        150
c_E[j=1] (SFA coupling)                  +0.05      +0.13592           0     0.03189        150
c_E[j=2] (SFA coupling)                  +0.05      +0.14416           0     0.03062        150
c_0_E[j=0] (SFA offset)                     +0       -0.7737           0      0.1655        150
c_0_E[j=1] (SFA offset)                     +0       -1.0014           0      0.3186        150
c_0_E[j=2] (SFA offset)                     +0       -1.0468           0      0.3779        150
a_0 (threshold)                          +0.35      -0.82632           0      0.1613        300
readout_weight                     -2.8854e-05    +0.0014493     0.06604      0.3687       6675
readout_bias                                +0     +0.023748           0      0.1002         89
ic.ic (per-variant init)              +0.10233      +0.10233      0.3832      0.3832        900
```

\newpage

# srnn-e-only-skip-per-neuron

![Tau evolution](srnn-e-only-skip-per-neuron/tau_evolution.png){ width=95% }

![W\_EI evolution](srnn-e-only-skip-per-neuron/W_EI_evolution.png){ width=95% }

\newpage

# srnn-e-only-skip-per-neuron — parameter table

```text
# rk4-wout-300e — variant srnn-e-only-skip-per-neuron (k=1)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                              +1      +0.37529           —           —     scalar
tau_d (s)                                 +0.1    +0.0095989           0    0.004796        300
tau_a_E[j=0] (s)                         +0.25     +0.022312           0     0.01249        150
tau_a_E[j=1] (s)                       +4.3832      +0.59099           0     0.05747        150
tau_a_E[j=2] (s)                           +10       +1.3304           0      0.0452        150
tau_b_rec_E (s)                             +1      +0.13967           0     0.03327        150
tau_b_rel_E (s)                          +0.25       +0.4179           0      0.1362        150
W_eff (E src, signed)                 +0.23267       +0.1623     0.07641     0.06233      15048
W_eff (I src, signed)                 -0.31062      -0.21008     0.07724     0.06407      14977
|W_eff| (E src)                       +0.23267       +0.1623     0.07641     0.06233      15048
|W_eff| (I src)                       +0.31062      +0.21008     0.07724     0.06407      14977
W_in_eff (input neurons)           -0.00084237    -0.0003466      0.1008     0.01914       6675
c_E[j=0] (SFA coupling)                  +0.05      +0.14153           0     0.07546        150
c_E[j=1] (SFA coupling)                  +0.05     +0.082642           0     0.01972        150
c_E[j=2] (SFA coupling)                  +0.05     +0.085951           0     0.01753        150
c_0_E[j=0] (SFA offset)                     +0      -0.37156           0      0.1258        150
c_0_E[j=1] (SFA offset)                     +0        -0.416           0      0.2162        150
c_0_E[j=2] (SFA offset)                     +0      -0.41887           0      0.2292        150
a_0 (threshold)                          +0.35     -0.057435           0      0.1178        300
readout_weight                     +0.00078219    -0.0061936     0.06654      0.1652       6675
readout_bias                                +0      +0.12832           0      0.1694         89
ic.ic (per-variant init)              +0.10092      +0.10092      0.3832      0.3832        900
```

\newpage

# srnn-sfa-e-only-per-neuron

![Tau evolution](srnn-sfa-e-only-per-neuron/tau_evolution.png){ width=95% }

![W\_EI evolution](srnn-sfa-e-only-per-neuron/W_EI_evolution.png){ width=95% }

\newpage

# srnn-sfa-e-only-per-neuron — parameter table

```text
# rk4-wout-300e — variant srnn-sfa-e-only-per-neuron (k=2)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                              +1       +0.3198           —           —     scalar
tau_d (s)                                 +0.1     +0.005601           0    0.002688        300
tau_a_E[j=0] (s)                         +0.25     +0.013493           0    0.006264        150
tau_a_E[j=1] (s)                       +4.3832      +0.47568           0     0.06381        150
tau_a_E[j=2] (s)                           +10       +1.0598           0     0.06605        150
tau_b_rec_E (s)                             +1       +0.3198           0           0        150
tau_b_rel_E (s)                          +0.25      +0.07995           0           0        150
W_eff (E src, signed)                 +0.23267      +0.16148     0.07641      0.0616      15048
W_eff (I src, signed)                 -0.31062      -0.21457     0.07724     0.06388      14977
|W_eff| (E src)                       +0.23267      +0.16148     0.07641      0.0616      15048
|W_eff| (I src)                       +0.31062      +0.21457     0.07724     0.06388      14977
W_in_eff (input neurons)           -0.00015382   -0.00067693     0.09934       0.044       6675
c_E[j=0] (SFA coupling)                  +0.05      +0.45524           0      0.3174        150
c_E[j=1] (SFA coupling)                  +0.05      +0.17414           0     0.05186        150
c_E[j=2] (SFA coupling)                  +0.05      +0.18888           0     0.04269        150
c_0_E[j=0] (SFA offset)                     +0      -0.39567           0       0.179        150
c_0_E[j=1] (SFA offset)                     +0      -0.40352           0      0.2373        150
c_0_E[j=2] (SFA offset)                     +0      -0.40179           0      0.2497        150
a_0 (threshold)                          +0.35      +0.25068           0      0.1195        300
readout_weight                     -0.00098972     +0.003701     0.06678      0.2512       6675
readout_bias                                +0      -0.06826           0      0.1322         89
ic.ic (per-variant init)             +0.098719     +0.098719      0.4415      0.4415        900
```

\newpage

# srnn-sfa-e-only-skip-per-neuron

![Tau evolution](srnn-sfa-e-only-skip-per-neuron/tau_evolution.png){ width=95% }

![W\_EI evolution](srnn-sfa-e-only-skip-per-neuron/W_EI_evolution.png){ width=95% }

\newpage

# srnn-sfa-e-only-skip-per-neuron — parameter table

```text
# rk4-wout-300e — variant srnn-sfa-e-only-skip-per-neuron (k=3)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                              +1      +0.38706           —           —     scalar
tau_d (s)                                 +0.1     +0.011607           0     0.00269        300
tau_a_E[j=0] (s)                         +0.25     +0.070031           0     0.01582        150
tau_a_E[j=1] (s)                       +4.3832       +1.3521           0     0.05551        150
tau_a_E[j=2] (s)                           +10       +3.1278           0     0.02274        150
tau_b_rec_E (s)                             +1      +0.38706           0           0        150
tau_b_rel_E (s)                          +0.25     +0.096764           0           0        150
W_eff (E src, signed)                 +0.23267      +0.20496     0.07641     0.07074      15048
W_eff (I src, signed)                 -0.31062      -0.27248     0.07724     0.07366      14977
|W_eff| (E src)                       +0.23267      +0.20496     0.07641     0.07074      15048
|W_eff| (I src)                       +0.31062      +0.27248     0.07724     0.07366      14977
W_in_eff (input neurons)            -0.0011621   -0.00046372     0.09962     0.03058       6675
c_E[j=0] (SFA coupling)                  +0.05     +0.060161           0     0.02322        150
c_E[j=1] (SFA coupling)                  +0.05     +0.053451           0     0.01059        150
c_E[j=2] (SFA coupling)                  +0.05     +0.053542           0     0.01022        150
c_0_E[j=0] (SFA offset)                     +0      -0.05172           0      0.1646        150
c_0_E[j=1] (SFA offset)                     +0     -0.049636           0      0.1857        150
c_0_E[j=2] (SFA offset)                     +0     -0.048053           0      0.1822        150
a_0 (threshold)                          +0.35      +0.28874           0      0.1135        300
readout_weight                     +0.00016801    -0.0040284      0.0673      0.1052       6675
readout_bias                                +0     +0.076065           0      0.1312         89
ic.ic (per-variant init)              +0.12557      +0.12557        0.42        0.42        900
```

\newpage

# srnn-std-e-only-per-neuron

![Tau evolution](srnn-std-e-only-per-neuron/tau_evolution.png){ width=95% }

![W\_EI evolution](srnn-std-e-only-per-neuron/W_EI_evolution.png){ width=95% }

\newpage

# srnn-std-e-only-per-neuron — parameter table

```text
# rk4-wout-300e — variant srnn-std-e-only-per-neuron (k=4)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                              +1      +0.33251           —           —     scalar
tau_d (s)                                 +0.1    +0.0051415           0    0.002853        300
tau_b_rec_E (s)                             +1       +1.7897           0      0.5016        150
tau_b_rel_E (s)                          +0.25      +0.46015           0      0.1359        150
W_eff (E src, signed)                 +0.23267      +0.20924     0.07641      0.0974      15048
W_eff (I src, signed)                 -0.31062      -0.25784     0.07724     0.08169      14977
|W_eff| (E src)                       +0.23267      +0.20924     0.07641      0.0974      15048
|W_eff| (I src)                       +0.31062      +0.25784     0.07724     0.08169      14977
W_in_eff (input neurons)           +0.00019944   -6.4633e-07     0.09994     0.03827       6675
a_0 (threshold)                          +0.35      -0.85975           0      0.1836        300
readout_weight                     +0.00074661   -0.00094686     0.06657      0.3627       6675
readout_bias                                +0     -0.080185           0      0.2094         89
ic.ic (per-variant init)             +0.090198     +0.090198      0.3825      0.3825        900
```

\newpage

# srnn-std-e-only-skip-per-neuron

![Tau evolution](srnn-std-e-only-skip-per-neuron/tau_evolution.png){ width=95% }

![W\_EI evolution](srnn-std-e-only-skip-per-neuron/W_EI_evolution.png){ width=95% }

\newpage

# srnn-std-e-only-skip-per-neuron — parameter table

```text
# rk4-wout-300e — variant srnn-std-e-only-skip-per-neuron (k=5)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                              +1      +0.37503           —           —     scalar
tau_d (s)                                 +0.1    +0.0092527           0    0.004334        300
tau_b_rec_E (s)                             +1      +0.13503           0     0.02779        150
tau_b_rel_E (s)                          +0.25      +0.39409           0      0.1122        150
W_eff (E src, signed)                 +0.23267      +0.16661     0.07641     0.06346      15048
W_eff (I src, signed)                 -0.31062      -0.21652     0.07724     0.06415      14977
|W_eff| (E src)                       +0.23267      +0.16661     0.07641     0.06346      15048
|W_eff| (I src)                       +0.31062      +0.21652     0.07724     0.06415      14977
W_in_eff (input neurons)           +0.00036119   +0.00025925     0.09956     0.01927       6675
a_0 (threshold)                          +0.35     -0.031354           0      0.1263        300
readout_weight                     +0.00055372    -0.0050808     0.06572      0.1581       6675
readout_bias                                +0     +0.030615           0      0.1208         89
ic.ic (per-variant init)             +0.082727     +0.082727      0.3836      0.3836        900
```
