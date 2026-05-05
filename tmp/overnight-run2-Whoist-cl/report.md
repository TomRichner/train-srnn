# Run: overnight-run2-Whoist-cl

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
# overnight-run2-Whoist-cl — variant srnn-e-only-per-neuron (k=0)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                              +1      +0.39706           —           —     scalar
tau_d (s)                                 +0.1    +0.0039354           0    0.003722        300
tau_a_E (s)                            +4.8777       +0.6065       3.996      0.5346        450
tau_b_rec_E (s)                             +1       +12.939           0       5.352        150
tau_b_rel_E (s)                          +0.25      +0.37073           0     0.05959        150
W_eff (E src, signed)                 +0.23267      +0.42072     0.07641       0.363      15048
W_eff (I src, signed)                 -0.31062      -0.39413     0.07724      0.1694      14977
|W_eff| (E src)                       +0.23267      +0.42072     0.07641       0.363      15048
|W_eff| (I src)                       +0.31062      +0.39413     0.07724      0.1694      14977
W_in_eff (input neurons)            -0.0003386   +0.00014097      0.1012     0.07117       6675
c_E (SFA coupling)                       +0.05      +0.34718           0      0.4698        450
c_0_E (SFA offset)                          +0       -1.8195           0      0.6839        450
a_0 (threshold)                          +0.35       -1.1487           0      0.2353        300
readout_weight                     -2.8854e-05      -0.01866     0.06604       1.029       6675
readout_bias                                +0     +0.030146           0      0.1201         89
ic.ic (per-variant init)              +0.10386      +0.10386      0.3828      0.3828        900
```

\newpage

# srnn-e-only-skip-per-neuron

![Tau evolution](srnn-e-only-skip-per-neuron/tau_evolution.png){ width=95% }

![W\_EI evolution](srnn-e-only-skip-per-neuron/W_EI_evolution.png){ width=95% }

\newpage

# srnn-e-only-skip-per-neuron — parameter table

```text
# overnight-run2-Whoist-cl — variant srnn-e-only-skip-per-neuron (k=1)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                              +1      +0.34822           —           —     scalar
tau_d (s)                                 +0.1    +0.0070211           0    0.005324        300
tau_a_E (s)                            +4.8777      +0.39992       3.996      0.3518        450
tau_b_rec_E (s)                             +1      +0.89423           0      0.3454        150
tau_b_rel_E (s)                          +0.25      +0.41824           0      0.1665        150
W_eff (E src, signed)                 +0.23267      +0.18583     0.07641     0.09793      15048
W_eff (I src, signed)                 -0.31062       -0.2346     0.07724      0.1038      14977
|W_eff| (E src)                       +0.23267      +0.18583     0.07641     0.09793      15048
|W_eff| (I src)                       +0.31062       +0.2346     0.07724      0.1038      14977
W_in_eff (input neurons)           -0.00084237   -0.00032963      0.1008     0.03327       6675
c_E (SFA coupling)                       +0.05      +0.16025           0      0.1934        450
c_0_E (SFA offset)                          +0      -0.51227           0      0.2784        450
a_0 (threshold)                          +0.35      -0.45279           0      0.1396        300
readout_weight                     +0.00078219    -0.0049355     0.06654      0.2849       6675
readout_bias                                +0      +0.14971           0      0.2096         89
ic.ic (per-variant init)              +0.10327      +0.10327      0.3829      0.3829        900
```

\newpage

# srnn-sfa-e-only-per-neuron

![Tau evolution](srnn-sfa-e-only-per-neuron/tau_evolution.png){ width=95% }

![W\_EI evolution](srnn-sfa-e-only-per-neuron/W_EI_evolution.png){ width=95% }

\newpage

# srnn-sfa-e-only-per-neuron — parameter table

```text
# overnight-run2-Whoist-cl — variant srnn-sfa-e-only-per-neuron (k=2)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                              +1      +0.22226           —           —     scalar
tau_d (s)                                 +0.1    +0.0029146           0    0.002612        300
tau_a_E (s)                            +4.8777      +0.39283       3.996      0.3336        450
tau_b_rec_E (s)                             +1      +0.22226           0           0        150
tau_b_rel_E (s)                          +0.25     +0.055565           0           0        150
W_eff (E src, signed)                 +0.23267      +0.13375     0.07641     0.06408      15048
W_eff (I src, signed)                 -0.31062      -0.17607     0.07724      0.0687      14977
|W_eff| (E src)                       +0.23267      +0.13375     0.07641     0.06408      15048
|W_eff| (I src)                       +0.31062      +0.17607     0.07724      0.0687      14977
W_in_eff (input neurons)           -0.00015382   -0.00086168     0.09934     0.06373       6675
c_E (SFA coupling)                       +0.05      +0.35569           0      0.4511        450
c_0_E (SFA offset)                          +0      -0.25393           0        0.26        450
a_0 (threshold)                          +0.35    +0.0098035           0       0.131        300
readout_weight                     -0.00098972    +0.0069969     0.06678      0.4859       6675
readout_bias                                +0      -0.13424           0      0.2878         89
ic.ic (per-variant init)              +0.13078      +0.13078      0.4172      0.4172        900
```

\newpage

# srnn-sfa-e-only-skip-per-neuron

![Tau evolution](srnn-sfa-e-only-skip-per-neuron/tau_evolution.png){ width=95% }

![W\_EI evolution](srnn-sfa-e-only-skip-per-neuron/W_EI_evolution.png){ width=95% }

\newpage

# srnn-sfa-e-only-skip-per-neuron — parameter table

```text
# overnight-run2-Whoist-cl — variant srnn-sfa-e-only-skip-per-neuron (k=3)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                              +1      +0.23219           —           —     scalar
tau_d (s)                                 +0.1    +0.0045342           0    0.003217        300
tau_a_E (s)                            +4.8777      +0.19658       3.996      0.1663        450
tau_b_rec_E (s)                             +1      +0.23219           0           0        150
tau_b_rel_E (s)                          +0.25     +0.058049           0           0        150
W_eff (E src, signed)                 +0.23267      +0.16382     0.07641     0.06763      15048
W_eff (I src, signed)                 -0.31062       -0.2183     0.07724     0.07632      14977
|W_eff| (E src)                       +0.23267      +0.16382     0.07641     0.06763      15048
|W_eff| (I src)                       +0.31062       +0.2183     0.07724     0.07632      14977
W_in_eff (input neurons)            -0.0011621   +9.8282e-05     0.09962     0.03324       6675
c_E (SFA coupling)                       +0.05     +0.068387           0     0.04854        450
c_0_E (SFA offset)                          +0      -0.21228           0      0.2366        450
a_0 (threshold)                          +0.35      +0.18645           0      0.1337        300
readout_weight                     +0.00016801     -0.010457      0.0673      0.2114       6675
readout_bias                                +0      +0.10138           0      0.2415         89
ic.ic (per-variant init)              +0.09231      +0.09231      0.5294      0.5294        900
```

\newpage

# srnn-std-e-only-per-neuron

![Tau evolution](srnn-std-e-only-per-neuron/tau_evolution.png){ width=95% }

![W\_EI evolution](srnn-std-e-only-per-neuron/W_EI_evolution.png){ width=95% }

\newpage

# srnn-std-e-only-per-neuron — parameter table

```text
# overnight-run2-Whoist-cl — variant srnn-std-e-only-per-neuron (k=4)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                              +1      +0.30412           —           —     scalar
tau_d (s)                                 +0.1    +0.0024888           0    0.002472        300
tau_a_E (s)                           +0.69315       +0.2108           0           0        450
tau_b_rec_E (s)                             +1       +7.0621           0       3.241        150
tau_b_rel_E (s)                          +0.25      +0.39294           0      0.0579        150
W_eff (E src, signed)                 +0.23267      +0.30903     0.07641      0.2289      15048
W_eff (I src, signed)                 -0.31062      -0.30695     0.07724       0.126      14977
|W_eff| (E src)                       +0.23267      +0.30903     0.07641      0.2289      15048
|W_eff| (I src)                       +0.31062      +0.30695     0.07724       0.126      14977
W_in_eff (input neurons)           +0.00019944   -0.00042256     0.09994     0.05212       6675
c_E (SFA coupling)                       +0.05         +0.05           0           0        450
c_0_E (SFA offset)                          +0            +0           0           0        450
a_0 (threshold)                          +0.35       -1.6603           0      0.2753        300
readout_weight                     +0.00074661    -0.0033378     0.06657      0.9113       6675
readout_bias                                +0     +0.013818           0      0.1603         89
ic.ic (per-variant init)             +0.090819     +0.090819      0.3835      0.3835        900
```

\newpage

# srnn-std-e-only-skip-per-neuron

![Tau evolution](srnn-std-e-only-skip-per-neuron/tau_evolution.png){ width=95% }

![W\_EI evolution](srnn-std-e-only-skip-per-neuron/W_EI_evolution.png){ width=95% }

\newpage

# srnn-std-e-only-skip-per-neuron — parameter table

```text
# overnight-run2-Whoist-cl — variant srnn-std-e-only-skip-per-neuron (k=5)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                              +1      +0.30196           —           —     scalar
tau_d (s)                                 +0.1    +0.0057643           0    0.004652        300
tau_a_E (s)                           +0.69315       +0.2093           0           0        450
tau_b_rec_E (s)                             +1      +0.33406           0      0.1223        150
tau_b_rel_E (s)                          +0.25      +0.47352           0      0.2022        150
W_eff (E src, signed)                 +0.23267      +0.16328     0.07641     0.08844      15048
W_eff (I src, signed)                 -0.31062      -0.20913     0.07724      0.0967      14977
|W_eff| (E src)                       +0.23267      +0.16328     0.07641     0.08844      15048
|W_eff| (I src)                       +0.31062      +0.20913     0.07724      0.0967      14977
W_in_eff (input neurons)           +0.00036119   +0.00022384     0.09956     0.02544       6675
c_E (SFA coupling)                       +0.05         +0.05           0           0        450
c_0_E (SFA offset)                          +0            +0           0           0        450
a_0 (threshold)                          +0.35      -0.33197           0      0.1515        300
readout_weight                     +0.00055372   -0.00094974     0.06572       0.279       6675
readout_bias                                +0     -0.034855           0      0.1837         89
ic.ic (per-variant init)             +0.083543     +0.083543       0.384       0.384        900
```
