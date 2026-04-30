# Run: alpha-ramp-150e

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
# alpha-ramp-150e — variant srnn-e-only-per-neuron (k=0)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                              +1      +0.66543           —           —     scalar
tau_d (s)                                 +0.1     +0.035117           0    0.008453        300
tau_a_E (s)                            +4.8777       +1.7311       3.996       1.418        450
tau_b_rec_E (s)                             +1      +0.35754           0     0.04294        150
tau_b_rel_E (s)                          +0.25      +0.32035           0     0.04641        150
W_eff (E src, signed)                 +0.23267      +0.21389     0.07641     0.07508      15048
W_eff (I src, signed)                 -0.31062      -0.28226     0.07724     0.07799      14977
|W_eff| (E src)                       +0.23267      +0.21389     0.07641     0.07508      15048
|W_eff| (I src)                       +0.31062      +0.28226     0.07724     0.07799      14977
W_in_eff (input neurons)            -0.0003386     +0.001217      0.1012      0.1355       6675
c_E (SFA coupling)                       +0.05      +0.06368           0     0.01052        450
c_0_E (SFA offset)                          +0      -0.28228           0      0.1187        450
a_0 (threshold)                          +0.35      +0.17353           0     0.09971        300
readout_weight                     -2.8854e-05    -0.0041653     0.06604      0.1689       6675
readout_bias                                +0    -0.0014186           0      0.1764         89
ic.ic (per-variant init)             +0.098875     -0.010233      0.3842      0.5166        900
```

\newpage

# srnn-e-only-skip-per-neuron

![Tau evolution](srnn-e-only-skip-per-neuron/tau_evolution.png){ width=95% }

![W\_EI evolution](srnn-e-only-skip-per-neuron/W_EI_evolution.png){ width=95% }

\newpage

# srnn-e-only-skip-per-neuron — parameter table

```text
# alpha-ramp-150e — variant srnn-e-only-skip-per-neuron (k=1)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                              +1      +0.57849           —           —     scalar
tau_d (s)                                 +0.1     +0.025282           0    0.006126        300
tau_a_E (s)                            +4.8777        +1.585       3.996        1.31        450
tau_b_rec_E (s)                             +1      +0.35732           0     0.04221        150
tau_b_rel_E (s)                          +0.25      +0.18424           0     0.02491        150
W_eff (E src, signed)                 +0.23267      +0.20765     0.07641     0.07126      15048
W_eff (I src, signed)                 -0.31062      -0.27481     0.07724     0.07415      14977
|W_eff| (E src)                       +0.23267      +0.20765     0.07641     0.07126      15048
|W_eff| (I src)                       +0.31062      +0.27481     0.07724     0.07415      14977
W_in_eff (input neurons)           -0.00084237   -0.00087285      0.1008     0.04768       6675
c_E (SFA coupling)                       +0.05     +0.076048           0     0.01365        450
c_0_E (SFA offset)                          +0      -0.20905           0      0.1452        450
a_0 (threshold)                          +0.35      +0.14821           0      0.1304        300
readout_weight                     +0.00078219    -0.0072867     0.06654       0.109       6675
readout_bias                                +0      +0.11416           0      0.1458         89
ic.ic (per-variant init)             +0.098961     -0.033766       0.387      0.4815        900
```

\newpage

# srnn-sfa-e-only-per-neuron

![Tau evolution](srnn-sfa-e-only-per-neuron/tau_evolution.png){ width=95% }

![W\_EI evolution](srnn-sfa-e-only-per-neuron/W_EI_evolution.png){ width=95% }

\newpage

# srnn-sfa-e-only-per-neuron — parameter table

```text
# alpha-ramp-150e — variant srnn-sfa-e-only-per-neuron (k=2)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                              +1       +0.6809           —           —     scalar
tau_d (s)                                 +0.1     +0.037426           0    0.007794        300
tau_a_E (s)                            +4.8777       +2.4981       3.996       2.047        450
tau_b_rec_E (s)                             +1       +0.6809           0           0        150
tau_b_rel_E (s)                          +0.25      +0.17023           0           0        150
W_eff (E src, signed)                 +0.23267      +0.21012     0.07641     0.07208      15048
W_eff (I src, signed)                 -0.31062      -0.27813     0.07724     0.07452      14977
|W_eff| (E src)                       +0.23267      +0.21012     0.07641     0.07208      15048
|W_eff| (I src)                       +0.31062      +0.27813     0.07724     0.07452      14977
W_in_eff (input neurons)           -0.00015382    -0.0021955     0.09934      0.1421       6675
c_E (SFA coupling)                       +0.05     +0.051488           0    0.005558        450
c_0_E (SFA offset)                          +0      -0.17221           0       0.103        450
a_0 (threshold)                          +0.35      +0.35374           0     0.06641        300
readout_weight                     -0.00098972    +0.0074873     0.06678      0.1426       6675
readout_bias                                +0     -0.047463           0     0.08236         89
ic.ic (per-variant init)              +0.13078       +0.0259      0.4172      0.4526        900
```

\newpage

# srnn-sfa-e-only-skip-per-neuron

![Tau evolution](srnn-sfa-e-only-skip-per-neuron/tau_evolution.png){ width=95% }

![W\_EI evolution](srnn-sfa-e-only-skip-per-neuron/W_EI_evolution.png){ width=95% }

\newpage

# srnn-sfa-e-only-skip-per-neuron — parameter table

```text
# alpha-ramp-150e — variant srnn-sfa-e-only-skip-per-neuron (k=3)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                              +1      +0.80508           —           —     scalar
tau_d (s)                                 +0.1     +0.058983           0    0.004545        300
tau_a_E (s)                            +4.8777        +3.668       3.996       3.011        450
tau_b_rec_E (s)                             +1      +0.80508           0           0        150
tau_b_rel_E (s)                          +0.25      +0.20127           0           0        150
W_eff (E src, signed)                 +0.23267      +0.22845     0.07641     0.07569      15048
W_eff (I src, signed)                 -0.31062      -0.30485     0.07724     0.07676      14977
|W_eff| (E src)                       +0.23267      +0.22845     0.07641     0.07569      15048
|W_eff| (I src)                       +0.31062      +0.30485     0.07724     0.07676      14977
W_in_eff (input neurons)            -0.0011621    -0.0031755     0.09962     0.08641       6675
c_E (SFA coupling)                       +0.05     +0.052814           0    0.004382        450
c_0_E (SFA offset)                          +0      +0.02968           0     0.06652        450
a_0 (threshold)                          +0.35      +0.36487           0     0.06577        300
readout_weight                     +0.00016801    -0.0029894      0.0673      0.0635       6675
readout_bias                                +0     +0.032153           0     0.04143         89
ic.ic (per-variant init)              +0.09231     +0.090339      0.5294      0.4444        900
```

\newpage

# srnn-std-e-only-per-neuron

![Tau evolution](srnn-std-e-only-per-neuron/tau_evolution.png){ width=95% }

![W\_EI evolution](srnn-std-e-only-per-neuron/W_EI_evolution.png){ width=95% }

\newpage

# srnn-std-e-only-per-neuron — parameter table

```text
# alpha-ramp-150e — variant srnn-std-e-only-per-neuron (k=4)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                              +1      +0.72008           —           —     scalar
tau_d (s)                                 +0.1     +0.041684           0    0.008946        300
tau_a_E (s)                           +0.69315      +0.49912           0           0        450
tau_b_rec_E (s)                             +1      +0.38513           0      0.0375        150
tau_b_rel_E (s)                          +0.25      +0.34002           0     0.04036        150
W_eff (E src, signed)                 +0.23267      +0.23701     0.07641     0.08288      15048
W_eff (I src, signed)                 -0.31062      -0.31315     0.07724     0.08562      14977
|W_eff| (E src)                       +0.23267      +0.23701     0.07641     0.08288      15048
|W_eff| (I src)                       +0.31062      +0.31315     0.07724     0.08562      14977
W_in_eff (input neurons)           +0.00019944    -0.0017695     0.09994      0.1438       6675
c_E (SFA coupling)                       +0.05         +0.05           0           0        450
c_0_E (SFA offset)                          +0            +0           0           0        450
a_0 (threshold)                          +0.35      +0.23747           0     0.08906        300
readout_weight                     +0.00074661     +0.008098     0.06657      0.1618       6675
readout_bias                                +0     -0.029767           0      0.1688         89
ic.ic (per-variant init)             +0.087739     +0.035925      0.3868      0.4518        900
```

\newpage

# srnn-std-e-only-skip-per-neuron

![Tau evolution](srnn-std-e-only-skip-per-neuron/tau_evolution.png){ width=95% }

![W\_EI evolution](srnn-std-e-only-skip-per-neuron/W_EI_evolution.png){ width=95% }

\newpage

# srnn-std-e-only-skip-per-neuron — parameter table

```text
# alpha-ramp-150e — variant srnn-std-e-only-skip-per-neuron (k=5)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                              +1      +0.58789           —           —     scalar
tau_d (s)                                 +0.1      +0.02679           0    0.005781        300
tau_a_E (s)                           +0.69315      +0.40749           0           0        450
tau_b_rec_E (s)                             +1      +0.39887           0     0.03951        150
tau_b_rel_E (s)                          +0.25      +0.17126           0     0.02144        150
W_eff (E src, signed)                 +0.23267      +0.21621     0.07641     0.07365      15048
W_eff (I src, signed)                 -0.31062      -0.28576     0.07724     0.07655      14977
|W_eff| (E src)                       +0.23267      +0.21621     0.07641     0.07365      15048
|W_eff| (I src)                       +0.31062      +0.28576     0.07724     0.07655      14977
W_in_eff (input neurons)           +0.00036119   +0.00024403     0.09956      0.0587       6675
c_E (SFA coupling)                       +0.05         +0.05           0           0        450
c_0_E (SFA offset)                          +0            +0           0           0        450
a_0 (threshold)                          +0.35      +0.19603           0      0.1144        300
readout_weight                     +0.00055372     -0.012366     0.06572     0.09322       6675
readout_bias                                +0     +0.090316           0      0.1209         89
ic.ic (per-variant init)             +0.089141     +0.013634      0.3857      0.4678        900
```
