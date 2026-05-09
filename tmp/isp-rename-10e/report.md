# Run: isp-rename-10e

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
# isp-rename-10e — variant srnn-e-only-per-neuron (k=0)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                              +1      +0.93981           —           —     scalar
tau_d (s)                                 +0.1     +0.084968           0    0.002474        300
tau_a_E[j=0] (s)                         +0.25      +0.24145           0    0.003984        150
tau_a_E[j=1] (s)                       +4.3832       +4.2053           0    0.006571        150
tau_a_E[j=2] (s)                           +10       +9.5948           0    0.002447        150
tau_b_rec_E (s)                             +1       +0.9643           0     0.01153        150
tau_b_rel_E (s)                          +0.25      +0.25128           0    0.004998        150
W_eff (E src, signed)                 +0.23267      +0.22938     0.07641     0.07542      15048
W_eff (I src, signed)                 -0.31062      -0.30543     0.07724     0.07612      14977
|W_eff| (E src)                       +0.23267      +0.22938     0.07641     0.07542      15048
|W_eff| (I src)                       +0.31062      +0.30543     0.07724     0.07612      14977
W_in_eff (input neurons)            -0.0003386   -0.00033055      0.1012      0.1105       6675
c_E[j=0] (SFA coupling)                  +0.05      +0.04879           0   0.0008542        150
c_E[j=1] (SFA coupling)                  +0.05     +0.048835           0    0.000923        150
c_E[j=2] (SFA coupling)                  +0.05     +0.048769           0   0.0009456        150
c_0_E[j=0] (SFA offset)                     +0     -0.042226           0     0.02024        150
c_0_E[j=1] (SFA offset)                     +0     -0.042622           0     0.01952        150
c_0_E[j=2] (SFA offset)                     +0      -0.04238           0     0.01908        150
a_0 (threshold)                          +0.35      +0.31537           0       0.024        300
readout_weight                     -2.8854e-05   +0.00051796     0.06604      0.0711       6675
readout_bias                                +0    -0.0015425           0     0.02677         89
ic.ic (per-variant init)              +0.10386      +0.10386      0.3828      0.3828        900
```

\newpage

# srnn-e-only-skip-per-neuron

![Tau evolution](srnn-e-only-skip-per-neuron/tau_evolution.png){ width=95% }

![W\_EI evolution](srnn-e-only-skip-per-neuron/W_EI_evolution.png){ width=95% }

\newpage

# srnn-e-only-skip-per-neuron — parameter table

```text
# isp-rename-10e — variant srnn-e-only-skip-per-neuron (k=1)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                              +1      +0.99226           —           —     scalar
tau_d (s)                                 +0.1      +0.10119           0    0.001566        300
tau_a_E[j=0] (s)                         +0.25      +0.24083           0    0.001541        150
tau_a_E[j=1] (s)                       +4.3832        +4.265           0    0.002706        150
tau_a_E[j=2] (s)                           +10       +9.7362           0    0.001039        150
tau_b_rec_E (s)                             +1      +0.99639           0    0.003979        150
tau_b_rel_E (s)                          +0.25      +0.24191           0    0.001242        150
W_eff (E src, signed)                 +0.23267      +0.22838     0.07641     0.07504      15048
W_eff (I src, signed)                 -0.31062      -0.30571     0.07724     0.07603      14977
|W_eff| (E src)                       +0.23267      +0.22838     0.07641     0.07504      15048
|W_eff| (I src)                       +0.31062      +0.30571     0.07724     0.07603      14977
W_in_eff (input neurons)           -0.00084237    -0.0009372      0.1008     0.09914       6675
c_E[j=0] (SFA coupling)                  +0.05     +0.051808           0   0.0004028        150
c_E[j=1] (SFA coupling)                  +0.05     +0.051854           0   0.0004949        150
c_E[j=2] (SFA coupling)                  +0.05     +0.051913           0   0.0005438        150
c_0_E[j=0] (SFA offset)                     +0     +0.034778           0    0.007287        150
c_0_E[j=1] (SFA offset)                     +0     +0.034521           0    0.006991        150
c_0_E[j=2] (SFA offset)                     +0     +0.033955           0    0.006951        150
a_0 (threshold)                          +0.35      +0.37489           0     0.01633        300
readout_weight                     +0.00078219   +0.00083932     0.06654     0.06087       6675
readout_bias                                +0   +0.00013959           0    0.001394         89
ic.ic (per-variant init)              +0.10327      +0.10327      0.3829      0.3829        900
```

\newpage

# srnn-sfa-e-only-per-neuron

![Tau evolution](srnn-sfa-e-only-per-neuron/tau_evolution.png){ width=95% }

![W\_EI evolution](srnn-sfa-e-only-per-neuron/W_EI_evolution.png){ width=95% }

\newpage

# srnn-sfa-e-only-per-neuron — parameter table

```text
# isp-rename-10e — variant srnn-sfa-e-only-per-neuron (k=2)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                              +1      +0.99546           —           —     scalar
tau_d (s)                                 +0.1     +0.099016           0   0.0009727        300
tau_a_E[j=0] (s)                         +0.25      +0.24492           0    0.001412        150
tau_a_E[j=1] (s)                       +4.3832       +4.3127           0    0.004419        150
tau_a_E[j=2] (s)                           +10       +9.8466           0     0.00305        150
tau_b_rec_E (s)                             +1      +0.99546           0           0        150
tau_b_rel_E (s)                          +0.25      +0.24886           0           0        150
W_eff (E src, signed)                 +0.23267      +0.22988     0.07641     0.07553      15048
W_eff (I src, signed)                 -0.31062        -0.307     0.07724     0.07634      14977
|W_eff| (E src)                       +0.23267      +0.22988     0.07641     0.07553      15048
|W_eff| (I src)                       +0.31062        +0.307     0.07724     0.07634      14977
W_in_eff (input neurons)           -0.00015382   -1.6796e-05     0.09934      0.0982       6675
c_E[j=0] (SFA coupling)                  +0.05     +0.050858           0   0.0003478        150
c_E[j=1] (SFA coupling)                  +0.05     +0.050909           0   0.0004038        150
c_E[j=2] (SFA coupling)                  +0.05     +0.050954           0   0.0004537        150
c_0_E[j=0] (SFA offset)                     +0     +0.018318           0    0.008675        150
c_0_E[j=1] (SFA offset)                     +0     +0.018447           0    0.008812        150
c_0_E[j=2] (SFA offset)                     +0     +0.018429           0    0.008721        150
a_0 (threshold)                          +0.35       +0.3618           0     0.01082        300
readout_weight                     -0.00098972   +0.00020923     0.06678     0.06622       6675
readout_bias                                +0   +0.00052439           0    0.009408         89
ic.ic (per-variant init)              +0.13078      +0.13078      0.4172      0.4172        900
```

\newpage

# srnn-sfa-e-only-skip-per-neuron

![Tau evolution](srnn-sfa-e-only-skip-per-neuron/tau_evolution.png){ width=95% }

![W\_EI evolution](srnn-sfa-e-only-skip-per-neuron/W_EI_evolution.png){ width=95% }

\newpage

# srnn-sfa-e-only-skip-per-neuron — parameter table

```text
# isp-rename-10e — variant srnn-sfa-e-only-skip-per-neuron (k=3)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                              +1       +1.0145           —           —     scalar
tau_d (s)                                 +0.1      +0.10399           0    0.001161        300
tau_a_E[j=0] (s)                         +0.25      +0.24775           0    0.001364        150
tau_a_E[j=1] (s)                       +4.3832       +4.3767           0    0.004588        150
tau_a_E[j=2] (s)                           +10       +9.9957           0    0.003787        150
tau_b_rec_E (s)                             +1       +1.0145           0           0        150
tau_b_rel_E (s)                          +0.25      +0.25362           0           0        150
W_eff (E src, signed)                 +0.23267      +0.22954     0.07641     0.07541      15048
W_eff (I src, signed)                 -0.31062      -0.30682     0.07724      0.0763      14977
|W_eff| (E src)                       +0.23267      +0.22954     0.07641     0.07541      15048
|W_eff| (I src)                       +0.31062      +0.30682     0.07724      0.0763      14977
W_in_eff (input neurons)            -0.0011621   -0.00088393     0.09962     0.09614       6675
c_E[j=0] (SFA coupling)                  +0.05     +0.051154           0   0.0003285        150
c_E[j=1] (SFA coupling)                  +0.05      +0.05129           0   0.0004489        150
c_E[j=2] (SFA coupling)                  +0.05     +0.051399           0   0.0005162        150
c_0_E[j=0] (SFA offset)                     +0     +0.028053           0    0.007471        150
c_0_E[j=1] (SFA offset)                     +0     +0.028959           0    0.007489        150
c_0_E[j=2] (SFA offset)                     +0     +0.028963           0    0.007423        150
a_0 (threshold)                          +0.35      +0.36842           0     0.01321        300
readout_weight                     +0.00016801   +0.00019328      0.0673     0.06327       6675
readout_bias                                +0   +3.8623e-05           0    0.006911         89
ic.ic (per-variant init)              +0.09231      +0.09231      0.5294      0.5294        900
```

\newpage

# srnn-std-e-only-per-neuron

![Tau evolution](srnn-std-e-only-per-neuron/tau_evolution.png){ width=95% }

![W\_EI evolution](srnn-std-e-only-per-neuron/W_EI_evolution.png){ width=95% }

\newpage

# srnn-std-e-only-per-neuron — parameter table

```text
# isp-rename-10e — variant srnn-std-e-only-per-neuron (k=4)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                              +1      +0.94245           —           —     scalar
tau_d (s)                                 +0.1     +0.085709           0    0.002801        300
tau_b_rec_E (s)                             +1      +0.92759           0     0.01098        150
tau_b_rel_E (s)                          +0.25      +0.24852           0    0.004655        150
W_eff (E src, signed)                 +0.23267      +0.24012     0.07641     0.07896      15048
W_eff (I src, signed)                 -0.31062      -0.31955     0.07724     0.07956      14977
|W_eff| (E src)                       +0.23267      +0.24012     0.07641     0.07896      15048
|W_eff| (I src)                       +0.31062      +0.31955     0.07724     0.07956      14977
W_in_eff (input neurons)           +0.00019944   +0.00036307     0.09994      0.1077       6675
a_0 (threshold)                          +0.35      +0.32272           0     0.02254        300
readout_weight                     +0.00074661    +0.0024107     0.06657     0.07225       6675
readout_bias                                +0      -0.00404           0     0.02586         89
ic.ic (per-variant init)             +0.090819     +0.090819      0.3835      0.3835        900
```

\newpage

# srnn-std-e-only-skip-per-neuron

![Tau evolution](srnn-std-e-only-skip-per-neuron/tau_evolution.png){ width=95% }

![W\_EI evolution](srnn-std-e-only-skip-per-neuron/W_EI_evolution.png){ width=95% }

\newpage

# srnn-std-e-only-skip-per-neuron — parameter table

```text
# isp-rename-10e — variant srnn-std-e-only-skip-per-neuron (k=5)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                              +1      +0.99243           —           —     scalar
tau_d (s)                                 +0.1      +0.10137           0    0.001597        300
tau_b_rec_E (s)                             +1       +1.0013           0     0.00428        150
tau_b_rel_E (s)                          +0.25      +0.24153           0    0.001266        150
W_eff (E src, signed)                 +0.23267      +0.22772     0.07641     0.07482      15048
W_eff (I src, signed)                 -0.31062      -0.30492     0.07724     0.07584      14977
|W_eff| (E src)                       +0.23267      +0.22772     0.07641     0.07482      15048
|W_eff| (I src)                       +0.31062      +0.30492     0.07724     0.07584      14977
W_in_eff (input neurons)           +0.00036119   +0.00034556     0.09956     0.09879       6675
a_0 (threshold)                          +0.35      +0.37506           0     0.01671        300
readout_weight                     +0.00055372   -0.00031886     0.06572     0.05966       6675
readout_bias                                +0   -0.00033867           0    0.002078         89
ic.ic (per-variant init)             +0.083543     +0.083543       0.384       0.384        900
```
