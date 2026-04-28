# Run: overnight9h

## Log-log loss curves

![skip variants — log-log loss/metric](log_log_curves_skip.png){ width=95% }

\newpage

# srnn-e-only-skip

![Tau evolution](srnn-e-only-skip/tau_evolution.png){ width=95% }

![W\_EI evolution](srnn-e-only-skip/W_EI_evolution.png){ width=95% }

\newpage

# srnn-e-only-skip — parameter table

```text
# overnight9h — variant srnn-e-only-skip (k=0)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                              +1      +0.79508           —           —     scalar
tau_d (s)                                 +0.1     +0.065561           0           0        300
tau_a_E (s)                            +4.8777       +3.6887       3.996       3.022        450
tau_a_I (s)                           +0.69315      +0.55111           0           0        450
tau_b_rec_E (s)                             +1      +0.78867           0           0        150
tau_b_rel_E (s)                          +0.25      +0.19556           0           0        150
tau_b_rec_I (s)                             +1      +0.79508           0           0        150
tau_b_rel_I (s)                          +0.25      +0.19877           0           0        150
W_eff (E src, signed)                 +0.23267      +0.23063     0.07641       0.076      15048
W_eff (I src, signed)                 -0.31062      -0.30818     0.07724     0.07696      14977
|W_eff| (E src)                       +0.23267      +0.23063     0.07641       0.076      15048
|W_eff| (I src)                       +0.31062      +0.30818     0.07724     0.07696      14977
W_in_eff (input neurons)            -0.0003386    +0.0037668      0.1012      0.1113       6675
c_E (SFA coupling)                       +0.05     +0.051138           0           0        450
c_0_E (SFA offset)                          +0     +0.021846           0           0        450
c_I (SFA coupling)                       +0.05         +0.05           0           0        450
c_0_I (SFA offset)                          +0            +0           0           0        450
a_0 (threshold)                          +0.35      +0.36556           0           0        300
readout_weight                     -0.00041422    -0.0076553     0.06568      0.0598       6675
readout_bias                                +0     +0.012703           0     0.01728         89
ic.ic (per-variant init)              +0.15932      +0.17372      0.4111      0.4079       1500
```

\newpage

# srnn-e-only-skip-per-neuron

![Tau evolution](srnn-e-only-skip-per-neuron/tau_evolution.png){ width=95% }

![W\_EI evolution](srnn-e-only-skip-per-neuron/W_EI_evolution.png){ width=95% }

\newpage

# srnn-e-only-skip-per-neuron — parameter table

```text
# overnight9h — variant srnn-e-only-skip-per-neuron (k=1)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                              +1      +0.79081           —           —     scalar
tau_d (s)                                 +0.1     +0.060951           0     0.00457        300
tau_a_E (s)                            +4.8777       +3.6689       3.996        3.01        450
tau_a_I (s)                           +0.69315      +0.54815           0           0        450
tau_b_rec_E (s)                             +1      +0.77469           0     0.02205        150
tau_b_rel_E (s)                          +0.25      +0.19497           0    0.006761        150
tau_b_rec_I (s)                             +1      +0.79081           0           0        150
tau_b_rel_I (s)                          +0.25       +0.1977           0           0        150
W_eff (E src, signed)                 +0.23267      +0.23188     0.07641     0.07641      15048
W_eff (I src, signed)                 -0.31062      -0.30904     0.07724     0.07736      14977
|W_eff| (E src)                       +0.23267      +0.23188     0.07641     0.07641      15048
|W_eff| (I src)                       +0.31062      +0.30904     0.07724     0.07736      14977
W_in_eff (input neurons)           -0.00084237   +0.00059068      0.1008     0.08985       6675
c_E (SFA coupling)                       +0.05     +0.051772           0    0.001311        450
c_0_E (SFA offset)                          +0     +0.026169           0      0.0242        450
c_I (SFA coupling)                       +0.05         +0.05           0           0        450
c_0_I (SFA offset)                          +0            +0           0           0        450
a_0 (threshold)                          +0.35      +0.35272           0     0.04392        300
readout_weight                      -0.0016154    -0.0090911     0.06624     0.06309       6675
readout_bias                                +0     +0.027875           0     0.03333         89
ic.ic (per-variant init)              +0.15938      +0.16418      0.4127      0.4249       1500
```

\newpage

# srnn-e-only-skip-echo

![Tau evolution](srnn-e-only-skip-echo/tau_evolution.png){ width=95% }

![W\_EI evolution](srnn-e-only-skip-echo/W_EI_evolution.png){ width=95% }

\newpage

# srnn-e-only-skip-echo — parameter table

```text
# overnight9h — variant srnn-e-only-skip-echo (k=2)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                              +1       +0.7898           —           —     scalar
tau_d (s)                                 +0.1     +0.066188           0           0        300
tau_a_E (s)                            +4.8777        +3.637       3.996       2.979        450
tau_a_I (s)                           +0.69315      +0.54745           0           0        450
tau_b_rec_E (s)                             +1      +0.79227           0           0        150
tau_b_rel_E (s)                          +0.25      +0.19272           0           0        150
tau_b_rec_I (s)                             +1       +0.7898           0           0        150
tau_b_rel_I (s)                          +0.25      +0.19745           0           0        150
W_eff (E src, signed)                 +0.23267      +0.22739     0.07641     0.07468      15048
W_eff (I src, signed)                 -0.31062      -0.30357     0.07724     0.07549      14977
|W_eff| (E src)                       +0.23267      +0.22739     0.07641     0.07468      15048
|W_eff| (I src)                       +0.31062      +0.30357     0.07724     0.07549      14977
W_in_eff (input neurons)           -0.00015382    +0.0030492     0.09934      0.1059       6675
c_E (SFA coupling)                       +0.05     +0.051541           0           0        450
c_0_E (SFA offset)                          +0     +0.027732           0           0        450
c_I (SFA coupling)                       +0.05         +0.05           0           0        450
c_0_I (SFA offset)                          +0            +0           0           0        450
a_0 (threshold)                          +0.35      +0.37416           0           0        300
readout_weight                     -0.00011373    -0.0096042     0.06687     0.05935       6675
readout_bias                                +0     +0.010302           0     0.01316         89
ic.ic (per-variant init)              +0.15962      +0.17629      0.4124      0.4045       1500
```

\newpage

# srnn-no-dales-skip

![Tau evolution](srnn-no-dales-skip/tau_evolution.png){ width=95% }

![W\_EI evolution](srnn-no-dales-skip/W_EI_evolution.png){ width=95% }

\newpage

# srnn-no-dales-skip — parameter table

```text
# overnight9h — variant srnn-no-dales-skip (k=3)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                              +1        +0.797           —           —     scalar
tau_d (s)                                 +0.1      +0.06505           0           0        300
tau_a_E (s)                            +4.8777       +3.7314       3.996       3.057        450
tau_a_I (s)                            +4.8777       +3.5061       3.996       2.872        450
tau_b_rec_E (s)                             +1      +0.79268           0           0        150
tau_b_rel_E (s)                          +0.25      +0.19763           0           0        150
tau_b_rec_I (s)                             +1      +0.79265           0           0        150
tau_b_rel_I (s)                          +0.25      +0.19557           0           0        150
W_eff (E src, signed)                  +0.2326      +0.23221     0.07663     0.08022      15048
W_eff (I src, signed)                 -0.31062      -0.30686     0.07727     0.08275      14977
|W_eff| (E src)                       +0.23267      +0.23239     0.07641     0.07972      15048
|W_eff| (I src)                       +0.31062      +0.30696     0.07724      0.0824      14977
W_in_eff (input neurons)            -0.0011621    +0.0049221     0.09962      0.1037       6675
c_E (SFA coupling)                       +0.05     +0.050413           0           0        450
c_0_E (SFA offset)                          +0    +0.0083969           0           0        450
c_I (SFA coupling)                       +0.05      +0.05011           0           0        450
c_0_I (SFA offset)                          +0     -0.012404           0           0        450
a_0 (threshold)                          +0.35      +0.35191           0           0        300
readout_weight                      -0.0016591     -0.012663     0.06668     0.06659       6675
readout_bias                                +0     +0.020652           0     0.02351         89
ic.ic (per-variant init)              +0.16747      +0.16416      0.3627      0.3999       1500
```

\newpage

# srnn-no-adapt-no-dales-skip

![Tau evolution](srnn-no-adapt-no-dales-skip/tau_evolution.png){ width=95% }

![W\_EI evolution](srnn-no-adapt-no-dales-skip/W_EI_evolution.png){ width=95% }

\newpage

# srnn-no-adapt-no-dales-skip — parameter table

```text
# overnight9h — variant srnn-no-adapt-no-dales-skip (k=4)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                              +1       +1.0121           —           —     scalar
tau_d (s)                                 +0.1      +0.10317           0           0        300
tau_a_E (s)                           +0.69315      +0.70155           0           0        450
tau_a_I (s)                           +0.69315      +0.70155           0           0        450
tau_b_rec_E (s)                             +1       +1.0121           0           0        150
tau_b_rel_E (s)                          +0.25      +0.25303           0           0        150
tau_b_rec_I (s)                             +1       +1.0121           0           0        150
tau_b_rel_I (s)                          +0.25      +0.25303           0           0        150
W_eff (E src, signed)                  +0.2326      +0.23059     0.07663     0.07752      15048
W_eff (I src, signed)                 -0.31062      -0.30708     0.07727     0.08045      14977
|W_eff| (E src)                       +0.23267      +0.23066     0.07641     0.07732      15048
|W_eff| (I src)                       +0.31062      +0.30716     0.07724     0.08013      14977
W_in_eff (input neurons)           +0.00019944    +0.0007215     0.09994     0.09496       6675
c_E (SFA coupling)                       +0.05         +0.05           0           0        450
c_0_E (SFA offset)                          +0            +0           0           0        450
c_I (SFA coupling)                       +0.05         +0.05           0           0        450
c_0_I (SFA offset)                          +0            +0           0           0        450
a_0 (threshold)                          +0.35      +0.36811           0           0        300
readout_weight                     +0.00079717   +0.00042102     0.06573     0.06185       6675
readout_bias                                +0    +0.0006623           0    0.002357         89
ic.ic (per-variant init)              +0.14435      +0.16403      0.4566      0.4252       1500
```

\newpage

# srnn-skip

![Tau evolution](srnn-skip/tau_evolution.png){ width=95% }

![W\_EI evolution](srnn-skip/W_EI_evolution.png){ width=95% }

\newpage

# srnn-skip — parameter table

```text
# overnight9h — variant srnn-skip (k=5)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                              +1      +0.82291           —           —     scalar
tau_d (s)                                 +0.1     +0.069289           0           0        300
tau_a_E (s)                            +4.8777       +3.8078       3.996       3.119        450
tau_a_I (s)                            +4.8777       +3.5291       3.996       2.891        450
tau_b_rec_E (s)                             +1       +0.8256           0           0        150
tau_b_rel_E (s)                          +0.25      +0.20204           0           0        150
tau_b_rec_I (s)                             +1       +0.8156           0           0        150
tau_b_rel_I (s)                          +0.25      +0.20033           0           0        150
W_eff (E src, signed)                 +0.23267      +0.22743     0.07641     0.07487      15048
W_eff (I src, signed)                 -0.31062      -0.30388     0.07724     0.07581      14977
|W_eff| (E src)                       +0.23267      +0.22743     0.07641     0.07487      15048
|W_eff| (I src)                       +0.31062      +0.30388     0.07724     0.07581      14977
W_in_eff (input neurons)            +0.0005819      +0.00453     0.09954      0.1013       6675
c_E (SFA coupling)                       +0.05     +0.051033           0           0        450
c_0_E (SFA offset)                          +0     +0.021421           0           0        450
c_I (SFA coupling)                       +0.05     +0.049782           0           0        450
c_0_I (SFA offset)                          +0     -0.024307           0           0        450
a_0 (threshold)                          +0.35      +0.36655           0           0        300
readout_weight                     -0.00025143    -0.0087016     0.06686     0.06239       6675
readout_bias                                +0     +0.011804           0     0.01621         89
ic.ic (per-variant init)              +0.17068      +0.16826      0.3591      0.3885       1500
```

\newpage

# srnn-sfa-e-only-skip

![Tau evolution](srnn-sfa-e-only-skip/tau_evolution.png){ width=95% }

![W\_EI evolution](srnn-sfa-e-only-skip/W_EI_evolution.png){ width=95% }

\newpage

# srnn-sfa-e-only-skip — parameter table

```text
# overnight9h — variant srnn-sfa-e-only-skip (k=6)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                              +1        +1.016           —           —     scalar
tau_d (s)                                 +0.1       +0.1041           0           0        300
tau_a_E (s)                            +4.8777       +4.8536       3.996       3.976        450
tau_a_I (s)                           +0.69315      +0.70424           0           0        450
tau_b_rec_E (s)                             +1        +1.016           0           0        150
tau_b_rel_E (s)                          +0.25        +0.254           0           0        150
tau_b_rec_I (s)                             +1        +1.016           0           0        150
tau_b_rel_I (s)                          +0.25        +0.254           0           0        150
W_eff (E src, signed)                 +0.23267       +0.2287     0.07641     0.07517      15048
W_eff (I src, signed)                 -0.31062       -0.3059     0.07724     0.07613      14977
|W_eff| (E src)                       +0.23267       +0.2287     0.07641     0.07517      15048
|W_eff| (I src)                       +0.31062       +0.3059     0.07724     0.07613      14977
W_in_eff (input neurons)           -0.00083129   -0.00069762     0.09974     0.09659       6675
c_E (SFA coupling)                       +0.05     +0.050765           0           0        450
c_0_E (SFA offset)                          +0     +0.023817           0           0        450
c_I (SFA coupling)                       +0.05         +0.05           0           0        450
c_0_I (SFA offset)                          +0            +0           0           0        450
a_0 (threshold)                          +0.35      +0.37963           0           0        300
readout_weight                       +0.001119   -0.00030331     0.06703     0.06107       6675
readout_bias                                +0   -0.00045903           0     0.00227         89
ic.ic (per-variant init)              +0.16079      +0.17861      0.4384      0.4194       1500
```

\newpage

# srnn-std-e-only-skip

![Tau evolution](srnn-std-e-only-skip/tau_evolution.png){ width=95% }

![W\_EI evolution](srnn-std-e-only-skip/W_EI_evolution.png){ width=95% }

\newpage

# srnn-std-e-only-skip — parameter table

```text
# overnight9h — variant srnn-std-e-only-skip (k=7)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                              +1      +0.79855           —           —     scalar
tau_d (s)                                 +0.1     +0.067139           0           0        300
tau_a_E (s)                           +0.69315      +0.55352           0           0        450
tau_a_I (s)                           +0.69315      +0.55352           0           0        450
tau_b_rec_E (s)                             +1       +0.7986           0           0        150
tau_b_rel_E (s)                          +0.25      +0.19626           0           0        150
tau_b_rec_I (s)                             +1      +0.79855           0           0        150
tau_b_rel_I (s)                          +0.25      +0.19964           0           0        150
W_eff (E src, signed)                 +0.23267       +0.2305     0.07641     0.07589      15048
W_eff (I src, signed)                 -0.31062      -0.30824     0.07724     0.07693      14977
|W_eff| (E src)                       +0.23267       +0.2305     0.07641     0.07589      15048
|W_eff| (I src)                       +0.31062      +0.30824     0.07724     0.07693      14977
W_in_eff (input neurons)            +0.0012099    +0.0048252     0.09923      0.1096       6675
c_E (SFA coupling)                       +0.05         +0.05           0           0        450
c_0_E (SFA offset)                          +0            +0           0           0        450
c_I (SFA coupling)                       +0.05         +0.05           0           0        450
c_0_I (SFA offset)                          +0            +0           0           0        450
a_0 (threshold)                          +0.35      +0.37122           0           0        300
readout_weight                     +0.00034909    -0.0058602     0.06671     0.06043       6675
readout_bias                                +0     +0.012499           0     0.01634         89
ic.ic (per-variant init)              +0.15334      +0.16026      0.4128      0.4104       1500
```
