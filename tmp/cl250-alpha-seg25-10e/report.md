# Run: cl250-alpha-seg25-10e

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
# cl250-alpha-seg25-10e — variant srnn-e-only-per-neuron (k=0)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                              +1      +0.97915           —           —     scalar
tau_d (s)                                 +0.1     +0.095697           0    0.001153        300
tau_a_E (s)                            +4.8777        +4.767       3.996       3.906        450
tau_b_rec_E (s)                             +1      +0.98053           0    0.004719        150
tau_b_rel_E (s)                          +0.25      +0.24447           0    0.001872        150
W_eff (E src, signed)                 +0.23267      +0.22952     0.07641     0.07538      15048
W_eff (I src, signed)                 -0.31062      -0.30635     0.07724      0.0762      14977
|W_eff| (E src)                       +0.23267      +0.22952     0.07641     0.07538      15048
|W_eff| (I src)                       +0.31062      +0.30635     0.07724      0.0762      14977
W_in_eff (input neurons)            -0.0003386   -0.00036664      0.1012      0.1036       6675
c_E (SFA coupling)                       +0.05      +0.05011           0   0.0004296        450
c_0_E (SFA offset)                          +0    -0.0020587           0     0.01032        450
a_0 (threshold)                          +0.35      +0.35122           0     0.01155        300
readout_weight                     -2.8854e-05   -3.1925e-05     0.06604      0.0669       6675
readout_bias                                +0   +0.00081339           0     0.01411         89
ic.ic (per-variant init)             +0.098875      +0.10327      0.3842      0.3848        900
```

\newpage

# srnn-e-only-skip-per-neuron

![Tau evolution](srnn-e-only-skip-per-neuron/tau_evolution.png){ width=95% }

![W\_EI evolution](srnn-e-only-skip-per-neuron/W_EI_evolution.png){ width=95% }

\newpage

# srnn-e-only-skip-per-neuron — parameter table

```text
# cl250-alpha-seg25-10e — variant srnn-e-only-skip-per-neuron (k=1)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                              +1       +1.0041           —           —     scalar
tau_d (s)                                 +0.1      +0.10235           0    0.001396        300
tau_a_E (s)                            +4.8777       +4.8207       3.996       3.951        450
tau_b_rec_E (s)                             +1       +1.0225           0    0.003211        150
tau_b_rel_E (s)                          +0.25      +0.24539           0    0.001116        150
W_eff (E src, signed)                 +0.23267      +0.22911     0.07641     0.07528      15048
W_eff (I src, signed)                 -0.31062      -0.30667     0.07724     0.07627      14977
|W_eff| (E src)                       +0.23267      +0.22911     0.07641     0.07528      15048
|W_eff| (I src)                       +0.31062      +0.30667     0.07724     0.07627      14977
W_in_eff (input neurons)           -0.00084237   -0.00093004      0.1008      0.0987       6675
c_E (SFA coupling)                       +0.05     +0.051356           0   0.0003076        450
c_0_E (SFA offset)                          +0     +0.028555           0    0.005884        450
a_0 (threshold)                          +0.35      +0.36955           0     0.01429        300
readout_weight                     +0.00078219   +0.00093101     0.06654     0.06129       6675
readout_bias                                +0    +9.946e-05           0    0.002534         89
ic.ic (per-variant init)             +0.098961      +0.12409       0.387       0.396        900
```

\newpage

# srnn-sfa-e-only-per-neuron

![Tau evolution](srnn-sfa-e-only-per-neuron/tau_evolution.png){ width=95% }

![W\_EI evolution](srnn-sfa-e-only-per-neuron/W_EI_evolution.png){ width=95% }

\newpage

# srnn-sfa-e-only-per-neuron — parameter table

```text
# cl250-alpha-seg25-10e — variant srnn-sfa-e-only-per-neuron (k=2)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                              +1       +1.0069           —           —     scalar
tau_d (s)                                 +0.1      +0.10184           0   0.0009093        300
tau_a_E (s)                            +4.8777       +4.8558       3.996        3.98        450
tau_b_rec_E (s)                             +1       +1.0069           0           0        150
tau_b_rel_E (s)                          +0.25      +0.25172           0           0        150
W_eff (E src, signed)                 +0.23267      +0.23026     0.07641     0.07565      15048
W_eff (I src, signed)                 -0.31062      -0.30761     0.07724      0.0765      14977
|W_eff| (E src)                       +0.23267      +0.23026     0.07641     0.07565      15048
|W_eff| (I src)                       +0.31062      +0.30761     0.07724      0.0765      14977
W_in_eff (input neurons)           -0.00015382    -6.446e-05     0.09934     0.09766       6675
c_E (SFA coupling)                       +0.05     +0.050861           0   0.0002694        450
c_0_E (SFA offset)                          +0     +0.019467           0     0.00589        450
a_0 (threshold)                          +0.35      +0.36162           0     0.01022        300
readout_weight                     -0.00098972    -0.0002994     0.06678     0.06555       6675
readout_bias                                +0    +0.0010083           0    0.008808         89
ic.ic (per-variant init)              +0.13078      +0.12284      0.4172      0.4019        900
```

\newpage

# srnn-sfa-e-only-skip-per-neuron

![Tau evolution](srnn-sfa-e-only-skip-per-neuron/tau_evolution.png){ width=95% }

![W\_EI evolution](srnn-sfa-e-only-skip-per-neuron/W_EI_evolution.png){ width=95% }

\newpage

# srnn-sfa-e-only-skip-per-neuron — parameter table

```text
# cl250-alpha-seg25-10e — variant srnn-sfa-e-only-skip-per-neuron (k=3)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                              +1       +1.0133           —           —     scalar
tau_d (s)                                 +0.1      +0.10347           0    0.001138        300
tau_a_E (s)                            +4.8777       +4.8751       3.996       3.997        450
tau_b_rec_E (s)                             +1       +1.0133           0           0        150
tau_b_rel_E (s)                          +0.25      +0.25333           0           0        150
W_eff (E src, signed)                 +0.23267      +0.22987     0.07641     0.07553      15048
W_eff (I src, signed)                 -0.31062      -0.30725     0.07724     0.07641      14977
|W_eff| (E src)                       +0.23267      +0.22987     0.07641     0.07553      15048
|W_eff| (I src)                       +0.31062      +0.30725     0.07724     0.07641      14977
W_in_eff (input neurons)            -0.0011621   -0.00082511     0.09962     0.09705       6675
c_E (SFA coupling)                       +0.05     +0.051057           0   0.0002806        450
c_0_E (SFA offset)                          +0      +0.02456           0    0.005836        450
a_0 (threshold)                          +0.35      +0.36477           0     0.01225        300
readout_weight                     +0.00016801   +0.00010628      0.0673     0.06379       6675
readout_bias                                +0   +4.3758e-05           0    0.007891         89
ic.ic (per-variant init)              +0.09231      +0.12403      0.5294      0.4013        900
```

\newpage

# srnn-std-e-only-per-neuron

![Tau evolution](srnn-std-e-only-per-neuron/tau_evolution.png){ width=95% }

![W\_EI evolution](srnn-std-e-only-per-neuron/W_EI_evolution.png){ width=95% }

\newpage

# srnn-std-e-only-per-neuron — parameter table

```text
# cl250-alpha-seg25-10e — variant srnn-std-e-only-per-neuron (k=4)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                              +1      +0.98542           —           —     scalar
tau_d (s)                                 +0.1     +0.097078           0     0.00107        300
tau_a_E (s)                           +0.69315      +0.68304           0           0        450
tau_b_rec_E (s)                             +1      +0.98822           0    0.005042        150
tau_b_rel_E (s)                          +0.25      +0.24516           0     0.00188        150
W_eff (E src, signed)                 +0.23267      +0.23029     0.07641     0.07564      15048
W_eff (I src, signed)                 -0.31062      -0.30749     0.07724     0.07645      14977
|W_eff| (E src)                       +0.23267      +0.23029     0.07641     0.07564      15048
|W_eff| (I src)                       +0.31062      +0.30749     0.07724     0.07645      14977
W_in_eff (input neurons)           +0.00019944   +0.00032829     0.09994      0.1017       6675
c_E (SFA coupling)                       +0.05         +0.05           0           0        450
c_0_E (SFA offset)                          +0            +0           0           0        450
a_0 (threshold)                          +0.35      +0.35382           0     0.01007        300
readout_weight                     +0.00074661     +0.001915     0.06657     0.06715       6675
readout_bias                                +0    -0.0024638           0     0.01364         89
ic.ic (per-variant init)             +0.087739     +0.092387      0.3868      0.3916        900
```

\newpage

# srnn-std-e-only-skip-per-neuron

![Tau evolution](srnn-std-e-only-skip-per-neuron/tau_evolution.png){ width=95% }

![W\_EI evolution](srnn-std-e-only-skip-per-neuron/W_EI_evolution.png){ width=95% }

\newpage

# srnn-std-e-only-skip-per-neuron — parameter table

```text
# cl250-alpha-seg25-10e — variant srnn-std-e-only-skip-per-neuron (k=5)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                              +1       +1.0007           —           —     scalar
tau_d (s)                                 +0.1      +0.10209           0    0.001385        300
tau_a_E (s)                           +0.69315      +0.69366           0           0        450
tau_b_rec_E (s)                             +1       +1.0198           0    0.003154        150
tau_b_rel_E (s)                          +0.25      +0.24446           0    0.001079        150
W_eff (E src, signed)                 +0.23267      +0.22886     0.07641      0.0752      15048
W_eff (I src, signed)                 -0.31062      -0.30635     0.07724     0.07619      14977
|W_eff| (E src)                       +0.23267      +0.22886     0.07641      0.0752      15048
|W_eff| (I src)                       +0.31062      +0.30635     0.07724     0.07619      14977
W_in_eff (input neurons)           +0.00036119   +0.00031513     0.09956     0.09858       6675
c_E (SFA coupling)                       +0.05         +0.05           0           0        450
c_0_E (SFA offset)                          +0            +0           0           0        450
a_0 (threshold)                          +0.35      +0.36931           0     0.01434        300
readout_weight                     +0.00055372   -0.00044298     0.06572     0.06039       6675
readout_bias                                +0   -0.00058261           0     0.00371         89
ic.ic (per-variant init)             +0.089141      +0.11144      0.3857      0.3981        900
```
