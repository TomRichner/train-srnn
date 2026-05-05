# Run: overnight-run1-Whoist

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
# overnight-run1-Whoist — variant srnn-e-only-per-neuron (k=0)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                              +1      +0.32738           —           —     scalar
tau_d (s)                                 +0.1    +0.0029804           0    0.002336        300
tau_a_E (s)                            +4.8777      +0.54953       3.996      0.4781        450
tau_b_rec_E (s)                             +1       +7.6743           0       3.204        150
tau_b_rel_E (s)                          +0.25      +0.37496           0     0.06777        150
W_eff (E src, signed)                 +0.23267      +0.37141     0.07641      0.3083      15048
W_eff (I src, signed)                 -0.31062      -0.36737     0.07724      0.1441      14977
|W_eff| (E src)                       +0.23267      +0.37141     0.07641      0.3083      15048
|W_eff| (I src)                       +0.31062      +0.36737     0.07724      0.1441      14977
W_in_eff (input neurons)            -0.0003386   -0.00011137      0.1012     0.06014       6675
c_E (SFA coupling)                       +0.05      +0.41231           0      0.5972        450
c_0_E (SFA offset)                          +0       -1.5192           0      0.6642        450
a_0 (threshold)                          +0.35       -1.3069           0      0.2358        300
readout_weight                     -2.8854e-05    -0.0088719     0.06604       1.075       6675
readout_bias                                +0     +0.036947           0      0.1891         89
ic.ic (per-variant init)              +0.10386      +0.10386      0.3828      0.3828        900
```

\newpage

# srnn-e-only-skip-per-neuron

![Tau evolution](srnn-e-only-skip-per-neuron/tau_evolution.png){ width=95% }

![W\_EI evolution](srnn-e-only-skip-per-neuron/W_EI_evolution.png){ width=95% }

\newpage

# srnn-e-only-skip-per-neuron — parameter table

```text
# overnight-run1-Whoist — variant srnn-e-only-skip-per-neuron (k=1)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                              +1      +0.39829           —           —     scalar
tau_d (s)                                 +0.1    +0.0070567           0    0.005277        300
tau_a_E (s)                            +4.8777      +0.48719       3.996       0.423        450
tau_b_rec_E (s)                             +1       +1.7102           0      0.5535        150
tau_b_rel_E (s)                          +0.25      +0.37069           0      0.1286        150
W_eff (E src, signed)                 +0.23267      +0.21163     0.07641       0.117      15048
W_eff (I src, signed)                 -0.31062      -0.26258     0.07724       0.116      14977
|W_eff| (E src)                       +0.23267      +0.21163     0.07641       0.117      15048
|W_eff| (I src)                       +0.31062      +0.26258     0.07724       0.116      14977
W_in_eff (input neurons)           -0.00084237     -0.000566      0.1008     0.03338       6675
c_E (SFA coupling)                       +0.05       +0.1532           0      0.1593        450
c_0_E (SFA offset)                          +0      -0.59708           0      0.3098        450
a_0 (threshold)                          +0.35      -0.53513           0      0.1452        300
readout_weight                     +0.00078219    -0.0020769     0.06654      0.3112       6675
readout_bias                                +0      +0.14165           0      0.2093         89
ic.ic (per-variant init)              +0.10327      +0.10327      0.3829      0.3829        900
```

\newpage

# srnn-sfa-e-only-per-neuron

![Tau evolution](srnn-sfa-e-only-per-neuron/tau_evolution.png){ width=95% }

![W\_EI evolution](srnn-sfa-e-only-per-neuron/W_EI_evolution.png){ width=95% }

\newpage

# srnn-sfa-e-only-per-neuron — parameter table

```text
# overnight-run1-Whoist — variant srnn-sfa-e-only-per-neuron (k=2)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                              +1      +0.18512           —           —     scalar
tau_d (s)                                 +0.1    +0.0023674           0    0.002576        300
tau_a_E (s)                            +4.8777      +0.56125       3.996      0.4747        450
tau_b_rec_E (s)                             +1      +0.18512           0           0        150
tau_b_rel_E (s)                          +0.25      +0.04628           0           0        150
W_eff (E src, signed)                 +0.23267      +0.11601     0.07641      0.0592      15048
W_eff (I src, signed)                 -0.31062      -0.15321     0.07724     0.06234      14977
|W_eff| (E src)                       +0.23267      +0.11601     0.07641      0.0592      15048
|W_eff| (I src)                       +0.31062      +0.15321     0.07724     0.06234      14977
W_in_eff (input neurons)           -0.00015382    -0.0010107     0.09934     0.05065       6675
c_E (SFA coupling)                       +0.05      +0.41158           0      0.6209        450
c_0_E (SFA offset)                          +0      -0.28508           0      0.2741        450
a_0 (threshold)                          +0.35     +0.087299           0      0.1382        300
readout_weight                     -0.00098972    +0.0081173     0.06678      0.5427       6675
readout_bias                                +0       -0.1264           0      0.2967         89
ic.ic (per-variant init)              +0.13078      +0.13078      0.4172      0.4172        900
```

\newpage

# srnn-sfa-e-only-skip-per-neuron

![Tau evolution](srnn-sfa-e-only-skip-per-neuron/tau_evolution.png){ width=95% }

![W\_EI evolution](srnn-sfa-e-only-skip-per-neuron/W_EI_evolution.png){ width=95% }

\newpage

# srnn-sfa-e-only-skip-per-neuron — parameter table

```text
# overnight-run1-Whoist — variant srnn-sfa-e-only-skip-per-neuron (k=3)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                              +1       +0.2414           —           —     scalar
tau_d (s)                                 +0.1    +0.0048985           0    0.003099        300
tau_a_E (s)                            +4.8777      +0.24446       3.996      0.2074        450
tau_b_rec_E (s)                             +1       +0.2414           0           0        150
tau_b_rel_E (s)                          +0.25      +0.06035           0           0        150
W_eff (E src, signed)                 +0.23267      +0.15765     0.07641      0.0644      15048
W_eff (I src, signed)                 -0.31062      -0.20829     0.07724     0.07216      14977
|W_eff| (E src)                       +0.23267      +0.15765     0.07641      0.0644      15048
|W_eff| (I src)                       +0.31062      +0.20829     0.07724     0.07216      14977
W_in_eff (input neurons)            -0.0011621   -0.00010726     0.09962     0.02913       6675
c_E (SFA coupling)                       +0.05     +0.072426           0     0.04932        450
c_0_E (SFA offset)                          +0      -0.31112           0      0.2576        450
a_0 (threshold)                          +0.35      +0.22147           0      0.1239        300
readout_weight                     +0.00016801    -0.0088109      0.0673      0.2104       6675
readout_bias                                +0       +0.1166           0      0.1961         89
ic.ic (per-variant init)              +0.09231      +0.09231      0.5294      0.5294        900
```

\newpage

# srnn-std-e-only-per-neuron

![Tau evolution](srnn-std-e-only-per-neuron/tau_evolution.png){ width=95% }

![W\_EI evolution](srnn-std-e-only-per-neuron/W_EI_evolution.png){ width=95% }

\newpage

# srnn-std-e-only-per-neuron — parameter table

```text
# overnight-run1-Whoist — variant srnn-std-e-only-per-neuron (k=4)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                              +1      +0.25008           —           —     scalar
tau_d (s)                                 +0.1    +0.0017287           0    0.001449        300
tau_a_E (s)                           +0.69315      +0.17334           0           0        450
tau_b_rec_E (s)                             +1       +3.9623           0       1.862        150
tau_b_rel_E (s)                          +0.25      +0.42512           0     0.07687        150
W_eff (E src, signed)                 +0.23267      +0.23364     0.07641      0.1586      15048
W_eff (I src, signed)                 -0.31062      -0.25145     0.07724     0.09947      14977
|W_eff| (E src)                       +0.23267      +0.23364     0.07641      0.1586      15048
|W_eff| (I src)                       +0.31062      +0.25145     0.07724     0.09947      14977
W_in_eff (input neurons)           +0.00019944   -0.00031341     0.09994     0.03871       6675
c_E (SFA coupling)                       +0.05         +0.05           0           0        450
c_0_E (SFA offset)                          +0            +0           0           0        450
a_0 (threshold)                          +0.35       -1.4128           0      0.2758        300
readout_weight                     +0.00074661    -0.0069444     0.06657      0.8627       6675
readout_bias                                +0     -0.019712           0      0.1602         89
ic.ic (per-variant init)             +0.090819     +0.090819      0.3835      0.3835        900
```

\newpage

# srnn-std-e-only-skip-per-neuron

![Tau evolution](srnn-std-e-only-skip-per-neuron/tau_evolution.png){ width=95% }

![W\_EI evolution](srnn-std-e-only-skip-per-neuron/W_EI_evolution.png){ width=95% }

\newpage

# srnn-std-e-only-skip-per-neuron — parameter table

```text
# overnight-run1-Whoist — variant srnn-std-e-only-skip-per-neuron (k=5)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                              +1      +0.31356           —           —     scalar
tau_d (s)                                 +0.1    +0.0058463           0    0.004701        300
tau_a_E (s)                           +0.69315      +0.21734           0           0        450
tau_b_rec_E (s)                             +1      +0.44648           0      0.1566        150
tau_b_rel_E (s)                          +0.25      +0.37733           0      0.1397        150
W_eff (E src, signed)                 +0.23267      +0.16943     0.07641     0.08499      15048
W_eff (I src, signed)                 -0.31062      -0.21666     0.07724     0.09173      14977
|W_eff| (E src)                       +0.23267      +0.16943     0.07641     0.08499      15048
|W_eff| (I src)                       +0.31062      +0.21666     0.07724     0.09173      14977
W_in_eff (input neurons)           +0.00036119   +8.7197e-05     0.09956     0.02451       6675
c_E (SFA coupling)                       +0.05         +0.05           0           0        450
c_0_E (SFA offset)                          +0            +0           0           0        450
a_0 (threshold)                          +0.35      -0.33933           0      0.1586        300
readout_weight                     +0.00055372    -0.0098317     0.06572      0.2858       6675
readout_bias                                +0    -0.0034242           0      0.1832         89
ic.ic (per-variant init)             +0.083543     +0.083543       0.384       0.384        900
```
