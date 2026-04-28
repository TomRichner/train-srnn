# Run: overnight-cl250

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
# overnight-cl250 — variant srnn-e-only-per-neuron (k=0)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                              +1      +0.75694           —           —     scalar
tau_d (s)                                 +0.1     +0.047285           0     0.00829        300
tau_a_E (s)                            +4.8777       +2.5842       3.996       2.116        450
tau_b_rec_E (s)                             +1      +0.45844           0     0.04265        150
tau_b_rel_E (s)                          +0.25      +0.32092           0     0.03539        150
W_eff (E src, signed)                 +0.23267      +0.24787     0.07641     0.08534      15048
W_eff (I src, signed)                 -0.31062      -0.32687     0.07724     0.08695      14977
|W_eff| (E src)                       +0.23267      +0.24787     0.07641     0.08534      15048
|W_eff| (I src)                       +0.31062      +0.32687     0.07724     0.08695      14977
W_in_eff (input neurons)            -0.0003386    +0.0013842      0.1012      0.1342       6675
c_E (SFA coupling)                       +0.05      +0.05476           0    0.006898        450
c_0_E (SFA offset)                          +0      -0.27431           0     0.09773        450
a_0 (threshold)                          +0.35      +0.16777           0     0.08514        300
readout_weight                     -2.8854e-05    -0.0057749     0.06604      0.1456       6675
readout_bias                                +0    +0.0044741           0      0.1548         89
ic.ic (per-variant init)             +0.098875   -0.00036654      0.3842      0.5642        900
```

\newpage

# srnn-e-only-skip-per-neuron

![Tau evolution](srnn-e-only-skip-per-neuron/tau_evolution.png){ width=95% }

![W\_EI evolution](srnn-e-only-skip-per-neuron/W_EI_evolution.png){ width=95% }

\newpage

# srnn-e-only-skip-per-neuron — parameter table

```text
# overnight-cl250 — variant srnn-e-only-skip-per-neuron (k=1)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                              +1      +0.71474           —           —     scalar
tau_d (s)                                 +0.1     +0.045103           0     0.00597        300
tau_a_E (s)                            +4.8777       +2.9855       3.996       2.453        450
tau_b_rec_E (s)                             +1      +0.64798           0     0.02907        150
tau_b_rel_E (s)                          +0.25      +0.17806           0    0.008715        150
W_eff (E src, signed)                 +0.23267      +0.22277     0.07641     0.07386      15048
W_eff (I src, signed)                 -0.31062      -0.29646     0.07724     0.07555      14977
|W_eff| (E src)                       +0.23267      +0.22277     0.07641     0.07386      15048
|W_eff| (I src)                       +0.31062      +0.29646     0.07724     0.07555      14977
W_in_eff (input neurons)           -0.00084237     +0.001791      0.1008      0.0739       6675
c_E (SFA coupling)                       +0.05     +0.052935           0    0.003242        450
c_0_E (SFA offset)                          +0    -0.0012899           0     0.05942        450
a_0 (threshold)                          +0.35      +0.28407           0     0.07765        300
readout_weight                     +0.00078219     -0.013388     0.06654     0.07669       6675
readout_bias                                +0     +0.075552           0     0.08734         89
ic.ic (per-variant init)             +0.098961     +0.064123       0.387      0.4347        900
```

\newpage

# srnn-sfa-e-only-per-neuron

![Tau evolution](srnn-sfa-e-only-per-neuron/tau_evolution.png){ width=95% }

![W\_EI evolution](srnn-sfa-e-only-per-neuron/W_EI_evolution.png){ width=95% }

\newpage

# srnn-sfa-e-only-per-neuron — parameter table

```text
# overnight-cl250 — variant srnn-sfa-e-only-per-neuron (k=2)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                              +1      +0.77065           —           —     scalar
tau_d (s)                                 +0.1     +0.050201           0    0.006832        300
tau_a_E (s)                            +4.8777       +3.2983       3.996       2.703        450
tau_b_rec_E (s)                             +1      +0.77065           0           0        150
tau_b_rel_E (s)                          +0.25      +0.19266           0           0        150
W_eff (E src, signed)                 +0.23267      +0.24106     0.07641     0.08131      15048
W_eff (I src, signed)                 -0.31062      -0.31873     0.07724     0.08281      14977
|W_eff| (E src)                       +0.23267      +0.24106     0.07641     0.08131      15048
|W_eff| (I src)                       +0.31062      +0.31873     0.07724     0.08281      14977
W_in_eff (input neurons)           -0.00015382    -0.0020851     0.09934      0.1374       6675
c_E (SFA coupling)                       +0.05     +0.051482           0    0.003646        450
c_0_E (SFA offset)                          +0     -0.079732           0     0.06781        450
a_0 (threshold)                          +0.35      +0.34909           0     0.05572        300
readout_weight                     -0.00098972    +0.0052418     0.06678      0.1206       6675
readout_bias                                +0     -0.026226           0     0.06046         89
ic.ic (per-variant init)              +0.13078     +0.065655      0.4172      0.4345        900
```

\newpage

# srnn-sfa-e-only-skip-per-neuron

![Tau evolution](srnn-sfa-e-only-skip-per-neuron/tau_evolution.png){ width=95% }

![W\_EI evolution](srnn-sfa-e-only-skip-per-neuron/W_EI_evolution.png){ width=95% }

\newpage

# srnn-sfa-e-only-skip-per-neuron — parameter table

```text
# overnight-cl250 — variant srnn-sfa-e-only-skip-per-neuron (k=3)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                              +1        +1.016           —           —     scalar
tau_d (s)                                 +0.1      +0.10403           0    0.003474        300
tau_a_E (s)                            +4.8777       +4.8605       3.996       3.986        450
tau_b_rec_E (s)                             +1        +1.016           0           0        150
tau_b_rel_E (s)                          +0.25        +0.254           0           0        150
W_eff (E src, signed)                 +0.23267      +0.22921     0.07641     0.07535      15048
W_eff (I src, signed)                 -0.31062      -0.30625     0.07724     0.07637      14977
|W_eff| (E src)                       +0.23267      +0.22921     0.07641     0.07535      15048
|W_eff| (I src)                       +0.31062      +0.30625     0.07724     0.07637      14977
W_in_eff (input neurons)            -0.0011621    -0.0014667     0.09962     0.09388       6675
c_E (SFA coupling)                       +0.05     +0.051699           0    0.001434        450
c_0_E (SFA offset)                          +0     +0.036936           0     0.02567        450
a_0 (threshold)                          +0.35      +0.37426           0     0.03578        300
readout_weight                     +0.00016801    -0.0010738      0.0673     0.06088       6675
readout_bias                                +0    +0.0022042           0    0.003744         89
ic.ic (per-variant init)              +0.09231      +0.11729      0.5294      0.4143        900
```

\newpage

# srnn-std-e-only-per-neuron

![Tau evolution](srnn-std-e-only-per-neuron/tau_evolution.png){ width=95% }

![W\_EI evolution](srnn-std-e-only-per-neuron/W_EI_evolution.png){ width=95% }

\newpage

# srnn-std-e-only-per-neuron — parameter table

```text
# overnight-cl250 — variant srnn-std-e-only-per-neuron (k=4)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                              +1      +0.79317           —           —     scalar
tau_d (s)                                 +0.1     +0.052401           0    0.007879        300
tau_a_E (s)                           +0.69315      +0.54978           0           0        450
tau_b_rec_E (s)                             +1      +0.49281           0     0.03874        150
tau_b_rel_E (s)                          +0.25      +0.32451           0     0.03178        150
W_eff (E src, signed)                 +0.23267      +0.27677     0.07641     0.09482      15048
W_eff (I src, signed)                 -0.31062      -0.36449     0.07724     0.09683      14977
|W_eff| (E src)                       +0.23267      +0.27677     0.07641     0.09482      15048
|W_eff| (I src)                       +0.31062      +0.36449     0.07724     0.09683      14977
W_in_eff (input neurons)           +0.00019944   -0.00091659     0.09994      0.1406       6675
c_E (SFA coupling)                       +0.05         +0.05           0           0        450
c_0_E (SFA offset)                          +0            +0           0           0        450
a_0 (threshold)                          +0.35      +0.22171           0     0.08034        300
readout_weight                     +0.00074661    +0.0096444     0.06657      0.1413       6675
readout_bias                                +0     -0.022083           0      0.1547         89
ic.ic (per-variant init)             +0.087739     +0.028283      0.3868      0.4661        900
```

\newpage

# srnn-std-e-only-skip-per-neuron

![Tau evolution](srnn-std-e-only-skip-per-neuron/tau_evolution.png){ width=95% }

![W\_EI evolution](srnn-std-e-only-skip-per-neuron/W_EI_evolution.png){ width=95% }

\newpage

# srnn-std-e-only-skip-per-neuron — parameter table

```text
# overnight-cl250 — variant srnn-std-e-only-skip-per-neuron (k=5)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                              +1      +0.73075           —           —     scalar
tau_d (s)                                 +0.1     +0.048476           0    0.004967        300
tau_a_E (s)                           +0.69315      +0.50652           0           0        450
tau_b_rec_E (s)                             +1      +0.68599           0     0.02816        150
tau_b_rel_E (s)                          +0.25      +0.18328           0    0.008936        150
W_eff (E src, signed)                 +0.23267      +0.24529     0.07641     0.08121      15048
W_eff (I src, signed)                 -0.31062       -0.3268     0.07724     0.08245      14977
|W_eff| (E src)                       +0.23267      +0.24529     0.07641     0.08121      15048
|W_eff| (I src)                       +0.31062       +0.3268     0.07724     0.08245      14977
W_in_eff (input neurons)           +0.00036119   -0.00069182     0.09956     0.08771       6675
c_E (SFA coupling)                       +0.05         +0.05           0           0        450
c_0_E (SFA offset)                          +0            +0           0           0        450
a_0 (threshold)                          +0.35       +0.3437           0     0.05943        300
readout_weight                     +0.00055372    -0.0083536     0.06572     0.06826       6675
readout_bias                                +0     +0.024059           0     0.04786         89
ic.ic (per-variant init)             +0.089141      +0.07427      0.3857      0.4266        900
```
