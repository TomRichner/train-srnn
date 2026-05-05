# Run: filt150-whoist-cl

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
# filt150-whoist-cl — variant srnn-e-only-per-neuron (k=0)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                              +1      +0.45741           —           —     scalar
tau_d (s)                                 +0.1     +0.014046           0    0.004683        300
tau_a_E[j=0] (s)                         +0.25     +0.034603           0    0.008747        150
tau_a_E[j=1] (s)                       +4.3832      +0.78163           0     0.03361        150
tau_a_E[j=2] (s)                           +10       +1.7562           0     0.02638        150
tau_b_rec_E (s)                             +1      +0.51562           0      0.0617        150
tau_b_rel_E (s)                          +0.25      +0.32634           0     0.05398        150
W_eff (E src, signed)                 +0.23267       +0.2002     0.07641     0.07165      15048
W_eff (I src, signed)                 -0.31062      -0.26116     0.07724      0.0718      14977
|W_eff| (E src)                       +0.23267       +0.2002     0.07641     0.07165      15048
|W_eff| (I src)                       +0.31062      +0.26116     0.07724      0.0718      14977
W_in_eff (input neurons)            -0.0003386   +0.00077285      0.1012     0.08464       6675
c_E[j=0] (SFA coupling)                  +0.05      +0.10078           0     0.02885        150
c_E[j=1] (SFA coupling)                  +0.05     +0.088987           0     0.01577        150
c_E[j=2] (SFA coupling)                  +0.05     +0.090157           0     0.01536        150
c_0_E[j=0] (SFA offset)                     +0      -0.55652           0      0.1215        150
c_0_E[j=1] (SFA offset)                     +0       -0.6329           0      0.1548        150
c_0_E[j=2] (SFA offset)                     +0      -0.64094           0      0.1601        150
a_0 (threshold)                          +0.35      -0.23773           0      0.1249        300
readout_weight                     -2.8854e-05    -0.0045077     0.06604      0.2072       6675
readout_bias                                +0    +0.0068269           0      0.1851         89
ic.ic (per-variant init)              +0.10386      +0.10386      0.3828      0.3828        900
```

\newpage

# srnn-e-only-skip-per-neuron

![Tau evolution](srnn-e-only-skip-per-neuron/tau_evolution.png){ width=95% }

![W\_EI evolution](srnn-e-only-skip-per-neuron/W_EI_evolution.png){ width=95% }

\newpage

# srnn-e-only-skip-per-neuron — parameter table

```text
# filt150-whoist-cl — variant srnn-e-only-skip-per-neuron (k=1)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                              +1      +0.47578           —           —     scalar
tau_d (s)                                 +0.1     +0.016737           0     0.00425        300
tau_a_E[j=0] (s)                         +0.25     +0.048089           0     0.01173        150
tau_a_E[j=1] (s)                       +4.3832       +1.0284           0     0.01272        150
tau_a_E[j=2] (s)                           +10       +2.3605           0    0.003405        150
tau_b_rec_E (s)                             +1      +0.18445           0     0.02652        150
tau_b_rel_E (s)                          +0.25      +0.15944           0     0.01977        150
W_eff (E src, signed)                 +0.23267      +0.18737     0.07641     0.06416      15048
W_eff (I src, signed)                 -0.31062      -0.24715     0.07724     0.06573      14977
|W_eff| (E src)                       +0.23267      +0.18737     0.07641     0.06416      15048
|W_eff| (I src)                       +0.31062      +0.24715     0.07724     0.06573      14977
W_in_eff (input neurons)           -0.00084237   -0.00018698      0.1008     0.04121       6675
c_E[j=0] (SFA coupling)                  +0.05     +0.087854           0     0.02116        150
c_E[j=1] (SFA coupling)                  +0.05     +0.073544           0     0.00723        150
c_E[j=2] (SFA coupling)                  +0.05     +0.073719           0    0.007041        150
c_0_E[j=0] (SFA offset)                     +0      -0.16547           0       0.119        150
c_0_E[j=1] (SFA offset)                     +0      -0.18917           0      0.1404        150
c_0_E[j=2] (SFA offset)                     +0      -0.19005           0      0.1373        150
a_0 (threshold)                          +0.35      +0.17463           0      0.1116        300
readout_weight                     +0.00078219    -0.0060208     0.06654      0.1114       6675
readout_bias                                +0      +0.11339           0      0.1413         89
ic.ic (per-variant init)              +0.10327      +0.10327      0.3829      0.3829        900
```

\newpage

# srnn-sfa-e-only-per-neuron

![Tau evolution](srnn-sfa-e-only-per-neuron/tau_evolution.png){ width=95% }

![W\_EI evolution](srnn-sfa-e-only-per-neuron/W_EI_evolution.png){ width=95% }

\newpage

# srnn-sfa-e-only-per-neuron — parameter table

```text
# filt150-whoist-cl — variant srnn-sfa-e-only-per-neuron (k=2)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                              +1      +0.48155           —           —     scalar
tau_d (s)                                 +0.1     +0.016454           0    0.004318        300
tau_a_E[j=0] (s)                         +0.25     +0.059444           0     0.01066        150
tau_a_E[j=1] (s)                       +4.3832       +1.1899           0     0.03563        150
tau_a_E[j=2] (s)                           +10       +2.7205           0     0.03012        150
tau_b_rec_E (s)                             +1      +0.48155           0           0        150
tau_b_rel_E (s)                          +0.25      +0.12039           0           0        150
W_eff (E src, signed)                 +0.23267      +0.15906     0.07641     0.05404      15048
W_eff (I src, signed)                 -0.31062      -0.21111     0.07724     0.05514      14977
|W_eff| (E src)                       +0.23267      +0.15906     0.07641     0.05404      15048
|W_eff| (I src)                       +0.31062      +0.21111     0.07724     0.05514      14977
W_in_eff (input neurons)           -0.00015382    -0.0018988     0.09934      0.1058       6675
c_E[j=0] (SFA coupling)                  +0.05     +0.059055           0     0.01004        150
c_E[j=1] (SFA coupling)                  +0.05     +0.055178           0    0.005836        150
c_E[j=2] (SFA coupling)                  +0.05     +0.054901           0    0.005454        150
c_0_E[j=0] (SFA offset)                     +0      -0.12002           0     0.09616        150
c_0_E[j=1] (SFA offset)                     +0      -0.14143           0      0.1268        150
c_0_E[j=2] (SFA offset)                     +0      -0.14277           0      0.1289        150
a_0 (threshold)                          +0.35      +0.25079           0     0.07603        300
readout_weight                     -0.00098972    +0.0080247     0.06678      0.1459       6675
readout_bias                                +0     -0.049068           0     0.07967         89
ic.ic (per-variant init)              +0.13078      +0.13078      0.4172      0.4172        900
```

\newpage

# srnn-sfa-e-only-skip-per-neuron

![Tau evolution](srnn-sfa-e-only-skip-per-neuron/tau_evolution.png){ width=95% }

![W\_EI evolution](srnn-sfa-e-only-skip-per-neuron/W_EI_evolution.png){ width=95% }

\newpage

# srnn-sfa-e-only-skip-per-neuron — parameter table

```text
# filt150-whoist-cl — variant srnn-sfa-e-only-skip-per-neuron (k=3)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                              +1       +1.0025           —           —     scalar
tau_d (s)                                 +0.1      +0.10099           0    0.004256        300
tau_a_E[j=0] (s)                         +0.25      +0.24063           0     0.01187        150
tau_a_E[j=1] (s)                       +4.3832       +4.2843           0      0.0225        150
tau_a_E[j=2] (s)                           +10       +9.7946           0    0.008763        150
tau_b_rec_E (s)                             +1       +1.0025           0           0        150
tau_b_rel_E (s)                          +0.25      +0.25062           0           0        150
W_eff (E src, signed)                 +0.23267      +0.22876     0.07641     0.07536      15048
W_eff (I src, signed)                 -0.31062      -0.30562     0.07724      0.0763      14977
|W_eff| (E src)                       +0.23267      +0.22876     0.07641     0.07536      15048
|W_eff| (I src)                       +0.31062      +0.30562     0.07724      0.0763      14977
W_in_eff (input neurons)            -0.0011621     -0.001351     0.09962     0.09224       6675
c_E[j=0] (SFA coupling)                  +0.05     +0.052185           0    0.006185        150
c_E[j=1] (SFA coupling)                  +0.05     +0.052166           0    0.002861        150
c_E[j=2] (SFA coupling)                  +0.05     +0.052257           0    0.002554        150
c_0_E[j=0] (SFA offset)                     +0     +0.041002           0     0.03882        150
c_0_E[j=1] (SFA offset)                     +0     +0.043201           0     0.03161        150
c_0_E[j=2] (SFA offset)                     +0     +0.043341           0      0.0294        150
a_0 (threshold)                          +0.35      +0.38207           0     0.03816        300
readout_weight                     +0.00016801    -0.0010836      0.0673     0.06018       6675
readout_bias                                +0    +0.0053543           0    0.008919         89
ic.ic (per-variant init)              +0.09231      +0.09231      0.5294      0.5294        900
```

\newpage

# srnn-std-e-only-per-neuron

![Tau evolution](srnn-std-e-only-per-neuron/tau_evolution.png){ width=95% }

![W\_EI evolution](srnn-std-e-only-per-neuron/W_EI_evolution.png){ width=95% }

\newpage

# srnn-std-e-only-per-neuron — parameter table

```text
# filt150-whoist-cl — variant srnn-std-e-only-per-neuron (k=4)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                              +1      +0.53603           —           —     scalar
tau_d (s)                                 +0.1     +0.020048           0    0.005587        300
tau_b_rec_E (s)                             +1      +0.57696           0     0.06774        150
tau_b_rel_E (s)                          +0.25      +0.34857           0     0.05015        150
W_eff (E src, signed)                 +0.23267       +0.2054     0.07641     0.07192      15048
W_eff (I src, signed)                 -0.31062      -0.27018     0.07724     0.07255      14977
|W_eff| (E src)                       +0.23267       +0.2054     0.07641     0.07192      15048
|W_eff| (I src)                       +0.31062      +0.27018     0.07724     0.07255      14977
W_in_eff (input neurons)           +0.00019944    -0.0013044     0.09994      0.1025       6675
a_0 (threshold)                          +0.35      -0.11619           0      0.1064        300
readout_weight                     +0.00074661    +0.0088996     0.06657      0.1846       6675
readout_bias                                +0     -0.064354           0      0.1847         89
ic.ic (per-variant init)             +0.090819     +0.090819      0.3835      0.3835        900
```

\newpage

# srnn-std-e-only-skip-per-neuron

![Tau evolution](srnn-std-e-only-skip-per-neuron/tau_evolution.png){ width=95% }

![W\_EI evolution](srnn-std-e-only-skip-per-neuron/W_EI_evolution.png){ width=95% }

\newpage

# srnn-std-e-only-skip-per-neuron — parameter table

```text
# filt150-whoist-cl — variant srnn-std-e-only-skip-per-neuron (k=5)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                              +1      +0.50341           —           —     scalar
tau_d (s)                                 +0.1     +0.019157           0    0.004312        300
tau_b_rec_E (s)                             +1      +0.22586           0     0.03093        150
tau_b_rel_E (s)                          +0.25      +0.15146           0     0.01657        150
W_eff (E src, signed)                 +0.23267       +0.2203     0.07641     0.07476      15048
W_eff (I src, signed)                 -0.31062      -0.29216     0.07724     0.07702      14977
|W_eff| (E src)                       +0.23267       +0.2203     0.07641     0.07476      15048
|W_eff| (I src)                       +0.31062      +0.29216     0.07724     0.07702      14977
W_in_eff (input neurons)           +0.00036119   +0.00040633     0.09956     0.04801       6675
a_0 (threshold)                          +0.35      +0.25135           0      0.1014        300
readout_weight                     +0.00055372   +0.00015984     0.06572     0.09561       6675
readout_bias                                +0     -0.028081           0      0.1108         89
ic.ic (per-variant init)             +0.083543     +0.083543       0.384       0.384        900
```
