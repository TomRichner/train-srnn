# Run: prod-1hr-b48-k6-bf16

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
# prod-1hr-b48-k6-bf16 — variant srnn-e-only-per-neuron (k=0)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                              +1      +0.84426           —           —     scalar
tau_d (s)                                 +0.1     +0.062495           0    0.005043        300
tau_a_E (s)                            +4.8777       +4.0495       3.996       3.315        450
tau_b_rec_E (s)                             +1      +0.77674           0      0.0319        150
tau_b_rel_E (s)                          +0.25      +0.28402           0     0.01541        150
W_eff (E src, signed)                 +0.23267      +0.25018     0.07641      0.0832      15048
W_eff (I src, signed)                 -0.31062      -0.33081     0.07724     0.08365      14977
|W_eff| (E src)                       +0.23267      +0.25018     0.07641      0.0832      15048
|W_eff| (I src)                       +0.31062      +0.33081     0.07724     0.08365      14977
W_in_eff (input neurons)            -0.0003386   +7.8613e-05      0.1012      0.1224       6675
c_E (SFA coupling)                       +0.05     +0.057177           0    0.003967        450
c_0_E (SFA offset)                          +0      -0.22391           0     0.05036        450
a_0 (threshold)                          +0.35      +0.15498           0     0.06464        300
readout_weight                     -2.8854e-05    -0.0032506     0.06604      0.0984       6675
readout_bias                                +0    -0.0050603           0      0.0857         89
ic.ic (per-variant init)              +0.10386      +0.10386      0.3828      0.3828        900
```

\newpage

# srnn-e-only-skip-per-neuron

![Tau evolution](srnn-e-only-skip-per-neuron/tau_evolution.png){ width=95% }

![W\_EI evolution](srnn-e-only-skip-per-neuron/W_EI_evolution.png){ width=95% }

\newpage

# srnn-e-only-skip-per-neuron — parameter table

```text
# prod-1hr-b48-k6-bf16 — variant srnn-e-only-skip-per-neuron (k=1)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                              +1      +0.83888           —           —     scalar
tau_d (s)                                 +0.1      +0.06641           0    0.004549        300
tau_a_E (s)                            +4.8777       +3.6655       3.996       3.007        450
tau_b_rec_E (s)                             +1      +0.69932           0     0.01853        150
tau_b_rel_E (s)                          +0.25      +0.20993           0    0.005072        150
W_eff (E src, signed)                 +0.23267      +0.23311     0.07641     0.07676      15048
W_eff (I src, signed)                 -0.31062      -0.31177     0.07724     0.07791      14977
|W_eff| (E src)                       +0.23267      +0.23311     0.07641     0.07676      15048
|W_eff| (I src)                       +0.31062      +0.31177     0.07724     0.07791      14977
W_in_eff (input neurons)           -0.00084237    +0.0013641      0.1008     0.08759       6675
c_E (SFA coupling)                       +0.05     +0.052384           0    0.001517        450
c_0_E (SFA offset)                          +0     +0.014492           0     0.03113        450
a_0 (threshold)                          +0.35      +0.33015           0     0.04215        300
readout_weight                     +0.00078219     -0.010057     0.06654     0.06103       6675
readout_bias                                +0     +0.026863           0     0.03419         89
ic.ic (per-variant init)              +0.10327      +0.10327      0.3829      0.3829        900
```

\newpage

# srnn-sfa-e-only-per-neuron

![Tau evolution](srnn-sfa-e-only-per-neuron/tau_evolution.png){ width=95% }

![W\_EI evolution](srnn-sfa-e-only-per-neuron/W_EI_evolution.png){ width=95% }

\newpage

# srnn-sfa-e-only-per-neuron — parameter table

```text
# prod-1hr-b48-k6-bf16 — variant srnn-sfa-e-only-per-neuron (k=2)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                              +1      +0.86885           —           —     scalar
tau_d (s)                                 +0.1     +0.068555           0    0.003936        300
tau_a_E (s)                            +4.8777       +3.9551       3.996       3.244        450
tau_b_rec_E (s)                             +1      +0.86885           0           0        150
tau_b_rel_E (s)                          +0.25      +0.21721           0           0        150
W_eff (E src, signed)                 +0.23267      +0.22467     0.07641     0.07414      15048
W_eff (I src, signed)                 -0.31062      -0.29933     0.07724     0.07477      14977
|W_eff| (E src)                       +0.23267      +0.22467     0.07641     0.07414      15048
|W_eff| (I src)                       +0.31062      +0.29933     0.07724     0.07477      14977
W_in_eff (input neurons)           -0.00015382   -2.4072e-05     0.09934      0.1176       6675
c_E (SFA coupling)                       +0.05     +0.051522           0    0.001539        450
c_0_E (SFA offset)                          +0     -0.027567           0     0.03823        450
a_0 (threshold)                          +0.35      +0.34634           0      0.0325        300
readout_weight                     -0.00098972   +0.00088098     0.06678      0.0803       6675
readout_bias                                +0    -0.0010009           0     0.02886         89
ic.ic (per-variant init)              +0.13078      +0.13078      0.4172      0.4172        900
```

\newpage

# srnn-sfa-e-only-skip-per-neuron

![Tau evolution](srnn-sfa-e-only-skip-per-neuron/tau_evolution.png){ width=95% }

![W\_EI evolution](srnn-sfa-e-only-skip-per-neuron/W_EI_evolution.png){ width=95% }

\newpage

# srnn-sfa-e-only-skip-per-neuron — parameter table

```text
# prod-1hr-b48-k6-bf16 — variant srnn-sfa-e-only-skip-per-neuron (k=3)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                              +1        +1.024           —           —     scalar
tau_d (s)                                 +0.1      +0.10643           0    0.001852        300
tau_a_E (s)                            +4.8777       +4.8957       3.996       4.014        450
tau_b_rec_E (s)                             +1        +1.024           0           0        150
tau_b_rel_E (s)                          +0.25      +0.25599           0           0        150
W_eff (E src, signed)                 +0.23267      +0.22905     0.07641     0.07524      15048
W_eff (I src, signed)                 -0.31062      -0.30623     0.07724     0.07617      14977
|W_eff| (E src)                       +0.23267      +0.22905     0.07641     0.07524      15048
|W_eff| (I src)                       +0.31062      +0.30623     0.07724     0.07617      14977
W_in_eff (input neurons)            -0.0011621   -0.00078498     0.09962     0.09376       6675
c_E (SFA coupling)                       +0.05      +0.05166           0   0.0006848        450
c_0_E (SFA offset)                          +0     +0.040759           0     0.01187        450
a_0 (threshold)                          +0.35       +0.3793           0     0.02108        300
readout_weight                     +0.00016801   -0.00022955      0.0673     0.06042       6675
readout_bias                                +0   -0.00013023           0     0.00185         89
ic.ic (per-variant init)              +0.09231      +0.09231      0.5294      0.5294        900
```

\newpage

# srnn-std-e-only-per-neuron

![Tau evolution](srnn-std-e-only-per-neuron/tau_evolution.png){ width=95% }

![W\_EI evolution](srnn-std-e-only-per-neuron/W_EI_evolution.png){ width=95% }

\newpage

# srnn-std-e-only-per-neuron — parameter table

```text
# prod-1hr-b48-k6-bf16 — variant srnn-std-e-only-per-neuron (k=4)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                              +1      +0.84706           —           —     scalar
tau_d (s)                                 +0.1     +0.062714           0    0.005576        300
tau_a_E (s)                           +0.69315      +0.58714           0           0        450
tau_b_rec_E (s)                             +1      +0.74452           0     0.03208        150
tau_b_rel_E (s)                          +0.25      +0.27741           0     0.01408        150
W_eff (E src, signed)                 +0.23267      +0.27438     0.07641     0.09116      15048
W_eff (I src, signed)                 -0.31062      -0.36304     0.07724      0.0915      14977
|W_eff| (E src)                       +0.23267      +0.27438     0.07641     0.09116      15048
|W_eff| (I src)                       +0.31062      +0.36304     0.07724      0.0915      14977
W_in_eff (input neurons)           +0.00019944   -0.00017918     0.09994      0.1175       6675
c_E (SFA coupling)                       +0.05         +0.05           0           0        450
c_0_E (SFA offset)                          +0            +0           0           0        450
a_0 (threshold)                          +0.35      +0.18151           0     0.05688        300
readout_weight                     +0.00074661    +0.0080122     0.06657     0.09976       6675
readout_bias                                +0      -0.01502           0     0.09724         89
ic.ic (per-variant init)             +0.090819     +0.090819      0.3835      0.3835        900
```

\newpage

# srnn-std-e-only-skip-per-neuron

![Tau evolution](srnn-std-e-only-skip-per-neuron/tau_evolution.png){ width=95% }

![W\_EI evolution](srnn-std-e-only-skip-per-neuron/W_EI_evolution.png){ width=95% }

\newpage

# srnn-std-e-only-skip-per-neuron — parameter table

```text
# prod-1hr-b48-k6-bf16 — variant srnn-std-e-only-skip-per-neuron (k=5)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                              +1      +0.85207           —           —     scalar
tau_d (s)                                 +0.1     +0.069634           0    0.003489        300
tau_a_E (s)                           +0.69315      +0.59061           0           0        450
tau_b_rec_E (s)                             +1      +0.74039           0     0.01912        150
tau_b_rel_E (s)                          +0.25      +0.21009           0    0.004211        150
W_eff (E src, signed)                 +0.23267      +0.23812     0.07641     0.07839      15048
W_eff (I src, signed)                 -0.31062      -0.31836     0.07724     0.07932      14977
|W_eff| (E src)                       +0.23267      +0.23812     0.07641     0.07839      15048
|W_eff| (I src)                       +0.31062      +0.31836     0.07724     0.07932      14977
W_in_eff (input neurons)           +0.00036119   -0.00043907     0.09956       0.095       6675
c_E (SFA coupling)                       +0.05         +0.05           0           0        450
c_0_E (SFA offset)                          +0            +0           0           0        450
a_0 (threshold)                          +0.35      +0.36411           0      0.0305        300
readout_weight                     +0.00055372    -0.0019413     0.06572     0.06001       6675
readout_bias                                +0   +0.00020188           0     0.02103         89
ic.ic (per-variant init)             +0.083543     +0.083543       0.384       0.384        900
```
