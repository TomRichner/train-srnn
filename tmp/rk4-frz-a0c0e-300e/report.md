# Run: rk4-frz-a0c0e-300e

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
# rk4-frz-a0c0e-300e — variant srnn-e-only-per-neuron (k=0)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                              +1      +0.34698           —           —     scalar
tau_d (s)                                 +0.1     +0.005996           0    0.003382        300
tau_a_E[j=0] (s)                         +0.25     +0.012352           0    0.006063        150
tau_a_E[j=1] (s)                       +4.3832       +0.4238           0     0.07366        150
tau_a_E[j=2] (s)                           +10      +0.98354           0      0.0621        150
tau_b_rec_E (s)                             +1       +1.4691           0      0.6074        150
tau_b_rel_E (s)                          +0.25      +0.78938           0      0.3671        150
W_eff (E src, signed)                 +0.23267      +0.28128     0.07641      0.1339      15048
W_eff (I src, signed)                 -0.31062      -0.33865     0.07724      0.1118      14977
|W_eff| (E src)                       +0.23267      +0.28128     0.07641      0.1339      15048
|W_eff| (I src)                       +0.31062      +0.33865     0.07724      0.1118      14977
W_in_eff (input neurons)            -0.0003386   -5.5226e-05      0.1012     0.03547       6675
c_E[j=0] (SFA coupling)                  +0.05      +0.12817           0     0.09836        150
c_E[j=1] (SFA coupling)                  +0.05      +0.03279           0      0.0176        150
c_E[j=2] (SFA coupling)                  +0.05     +0.034775           0     0.01713        150
c_0_E[j=0] (SFA offset)                     +0            +0           0           0        150
c_0_E[j=1] (SFA offset)                     +0            +0           0           0        150
c_0_E[j=2] (SFA offset)                     +0            +0           0           0        150
a_0 (threshold)                          +0.35         +0.35           0           0        300
readout_weight                     -2.8854e-05    +0.0019011     0.06604      0.3711       6675
readout_bias                                +0    -0.0090848           0      0.1653         89
ic.ic (per-variant init)              +0.10233      +0.10233      0.3832      0.3832        900
```

\newpage

# srnn-e-only-skip-per-neuron

![Tau evolution](srnn-e-only-skip-per-neuron/tau_evolution.png){ width=95% }

![W\_EI evolution](srnn-e-only-skip-per-neuron/W_EI_evolution.png){ width=95% }

\newpage

# srnn-e-only-skip-per-neuron — parameter table

```text
# rk4-frz-a0c0e-300e — variant srnn-e-only-skip-per-neuron (k=1)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                              +1      +0.37391           —           —     scalar
tau_d (s)                                 +0.1    +0.0092588           0     0.00493        300
tau_a_E[j=0] (s)                         +0.25     +0.017057           0    0.008919        150
tau_a_E[j=1] (s)                       +4.3832      +0.50498           0     0.04034        150
tau_a_E[j=2] (s)                           +10       +1.1382           0     0.02469        150
tau_b_rec_E (s)                             +1      +0.13224           0     0.03099        150
tau_b_rel_E (s)                          +0.25      +0.41067           0      0.1333        150
W_eff (E src, signed)                 +0.23267      +0.16725     0.07641     0.06634      15048
W_eff (I src, signed)                 -0.31062      -0.21189     0.07724     0.06713      14977
|W_eff| (E src)                       +0.23267      +0.16725     0.07641     0.06634      15048
|W_eff| (I src)                       +0.31062      +0.21189     0.07724     0.06713      14977
W_in_eff (input neurons)           -0.00084237   -0.00022745      0.1008     0.01649       6675
c_E[j=0] (SFA coupling)                  +0.05     +0.061055           0     0.03309        150
c_E[j=1] (SFA coupling)                  +0.05     +0.032305           0    0.009822        150
c_E[j=2] (SFA coupling)                  +0.05     +0.033062           0     0.00887        150
c_0_E[j=0] (SFA offset)                     +0            +0           0           0        150
c_0_E[j=1] (SFA offset)                     +0            +0           0           0        150
c_0_E[j=2] (SFA offset)                     +0            +0           0           0        150
a_0 (threshold)                          +0.35         +0.35           0           0        300
readout_weight                     +0.00078219    -0.0088944     0.06654      0.1712       6675
readout_bias                                +0     +0.096091           0       0.127         89
ic.ic (per-variant init)              +0.10092      +0.10092      0.3832      0.3832        900
```

\newpage

# srnn-sfa-e-only-per-neuron

![Tau evolution](srnn-sfa-e-only-per-neuron/tau_evolution.png){ width=95% }

![W\_EI evolution](srnn-sfa-e-only-per-neuron/W_EI_evolution.png){ width=95% }

\newpage

# srnn-sfa-e-only-per-neuron — parameter table

```text
# rk4-frz-a0c0e-300e — variant srnn-sfa-e-only-per-neuron (k=2)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                              +1      +0.30672           —           —     scalar
tau_d (s)                                 +0.1    +0.0050946           0    0.002524        300
tau_a_E[j=0] (s)                         +0.25     +0.012626           0    0.005371        150
tau_a_E[j=1] (s)                       +4.3832        +0.389           0     0.06346        150
tau_a_E[j=2] (s)                           +10      +0.91717           0     0.05339        150
tau_b_rec_E (s)                             +1      +0.30672           0           0        150
tau_b_rel_E (s)                          +0.25     +0.076681           0           0        150
W_eff (E src, signed)                 +0.23267      +0.14209     0.07641     0.05529      15048
W_eff (I src, signed)                 -0.31062       -0.1872     0.07724     0.05725      14977
|W_eff| (E src)                       +0.23267      +0.14209     0.07641     0.05529      15048
|W_eff| (I src)                       +0.31062       +0.1872     0.07724     0.05725      14977
W_in_eff (input neurons)           -0.00015382   -0.00060358     0.09934     0.03697       6675
c_E[j=0] (SFA coupling)                  +0.05      +0.16302           0     0.08614        150
c_E[j=1] (SFA coupling)                  +0.05     +0.078942           0     0.02105        150
c_E[j=2] (SFA coupling)                  +0.05     +0.077687           0     0.01942        150
c_0_E[j=0] (SFA offset)                     +0            +0           0           0        150
c_0_E[j=1] (SFA offset)                     +0            +0           0           0        150
c_0_E[j=2] (SFA offset)                     +0            +0           0           0        150
a_0 (threshold)                          +0.35         +0.35           0           0        300
readout_weight                     -0.00098972    +0.0033037     0.06678      0.2511       6675
readout_bias                                +0     -0.056828           0      0.1019         89
ic.ic (per-variant init)             +0.098719     +0.098719      0.4415      0.4415        900
```

\newpage

# srnn-sfa-e-only-skip-per-neuron

![Tau evolution](srnn-sfa-e-only-skip-per-neuron/tau_evolution.png){ width=95% }

![W\_EI evolution](srnn-sfa-e-only-skip-per-neuron/W_EI_evolution.png){ width=95% }

\newpage

# srnn-sfa-e-only-skip-per-neuron — parameter table

```text
# rk4-frz-a0c0e-300e — variant srnn-sfa-e-only-skip-per-neuron (k=3)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                              +1      +0.35111           —           —     scalar
tau_d (s)                                 +0.1    +0.0091446           0    0.002498        300
tau_a_E[j=0] (s)                         +0.25     +0.045267           0     0.01344        150
tau_a_E[j=1] (s)                       +4.3832      +0.94338           0     0.05165        150
tau_a_E[j=2] (s)                           +10        +2.211           0     0.02075        150
tau_b_rec_E (s)                             +1      +0.35111           0           0        150
tau_b_rel_E (s)                          +0.25     +0.087779           0           0        150
W_eff (E src, signed)                 +0.23267      +0.14995     0.07641     0.05247      15048
W_eff (I src, signed)                 -0.31062      -0.19841     0.07724     0.05476      14977
|W_eff| (E src)                       +0.23267      +0.14995     0.07641     0.05247      15048
|W_eff| (I src)                       +0.31062      +0.19841     0.07724     0.05476      14977
W_in_eff (input neurons)            -0.0011621   -0.00028593     0.09962     0.03203       6675
c_E[j=0] (SFA coupling)                  +0.05     +0.052788           0     0.01142        150
c_E[j=1] (SFA coupling)                  +0.05     +0.043313           0    0.008855        150
c_E[j=2] (SFA coupling)                  +0.05     +0.043164           0    0.009024        150
c_0_E[j=0] (SFA offset)                     +0            +0           0           0        150
c_0_E[j=1] (SFA offset)                     +0            +0           0           0        150
c_0_E[j=2] (SFA offset)                     +0            +0           0           0        150
a_0 (threshold)                          +0.35         +0.35           0           0        300
readout_weight                     +0.00016801   +0.00093434      0.0673      0.1085       6675
readout_bias                                +0    -0.0040811           0     0.07401         89
ic.ic (per-variant init)              +0.12557      +0.12557        0.42        0.42        900
```

\newpage

# srnn-std-e-only-per-neuron

![Tau evolution](srnn-std-e-only-per-neuron/tau_evolution.png){ width=95% }

![W\_EI evolution](srnn-std-e-only-per-neuron/W_EI_evolution.png){ width=95% }

\newpage

# srnn-std-e-only-per-neuron — parameter table

```text
# rk4-frz-a0c0e-300e — variant srnn-std-e-only-per-neuron (k=4)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                              +1      +0.33416           —           —     scalar
tau_d (s)                                 +0.1    +0.0053372           0    0.002804        300
tau_b_rec_E (s)                             +1       +2.1426           0      0.9039        150
tau_b_rel_E (s)                          +0.25       +0.7232           0       0.321        150
W_eff (E src, signed)                 +0.23267      +0.26155     0.07641      0.1251      15048
W_eff (I src, signed)                 -0.31062      -0.30953     0.07724      0.1024      14977
|W_eff| (E src)                       +0.23267      +0.26155     0.07641      0.1251      15048
|W_eff| (I src)                       +0.31062      +0.30953     0.07724      0.1024      14977
W_in_eff (input neurons)           +0.00019944   -0.00017023     0.09994     0.03078       6675
a_0 (threshold)                          +0.35         +0.35           0           0        300
readout_weight                     +0.00074661    -0.0029953     0.06657      0.3847       6675
readout_bias                                +0     -0.057122           0      0.2424         89
ic.ic (per-variant init)             +0.090198     +0.090198      0.3825      0.3825        900
```

\newpage

# srnn-std-e-only-skip-per-neuron

![Tau evolution](srnn-std-e-only-skip-per-neuron/tau_evolution.png){ width=95% }

![W\_EI evolution](srnn-std-e-only-skip-per-neuron/W_EI_evolution.png){ width=95% }

\newpage

# srnn-std-e-only-skip-per-neuron — parameter table

```text
# rk4-frz-a0c0e-300e — variant srnn-std-e-only-skip-per-neuron (k=5)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                              +1      +0.37143           —           —     scalar
tau_d (s)                                 +0.1    +0.0091822           0    0.004845        300
tau_b_rec_E (s)                             +1      +0.11387           0      0.0255        150
tau_b_rel_E (s)                          +0.25      +0.42433           0      0.1359        150
W_eff (E src, signed)                 +0.23267      +0.16768     0.07641     0.06635      15048
W_eff (I src, signed)                 -0.31062       -0.2127     0.07724     0.06666      14977
|W_eff| (E src)                       +0.23267      +0.16768     0.07641     0.06635      15048
|W_eff| (I src)                       +0.31062       +0.2127     0.07724     0.06666      14977
W_in_eff (input neurons)           +0.00036119   -4.0943e-05     0.09956      0.0164       6675
a_0 (threshold)                          +0.35         +0.35           0           0        300
readout_weight                     +0.00055372    -0.0045908     0.06572      0.1673       6675
readout_bias                                +0     +0.039199           0       0.089         89
ic.ic (per-variant init)             +0.082727     +0.082727      0.3836      0.3836        900
```
