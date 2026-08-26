# Run: frz150-a0c0e-cl

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
# frz150-a0c0e-cl — variant srnn-e-only-per-neuron (k=0)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                              +1      +0.34039           —           —     scalar
tau_d (s)                                 +0.1     +0.007162           0    0.003174        300
tau_a_E[j=0] (s)                         +0.25     +0.016864           0    0.005739        150
tau_a_E[j=1] (s)                       +4.3832      +0.43233           0     0.02709        150
tau_a_E[j=2] (s)                           +10      +0.96551           0     0.02078        150
tau_b_rec_E (s)                             +1      +0.66645           0      0.1162        150
tau_b_rel_E (s)                          +0.25      +0.39631           0     0.08465        150
W_eff (E src, signed)                 +0.23267      +0.23196     0.07641      0.0902      15048
W_eff (I src, signed)                 -0.31062      -0.29324     0.07724     0.08764      14977
|W_eff| (E src)                       +0.23267      +0.23196     0.07641      0.0902      15048
|W_eff| (I src)                       +0.31062      +0.29324     0.07724     0.08764      14977
W_in_eff (input neurons)            -0.0003386    +0.0012009      0.1012     0.06848       6675
c_E[j=0] (SFA coupling)                  +0.05     +0.026238           0      0.0128        150
c_E[j=1] (SFA coupling)                  +0.05     +0.018407           0    0.004744        150
c_E[j=2] (SFA coupling)                  +0.05     +0.018962           0    0.004853        150
c_0_E[j=0] (SFA offset)                     +0            +0           0           0        150
c_0_E[j=1] (SFA offset)                     +0            +0           0           0        150
c_0_E[j=2] (SFA offset)                     +0            +0           0           0        150
a_0 (threshold)                          +0.35         +0.35           0           0        300
readout_weight                     -2.8854e-05    -0.0023459     0.06604      0.2778       6675
readout_bias                                +0     -0.027527           0      0.1935         89
ic.ic (per-variant init)              +0.10386      +0.10386      0.3828      0.3828        900
```

\newpage

# srnn-e-only-skip-per-neuron

![Tau evolution](srnn-e-only-skip-per-neuron/tau_evolution.png){ width=95% }

![W\_EI evolution](srnn-e-only-skip-per-neuron/W_EI_evolution.png){ width=95% }

\newpage

# srnn-e-only-skip-per-neuron — parameter table

```text
# frz150-a0c0e-cl — variant srnn-e-only-skip-per-neuron (k=1)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                              +1      +0.37475           —           —     scalar
tau_d (s)                                 +0.1    +0.0093965           0    0.004008        300
tau_a_E[j=0] (s)                         +0.25     +0.018205           0      0.0071        150
tau_a_E[j=1] (s)                       +4.3832      +0.50243           0     0.01512        150
tau_a_E[j=2] (s)                           +10       +1.1475           0    0.006022        150
tau_b_rec_E (s)                             +1       +0.1144           0      0.0215        150
tau_b_rel_E (s)                          +0.25       +0.2291           0     0.05166        150
W_eff (E src, signed)                 +0.23267      +0.16102     0.07641     0.06004      15048
W_eff (I src, signed)                 -0.31062      -0.20807     0.07724     0.06089      14977
|W_eff| (E src)                       +0.23267      +0.16102     0.07641     0.06004      15048
|W_eff| (I src)                       +0.31062      +0.20807     0.07724     0.06089      14977
W_in_eff (input neurons)           -0.00084237   -0.00021733      0.1008     0.02974       6675
c_E[j=0] (SFA coupling)                  +0.05     +0.064445           0     0.02727        150
c_E[j=1] (SFA coupling)                  +0.05     +0.038848           0    0.009048        150
c_E[j=2] (SFA coupling)                  +0.05     +0.039301           0     0.00884        150
c_0_E[j=0] (SFA offset)                     +0            +0           0           0        150
c_0_E[j=1] (SFA offset)                     +0            +0           0           0        150
c_0_E[j=2] (SFA offset)                     +0            +0           0           0        150
a_0 (threshold)                          +0.35         +0.35           0           0        300
readout_weight                     +0.00078219    -0.0041568     0.06654      0.1669       6675
readout_bias                                +0     +0.075051           0     0.09914         89
ic.ic (per-variant init)              +0.10327      +0.10327      0.3829      0.3829        900
```

\newpage

# srnn-sfa-e-only-per-neuron

![Tau evolution](srnn-sfa-e-only-per-neuron/tau_evolution.png){ width=95% }

![W\_EI evolution](srnn-sfa-e-only-per-neuron/W_EI_evolution.png){ width=95% }

\newpage

# srnn-sfa-e-only-per-neuron — parameter table

```text
# frz150-a0c0e-cl — variant srnn-sfa-e-only-per-neuron (k=2)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                              +1      +0.31358           —           —     scalar
tau_d (s)                                 +0.1    +0.0061526           0    0.002533        300
tau_a_E[j=0] (s)                         +0.25      +0.01798           0    0.005394        150
tau_a_E[j=1] (s)                       +4.3832       +0.4452           0     0.02597        150
tau_a_E[j=2] (s)                           +10       +1.0277           0      0.0222        150
tau_b_rec_E (s)                             +1      +0.31358           0           0        150
tau_b_rel_E (s)                          +0.25     +0.078396           0           0        150
W_eff (E src, signed)                 +0.23267      +0.13831     0.07641     0.04941      15048
W_eff (I src, signed)                 -0.31062      -0.18211     0.07724     0.05093      14977
|W_eff| (E src)                       +0.23267      +0.13831     0.07641     0.04941      15048
|W_eff| (I src)                       +0.31062      +0.18211     0.07724     0.05093      14977
W_in_eff (input neurons)           -0.00015382    -0.0014175     0.09934     0.05862       6675
c_E[j=0] (SFA coupling)                  +0.05     +0.059952           0     0.01667        150
c_E[j=1] (SFA coupling)                  +0.05     +0.044852           0    0.007163        150
c_E[j=2] (SFA coupling)                  +0.05     +0.044457           0    0.007396        150
c_0_E[j=0] (SFA offset)                     +0            +0           0           0        150
c_0_E[j=1] (SFA offset)                     +0            +0           0           0        150
c_0_E[j=2] (SFA offset)                     +0            +0           0           0        150
a_0 (threshold)                          +0.35         +0.35           0           0        300
readout_weight                     -0.00098972    +0.0075209     0.06678      0.2081       6675
readout_bias                                +0      -0.06755           0      0.1037         89
ic.ic (per-variant init)              +0.13078      +0.13078      0.4172      0.4172        900
```

\newpage

# srnn-sfa-e-only-skip-per-neuron

![Tau evolution](srnn-sfa-e-only-skip-per-neuron/tau_evolution.png){ width=95% }

![W\_EI evolution](srnn-sfa-e-only-skip-per-neuron/W_EI_evolution.png){ width=95% }

\newpage

# srnn-sfa-e-only-skip-per-neuron — parameter table

```text
# frz150-a0c0e-cl — variant srnn-sfa-e-only-skip-per-neuron (k=3)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                              +1      +0.88066           —           —     scalar
tau_d (s)                                 +0.1      +0.07556           0    0.004313        300
tau_a_E[j=0] (s)                         +0.25      +0.20439           0     0.01152        150
tau_a_E[j=1] (s)                       +4.3832       +3.6736           0     0.03808        150
tau_a_E[j=2] (s)                           +10       +8.4115           0     0.01464        150
tau_b_rec_E (s)                             +1      +0.88066           0           0        150
tau_b_rel_E (s)                          +0.25      +0.22016           0           0        150
W_eff (E src, signed)                 +0.23267      +0.22712     0.07641     0.07492      15048
W_eff (I src, signed)                 -0.31062      -0.30339     0.07724      0.0758      14977
|W_eff| (E src)                       +0.23267      +0.22712     0.07641     0.07492      15048
|W_eff| (I src)                       +0.31062      +0.30339     0.07724      0.0758      14977
W_in_eff (input neurons)            -0.0011621   -0.00093949     0.09962     0.09398       6675
c_E[j=0] (SFA coupling)                  +0.05     +0.052804           0     0.00683        150
c_E[j=1] (SFA coupling)                  +0.05      +0.05201           0   0.0008557        150
c_E[j=2] (SFA coupling)                  +0.05     +0.051995           0    0.001086        150
c_0_E[j=0] (SFA offset)                     +0            +0           0           0        150
c_0_E[j=1] (SFA offset)                     +0            +0           0           0        150
c_0_E[j=2] (SFA offset)                     +0            +0           0           0        150
a_0 (threshold)                          +0.35         +0.35           0           0        300
readout_weight                     +0.00016801    -0.0043514      0.0673     0.06344       6675
readout_bias                                +0    +0.0042362           0    0.008181         89
ic.ic (per-variant init)              +0.09231      +0.09231      0.5294      0.5294        900
```

\newpage

# srnn-std-e-only-per-neuron

![Tau evolution](srnn-std-e-only-per-neuron/tau_evolution.png){ width=95% }

![W\_EI evolution](srnn-std-e-only-per-neuron/W_EI_evolution.png){ width=95% }

\newpage

# srnn-std-e-only-per-neuron — parameter table

```text
# frz150-a0c0e-cl — variant srnn-std-e-only-per-neuron (k=4)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                              +1      +0.47323           —           —     scalar
tau_d (s)                                 +0.1     +0.013463           0     0.00467        300
tau_b_rec_E (s)                             +1      +0.72903           0      0.1019        150
tau_b_rel_E (s)                          +0.25      +0.44383           0     0.07382        150
W_eff (E src, signed)                 +0.23267      +0.22068     0.07641     0.07992      15048
W_eff (I src, signed)                 -0.31062      -0.28579     0.07724     0.08107      14977
|W_eff| (E src)                       +0.23267      +0.22068     0.07641     0.07992      15048
|W_eff| (I src)                       +0.31062      +0.28579     0.07724     0.08107      14977
W_in_eff (input neurons)           +0.00019944    -0.0012099     0.09994     0.09731       6675
a_0 (threshold)                          +0.35         +0.35           0           0        300
readout_weight                     +0.00074661    +0.0073141     0.06657      0.2248       6675
readout_bias                                +0     -0.050919           0      0.2471         89
ic.ic (per-variant init)             +0.090819     +0.090819      0.3835      0.3835        900
```

\newpage

# srnn-std-e-only-skip-per-neuron

![Tau evolution](srnn-std-e-only-skip-per-neuron/tau_evolution.png){ width=95% }

![W\_EI evolution](srnn-std-e-only-skip-per-neuron/W_EI_evolution.png){ width=95% }

\newpage

# srnn-std-e-only-skip-per-neuron — parameter table

```text
# frz150-a0c0e-cl — variant srnn-std-e-only-skip-per-neuron (k=5)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                              +1      +0.37754           —           —     scalar
tau_d (s)                                 +0.1    +0.0096444           0    0.003793        300
tau_b_rec_E (s)                             +1      +0.12067           0     0.02221        150
tau_b_rel_E (s)                          +0.25      +0.21729           0     0.04691        150
W_eff (E src, signed)                 +0.23267      +0.16627     0.07641     0.06157      15048
W_eff (I src, signed)                 -0.31062       -0.2153     0.07724     0.06187      14977
|W_eff| (E src)                       +0.23267      +0.16627     0.07641     0.06157      15048
|W_eff| (I src)                       +0.31062       +0.2153     0.07724     0.06187      14977
W_in_eff (input neurons)           +0.00036119   +0.00041135     0.09956     0.03142       6675
a_0 (threshold)                          +0.35         +0.35           0           0        300
readout_weight                     +0.00055372    -0.0028452     0.06572      0.1506       6675
readout_bias                                +0    +0.0040465           0     0.06836         89
ic.ic (per-variant init)             +0.083543     +0.083543       0.384       0.384        900
```
