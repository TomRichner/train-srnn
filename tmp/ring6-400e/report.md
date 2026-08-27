# Run: ring6-400e

## Log-log loss curves

![skip variants — log-log loss/metric](log_log_curves_skip.png){ width=95% }

![no-skip variants — log-log loss/metric](log_log_curves_no-skip.png){ width=95% }

\newpage

# srnn-no-adapt-no-dales

![Tau evolution](srnn-no-adapt-no-dales/tau_evolution.png){ width=95% }

![W\_EI evolution](srnn-no-adapt-no-dales/W_EI_evolution.png){ width=95% }

![W\_io evolution](srnn-no-adapt-no-dales/W_io_evolution.png){ width=95% }

![Offsets and threshold](srnn-no-adapt-no-dales/offsets_evolution.png){ width=95% }

\newpage

# srnn-no-adapt-no-dales — parameter table

```text
# ring6-400e — variant srnn-no-adapt-no-dales (k=0)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                              +1      +0.67424           —           —     scalar
tau_d (s)                                 +0.1     +0.038251           0           0        300
tau_b_rec_E (s)                             +1      +0.67424           0           0        150
tau_b_rel_E (s)                          +0.25      +0.16856           0           0        150
tau_b_rec_I (s)                             +1      +0.67424           0           0        150
tau_b_rel_I (s)                          +0.25      +0.16856           0           0        150
W_eff (E src, signed)                  +0.2326      +0.23755     0.07663      0.1367      15048
W_eff (I src, signed)                 -0.31062      -0.31561     0.07727      0.1617      14977
|W_eff| (E src)                       +0.23267       +0.2471     0.07641      0.1186      15048
|W_eff| (I src)                       +0.31062      +0.32459     0.07724      0.1429      14977
W_in_eff (input neurons)            +0.0038159    -0.0053236      0.1019      0.2185       1275
a_0 (threshold)                          +0.35       +0.3017           0           0        300
W_out_eff (readout × gain)          +0.0011253     -0.010918     0.06572      0.3555       1275
readout_bias                                +0     +0.063545           0      0.1212         17
ic.ic (per-variant init)              +0.13192      +0.13192      0.5265      0.5265       1500
```

\newpage

# srnn-no-adapt

![Tau evolution](srnn-no-adapt/tau_evolution.png){ width=95% }

![W\_EI evolution](srnn-no-adapt/W_EI_evolution.png){ width=95% }

![W\_io evolution](srnn-no-adapt/W_io_evolution.png){ width=95% }

![Offsets and threshold](srnn-no-adapt/offsets_evolution.png){ width=95% }

\newpage

# srnn-no-adapt — parameter table

```text
# ring6-400e — variant srnn-no-adapt (k=1)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                              +1      +0.64079           —           —     scalar
tau_d (s)                                 +0.1     +0.034076           0           0        300
tau_b_rec_E (s)                             +1      +0.64079           0           0        150
tau_b_rel_E (s)                          +0.25       +0.1602           0           0        150
tau_b_rec_I (s)                             +1      +0.64079           0           0        150
tau_b_rel_I (s)                          +0.25       +0.1602           0           0        150
W_eff (E src, signed)                 +0.23267      +0.21199     0.07641     0.08059      15048
W_eff (I src, signed)                 -0.31062      -0.28378     0.07724     0.09891      14977
|W_eff| (E src)                       +0.23267      +0.21199     0.07641     0.08059      15048
|W_eff| (I src)                       +0.31062      +0.28378     0.07724     0.09891      14977
W_in_eff (input neurons)           +0.00075891    -0.0015403      0.1012      0.2295       1275
a_0 (threshold)                          +0.35      +0.27806           0           0        300
W_out_eff (readout × gain)          +0.0034456    -0.0094228     0.06605      0.4501       1275
readout_bias                                +0     +0.013928           0      0.2061         17
ic.ic (per-variant init)              +0.13096      +0.13096      0.4725      0.4725       1500
```

\newpage

# srnn-no-dales-skip

![Tau evolution](srnn-no-dales-skip/tau_evolution.png){ width=95% }

![W\_EI evolution](srnn-no-dales-skip/W_EI_evolution.png){ width=95% }

![W\_io evolution](srnn-no-dales-skip/W_io_evolution.png){ width=95% }

![Offsets and threshold](srnn-no-dales-skip/offsets_evolution.png){ width=95% }

\newpage

# srnn-no-dales-skip — parameter table

```text
# ring6-400e — variant srnn-no-dales-skip (k=2)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                              +1      +0.71013           —           —     scalar
tau_d (s)                                 +0.1     +0.043403           0           0        300
tau_a_E[j=0] (s)                         +0.25      +0.10724           0           0        150
tau_a_E[j=1] (s)                       +1.5896      +0.68185           0           0        150
tau_a_E[j=2] (s)                            +4       +1.7158           0           0        150
tau_a_I[j=0] (s)                         +0.25      +0.15645           0           0        150
tau_a_I[j=1] (s)                       +1.5896      +0.99477           0           0        150
tau_a_I[j=2] (s)                            +4       +2.5032           0           0        150
tau_b_rec_E (s)                             +1      +0.33078           0           0        150
tau_b_rel_E (s)                          +0.25      +0.32892           0           0        150
tau_b_rec_I (s)                             +1      +0.26886           0           0        150
tau_b_rel_I (s)                          +0.25      +0.31483           0           0        150
W_eff (E src, signed)                  +0.2326      +0.38439     0.07663      0.2588      15048
W_eff (I src, signed)                 -0.31062      -0.52445     0.07727      0.2847      14977
|W_eff| (E src)                       +0.23267      +0.40465     0.07641      0.2258      15048
|W_eff| (I src)                       +0.31062      +0.53241     0.07724      0.2695      14977
W_in_eff (input neurons)           -0.00021194     +0.017065     0.09952      0.2314       1275
c_E[j=0] (SFA coupling)                  +0.05     +0.041108           0           0        150
c_E[j=1] (SFA coupling)                  +0.05     +0.041108           0           0        150
c_E[j=2] (SFA coupling)                  +0.05     +0.041108           0           0        150
c_0_E[j=0] (SFA offset)                     +0     -0.086949           0           0        150
c_0_E[j=1] (SFA offset)                     +0     -0.086949           0           0        150
c_0_E[j=2] (SFA offset)                     +0     -0.086949           0           0        150
c_I[j=0] (SFA coupling)                  +0.05      +0.04975           0           0        150
c_I[j=1] (SFA coupling)                  +0.05      +0.04975           0           0        150
c_I[j=2] (SFA coupling)                  +0.05      +0.04975           0           0        150
c_0_I[j=0] (SFA offset)                     +0     +0.070548           0           0        150
c_0_I[j=1] (SFA offset)                     +0     +0.070548           0           0        150
c_0_I[j=2] (SFA offset)                     +0     +0.070548           0           0        150
a_0 (threshold)                          +0.35      +0.27168           0           0        300
W_out_eff (readout × gain)         -0.00031521   -8.2305e-05     0.06595      0.4076       1275
readout_bias                                +0    +0.0077873           0      0.0891         17
ic.ic (per-variant init)              +0.17587      +0.17587      0.3587      0.3587       1500
```

\newpage

# srnn-no-adapt-no-dales-skip

![Tau evolution](srnn-no-adapt-no-dales-skip/tau_evolution.png){ width=95% }

![W\_EI evolution](srnn-no-adapt-no-dales-skip/W_EI_evolution.png){ width=95% }

![W\_io evolution](srnn-no-adapt-no-dales-skip/W_io_evolution.png){ width=95% }

![Offsets and threshold](srnn-no-adapt-no-dales-skip/offsets_evolution.png){ width=95% }

\newpage

# srnn-no-adapt-no-dales-skip — parameter table

```text
# ring6-400e — variant srnn-no-adapt-no-dales-skip (k=3)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                              +1      +0.64821           —           —     scalar
tau_d (s)                                 +0.1     +0.034793           0           0        300
tau_b_rec_E (s)                             +1      +0.64821           0           0        150
tau_b_rel_E (s)                          +0.25      +0.16205           0           0        150
tau_b_rec_I (s)                             +1      +0.64821           0           0        150
tau_b_rel_I (s)                          +0.25      +0.16205           0           0        150
W_eff (E src, signed)                  +0.2326      +0.33374     0.07663      0.1849      15048
W_eff (I src, signed)                 -0.31062      -0.45023     0.07727      0.2228      14977
|W_eff| (E src)                       +0.23267      +0.34538     0.07641      0.1622      15048
|W_eff| (I src)                       +0.31062      +0.45904     0.07724      0.2041      14977
W_in_eff (input neurons)            +0.0028809      +0.00338      0.1007      0.1897       1275
a_0 (threshold)                          +0.35       +0.3357           0           0        300
W_out_eff (readout × gain)          +0.0003819    +0.0081736      0.0651      0.3561       1275
readout_bias                                +0        -0.031           0      0.1233         17
ic.ic (per-variant init)              +0.13418      +0.13418      0.4813      0.4813       1500
```

\newpage

# srnn

![Tau evolution](srnn/tau_evolution.png){ width=95% }

![W\_EI evolution](srnn/W_EI_evolution.png){ width=95% }

![W\_io evolution](srnn/W_io_evolution.png){ width=95% }

![Offsets and threshold](srnn/offsets_evolution.png){ width=95% }

\newpage

# srnn — parameter table

```text
# ring6-400e — variant srnn (k=4)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                              +1      +0.63703           —           —     scalar
tau_d (s)                                 +0.1     +0.033836           0           0        300
tau_a_E[j=0] (s)                         +0.25     +0.044977           0           0        150
tau_a_E[j=1] (s)                       +1.5896      +0.28598           0           0        150
tau_a_E[j=2] (s)                            +4      +0.71963           0           0        150
tau_a_I[j=0] (s)                         +0.25     +0.049557           0           0        150
tau_a_I[j=1] (s)                       +1.5896       +0.3151           0           0        150
tau_a_I[j=2] (s)                            +4      +0.79291           0           0        150
tau_b_rec_E (s)                             +1      +0.25157           0           0        150
tau_b_rel_E (s)                          +0.25      +0.40665           0           0        150
tau_b_rec_I (s)                             +1      +0.21045           0           0        150
tau_b_rel_I (s)                          +0.25      +0.40069           0           0        150
W_eff (E src, signed)                 +0.23267      +0.26156     0.07641      0.1139      15048
W_eff (I src, signed)                 -0.31062       -0.3557     0.07724      0.1482      14977
|W_eff| (E src)                       +0.23267      +0.26156     0.07641      0.1139      15048
|W_eff| (I src)                       +0.31062       +0.3557     0.07724      0.1482      14977
W_in_eff (input neurons)             +0.004538     +0.030117     0.09841      0.2398       1275
c_E[j=0] (SFA coupling)                  +0.05     +0.020863           0           0        150
c_E[j=1] (SFA coupling)                  +0.05     +0.020863           0           0        150
c_E[j=2] (SFA coupling)                  +0.05     +0.020863           0           0        150
c_0_E[j=0] (SFA offset)                     +0      -0.22889           0           0        150
c_0_E[j=1] (SFA offset)                     +0      -0.22889           0           0        150
c_0_E[j=2] (SFA offset)                     +0      -0.22889           0           0        150
c_I[j=0] (SFA coupling)                  +0.05     +0.038609           0           0        150
c_I[j=1] (SFA coupling)                  +0.05     +0.038609           0           0        150
c_I[j=2] (SFA coupling)                  +0.05     +0.038609           0           0        150
c_0_I[j=0] (SFA offset)                     +0      +0.21352           0           0        150
c_0_I[j=1] (SFA offset)                     +0      +0.21352           0           0        150
c_0_I[j=2] (SFA offset)                     +0      +0.21352           0           0        150
a_0 (threshold)                          +0.35      +0.18983           0           0        300
W_out_eff (readout × gain)           -0.001299     -0.018359       0.067       0.648       1275
readout_bias                                +0     +0.050641           0      0.1973         17
ic.ic (per-variant init)               +0.1766       +0.1766      0.3557      0.3557       1500
```

\newpage

# srnn-skip

![Tau evolution](srnn-skip/tau_evolution.png){ width=95% }

![W\_EI evolution](srnn-skip/W_EI_evolution.png){ width=95% }

![W\_io evolution](srnn-skip/W_io_evolution.png){ width=95% }

![Offsets and threshold](srnn-skip/offsets_evolution.png){ width=95% }

\newpage

# srnn-skip — parameter table

```text
# ring6-400e — variant srnn-skip (k=5)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                              +1      +0.67934           —           —     scalar
tau_d (s)                                 +0.1     +0.039279           0           0        300
tau_a_E[j=0] (s)                         +0.25       +0.0314           0           0        150
tau_a_E[j=1] (s)                       +1.5896      +0.19965           0           0        150
tau_a_E[j=2] (s)                            +4      +0.50241           0           0        150
tau_a_I[j=0] (s)                         +0.25      +0.12391           0           0        150
tau_a_I[j=1] (s)                       +1.5896      +0.78789           0           0        150
tau_a_I[j=2] (s)                            +4       +1.9826           0           0        150
tau_b_rec_E (s)                             +1      +0.40795           0           0        150
tau_b_rel_E (s)                          +0.25      +0.26933           0           0        150
tau_b_rec_I (s)                             +1      +0.43798           0           0        150
tau_b_rel_I (s)                          +0.25      +0.24543           0           0        150
W_eff (E src, signed)                 +0.23267      +0.37937     0.07641      0.1551      15048
W_eff (I src, signed)                 -0.31062      -0.52007     0.07724      0.1957      14977
|W_eff| (E src)                       +0.23267      +0.37937     0.07641      0.1551      15048
|W_eff| (I src)                       +0.31062      +0.52007     0.07724      0.1957      14977
W_in_eff (input neurons)            -0.0036791     -0.015529     0.09946      0.2645       1275
c_E[j=0] (SFA coupling)                  +0.05     +0.051349           0           0        150
c_E[j=1] (SFA coupling)                  +0.05     +0.051349           0           0        150
c_E[j=2] (SFA coupling)                  +0.05     +0.051349           0           0        150
c_0_E[j=0] (SFA offset)                     +0      -0.10566           0           0        150
c_0_E[j=1] (SFA offset)                     +0      -0.10566           0           0        150
c_0_E[j=2] (SFA offset)                     +0      -0.10566           0           0        150
c_I[j=0] (SFA coupling)                  +0.05     +0.053623           0           0        150
c_I[j=1] (SFA coupling)                  +0.05     +0.053623           0           0        150
c_I[j=2] (SFA coupling)                  +0.05     +0.053623           0           0        150
c_0_I[j=0] (SFA offset)                     +0      +0.11367           0           0        150
c_0_I[j=1] (SFA offset)                     +0      +0.11367           0           0        150
c_0_I[j=2] (SFA offset)                     +0      +0.11367           0           0        150
a_0 (threshold)                          +0.35      +0.30654           0           0        300
W_out_eff (readout × gain)         +0.00099875   -0.00040111     0.06582       0.666       1275
readout_bias                                +0     -0.011576           0     0.08749         17
ic.ic (per-variant init)              +0.17477      +0.17477      0.3568      0.3568       1500
```
