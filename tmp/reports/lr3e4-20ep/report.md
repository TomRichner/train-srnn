# Run: lr3e4-20ep

## Log-log loss curves

![skip variants — log-log loss/metric](log_log_curves_skip.png){ width=95% }

\newpage

# srnn-e-only-skip

![Tau evolution](srnn-e-only-skip/tau_evolution.png){ width=95% }

![W\_EI evolution](srnn-e-only-skip/W_EI_evolution.png){ width=95% }

\newpage

# srnn-e-only-skip — parameter table

```text
# lr3e4-20ep — variant srnn-e-only-skip (k=0)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                              +1      +0.99641           —           —     scalar
tau_d (s)                                 +0.1      +0.10092           0           0        300
tau_a_E (s)                            +4.8777       +4.7793       3.996       3.915        450
tau_a_I (s)                           +0.69315      +0.69066           0           0        450
tau_b_rec_E (s)                             +1       +1.0099           0           0        150
tau_b_rel_E (s)                          +0.25      +0.24552           0           0        150
tau_b_rec_I (s)                             +1      +0.99641           0           0        150
tau_b_rel_I (s)                          +0.25       +0.2491           0           0        150
W_eff (E src, signed)                 +0.23267      +0.22885     0.07641      0.0752      15048
W_eff (I src, signed)                 -0.31062      -0.30636     0.07724      0.0762      14977
|W_eff| (E src)                       +0.23267      +0.22885     0.07641      0.0752      15048
|W_eff| (I src)                       +0.31062      +0.30636     0.07724      0.0762      14977
W_in_eff (input neurons)            -0.0003386   -0.00033955      0.1012     0.09962       6675
c_E (SFA coupling)                       +0.05     +0.050816           0           0        450
c_0_E (SFA offset)                          +0     +0.017052           0           0        450
c_I (SFA coupling)                       +0.05         +0.05           0           0        450
c_0_I (SFA offset)                          +0            +0           0           0        450
a_0 (threshold)                          +0.35      +0.36847           0           0        300
readout_weight                     -0.00041422   +0.00041058     0.06568     0.05901       6675
readout_bias                                +0    +0.0006861           0    0.003725         89
ic.ic (per-variant init)              +0.15932      +0.17082      0.4111      0.4078       1500
```

\newpage

# srnn-e-only-skip-per-neuron

![Tau evolution](srnn-e-only-skip-per-neuron/tau_evolution.png){ width=95% }

![W\_EI evolution](srnn-e-only-skip-per-neuron/W_EI_evolution.png){ width=95% }

\newpage

# srnn-e-only-skip-per-neuron — parameter table

```text
# lr3e4-20ep — variant srnn-e-only-skip-per-neuron (k=1)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                              +1       +0.9987           —           —     scalar
tau_d (s)                                 +0.1      +0.10126           0    0.001252        300
tau_a_E (s)                            +4.8777       +4.8026       3.996       3.936        450
tau_a_I (s)                           +0.69315      +0.69224           0           0        450
tau_b_rec_E (s)                             +1       +1.0141           0    0.002596        150
tau_b_rel_E (s)                          +0.25       +0.2448           0   0.0009132        150
tau_b_rec_I (s)                             +1       +0.9987           0           0        150
tau_b_rel_I (s)                          +0.25      +0.24967           0           0        150
W_eff (E src, signed)                 +0.23267      +0.22939     0.07641     0.07536      15048
W_eff (I src, signed)                 -0.31062      -0.30688     0.07724     0.07631      14977
|W_eff| (E src)                       +0.23267      +0.22939     0.07641     0.07536      15048
|W_eff| (I src)                       +0.31062      +0.30688     0.07724     0.07631      14977
W_in_eff (input neurons)           -0.00084237    -0.0011027      0.1008     0.09777       6675
c_E (SFA coupling)                       +0.05     +0.051195           0   0.0002591        450
c_0_E (SFA offset)                          +0     +0.024719           0    0.004712        450
c_I (SFA coupling)                       +0.05         +0.05           0           0        450
c_0_I (SFA offset)                          +0            +0           0           0        450
a_0 (threshold)                          +0.35       +0.3674           0     0.01287        300
readout_weight                      -0.0016154    -0.0015687     0.06624     0.06157       6675
readout_bias                                +0   +0.00041623           0     0.00353         89
ic.ic (per-variant init)              +0.15938      +0.17378      0.4127      0.4152       1500
```

\newpage

# srnn-e-only-skip-echo

![Tau evolution](srnn-e-only-skip-echo/tau_evolution.png){ width=95% }

![W\_EI evolution](srnn-e-only-skip-echo/W_EI_evolution.png){ width=95% }

\newpage

# srnn-e-only-skip-echo — parameter table

```text
# lr3e4-20ep — variant srnn-e-only-skip-echo (k=2)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                              +1      +0.98764           —           —     scalar
tau_d (s)                                 +0.1      +0.10008           0           0        300
tau_a_E (s)                            +4.8777       +4.7312       3.996       3.876        450
tau_a_I (s)                           +0.69315      +0.68458           0           0        450
tau_b_rec_E (s)                             +1       +1.0024           0           0        150
tau_b_rel_E (s)                          +0.25      +0.24305           0           0        150
tau_b_rec_I (s)                             +1      +0.98764           0           0        150
tau_b_rel_I (s)                          +0.25      +0.24691           0           0        150
W_eff (E src, signed)                 +0.23267       +0.2286     0.07641     0.07508      15048
W_eff (I src, signed)                 -0.31062      -0.30519     0.07724     0.07589      14977
|W_eff| (E src)                       +0.23267       +0.2286     0.07641     0.07508      15048
|W_eff| (I src)                       +0.31062      +0.30519     0.07724     0.07589      14977
W_in_eff (input neurons)           -0.00015382   -0.00019858     0.09934     0.09912       6675
c_E (SFA coupling)                       +0.05     +0.050897           0           0        450
c_0_E (SFA offset)                          +0     +0.018208           0           0        450
c_I (SFA coupling)                       +0.05         +0.05           0           0        450
c_0_I (SFA offset)                          +0            +0           0           0        450
a_0 (threshold)                          +0.35      +0.36933           0           0        300
readout_weight                     -0.00011373    -0.0011971     0.06687     0.06072       6675
readout_bias                                +0   +0.00037265           0    0.005328         89
ic.ic (per-variant init)              +0.15962      +0.17043      0.4124      0.4065       1500
```

\newpage

# srnn-no-dales-skip

![Tau evolution](srnn-no-dales-skip/tau_evolution.png){ width=95% }

![W\_EI evolution](srnn-no-dales-skip/W_EI_evolution.png){ width=95% }

\newpage

# srnn-no-dales-skip — parameter table

```text
# lr3e4-20ep — variant srnn-no-dales-skip (k=3)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                              +1      +0.99736           —           —     scalar
tau_d (s)                                 +0.1      +0.10099           0           0        300
tau_a_E (s)                            +4.8777       +4.8053       3.996       3.936        450
tau_a_I (s)                            +4.8777       +4.8626       3.996       3.983        450
tau_b_rec_E (s)                             +1       +1.0062           0           0        150
tau_b_rel_E (s)                          +0.25      +0.24688           0           0        150
tau_b_rec_I (s)                             +1       +0.9879           0           0        150
tau_b_rel_I (s)                          +0.25      +0.25119           0           0        150
W_eff (E src, signed)                  +0.2326      +0.22939     0.07663     0.07626      15048
W_eff (I src, signed)                 -0.31062      -0.30803     0.07727     0.07678      14977
|W_eff| (E src)                       +0.23267      +0.22946     0.07641     0.07606      15048
|W_eff| (I src)                       +0.31062      +0.30804     0.07724     0.07676      14977
W_in_eff (input neurons)            -0.0011621   -0.00080286     0.09962     0.09653       6675
c_E (SFA coupling)                       +0.05     +0.050473           0           0        450
c_0_E (SFA offset)                          +0     +0.011531           0           0        450
c_I (SFA coupling)                       +0.05     +0.049562           0           0        450
c_0_I (SFA offset)                          +0     -0.010542           0           0        450
a_0 (threshold)                          +0.35      +0.36292           0           0        300
readout_weight                      -0.0016591    -0.0012728     0.06668     0.06284       6675
readout_bias                                +0   +0.00088264           0    0.005516         89
ic.ic (per-variant init)              +0.16747      +0.16768      0.3627      0.3957       1500
```

\newpage

# srnn-no-adapt-no-dales-skip

![Tau evolution](srnn-no-adapt-no-dales-skip/tau_evolution.png){ width=95% }

![W\_EI evolution](srnn-no-adapt-no-dales-skip/W_EI_evolution.png){ width=95% }

\newpage

# srnn-no-adapt-no-dales-skip — parameter table

```text
# lr3e4-20ep — variant srnn-no-adapt-no-dales-skip (k=4)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                              +1       +1.0135           —           —     scalar
tau_d (s)                                 +0.1      +0.10354           0           0        300
tau_a_E (s)                           +0.69315      +0.70252           0           0        450
tau_a_I (s)                           +0.69315      +0.70252           0           0        450
tau_b_rec_E (s)                             +1       +1.0135           0           0        150
tau_b_rel_E (s)                          +0.25      +0.25338           0           0        150
tau_b_rec_I (s)                             +1       +1.0135           0           0        150
tau_b_rel_I (s)                          +0.25      +0.25338           0           0        150
W_eff (E src, signed)                  +0.2326      +0.23018     0.07663     0.07643      15048
W_eff (I src, signed)                 -0.31062      -0.30855     0.07727       0.077      14977
|W_eff| (E src)                       +0.23267      +0.23024     0.07641     0.07624      15048
|W_eff| (I src)                       +0.31062      +0.30856     0.07724     0.07698      14977
W_in_eff (input neurons)           +0.00019944   +0.00032314     0.09994     0.09709       6675
c_E (SFA coupling)                       +0.05         +0.05           0           0        450
c_0_E (SFA offset)                          +0            +0           0           0        450
c_I (SFA coupling)                       +0.05         +0.05           0           0        450
c_0_I (SFA offset)                          +0            +0           0           0        450
a_0 (threshold)                          +0.35      +0.36142           0           0        300
readout_weight                     +0.00079717    +0.0005836     0.06573     0.06295       6675
readout_bias                                +0   -0.00017099           0    0.006122         89
ic.ic (per-variant init)              +0.14435      +0.15979      0.4566      0.4303       1500
```

\newpage

# srnn-skip

![Tau evolution](srnn-skip/tau_evolution.png){ width=95% }

![W\_EI evolution](srnn-skip/W_EI_evolution.png){ width=95% }

\newpage

# srnn-skip — parameter table

```text
# lr3e4-20ep — variant srnn-skip (k=5)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                              +1        +1.005           —           —     scalar
tau_d (s)                                 +0.1      +0.10209           0           0        300
tau_a_E (s)                            +4.8777       +4.8194       3.996       3.948        450
tau_a_I (s)                            +4.8777       +4.9015       3.996       4.015        450
tau_b_rec_E (s)                             +1        +1.018           0           0        150
tau_b_rel_E (s)                          +0.25      +0.24775           0           0        150
tau_b_rec_I (s)                             +1      +0.99014           0           0        150
tau_b_rel_I (s)                          +0.25       +0.2542           0           0        150
W_eff (E src, signed)                 +0.23267      +0.22861     0.07641     0.07512      15048
W_eff (I src, signed)                 -0.31062      -0.30589     0.07724     0.07609      14977
|W_eff| (E src)                       +0.23267      +0.22861     0.07641     0.07512      15048
|W_eff| (I src)                       +0.31062      +0.30589     0.07724     0.07609      14977
W_in_eff (input neurons)            +0.0005819   +0.00038774     0.09954     0.09694       6675
c_E (SFA coupling)                       +0.05     +0.050741           0           0        450
c_0_E (SFA offset)                          +0     +0.016466           0           0        450
c_I (SFA coupling)                       +0.05     +0.049347           0           0        450
c_0_I (SFA offset)                          +0     -0.015484           0           0        450
a_0 (threshold)                          +0.35      +0.36768           0           0        300
readout_weight                     -0.00025143   -7.6565e-05     0.06686     0.06216       6675
readout_bias                                +0    +0.0011882           0    0.007026         89
ic.ic (per-variant init)              +0.17068       +0.1695      0.3591      0.3823       1500
```

\newpage

# srnn-sfa-e-only-skip

![Tau evolution](srnn-sfa-e-only-skip/tau_evolution.png){ width=95% }

![W\_EI evolution](srnn-sfa-e-only-skip/W_EI_evolution.png){ width=95% }

\newpage

# srnn-sfa-e-only-skip — parameter table

```text
# lr3e4-20ep — variant srnn-sfa-e-only-skip (k=6)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                              +1       +1.0113           —           —     scalar
tau_d (s)                                 +0.1      +0.10289           0           0        300
tau_a_E (s)                            +4.8777       +4.8623       3.996       3.983        450
tau_a_I (s)                           +0.69315        +0.701           0           0        450
tau_b_rec_E (s)                             +1       +1.0113           0           0        150
tau_b_rel_E (s)                          +0.25      +0.25283           0           0        150
tau_b_rec_I (s)                             +1       +1.0113           0           0        150
tau_b_rel_I (s)                          +0.25      +0.25283           0           0        150
W_eff (E src, signed)                 +0.23267      +0.22953     0.07641     0.07543      15048
W_eff (I src, signed)                 -0.31062      -0.30688     0.07724     0.07632      14977
|W_eff| (E src)                       +0.23267      +0.22953     0.07641     0.07543      15048
|W_eff| (I src)                       +0.31062      +0.30688     0.07724     0.07632      14977
W_in_eff (input neurons)           -0.00083129   -0.00065163     0.09974     0.09748       6675
c_E (SFA coupling)                       +0.05     +0.050589           0           0        450
c_0_E (SFA offset)                          +0     +0.014862           0           0        450
c_I (SFA coupling)                       +0.05         +0.05           0           0        450
c_0_I (SFA offset)                          +0            +0           0           0        450
a_0 (threshold)                          +0.35      +0.36598           0           0        300
readout_weight                       +0.001119   +0.00010576     0.06703     0.06326       6675
readout_bias                                +0    -0.0010375           0    0.007758         89
ic.ic (per-variant init)              +0.16079      +0.17166      0.4384      0.4265       1500
```

\newpage

# srnn-std-e-only-skip

![Tau evolution](srnn-std-e-only-skip/tau_evolution.png){ width=95% }

![W\_EI evolution](srnn-std-e-only-skip/W_EI_evolution.png){ width=95% }

\newpage

# srnn-std-e-only-skip — parameter table

```text
# lr3e4-20ep — variant srnn-std-e-only-skip (k=7)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                              +1      +0.99784           —           —     scalar
tau_d (s)                                 +0.1      +0.10125           0           0        300
tau_a_E (s)                           +0.69315      +0.69165           0           0        450
tau_a_I (s)                           +0.69315      +0.69165           0           0        450
tau_b_rec_E (s)                             +1       +1.0113           0           0        150
tau_b_rel_E (s)                          +0.25      +0.24593           0           0        150
tau_b_rec_I (s)                             +1      +0.99784           0           0        150
tau_b_rel_I (s)                          +0.25      +0.24946           0           0        150
W_eff (E src, signed)                 +0.23267      +0.22874     0.07641     0.07516      15048
W_eff (I src, signed)                 -0.31062      -0.30623     0.07724     0.07618      14977
|W_eff| (E src)                       +0.23267      +0.22874     0.07641     0.07516      15048
|W_eff| (I src)                       +0.31062      +0.30623     0.07724     0.07618      14977
W_in_eff (input neurons)            +0.0012099    +0.0015692     0.09923      0.0971       6675
c_E (SFA coupling)                       +0.05         +0.05           0           0        450
c_0_E (SFA offset)                          +0            +0           0           0        450
c_I (SFA coupling)                       +0.05         +0.05           0           0        450
c_0_I (SFA offset)                          +0            +0           0           0        450
a_0 (threshold)                          +0.35       +0.3683           0           0        300
readout_weight                     +0.00034909   +0.00053166     0.06671     0.06064       6675
readout_bias                                +0    -0.0004303           0    0.005114         89
ic.ic (per-variant init)              +0.15334      +0.15985      0.4128      0.4103       1500
```
