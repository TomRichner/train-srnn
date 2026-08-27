# Run: ring2x5-100e

## Log-log loss curves

![skip variants — log-log loss/metric](log_log_curves_skip.png){ width=95% }

\newpage

# srnn-no-dales-skip-seed1

![Tau evolution](srnn-no-dales-skip-seed1/tau_evolution.png){ width=95% }

![W\_EI evolution](srnn-no-dales-skip-seed1/W_EI_evolution.png){ width=95% }

![W\_io evolution](srnn-no-dales-skip-seed1/W_io_evolution.png){ width=95% }

![Offsets and threshold](srnn-no-dales-skip-seed1/offsets_evolution.png){ width=95% }

\newpage

# srnn-no-dales-skip-seed1 — parameter table

```text
# ring2x5-100e — variant srnn-no-dales-skip-seed1 (k=0)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                              +1      +0.78257           —           —     scalar
tau_d (s)                                 +0.1     +0.053436           0           0        300
tau_a_E[j=0] (s)                         +0.25      +0.16251           0           0        150
tau_a_E[j=1] (s)                       +1.5896       +1.0333           0           0        150
tau_a_E[j=2] (s)                            +4       +2.6002           0           0        150
tau_a_I[j=0] (s)                         +0.25       +0.2032           0           0        150
tau_a_I[j=1] (s)                       +1.5896        +1.292           0           0        150
tau_a_I[j=2] (s)                            +4       +3.2512           0           0        150
tau_b_rec_E (s)                             +1      +0.52569           0           0        150
tau_b_rel_E (s)                          +0.25      +0.30345           0           0        150
tau_b_rec_I (s)                             +1      +0.54329           0           0        150
tau_b_rel_I (s)                          +0.25      +0.27843           0           0        150
W_eff (E src, signed)                  +0.2326       +0.3696     0.07663      0.1988      15048
W_eff (I src, signed)                 -0.31062      -0.50436     0.07727       0.215      14977
|W_eff| (E src)                       +0.23267      +0.37571     0.07641       0.187      15048
|W_eff| (I src)                       +0.31062      +0.50589     0.07724      0.2114      14977
W_in_eff (input neurons)            +0.0038159    +0.0079425      0.1019      0.1988       1275
c_E[j=0] (SFA coupling)                  +0.05     +0.046151           0           0        150
c_E[j=1] (SFA coupling)                  +0.05     +0.046151           0           0        150
c_E[j=2] (SFA coupling)                  +0.05     +0.046151           0           0        150
c_0_E[j=0] (SFA offset)                     +0      -0.10598           0           0        150
c_0_E[j=1] (SFA offset)                     +0      -0.10598           0           0        150
c_0_E[j=2] (SFA offset)                     +0      -0.10598           0           0        150
c_I[j=0] (SFA coupling)                  +0.05     +0.052562           0           0        150
c_I[j=1] (SFA coupling)                  +0.05     +0.052562           0           0        150
c_I[j=2] (SFA coupling)                  +0.05     +0.052562           0           0        150
c_0_I[j=0] (SFA offset)                     +0      +0.10767           0           0        150
c_0_I[j=1] (SFA offset)                     +0      +0.10767           0           0        150
c_0_I[j=2] (SFA offset)                     +0      +0.10767           0           0        150
a_0 (threshold)                          +0.35      +0.32466           0           0        300
W_out_eff (readout × gain)         -0.00028065    -0.0018449     0.06573      0.3433       1275
readout_bias                                +0     +0.018247           0     0.09854         17
ic.ic (per-variant init)              +0.17657      +0.17657      0.3565      0.3565       1500
```

\newpage

# srnn-no-dales-skip-seed2

![Tau evolution](srnn-no-dales-skip-seed2/tau_evolution.png){ width=95% }

![W\_EI evolution](srnn-no-dales-skip-seed2/W_EI_evolution.png){ width=95% }

![W\_io evolution](srnn-no-dales-skip-seed2/W_io_evolution.png){ width=95% }

![Offsets and threshold](srnn-no-dales-skip-seed2/offsets_evolution.png){ width=95% }

\newpage

# srnn-no-dales-skip-seed2 — parameter table

```text
# ring2x5-100e — variant srnn-no-dales-skip-seed2 (k=1)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                              +1      +0.74556           —           —     scalar
tau_d (s)                                 +0.1     +0.048095           0           0        300
tau_a_E[j=0] (s)                         +0.25      +0.14442           0           0        150
tau_a_E[j=1] (s)                       +1.5896      +0.91826           0           0        150
tau_a_E[j=2] (s)                            +4       +2.3107           0           0        150
tau_a_I[j=0] (s)                         +0.25      +0.20204           0           0        150
tau_a_I[j=1] (s)                       +1.5896       +1.2846           0           0        150
tau_a_I[j=2] (s)                            +4       +3.2326           0           0        150
tau_b_rec_E (s)                             +1      +0.50994           0           0        150
tau_b_rel_E (s)                          +0.25      +0.26591           0           0        150
tau_b_rec_I (s)                             +1      +0.48718           0           0        150
tau_b_rel_I (s)                          +0.25      +0.27013           0           0        150
W_eff (E src, signed)                  +0.2326      +0.37996     0.07684       0.209      14953
W_eff (I src, signed)                 -0.30959      -0.51351      0.0767      0.2312      15212
|W_eff| (E src)                       +0.23264      +0.38717     0.07671      0.1953      14953
|W_eff| (I src)                       +0.30959      +0.51613      0.0767      0.2253      15212
W_in_eff (input neurons)           +0.00075891    +0.0092922      0.1012      0.1971       1275
c_E[j=0] (SFA coupling)                  +0.05     +0.045811           0           0        150
c_E[j=1] (SFA coupling)                  +0.05     +0.045811           0           0        150
c_E[j=2] (SFA coupling)                  +0.05     +0.045811           0           0        150
c_0_E[j=0] (SFA offset)                     +0     -0.085005           0           0        150
c_0_E[j=1] (SFA offset)                     +0     -0.085005           0           0        150
c_0_E[j=2] (SFA offset)                     +0     -0.085005           0           0        150
c_I[j=0] (SFA coupling)                  +0.05     +0.050912           0           0        150
c_I[j=1] (SFA coupling)                  +0.05     +0.050912           0           0        150
c_I[j=2] (SFA coupling)                  +0.05     +0.050912           0           0        150
c_0_I[j=0] (SFA offset)                     +0     +0.077828           0           0        150
c_0_I[j=1] (SFA offset)                     +0     +0.077828           0           0        150
c_0_I[j=2] (SFA offset)                     +0     +0.077828           0           0        150
a_0 (threshold)                          +0.35      +0.28603           0           0        300
W_out_eff (readout × gain)          +0.0024758     +0.004548     0.06765      0.3286       1275
readout_bias                                +0     -0.015984           0     0.07044         17
ic.ic (per-variant init)              +0.17736      +0.17736      0.3517      0.3517       1500
```

\newpage

# srnn-no-dales-skip-seed3

![Tau evolution](srnn-no-dales-skip-seed3/tau_evolution.png){ width=95% }

![W\_EI evolution](srnn-no-dales-skip-seed3/W_EI_evolution.png){ width=95% }

![W\_io evolution](srnn-no-dales-skip-seed3/W_io_evolution.png){ width=95% }

![Offsets and threshold](srnn-no-dales-skip-seed3/offsets_evolution.png){ width=95% }

\newpage

# srnn-no-dales-skip-seed3 — parameter table

```text
# ring2x5-100e — variant srnn-no-dales-skip-seed3 (k=2)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                              +1      +0.74651           —           —     scalar
tau_d (s)                                 +0.1     +0.047293           0           0        300
tau_a_E[j=0] (s)                         +0.25      +0.14805           0           0        150
tau_a_E[j=1] (s)                       +1.5896      +0.94136           0           0        150
tau_a_E[j=2] (s)                            +4       +2.3688           0           0        150
tau_a_I[j=0] (s)                         +0.25      +0.19338           0           0        150
tau_a_I[j=1] (s)                       +1.5896       +1.2296           0           0        150
tau_a_I[j=2] (s)                            +4       +3.0941           0           0        150
tau_b_rec_E (s)                             +1       +0.4932           0           0        150
tau_b_rel_E (s)                          +0.25      +0.29091           0           0        150
tau_b_rec_I (s)                             +1      +0.45378           0           0        150
tau_b_rel_I (s)                          +0.25      +0.28905           0           0        150
W_eff (E src, signed)                 +0.23242      +0.38084      0.0775      0.2078      14937
W_eff (I src, signed)                 -0.30903       -0.5231     0.07701      0.2305      14853
|W_eff| (E src)                       +0.23245      +0.38876      0.0774      0.1926      14937
|W_eff| (I src)                       +0.30903      +0.52518     0.07701      0.2257      14853
W_in_eff (input neurons)           -0.00021194     +0.011104     0.09952      0.2133       1275
c_E[j=0] (SFA coupling)                  +0.05      +0.04066           0           0        150
c_E[j=1] (SFA coupling)                  +0.05      +0.04066           0           0        150
c_E[j=2] (SFA coupling)                  +0.05      +0.04066           0           0        150
c_0_E[j=0] (SFA offset)                     +0      -0.10908           0           0        150
c_0_E[j=1] (SFA offset)                     +0      -0.10908           0           0        150
c_0_E[j=2] (SFA offset)                     +0      -0.10908           0           0        150
c_I[j=0] (SFA coupling)                  +0.05     +0.046994           0           0        150
c_I[j=1] (SFA coupling)                  +0.05     +0.046994           0           0        150
c_I[j=2] (SFA coupling)                  +0.05     +0.046994           0           0        150
c_0_I[j=0] (SFA offset)                     +0     +0.082597           0           0        150
c_0_I[j=1] (SFA offset)                     +0     +0.082597           0           0        150
c_0_I[j=2] (SFA offset)                     +0     +0.082597           0           0        150
a_0 (threshold)                          +0.35      +0.25368           0           0        300
W_out_eff (readout × gain)          +0.0021677    +0.0040971     0.06626      0.3399       1275
readout_bias                                +0   +0.00039762           0     0.07067         17
ic.ic (per-variant init)              +0.19622      +0.19622      0.3102      0.3102       1500
```

\newpage

# srnn-no-dales-skip-seed4

![Tau evolution](srnn-no-dales-skip-seed4/tau_evolution.png){ width=95% }

![W\_EI evolution](srnn-no-dales-skip-seed4/W_EI_evolution.png){ width=95% }

![W\_io evolution](srnn-no-dales-skip-seed4/W_io_evolution.png){ width=95% }

![Offsets and threshold](srnn-no-dales-skip-seed4/offsets_evolution.png){ width=95% }

\newpage

# srnn-no-dales-skip-seed4 — parameter table

```text
# ring2x5-100e — variant srnn-no-dales-skip-seed4 (k=3)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                              +1      +0.75064           —           —     scalar
tau_d (s)                                 +0.1      +0.04718           0           0        300
tau_a_E[j=0] (s)                         +0.25      +0.16412           0           0        150
tau_a_E[j=1] (s)                       +1.5896       +1.0435           0           0        150
tau_a_E[j=2] (s)                            +4       +2.6259           0           0        150
tau_a_I[j=0] (s)                         +0.25      +0.19358           0           0        150
tau_a_I[j=1] (s)                       +1.5896       +1.2308           0           0        150
tau_a_I[j=2] (s)                            +4       +3.0972           0           0        150
tau_b_rec_E (s)                             +1      +0.48549           0           0        150
tau_b_rel_E (s)                          +0.25      +0.32758           0           0        150
tau_b_rec_I (s)                             +1      +0.47179           0           0        150
tau_b_rel_I (s)                          +0.25      +0.28613           0           0        150
W_eff (E src, signed)                 +0.23246      +0.37357     0.07702      0.2129      15026
W_eff (I src, signed)                 -0.30974      -0.53144     0.07771      0.2316      14796
|W_eff| (E src)                       +0.23251      +0.38285     0.07687      0.1957      15026
|W_eff| (I src)                       +0.30974      +0.53362     0.07771      0.2265      14796
W_in_eff (input neurons)            +0.0028809     +0.010779      0.1007      0.2204       1275
c_E[j=0] (SFA coupling)                  +0.05      +0.04349           0           0        150
c_E[j=1] (SFA coupling)                  +0.05      +0.04349           0           0        150
c_E[j=2] (SFA coupling)                  +0.05      +0.04349           0           0        150
c_0_E[j=0] (SFA offset)                     +0      -0.14455           0           0        150
c_0_E[j=1] (SFA offset)                     +0      -0.14455           0           0        150
c_0_E[j=2] (SFA offset)                     +0      -0.14455           0           0        150
c_I[j=0] (SFA coupling)                  +0.05     +0.049767           0           0        150
c_I[j=1] (SFA coupling)                  +0.05     +0.049767           0           0        150
c_I[j=2] (SFA coupling)                  +0.05     +0.049767           0           0        150
c_0_I[j=0] (SFA offset)                     +0      +0.14577           0           0        150
c_0_I[j=1] (SFA offset)                     +0      +0.14577           0           0        150
c_0_I[j=2] (SFA offset)                     +0      +0.14577           0           0        150
a_0 (threshold)                          +0.35      +0.30752           0           0        300
W_out_eff (readout × gain)          +0.0026403    +5.934e-05     0.06667      0.3346       1275
readout_bias                                +0    +0.0077422           0      0.0963         17
ic.ic (per-variant init)              +0.18763      +0.18763      0.3347      0.3347       1500
```

\newpage

# srnn-no-dales-skip-seed5

![Tau evolution](srnn-no-dales-skip-seed5/tau_evolution.png){ width=95% }

![W\_EI evolution](srnn-no-dales-skip-seed5/W_EI_evolution.png){ width=95% }

![W\_io evolution](srnn-no-dales-skip-seed5/W_io_evolution.png){ width=95% }

![Offsets and threshold](srnn-no-dales-skip-seed5/offsets_evolution.png){ width=95% }

\newpage

# srnn-no-dales-skip-seed5 — parameter table

```text
# ring2x5-100e — variant srnn-no-dales-skip-seed5 (k=4)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                              +1       +0.7492           —           —     scalar
tau_d (s)                                 +0.1     +0.048723           0           0        300
tau_a_E[j=0] (s)                         +0.25      +0.16087           0           0        150
tau_a_E[j=1] (s)                       +1.5896       +1.0229           0           0        150
tau_a_E[j=2] (s)                            +4       +2.5739           0           0        150
tau_a_I[j=0] (s)                         +0.25      +0.19211           0           0        150
tau_a_I[j=1] (s)                       +1.5896       +1.2215           0           0        150
tau_a_I[j=2] (s)                            +4       +3.0737           0           0        150
tau_b_rec_E (s)                             +1      +0.48211           0           0        150
tau_b_rel_E (s)                          +0.25      +0.28065           0           0        150
tau_b_rec_I (s)                             +1      +0.52989           0           0        150
tau_b_rel_I (s)                          +0.25       +0.2603           0           0        150
W_eff (E src, signed)                 +0.23266      +0.37731      0.0779      0.2143      14942
W_eff (I src, signed)                 -0.31065      -0.52513     0.07739      0.2258      15085
|W_eff| (E src)                       +0.23271      +0.38681     0.07774      0.1967      14942
|W_eff| (I src)                       +0.31065      +0.52765     0.07738      0.2198      15085
W_in_eff (input neurons)             +0.004538    +0.0047061     0.09841      0.2198       1275
c_E[j=0] (SFA coupling)                  +0.05     +0.046511           0           0        150
c_E[j=1] (SFA coupling)                  +0.05     +0.046511           0           0        150
c_E[j=2] (SFA coupling)                  +0.05     +0.046511           0           0        150
c_0_E[j=0] (SFA offset)                     +0     -0.070028           0           0        150
c_0_E[j=1] (SFA offset)                     +0     -0.070028           0           0        150
c_0_E[j=2] (SFA offset)                     +0     -0.070028           0           0        150
c_I[j=0] (SFA coupling)                  +0.05     +0.049246           0           0        150
c_I[j=1] (SFA coupling)                  +0.05     +0.049246           0           0        150
c_I[j=2] (SFA coupling)                  +0.05     +0.049246           0           0        150
c_0_I[j=0] (SFA offset)                     +0      +0.05937           0           0        150
c_0_I[j=1] (SFA offset)                     +0      +0.05937           0           0        150
c_0_I[j=2] (SFA offset)                     +0      +0.05937           0           0        150
a_0 (threshold)                          +0.35      +0.31963           0           0        300
W_out_eff (readout × gain)          +0.0029913   -4.0578e-05     0.06652      0.3402       1275
readout_bias                                +0    +0.0037483           0     0.09928         17
ic.ic (per-variant init)              +0.20431      +0.20431       0.301       0.301       1500
```

\newpage

# srnn-no-adapt-no-dales-skip-seed1

![Tau evolution](srnn-no-adapt-no-dales-skip-seed1/tau_evolution.png){ width=95% }

![W\_EI evolution](srnn-no-adapt-no-dales-skip-seed1/W_EI_evolution.png){ width=95% }

![W\_io evolution](srnn-no-adapt-no-dales-skip-seed1/W_io_evolution.png){ width=95% }

![Offsets and threshold](srnn-no-adapt-no-dales-skip-seed1/offsets_evolution.png){ width=95% }

\newpage

# srnn-no-adapt-no-dales-skip-seed1 — parameter table

```text
# ring2x5-100e — variant srnn-no-adapt-no-dales-skip-seed1 (k=5)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                              +1      +0.73687           —           —     scalar
tau_d (s)                                 +0.1     +0.047043           0           0        300
tau_b_rec_E (s)                             +1      +0.73687           0           0        150
tau_b_rel_E (s)                          +0.25      +0.18422           0           0        150
tau_b_rec_I (s)                             +1      +0.73687           0           0        150
tau_b_rel_I (s)                          +0.25      +0.18422           0           0        150
W_eff (E src, signed)                  +0.2326      +0.30612     0.07663      0.1254      15048
W_eff (I src, signed)                 -0.31062      -0.40632     0.07727      0.1364      14977
|W_eff| (E src)                       +0.23267      +0.30778     0.07641      0.1213      15048
|W_eff| (I src)                       +0.31062      +0.40726     0.07724      0.1336      14977
W_in_eff (input neurons)            -0.0034346    -0.0050749      0.0996       0.177       1275
a_0 (threshold)                          +0.35      +0.34925           0           0        300
W_out_eff (readout × gain)          -0.0022654    -0.0019774     0.06674       0.232       1275
readout_bias                                +0    +0.0057666           0     0.03787         17
ic.ic (per-variant init)              +0.16154      +0.16154      0.4334      0.4334       1500
```

\newpage

# srnn-no-adapt-no-dales-skip-seed2

![Tau evolution](srnn-no-adapt-no-dales-skip-seed2/tau_evolution.png){ width=95% }

![W\_EI evolution](srnn-no-adapt-no-dales-skip-seed2/W_EI_evolution.png){ width=95% }

![W\_io evolution](srnn-no-adapt-no-dales-skip-seed2/W_io_evolution.png){ width=95% }

![Offsets and threshold](srnn-no-adapt-no-dales-skip-seed2/offsets_evolution.png){ width=95% }

\newpage

# srnn-no-adapt-no-dales-skip-seed2 — parameter table

```text
# ring2x5-100e — variant srnn-no-adapt-no-dales-skip-seed2 (k=6)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                              +1      +0.73705           —           —     scalar
tau_d (s)                                 +0.1     +0.047082           0           0        300
tau_b_rec_E (s)                             +1      +0.73705           0           0        150
tau_b_rel_E (s)                          +0.25      +0.18426           0           0        150
tau_b_rec_I (s)                             +1      +0.73705           0           0        150
tau_b_rel_I (s)                          +0.25      +0.18426           0           0        150
W_eff (E src, signed)                  +0.2326      +0.31887     0.07684      0.1306      14953
W_eff (I src, signed)                 -0.30959      -0.42724      0.0767      0.1363      15212
|W_eff| (E src)                       +0.23264      +0.32097     0.07671      0.1254      14953
|W_eff| (I src)                       +0.30959      +0.42835      0.0767      0.1328      15212
W_in_eff (input neurons)             +0.001514    +0.0053264      0.1004      0.1745       1275
a_0 (threshold)                          +0.35      +0.36249           0           0        300
W_out_eff (readout × gain)          -0.0015425    -0.0026665     0.06628      0.2205       1275
readout_bias                                +0     +0.011159           0     0.04235         17
ic.ic (per-variant init)              +0.16262      +0.16262      0.6757      0.6757       1500
```

\newpage

# srnn-no-adapt-no-dales-skip-seed3

![Tau evolution](srnn-no-adapt-no-dales-skip-seed3/tau_evolution.png){ width=95% }

![W\_EI evolution](srnn-no-adapt-no-dales-skip-seed3/W_EI_evolution.png){ width=95% }

![W\_io evolution](srnn-no-adapt-no-dales-skip-seed3/W_io_evolution.png){ width=95% }

![Offsets and threshold](srnn-no-adapt-no-dales-skip-seed3/offsets_evolution.png){ width=95% }

\newpage

# srnn-no-adapt-no-dales-skip-seed3 — parameter table

```text
# ring2x5-100e — variant srnn-no-adapt-no-dales-skip-seed3 (k=7)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                              +1      +0.65253           —           —     scalar
tau_d (s)                                 +0.1     +0.035421           0           0        300
tau_b_rec_E (s)                             +1      +0.65253           0           0        150
tau_b_rel_E (s)                          +0.25      +0.16313           0           0        150
tau_b_rec_I (s)                             +1      +0.65253           0           0        150
tau_b_rel_I (s)                          +0.25      +0.16313           0           0        150
W_eff (E src, signed)                 +0.23242      +0.37921      0.0775      0.1459      14937
W_eff (I src, signed)                 -0.30903      -0.50052     0.07701      0.1508      14853
|W_eff| (E src)                       +0.23245      +0.38056      0.0774      0.1423      14937
|W_eff| (I src)                       +0.30903      +0.50192     0.07701      0.1461      14853
W_in_eff (input neurons)            +0.0020464    -0.0052746      0.1001      0.2179       1275
a_0 (threshold)                          +0.35      +0.33272           0           0        300
W_out_eff (readout × gain)         -0.00031136    -0.0014694     0.06752      0.1619       1275
readout_bias                                +0    +0.0028744           0     0.01564         17
ic.ic (per-variant init)              +0.23263      +0.23263      0.7278      0.7278       1500
```

\newpage

# srnn-no-adapt-no-dales-skip-seed4

![Tau evolution](srnn-no-adapt-no-dales-skip-seed4/tau_evolution.png){ width=95% }

![W\_EI evolution](srnn-no-adapt-no-dales-skip-seed4/W_EI_evolution.png){ width=95% }

![W\_io evolution](srnn-no-adapt-no-dales-skip-seed4/W_io_evolution.png){ width=95% }

![Offsets and threshold](srnn-no-adapt-no-dales-skip-seed4/offsets_evolution.png){ width=95% }

\newpage

# srnn-no-adapt-no-dales-skip-seed4 — parameter table

```text
# ring2x5-100e — variant srnn-no-adapt-no-dales-skip-seed4 (k=8)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                              +1       +0.6646           —           —     scalar
tau_d (s)                                 +0.1      +0.03712           0           0        300
tau_b_rec_E (s)                             +1       +0.6646           0           0        150
tau_b_rel_E (s)                          +0.25      +0.16615           0           0        150
tau_b_rec_I (s)                             +1       +0.6646           0           0        150
tau_b_rel_I (s)                          +0.25      +0.16615           0           0        150
W_eff (E src, signed)                 +0.23246      +0.35828     0.07702      0.1416      15026
W_eff (I src, signed)                 -0.30974       -0.4791     0.07771      0.1529      14796
|W_eff| (E src)                       +0.23251      +0.36041     0.07687      0.1361      15026
|W_eff| (I src)                       +0.30974      +0.48103     0.07771      0.1467      14796
W_in_eff (input neurons)            -0.0011312    -0.0039372     0.09732      0.2103       1275
a_0 (threshold)                          +0.35      +0.36561           0           0        300
W_out_eff (readout × gain)          +0.0005506    +0.0050672      0.0661      0.1867       1275
readout_bias                                +0   -0.00089129           0     0.02409         17
ic.ic (per-variant init)              +0.11354      +0.11354      0.6578      0.6578       1500
```

\newpage

# srnn-no-adapt-no-dales-skip-seed5

![Tau evolution](srnn-no-adapt-no-dales-skip-seed5/tau_evolution.png){ width=95% }

![W\_EI evolution](srnn-no-adapt-no-dales-skip-seed5/W_EI_evolution.png){ width=95% }

![W\_io evolution](srnn-no-adapt-no-dales-skip-seed5/W_io_evolution.png){ width=95% }

![Offsets and threshold](srnn-no-adapt-no-dales-skip-seed5/offsets_evolution.png){ width=95% }

\newpage

# srnn-no-adapt-no-dales-skip-seed5 — parameter table

```text
# ring2x5-100e — variant srnn-no-adapt-no-dales-skip-seed5 (k=9)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                              +1      +0.75196           —           —     scalar
tau_d (s)                                 +0.1     +0.049365           0           0        300
tau_b_rec_E (s)                             +1      +0.75196           0           0        150
tau_b_rel_E (s)                          +0.25      +0.18799           0           0        150
tau_b_rec_I (s)                             +1      +0.75196           0           0        150
tau_b_rel_I (s)                          +0.25      +0.18799           0           0        150
W_eff (E src, signed)                 +0.23266      +0.31564      0.0779       0.129      14942
W_eff (I src, signed)                 -0.31065      -0.41783     0.07739      0.1379      15085
|W_eff| (E src)                       +0.23271      +0.31755     0.07774      0.1243      14942
|W_eff| (I src)                       +0.31065      +0.41872     0.07738      0.1352      15085
W_in_eff (input neurons)           -0.00056932    -0.0044341     0.09975      0.1856       1275
a_0 (threshold)                          +0.35      +0.34474           0           0        300
W_out_eff (readout × gain)          -0.0005528    +0.0016052     0.06596      0.2248       1275
readout_bias                                +0     +0.014223           0     0.08062         17
ic.ic (per-variant init)              +0.17253      +0.17253      0.7022      0.7022       1500
```
