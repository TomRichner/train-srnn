# Run: alpha-smoke-10e

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
# alpha-smoke-10e — variant srnn-e-only-per-neuron (k=0)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                              +1      +0.95745           —           —     scalar
tau_d (s)                                 +0.1     +0.091829           0    0.002637        150
tau_a_E (s)                            +4.8777        +4.569       3.996       3.747        225
tau_b_rec_E (s)                             +1      +0.96689           0     0.01197         75
tau_b_rel_E (s)                          +0.25      +0.23605           0    0.004715         75
W_eff (E src, signed)                 +0.32855      +0.31881      0.1087      0.1056       3729
W_eff (I src, signed)                 -0.44048      -0.42819      0.1104      0.1073       3698
|W_eff| (E src)                       +0.32855      +0.31881      0.1087      0.1056       3729
|W_eff| (I src)                       +0.44048      +0.42819      0.1104      0.1073       3698
W_in_eff (input neurons)           -0.00088642   -3.8661e-05     0.09775      0.0997       3293
c_E (SFA coupling)                       +0.05     +0.052033           0    0.001196        225
c_0_E (SFA offset)                          +0     +0.024165           0      0.0223        225
a_0 (threshold)                          +0.35      +0.37937           0     0.02651        150
readout_weight                      -0.0019584    -0.0019584     0.09524     0.09524       3293
readout_bias                                +0    -0.0022583           0     0.01842         89
ic.ic (per-variant init)             +0.084632      +0.11912       0.411       0.395        450
```

\newpage

# srnn-e-only-skip-per-neuron

![Tau evolution](srnn-e-only-skip-per-neuron/tau_evolution.png){ width=95% }

![W\_EI evolution](srnn-e-only-skip-per-neuron/W_EI_evolution.png){ width=95% }

\newpage

# srnn-e-only-skip-per-neuron — parameter table

```text
# alpha-smoke-10e — variant srnn-e-only-skip-per-neuron (k=1)
# Effective values (post-transform). Means/stds computed over the indicated dimension.
effective parameter                  init mean    final mean    init std   final std          n
-----------------------------------------------------------------------------------------------
tau_global (s)                              +1       +1.0306           —           —     scalar
tau_d (s)                                 +0.1      +0.10706           0     0.00292        150
tau_a_E (s)                            +4.8777       +4.8701       3.996       3.996        225
tau_b_rec_E (s)                             +1       +1.0598           0    0.005423         75
tau_b_rel_E (s)                          +0.25      +0.24799           0    0.001726         75
W_eff (E src, signed)                 +0.32855      +0.31865      0.1087      0.1054       3729
W_eff (I src, signed)                 -0.44048      -0.42959      0.1104      0.1077       3698
|W_eff| (E src)                       +0.32855      +0.31865      0.1087      0.1054       3729
|W_eff| (I src)                       +0.44048      +0.42959      0.1104      0.1077       3698
W_in_eff (input neurons)            +0.0025045    +0.0035252      0.0991     0.09251       3293
c_E (SFA coupling)                       +0.05     +0.053273           0   0.0008373        225
c_0_E (SFA offset)                          +0     +0.062207           0    0.009827        225
a_0 (threshold)                          +0.35       +0.3934           0     0.03105        150
readout_weight                      +0.0023626    +0.0023626     0.09405     0.09405       3293
readout_bias                                +0   -0.00097848           0    0.004751         89
ic.ic (per-variant init)             +0.084506       +0.1385      0.4111      0.3941        450
```
