# Gradient-norm probe: |∇θ|₂ vs bptt_chunk_len

Setup: K=2 batched cell ({srnn-skip, srnn-no-adapt-no-dales-skip}), batch=4,
window_len=1024, bptt_len=768, nominal-init weights (gains at 0, taus at their inits),
real seeg batch (subject 939, baseline). Single forward + backward, no optimizer step.
Only chunk_len varies; everything else is fixed.

## Results for `srnn-skip`

| param                       | cl=16    | cl=32    | cl=64    | cl=128   | cl=256   | cl=512   | 512/16 |
|-----------------------------|---------:|---------:|---------:|---------:|---------:|---------:|-------:|
| readout_weight              | 3.08e-03 | 3.08e-03 | 3.08e-03 | 3.08e-03 | 3.08e-03 | 3.08e-03 |   1.00 |
| a_0_scalar                  | 1.49e-02 | 1.42e-02 | 1.36e-02 | 1.30e-02 | 1.16e-02 | 1.10e-02 |   0.74 |
| W_in                        | 2.10e-04 | 5.81e-04 | 1.21e-03 | 2.11e-03 | 2.49e-03 | 2.47e-03 |  11.79 |
| W_raw                       | 1.08e-04 | 2.25e-04 | 4.51e-04 | 7.64e-04 | 8.79e-04 | 8.44e-04 |   7.83 |
| W_raw_gain                  | 3.13e-04 | 6.29e-04 | 9.99e-04 | 1.34e-03 | 1.46e-03 | 1.47e-03 |   4.71 |
| log_tau_d_gain              | 7.63e-05 | 2.16e-04 | 3.07e-04 | 5.73e-04 | 5.69e-04 | 5.43e-04 |   7.11 |
| log_c_E_gain                | 1.91e-04 | 2.61e-04 | 3.28e-04 | 3.63e-04 | 3.36e-04 | 3.15e-04 |   1.65 |
| log_tau_b_rec_E_gain        | 5.05e-05 | 1.41e-04 | 3.60e-04 | 8.16e-04 | 1.44e-03 | 1.88e-03 |  37.11 |
| log_tau_b_rel_E_gain        | 1.05e-04 | 2.83e-04 | 6.64e-04 | 1.23e-03 | 1.71e-03 | 1.94e-03 |  18.57 |
| **log_tau_a_E_gain**        | 7.20e-06 | 1.75e-05 | 3.71e-05 | 5.75e-05 | 6.58e-05 | 7.39e-05 |  10.26 |
| **log_tau_a_I_gain**        | 2.85e-07 | 2.79e-06 | 1.13e-05 | 2.03e-05 | 2.52e-05 | 3.09e-05 | 108.62 |
| log_tau_global              | 1.07e-05 | 5.38e-05 | 3.84e-05 | 1.40e-04 | 1.77e-04 | 2.36e-04 |  22.11 |

## Three regimes

1. **`readout_weight`: chunk-independent** — sits at the output, no temporal credit assignment, bit-identical across chunk_lens.

2. **Fast / dendritic / structural (`W_*`, `log_tau_d_gain`, `c_E_gain`, `a_0_scalar`): saturate around cl=128–256.** Gradient stops growing because the dendritic state has decayed by then (~5–7 τ_d). Current `bptt_chunk_len=128` is roughly the elbow.

3. **Slow / adaptation (`log_tau_a_E_gain`, `log_tau_a_I_gain`, STD `tau_b_*_gain`, `log_tau_global`): still climbing strongly at cl=512.** These gradients grow 10–100× when chunk extends 32×. SFA-I gradient at cl=16 is **2.85e-7** — effectively noise, Adam can't make use of it.

## Interpretation

The current `bptt_chunk_len=128` is well-tuned for the dendritic and recurrent paths
(`x` saturates), but ~10–100× too short for the SFA / STD / global-tau parameters
whose effects only show up after seconds (5 s SFA, 1 s STD recovery, etc.). The
slow params receive tiny, noisy gradients per backward pass, which explains why
in the actual training run their `*_gain` values barely budge from 0:

- `log_tau_a_I_gain` over 90 epochs: 0 → −0.13   (consistent with a slow drift driven by very small but consistent gradients)
- `log_c_E_gain` over 90 epochs: 0 → +0.02       (likewise)

The training-loss plateau is consistent with the network learning the autoregressive
skip residual via readout + threshold, while the slow internal state is too costly
to exploit (high optimization noise relative to signal).

## Recommendations

- Increase `bptt_chunk_len` to 384–512 (already gradient-checkpointed, so VRAM should accommodate).
- Or reduce slow τ inits (start `tau_a_E` at ~0.5 s instead of spread 0.25–10 s) so the SFA effect manifests within a 128-step window.
- Or train the slow-side params with a higher learning rate (parameter-group LR scaling).
