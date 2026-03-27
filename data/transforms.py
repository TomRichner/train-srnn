"""Time stretch, palindrome looping, and training/eval wrappers.

All functions operate on numpy arrays.  The training loop is responsible
for converting the final outputs to torch tensors.

Convention: arrays are batch-first (batch, seq_len, features) everywhere
in the public API, matching the dataset loaders.  Internal helper
functions note their axis conventions.
"""

import math
import numpy as np
from scipy.interpolate import PchipInterpolator


# ---------------------------------------------------------------------------
# Time stretch (PCHIP interpolation)
# ---------------------------------------------------------------------------

def random_stretch_factor(lo=0.25, hi=4.0, rng=None):
    """Sample a time-stretch factor from a log-uniform distribution.

    Args:
        lo: Minimum stretch factor.
        hi: Maximum stretch factor.
        rng: numpy RandomState (default: np.random).

    Returns:
        float in [lo, hi], log-uniformly sampled.
    """
    if rng is None:
        rng = np.random
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def time_stretch(x, y, factor, per_timestep_labels=True):
    """Stretch a single sequence by *factor* using PCHIP interpolation.

    Args:
        x: (seq_len, features) numpy array.
        y: (seq_len,) or (seq_len, label_dim) for per-timestep labels,
           or scalar / (label_dim,) for single-label tasks.
        factor: float > 0.  >1 = longer (slower), <1 = shorter (faster).
        per_timestep_labels: Whether y has a time dimension.

    Returns:
        x_new: (new_seq_len, features)
        y_new: Appropriately resampled labels.
    """
    if abs(factor - 1.0) < 1e-6:
        return x, y

    T = x.shape[0]
    T_new = max(2, int(round(T * factor)))

    t_orig = np.linspace(0.0, 1.0, T)
    t_new = np.linspace(0.0, 1.0, T_new)

    # Interpolate features (always float PCHIP)
    x_new = _pchip_resample(x, t_orig, t_new)

    if not per_timestep_labels:
        return x_new, y

    # Resample labels
    y_new = _resample_labels(y, t_orig, t_new, T)
    return x_new, y_new


def _pchip_resample(arr, t_orig, t_new):
    """Resample a 2-D array along axis 0 with PCHIP.

    Args:
        arr: (T, D) float array.
        t_orig: (T,) original sample positions.
        t_new: (T_new,) new sample positions.

    Returns:
        (T_new, D) float array.
    """
    if arr.ndim == 1:
        arr = arr[:, None]
        squeeze = True
    else:
        squeeze = False

    T_new = len(t_new)
    D = arr.shape[1]
    out = np.empty((T_new, D), dtype=arr.dtype)
    for col in range(D):
        interp = PchipInterpolator(t_orig, arr[:, col])
        out[:, col] = interp(t_new).astype(arr.dtype)

    if squeeze:
        out = out[:, 0]
    return out


def _resample_labels(y, t_orig, t_new, T):
    """Resample labels: nearest-neighbor for integers, PCHIP for floats."""
    if np.issubdtype(y.dtype, np.integer):
        indices = np.searchsorted(t_orig, t_new, side="right") - 1
        indices = np.clip(indices, 0, T - 1)
        return y[indices]
    else:
        return _pchip_resample(y, t_orig, t_new)


# ---------------------------------------------------------------------------
# Palindrome looping
# ---------------------------------------------------------------------------

def compute_n_loops(seq_len, min_loop_len=500, min_loops=5):
    """Compute number of palindrome fwd+bwd loop pairs needed.

    Each pair = one forward pass + one backward pass = 2 * seq_len steps.

    Args:
        seq_len: Length of the original (possibly stretched) sequence.
        min_loop_len: Minimum total looped length in timesteps.
        min_loops: Minimum number of fwd+bwd loop pairs.

    Returns:
        n_loops: int, number of fwd+bwd pairs.
    """
    return max(min_loops, math.ceil(min_loop_len / (2 * seq_len)))


def palindrome_loop(x, y, n_loops, per_timestep_labels=True):
    """Create palindrome-looped sequence: [fwd, bwd, fwd, bwd, ...].

    Args:
        x: (seq_len, features) numpy array.
        y: (seq_len, ...) or scalar labels.
        n_loops: Number of fwd+bwd pairs.
        per_timestep_labels: Whether y has a time dimension.

    Returns:
        x_looped: (n_loops * 2 * seq_len, features)
        y_looped: Mirrored labels (or unchanged scalar).
    """
    x_fwd = x
    x_bwd = x[::-1]

    x_pieces = []
    for _ in range(n_loops):
        x_pieces.append(x_fwd)
        x_pieces.append(x_bwd)
    x_looped = np.concatenate(x_pieces, axis=0)

    if not per_timestep_labels:
        return x_looped, y

    y_fwd = y
    y_bwd = y[::-1]
    y_pieces = []
    for _ in range(n_loops):
        y_pieces.append(y_fwd)
        y_pieces.append(y_bwd)
    y_looped = np.concatenate(y_pieces, axis=0)

    return x_looped, y_looped


def random_window(x_looped, y_looped, loop_len, rng,
                  n_bptt_loops=2, per_timestep_labels=True):
    """Extract a random window from a palindrome-looped sequence.

    Picks a random start offset within the first loop, producing a window
    of exactly (n_total_loops - 1) * loop_len timesteps.  Also returns
    the readout index (random point in the last loop) and the BPTT
    boundary.

    Args:
        x_looped: (T_total, features) palindrome-looped input.
        y_looped: (T_total, ...) or scalar labels.
        loop_len: Length of one fwd+bwd pair (2 * seq_len).
        rng: numpy RandomState.
        n_bptt_loops: Loops at the end to backpropagate through.
        per_timestep_labels: Whether y has a time dimension.

    Returns:
        x_win: (win_len, features)
        y_win: Windowed labels.
        readout_idx: int, index within window for readout.
        bptt_start_idx: int, where to start BPTT.
    """
    T_total = x_looped.shape[0]
    n_total_loops = T_total // loop_len

    if n_total_loops < 2:
        readout_idx = T_total - 1
        return x_looped, y_looped, readout_idx, 0

    # Random start within the first loop
    i = rng.randint(0, loop_len)
    end = T_total - (loop_len - i)
    win_len = end - i  # = (n_total_loops - 1) * loop_len

    x_win = x_looped[i:end]
    if per_timestep_labels:
        y_win = y_looped[i:end]
    else:
        y_win = y_looped

    # Readout: random point in the last loop
    last_loop_start = win_len - loop_len
    readout_idx = rng.randint(last_loop_start, win_len)

    # BPTT boundary
    bptt_len = n_bptt_loops * loop_len
    bptt_start_idx = max(0, win_len - bptt_len)

    return x_win, y_win, readout_idx, bptt_start_idx


# ---------------------------------------------------------------------------
# Training / eval wrappers (batch level)
# ---------------------------------------------------------------------------

def wrap_train_batch(batch_x, batch_y, rng,
                     stretch_lo=1.0, stretch_hi=1.0,
                     min_loops=5, min_loop_len=500,
                     per_timestep_labels=True):
    """Full training augmentation: stretch -> loop -> random window.

    Applies the same stretch factor and looping to every sequence in the
    batch, but each sequence is processed independently so that stretch
    is applied per-sample before batching for the loop/window stage.

    Args:
        batch_x: (batch, seq_len, features) numpy array.
        batch_y: (batch, seq_len) or (batch,) numpy array.
        rng: numpy RandomState.
        stretch_lo: Minimum stretch factor (1.0 = no stretch).
        stretch_hi: Maximum stretch factor.
        min_loops: Min palindrome loop pairs.
        min_loop_len: Min total looped sequence length.
        per_timestep_labels: Whether labels are per-timestep.

    Returns:
        aug_x: (batch, win_len, features) numpy array.
        aug_y: (batch, win_len) or (batch,) numpy array.
        readout_idx: int -- timestep index for readout.
        bptt_start_idx: int -- where BPTT should begin.
    """
    B = batch_x.shape[0]

    # 1. Time stretch (same factor for entire batch)
    do_stretch = (abs(stretch_lo - stretch_hi) > 1e-6 or
                  abs(stretch_lo - 1.0) > 1e-6)
    if do_stretch:
        factor = random_stretch_factor(stretch_lo, stretch_hi, rng)
    else:
        factor = 1.0

    stretched_xs = []
    stretched_ys = []
    for i in range(B):
        xi = batch_x[i]  # (seq_len, features)
        yi = batch_y[i] if per_timestep_labels else batch_y[i]
        xi_s, yi_s = time_stretch(xi, yi, factor, per_timestep_labels)
        stretched_xs.append(xi_s)
        stretched_ys.append(yi_s)

    # After stretch all sequences have the same length (same factor)
    eff_seq_len = stretched_xs[0].shape[0]
    n_loops = compute_n_loops(eff_seq_len, min_loop_len, min_loops)
    loop_len = 2 * eff_seq_len

    # 2. Palindrome loop + random window (per sample, but same rng state
    #    means same window offsets -- we want the same readout_idx for the
    #    whole batch, so we generate it once)
    #    To keep batch alignment, loop all, then window all with the same
    #    random offset.

    # Determine window params from first sample
    rng_snapshot = rng.get_state()

    # Loop and window the first sample to get dimensions / indices
    x0_l, y0_l = palindrome_loop(
        stretched_xs[0], stretched_ys[0], n_loops, per_timestep_labels)
    x0_w, y0_w, readout_idx, bptt_start_idx = random_window(
        x0_l, y0_l, loop_len, rng, per_timestep_labels=per_timestep_labels)

    # Now process all samples with the same random offsets
    # Reset rng for consistent windowing params, but we already consumed it
    # above.  We'll just re-derive the offset from the first call.
    # Since random_window only calls rng twice (for i and readout_idx),
    # we need to replicate the same i and readout_idx for all samples.
    # The simplest approach: compute the loop+window for all samples
    # using the same i and readout_idx we already got.

    T_total = x0_l.shape[0]
    n_total_loops = T_total // loop_len
    # Recover i from readout_idx and window length
    win_len = x0_w.shape[0]
    # i = T_total - loop_len - win_len + i ... actually just redo cleanly:

    # We already have readout_idx and bptt_start_idx from the first sample.
    # Now loop+window all other samples with the same slice.
    # Recover the start offset: win_len = (n_total_loops - 1) * loop_len
    # i + win_len = end, end = T_total - (loop_len - i)
    # => i = (T_total - win_len - loop_len) ... solve:
    # win_len = end - i = T_total - loop_len + i - i ... no.
    # From random_window: end = T_total - (loop_len - i), win_len = end - i
    # => win_len = T_total - loop_len
    # So i doesn't affect win_len! It only shifts the window.
    # We need to recover i. From end = T_total - (loop_len - i):
    # We know win_len and T_total, and win_len = end - i.
    # end = win_len + i, also end = T_total - loop_len + i
    # So win_len = T_total - loop_len.  i doesn't appear in win_len.
    # The slice is x_looped[i : i + win_len].
    # We can recover i from the snapshot... but it's simpler to just
    # loop+window each sample consistently.

    # Better approach: loop all, stack, then slice uniformly.
    looped_xs = []
    looped_ys = []
    for i in range(B):
        xl, yl = palindrome_loop(
            stretched_xs[i], stretched_ys[i], n_loops, per_timestep_labels)
        looped_xs.append(xl)
        looped_ys.append(yl)

    # Stack into batch arrays
    looped_xs = np.stack(looped_xs, axis=0)  # (B, T_total, feat)
    if per_timestep_labels:
        looped_ys = np.stack(looped_ys, axis=0)  # (B, T_total, ...)

    # Compute window slice indices (same for all samples)
    if n_total_loops >= 2:
        # Restore rng to get the same i as we got for sample 0
        rng.set_state(rng_snapshot)
        offset_i = rng.randint(0, loop_len)
        end_i = T_total - (loop_len - offset_i)
        win_len_check = end_i - offset_i

        last_loop_start = win_len_check - loop_len
        readout_idx = rng.randint(last_loop_start, win_len_check)
        bptt_len = 2 * loop_len
        bptt_start_idx = max(0, win_len_check - bptt_len)

        aug_x = looped_xs[:, offset_i:end_i]
        if per_timestep_labels:
            aug_y = looped_ys[:, offset_i:end_i]
        else:
            aug_y = batch_y  # single labels unchanged
    else:
        aug_x = looped_xs
        if per_timestep_labels:
            aug_y = looped_ys
        else:
            aug_y = batch_y
        readout_idx = T_total - 1
        bptt_start_idx = 0

    return aug_x, aug_y, readout_idx, bptt_start_idx


def wrap_eval_batch(batch_x, batch_y,
                    min_loops=5, min_loop_len=500,
                    per_timestep_labels=True):
    """Eval augmentation: palindrome loop (no stretch, deterministic window).

    Args:
        batch_x: (batch, seq_len, features) numpy array.
        batch_y: (batch, seq_len) or (batch,) numpy array.
        min_loops: Min palindrome loop pairs.
        min_loop_len: Min total looped sequence length.
        per_timestep_labels: Whether labels are per-timestep.

    Returns:
        looped_x: (batch, T_looped, features) numpy array.
        labels_at_readout: (batch,) or (batch, label_dim) -- label at readout.
        readout_idx: int -- last timestep of the looped sequence.
    """
    B = batch_x.shape[0]
    seq_len = batch_x.shape[1]
    n_loops = compute_n_loops(seq_len, min_loop_len, min_loops)

    looped_xs = []
    looped_ys = []
    for i in range(B):
        xl, yl = palindrome_loop(
            batch_x[i], batch_y[i], n_loops, per_timestep_labels)
        looped_xs.append(xl)
        if per_timestep_labels:
            looped_ys.append(yl)

    looped_x = np.stack(looped_xs, axis=0)  # (B, T_looped, feat)
    readout_idx = looped_x.shape[1] - 1

    if per_timestep_labels:
        looped_y = np.stack(looped_ys, axis=0)
        labels_at_readout = looped_y[:, readout_idx]
    else:
        labels_at_readout = batch_y

    return looped_x, labels_at_readout, readout_idx
