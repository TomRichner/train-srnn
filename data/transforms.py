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
# Batch-level time stretch
# ---------------------------------------------------------------------------

def time_stretch_batch(batch_x, batch_y, factor, per_timestep_labels=True):
    """Stretch an entire batch by the same factor.  Vectorized over batch.

    Args:
        batch_x: (B, T, F) numpy array.
        batch_y: (B, T, ...) or (B,) numpy array.
        factor: float > 0.
        per_timestep_labels: Whether y has a time dimension.

    Returns:
        x_new: (B, T_new, F)
        y_new: Appropriately resampled labels.
    """
    if abs(factor - 1.0) < 1e-6:
        return batch_x, batch_y

    B, T, F = batch_x.shape
    T_new = max(2, int(round(T * factor)))

    t_orig = np.linspace(0.0, 1.0, T)
    t_new = np.linspace(0.0, 1.0, T_new)

    # Reshape (B, T, F) → (T, B*F) so _pchip_resample handles all columns
    flat = np.ascontiguousarray(batch_x.transpose(1, 0, 2)).reshape(T, B * F)
    stretched_flat = _pchip_resample(flat, t_orig, t_new)  # (T_new, B*F)
    x_new = stretched_flat.reshape(T_new, B, F).transpose(1, 0, 2)  # (B, T_new, F)

    if not per_timestep_labels:
        return x_new, batch_y

    # Resample labels
    y_new = _resample_labels_batch(batch_y, t_orig, t_new, T)
    return x_new, y_new


def _resample_labels_batch(y, t_orig, t_new, T):
    """Resample batched labels: nearest-neighbor for ints, PCHIP for floats.

    Args:
        y: (B, T, ...) numpy array.
        t_orig: (T,) original sample positions.
        t_new: (T_new,) new sample positions.
        T: original sequence length.

    Returns:
        (B, T_new, ...) resampled labels.
    """
    if np.issubdtype(y.dtype, np.integer):
        indices = np.searchsorted(t_orig, t_new, side="right") - 1
        indices = np.clip(indices, 0, T - 1)
        return y[:, indices]  # (B, T_new, ...) via fancy indexing on axis 1

    # Float labels — PCHIP via reshape trick
    B = y.shape[0]
    if y.ndim == 2:
        # (B, T) → (T, B) → _pchip_resample → (T_new, B) → (B, T_new)
        flat = np.ascontiguousarray(y.T)  # (T, B)
        resampled = _pchip_resample(flat, t_orig, t_new)  # (T_new, B)
        return resampled.T  # (B, T_new)
    else:
        # (B, T, D) → (T, B*D) → _pchip_resample → (T_new, B*D) → (B, T_new, D)
        D = y.shape[2:]
        flat = np.ascontiguousarray(y.transpose(1, 0, *range(2, y.ndim))).reshape(T, -1)
        resampled = _pchip_resample(flat, t_orig, t_new)
        return resampled.reshape(len(t_new), B, *D).transpose(1, 0, *range(2, y.ndim))


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


def palindrome_loop_batch(x, y, n_loops, per_timestep_labels=True):
    """Palindrome-loop an entire batch.  No per-sample loop.

    Args:
        x: (B, T, F) numpy array.
        y: (B, T, ...) or (B,) numpy array.
        n_loops: Number of fwd+bwd pairs.
        per_timestep_labels: Whether y has a time dimension.

    Returns:
        x_looped: (B, n_loops * 2 * T, F)
        y_looped: Mirrored labels (or unchanged scalar array).
    """
    x_fwd = x
    x_bwd = x[:, ::-1, :]
    x_pieces = [x_fwd, x_bwd] * n_loops
    x_looped = np.concatenate(x_pieces, axis=1)

    if not per_timestep_labels:
        return x_looped, y

    y_fwd = y
    y_bwd = y[:, ::-1] if y.ndim >= 2 else y
    y_pieces = [y_fwd, y_bwd] * n_loops
    y_looped = np.concatenate(y_pieces, axis=1)

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

    All operations are vectorized over the batch dimension — no per-sample
    Python loops.

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
    # 1. Time stretch (vectorized) — 1 RNG call
    do_stretch = (abs(stretch_lo - stretch_hi) > 1e-6 or
                  abs(stretch_lo - 1.0) > 1e-6)
    if do_stretch:
        factor = random_stretch_factor(stretch_lo, stretch_hi, rng)
        batch_x, batch_y = time_stretch_batch(
            batch_x, batch_y, factor, per_timestep_labels)

    # 2. Palindrome loop (vectorized, no per-sample loop)
    seq_len = batch_x.shape[1]
    n_loops = compute_n_loops(seq_len, min_loop_len, min_loops)
    loop_len = 2 * seq_len
    looped_x, looped_y = palindrome_loop_batch(
        batch_x, batch_y, n_loops, per_timestep_labels)

    # 3. Random window (vectorized numpy slice) — 2 RNG calls
    T_total = looped_x.shape[1]
    n_total_loops = T_total // loop_len

    if n_total_loops < 2:
        readout_idx = T_total - 1
        if per_timestep_labels:
            return looped_x, looped_y, readout_idx, 0
        return looped_x, batch_y, readout_idx, 0

    offset = rng.randint(0, loop_len)
    end = T_total - (loop_len - offset)
    win_len = end - offset  # = (n_total_loops - 1) * loop_len

    aug_x = looped_x[:, offset:end]
    if per_timestep_labels:
        aug_y = looped_y[:, offset:end]
    else:
        aug_y = batch_y

    last_loop_start = win_len - loop_len
    readout_idx = rng.randint(last_loop_start, win_len)
    bptt_start_idx = max(0, win_len - 2 * loop_len)

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
    seq_len = batch_x.shape[1]
    n_loops = compute_n_loops(seq_len, min_loop_len, min_loops)

    looped_x, looped_y = palindrome_loop_batch(
        batch_x, batch_y, n_loops, per_timestep_labels)

    readout_idx = looped_x.shape[1] - 1

    if per_timestep_labels:
        labels_at_readout = looped_y[:, readout_idx]
    else:
        labels_at_readout = batch_y

    return looped_x, labels_at_readout, readout_idx
