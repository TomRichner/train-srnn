"""Augmentation for the windowed benchmark tasks: PCHIP time stretch, palindrome looping, windowing.

Everything works on batch-first numpy arrays ``(B, T, F)``; the trainer
converts the results to tensors.
"""

import math
import numpy as np
from scipy.interpolate import PchipInterpolator


# Time stretch (PCHIP interpolation)

def random_stretch_factor(lo=0.25, hi=4.0, rng=None):
    """Log-uniform stretch factor in ``[lo, hi]``."""
    if rng is None:
        rng = np.random
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def time_stretch(x, y, factor, per_timestep_labels=True):
    """Stretch one ``(T, F)`` sequence in time by ``factor`` with PCHIP; labels follow if per-timestep."""
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
    """PCHIP-resample ``(T, D)`` along axis 0 to ``(T_new, D)``."""
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


# Batch-level time stretch

def time_stretch_batch(batch_x, batch_y, factor, per_timestep_labels=True):
    """Stretch a whole ``(B, T, F)`` batch by one factor."""
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
    """Resample ``(B, T, ...)`` labels: nearest neighbour for integers, PCHIP for floats."""
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


# Palindrome looping

def compute_n_loops(seq_len, min_loop_len=500, min_loops=5):
    """Forward-plus-backward loop pairs needed to reach ``min_loop_len`` steps (at least ``min_loops``)."""
    return max(min_loops, math.ceil(min_loop_len / (2 * seq_len)))


def palindrome_loop(x, y, n_loops, per_timestep_labels=True):
    """``[x, x reversed] * n_loops`` along time, with labels mirrored the same way."""
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
    """Palindrome-loop a ``(B, T, F)`` batch along time."""
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
    """Random ``(n_loops - 1) * loop_len`` window of a looped sequence, with a readout index in its last loop."""
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


# Training / eval wrappers (batch level)

def wrap_train_batch(batch_x, batch_y, rng,
                     stretch_lo=1.0, stretch_hi=1.0,
                     window_len=1024, bptt_len=512,
                     per_timestep_labels=True,
                     no_augment=False,
                     loss_over_bptt=False):
    """Full training augmentation: stretch -> loop -> fixed-length window.

    Produces a fixed output length regardless of stretch factor, so
    torch.compile doesn't retrace on each batch's sequence length.

    All operations are vectorized over the batch dimension — no per-sample
    Python loops.

    Args:
        batch_x: (batch, seq_len, features) numpy array.
        batch_y: (batch, seq_len) or (batch,) numpy array.
        rng: numpy RandomState.
        stretch_lo: Minimum stretch factor (1.0 = no stretch).
        stretch_hi: Maximum stretch factor.
        window_len: Fixed output length in timesteps.
        bptt_len: BPTT horizon in timesteps (last bptt_len steps get grads).
        per_timestep_labels: Whether labels are per-timestep.

    Returns:
        aug_x: (batch, window_len, features) numpy array.
        aug_y: (batch, window_len) or (batch,) numpy array.
        readout_idx: int -- timestep index for readout.
        bptt_start_idx: int -- where BPTT should begin (constant = window_len - bptt_len).
    """
    if no_augment:
        if batch_x.shape[1] != window_len:
            raise ValueError(
                f"no_augment=True requires batch seq_len == window_len, "
                f"got seq_len={batch_x.shape[1]} window_len={window_len}")
        bptt_start_idx = max(0, window_len - bptt_len)
        if loss_over_bptt:
            # Slice over the whole grad region so loss sees every timestep
            # (teacher-forced 1-step-ahead at each step).
            readout_idx = slice(bptt_start_idx, window_len)
        else:
            readout_idx = rng.randint(window_len - bptt_len, window_len)
        return batch_x, batch_y, readout_idx, bptt_start_idx

    # 1. Time stretch (vectorized) — 1 RNG call
    do_stretch = (abs(stretch_lo - stretch_hi) > 1e-6 or
                  abs(stretch_lo - 1.0) > 1e-6)
    if do_stretch:
        factor = random_stretch_factor(stretch_lo, stretch_hi, rng)
        batch_x, batch_y = time_stretch_batch(
            batch_x, batch_y, factor, per_timestep_labels)

    # 2. Palindrome loop enough to have >= window_len + loop_len timesteps
    #    (extra loop_len gives room to randomize the offset)
    seq_len = batch_x.shape[1]
    loop_len = 2 * seq_len
    n_loops = max(1, math.ceil((window_len + loop_len) / loop_len))
    looped_x, looped_y = palindrome_loop_batch(
        batch_x, batch_y, n_loops, per_timestep_labels)

    # 3. Fixed-length window
    T_total = looped_x.shape[1]
    max_offset = T_total - window_len
    offset = rng.randint(0, max_offset + 1) if max_offset > 0 else 0
    aug_x = looped_x[:, offset:offset + window_len]
    if per_timestep_labels:
        aug_y = looped_y[:, offset:offset + window_len]
    else:
        aug_y = batch_y

    # 4. Readout. BPTT: fixed horizon — bptt_start_idx constant across batches.
    #    loss_over_bptt=True scores every timestep in the grad region instead
    #    of one sampled timestep (teacher-forced 1-step-ahead at each step).
    bptt_start_idx = max(0, window_len - bptt_len)
    if loss_over_bptt:
        readout_idx = slice(bptt_start_idx, window_len)
    else:
        last_window = min(loop_len, bptt_len)
        readout_idx = rng.randint(window_len - last_window, window_len)

    return aug_x, aug_y, readout_idx, bptt_start_idx


def wrap_eval_batch(batch_x, batch_y,
                    window_len=1024,
                    per_timestep_labels=True,
                    no_augment=False,
                    loss_over_bptt=False,
                    bptt_len=512):
    """Eval augmentation: palindrome loop + fixed-length window (no stretch).

    Produces (batch, window_len, F) for shape parity with wrap_train_batch.
    Takes the last window_len timesteps of the palindrome-looped array so
    readout happens on the most "settled" dynamics.

    Args:
        batch_x: (batch, seq_len, features) numpy array.
        batch_y: (batch, seq_len) or (batch,) numpy array.
        window_len: Fixed output length in timesteps.
        per_timestep_labels: Whether labels are per-timestep.

    Returns:
        eval_x: (batch, window_len, features) numpy array.
        labels_at_readout: (batch,) or (batch, label_dim) -- label at readout.
        readout_idx: int -- last timestep (window_len - 1).
    """
    if no_augment:
        if batch_x.shape[1] != window_len:
            raise ValueError(
                f"no_augment=True requires batch seq_len == window_len, "
                f"got seq_len={batch_x.shape[1]} window_len={window_len}")
        if loss_over_bptt:
            bptt_start_idx = max(0, window_len - bptt_len)
            readout_idx = slice(bptt_start_idx, window_len)
            if per_timestep_labels:
                labels_at_readout = batch_y[:, readout_idx]
            else:
                labels_at_readout = batch_y
        else:
            readout_idx = window_len - 1
            if per_timestep_labels:
                labels_at_readout = batch_y[:, readout_idx]
            else:
                labels_at_readout = batch_y
        return batch_x, labels_at_readout, readout_idx

    seq_len = batch_x.shape[1]
    loop_len = 2 * seq_len
    n_loops = max(1, math.ceil(window_len / loop_len))

    looped_x, looped_y = palindrome_loop_batch(
        batch_x, batch_y, n_loops, per_timestep_labels)

    T_total = looped_x.shape[1]
    start = max(0, T_total - window_len)
    eval_x = looped_x[:, start:start + window_len]

    # Mirror wrap_train_batch: score the whole grad region when requested, so
    # train and eval losses stay on the same footing.
    if loss_over_bptt:
        bptt_start_idx = max(0, window_len - bptt_len)
        readout_idx = slice(bptt_start_idx, window_len)
        if per_timestep_labels:
            labels_at_readout = looped_y[
                :, start + bptt_start_idx : start + window_len]
        else:
            labels_at_readout = batch_y
    else:
        readout_idx = window_len - 1
        if per_timestep_labels:
            labels_at_readout = looped_y[:, start + readout_idx]
        else:
            labels_at_readout = batch_y

    return eval_x, labels_at_readout, readout_idx
