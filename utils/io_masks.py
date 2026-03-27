"""Neuron partitioning utilities for I/O masking in structured RNNs."""

import numpy as np


def generate_neuron_partition(n, seed=0, frac_input=0.25, frac_output=0.25):
    """Partition neurons into input, interneuron, and output groups.

    Args:
        n: Total number of neurons.
        seed: Random seed for reproducible partitioning.
        frac_input: Fraction of neurons that receive external input.
        frac_output: Fraction of neurons used for readout.

    Returns:
        (input_indices, inter_indices, output_indices) -- each is an array of
        neuron indices.
    """
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n)
    n_input = max(1, int(n * frac_input))
    n_output = max(1, int(n * frac_output))
    return indices[:n_input], indices[n_input:-n_output], indices[-n_output:]


def make_input_mask(n, input_indices):
    """Binary mask (n,): 1 for input neurons, 0 for others.

    Applied to W_in rows to restrict which neurons receive input.
    """
    mask = np.zeros(n, dtype=np.float32)
    mask[input_indices] = 1.0
    return mask


def make_output_mask(n, output_indices):
    """Binary mask (n,): 1 for output neurons, 0 for others.

    Applied to hidden state before readout Dense layer.
    """
    mask = np.zeros(n, dtype=np.float32)
    mask[output_indices] = 1.0
    return mask
