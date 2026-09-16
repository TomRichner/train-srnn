"""Fixed-step ODE integrators over a tensor or a tuple of tensors."""
from __future__ import annotations

from typing import Callable, TypeVar

import torch
from torch.utils._pytree import tree_map

Y = TypeVar("Y")


def euler_step(f: Callable[[Y], Y], y: Y, h: float) -> Y:
    return tree_map(lambda a, k: a + h * k, y, f(y))


def rk4_step(f: Callable[[Y], Y], y: Y, h: float) -> Y:
    """Classical Runge-Kutta: ``y + (k1 + 2 k2 + 2 k3 + k4) / 6`` with ``k_i = h f(.)``."""
    k1 = tree_map(lambda k: h * k, f(y))
    k2 = tree_map(lambda k: h * k, f(tree_map(lambda a, b: a + 0.5 * b, y, k1)))
    k3 = tree_map(lambda k: h * k, f(tree_map(lambda a, b: a + 0.5 * b, y, k2)))
    k4 = tree_map(lambda k: h * k, f(tree_map(lambda a, b: a + b, y, k3)))
    return tree_map(lambda a, b, c, d, e: a + (b + 2.0 * c + 2.0 * d + e) / 6.0,
                    y, k1, k2, k3, k4)


def sra1_step(f: Callable[[Y], Y], y: Y, h: float) -> Y:
    """Exact zero-diffusion limit of MATLAB sde_fixed_step's SRA1 tableau."""
    k1 = f(y)
    k2 = f(tree_map(lambda a, k: a + (3.0 * h / 4.0) * k, y, k1))
    return tree_map(lambda a, b, c: a + (h / 3.0) * (b + 2.0 * c), y, k1, k2)
