"""Gradient-norm clipping per variant.

``clip_grad_norm_`` takes one norm over the whole model, so in a K-batched
model a single variant with a large gradient would shrink every other
variant's step. Every trainable tensor is ``(K, ...)``-shaped and nothing
couples the variants, so each slice is clipped on its own.
"""

from __future__ import annotations

from typing import Iterable

import torch


@torch.no_grad()
def clip_grad_norm_per_variant(
    parameters: Iterable[torch.nn.Parameter],
    max_norm: float,
    K: int,
) -> torch.Tensor:
    """Clip each variant's gradient slice to ``max_norm``; returns the ``(K,)`` pre-clip norms."""
    grads: list[torch.Tensor] = []
    sq: torch.Tensor | None = None

    for p in parameters:
        if p.grad is None:
            continue
        g = p.grad
        if g.dim() == 0 or g.shape[0] != K:
            raise ValueError(
                f"clip_grad_norm_per_variant expects every gradient to be "
                f"(K={K}, ...)-shaped; got {tuple(g.shape)}. A parameter "
                f"shared across variants cannot be clipped per variant."
            )
        s = (g.reshape(K, -1).float() ** 2).sum(1)
        sq = s if sq is None else sq + s
        grads.append(g)

    if sq is None:  # nothing had a gradient
        return torch.zeros(K)

    norms = sq.sqrt()
    scale = (max_norm / (norms + 1e-6)).clamp(max=1.0)
    for g in grads:
        g.mul_(scale.view(-1, *([1] * (g.dim() - 1))).to(g.dtype))
    return norms
