"""Per-variant gradient-norm clipping for K-batched ablation models.

``torch.nn.utils.clip_grad_norm_`` takes ONE norm over every parameter in the
model and rescales all gradients by a single factor.  In ``batched_ablations``
mode every trainable tensor is ``(K, ...)``-shaped, so that single factor is
shared by all K variants: one variant with a large gradient throttles the step
size of all the others (KnownIssues §14).

Because the leading axis is the variant axis and nothing else couples the
variants, the slices are separable and clipping can be done per variant with
one extra reduction per parameter and no host sync.
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
    """Clip each batched variant's gradient slice independently.

    Args:
        parameters: Iterable of parameters.  Every one carrying a gradient must
            be ``(K, ...)``-shaped — true for all ``SRNNCell`` /
            batched ``SequenceModel`` parameters.
        max_norm: Per-variant max L2 norm.
        K: Number of batched variants.

    Returns:
        ``(K,)`` tensor of the *pre-clip* gradient norms, one per variant.
        Useful for per-variant gradient-norm logging.

    Raises:
        ValueError: If a parameter with a gradient is not ``(K, ...)``-shaped,
            which would mean the variants share a trainable tensor and per-
            variant clipping is not well defined.
    """
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
