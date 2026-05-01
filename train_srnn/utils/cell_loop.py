"""Helpers for CUDA-graph-friendly per-timestep RNN loops.

Every loop site that calls ``cell(x_t, state)`` across ``T`` timesteps
follows the same pattern:

    out_seq = None
    for t in range(T):
        mark_cudagraph_step()
        out, state = cell(x[:, t, :], state)
        state = state.clone()
        if out_seq is None:
            out_seq = empty_time_buffer(out, T)
        out_seq[..., t, :] = out

This module owns the two non-trivial primitives so the contract lives in
one place. ``state.clone()`` is intentionally not wrapped — it's a single
method call and inlining keeps each call site self-explanatory.
"""
import torch


def mark_cudagraph_step() -> None:
    """Signal end-of-step to the CUDA-graph trees allocator.

    Required at the top of the loop body when ``cell`` is compiled with
    ``mode='reduce-overhead'`` so the graph manager can recycle static
    output buffers between iterations. No-op outside CUDA-graph capture,
    so safe to call unconditionally.
    """
    torch.compiler.cudagraph_mark_step_begin()


def empty_time_buffer(value: torch.Tensor, T: int) -> torch.Tensor:
    """Allocate ``(*value.shape[:-1], T, value.shape[-1])`` uninitialized.

    Matches ``value``'s dtype and device. Designed to be filled by
    slice-assign ``buf[..., t, :] = value`` across a ``T``-step loop.
    Slice-assign is autograd-safe (dispatches to ``index_put_``).
    """
    return torch.empty(
        value.shape[:-1] + (T, value.shape[-1]),
        device=value.device,
        dtype=value.dtype,
    )
