"""Numerical equivalence tests for cell_loop slice-assign vs. append+stack.

The continuous trainer and SequenceModel were refactored to write
per-timestep cell outputs into a pre-allocated ``(..., T, F)`` buffer by
slice-assign, replacing the prior ``outputs.append(out)`` +
``torch.stack(outputs, dim=-2)`` pattern. Both forward values and
gradients must be byte-identical — slice-assign dispatches to
``index_put_`` whose backward (``CopySlices``) is mathematically the
same as stack's backward.

This test pins that contract on a tiny generic RNN cell (GRUCell), so a
failure clearly indicates the slice-assign primitive itself is broken,
independent of any SRNN-specific behavior. The integration smoke tests
(the training smoke runs) cover
the full SRNN path.

Run: PYTHONPATH=. python scripts/test_cell_loop.py
"""
from __future__ import annotations

import sys

import torch
import torch.nn as nn

from train_srnn.utils.cell_loop import empty_time_buffer, mark_cudagraph_step


def _loop_append(cell: nn.Module, x: torch.Tensor, state: torch.Tensor):
    """Original pattern: list append + torch.stack."""
    outs = []
    for t in range(x.shape[1]):
        state = cell(x[:, t, :], state)
        outs.append(state)
    return torch.stack(outs, dim=-2), state


def _loop_slice(cell: nn.Module, x: torch.Tensor, state: torch.Tensor):
    """Refactored pattern: lazy pre-alloc + slice-assign."""
    T = x.shape[1]
    out_seq = None
    for t in range(T):
        mark_cudagraph_step()
        state = cell(x[:, t, :], state)
        if out_seq is None:
            out_seq = empty_time_buffer(state, T)
        out_seq[..., t, :] = state
    return out_seq, state


def _make_cell_and_inputs(*, B: int = 4, T: int = 8, F: int = 6, H: int = 5,
                          requires_grad: bool, seed: int = 0):
    torch.manual_seed(seed)
    cell = nn.GRUCell(F, H)
    x = torch.randn(B, T, F, requires_grad=requires_grad)
    state = torch.zeros(B, H)
    return cell, x, state


def test_forward_equivalence():
    """Forward outputs must be bit-identical."""
    cell, x, state = _make_cell_and_inputs(requires_grad=False)
    out_a, st_a = _loop_append(cell, x, state.clone())
    out_b, st_b = _loop_slice(cell, x, state.clone())
    assert torch.equal(out_a, out_b), \
        f"forward mismatch: max diff {(out_a - out_b).abs().max().item()}"
    assert torch.equal(st_a, st_b), "final state mismatch"
    print("[ok] forward equivalence")


def test_backward_equivalence():
    """Gradients into inputs and parameters must match within fp32 epsilon.

    The two backward paths are mathematically identical but may differ in
    fp32 reduction order (``torch.stack``'s backward unstacks and routes
    grad to each list element; ``index_put_``'s backward gathers via
    ``CopySlices``). Both paths sum the same per-step gradients, so any
    discrepancy is at single-ULP scale (≤ a few × 1e-7 for fp32).
    """
    cell_a, x_a, state_a = _make_cell_and_inputs(requires_grad=True, seed=1)
    cell_b, x_b, state_b = _make_cell_and_inputs(requires_grad=True, seed=1)

    for pa, pb in zip(cell_a.parameters(), cell_b.parameters()):
        assert torch.equal(pa, pb)

    out_a, _ = _loop_append(cell_a, x_a, state_a)
    out_b, _ = _loop_slice(cell_b, x_b, state_b)

    loss_a = (out_a * out_a).sum()
    loss_b = (out_b * out_b).sum()
    loss_a.backward()
    loss_b.backward()

    # fp32 ULP tolerance — losses are O(few × 10) here, so ~1e-6 absolute is safe.
    assert torch.allclose(x_a.grad, x_b.grad, atol=1e-6, rtol=1e-5), \
        f"input grad mismatch: max diff {(x_a.grad - x_b.grad).abs().max().item()}"
    for pa, pb in zip(cell_a.parameters(), cell_b.parameters()):
        ga, gb = pa.grad, pb.grad
        assert ga is not None and gb is not None
        assert torch.allclose(ga, gb, atol=1e-6, rtol=1e-5), \
            f"param grad mismatch on {pa.shape}: max diff {(ga - gb).abs().max().item()}"
    print("[ok] backward equivalence (input + parameter gradients within fp32 ULP)")


def test_buffer_dtype_device():
    """empty_time_buffer must inherit dtype + device from value."""
    v = torch.zeros(3, 5, dtype=torch.float32)
    buf = empty_time_buffer(v, T=7)
    assert buf.shape == (3, 7, 5)
    assert buf.dtype == torch.float32
    assert buf.device == v.device

    v_bf = torch.zeros(2, 3, 5, dtype=torch.bfloat16)
    buf_bf = empty_time_buffer(v_bf, T=4)
    assert buf_bf.shape == (2, 3, 4, 5)
    assert buf_bf.dtype == torch.bfloat16
    print("[ok] empty_time_buffer dtype/device")


def test_mark_cudagraph_step_noop():
    """mark_cudagraph_step must be safe to call outside compile capture."""
    # Call it many times — no error, no state.
    for _ in range(100):
        mark_cudagraph_step()
    print("[ok] mark_cudagraph_step is a no-op outside capture")


if __name__ == "__main__":
    test_forward_equivalence()
    test_backward_equivalence()
    test_buffer_dtype_device()
    test_mark_cudagraph_step_noop()
    print("\nAll cell_loop tests passed.")
    sys.exit(0)
