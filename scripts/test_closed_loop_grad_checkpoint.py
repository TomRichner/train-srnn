"""Gradient + forward equivalence under closed-loop grad_checkpoint.

This is the high-fidelity correctness test for the closed-loop checkpointing
path. We run the same forward+backward twice on identically-initialized
models — once with `grad_checkpoint=False`, once with `=True` — and verify
that per-parameter gradients match within tight numerical tolerance, and
that forward outputs are bit-identical.

If these pass, the checkpoint path's autograd plumbing is correct.

Run: PYTHONPATH=. python scripts/test_closed_loop_grad_checkpoint.py
or:  pytest scripts/test_closed_loop_grad_checkpoint.py -v
"""
from __future__ import annotations

import copy
import sys

import torch
from omegaconf import OmegaConf

from train_srnn.models.factory import build_batched_model, build_model


def _make_seeg_cfg(num_units: int = 16, n_features: int = 4,
                   no_sfa: bool = True) -> OmegaConf:
    """Minimal SEEG-shaped config matching test_closed_loop_forward.py."""
    model_cfg = {
        "type": "srnn",
        "name": "srnn-test",
        "num_units": num_units,
        "frac_E": 0.75,
        "alpha": 1.0 / 3.0,
        "level_of_chaos": 1.0,
        "solver": "rk4",
        "h": 0.005,
        "ode_unfolds": 1,
    }
    if no_sfa:
        model_cfg.update({"n_a_E": 0, "n_a_I": 0, "n_b_E": 0, "n_b_I": 0})
    return OmegaConf.create({
        "seed": 0,
        "size": num_units,
        "model": model_cfg,
        "task": {
            "task_type": "regression",
            "input_size": n_features,
            "output_size": n_features,
        },
    })


def _build_paired_models_single():
    """Build two SequenceModels initialized identically."""
    torch.manual_seed(0)
    cfg = _make_seeg_cfg()
    a = build_model(cfg)
    b = copy.deepcopy(a)
    return cfg, a, b


def _build_paired_models_batched(ablations):
    torch.manual_seed(0)
    cfg = _make_seeg_cfg()
    cfg.batched_ablations = ablations
    a = build_batched_model(cfg, ablations)
    b = copy.deepcopy(a)
    return cfg, a, b


def _zero_grads(model):
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()


def _run_pair(model_a, model_b, x, alpha, *, readout_idx, bptt_start_idx,
              bptt_chunk_len, grad_checkpoint_segment_len=None):
    """Run identical forward+backward on two models, one without ckpt and
    one with. Returns (y_a, y_b).
    """
    model_a.train(); model_b.train()
    _zero_grads(model_a); _zero_grads(model_b)

    y_a = model_a(x, readout_idx=readout_idx, alpha_schedule=alpha,
                  bptt_start_idx=bptt_start_idx, bptt_chunk_len=bptt_chunk_len,
                  grad_checkpoint=False)
    y_a.pow(2).sum().backward()

    y_b = model_b(x, readout_idx=readout_idx, alpha_schedule=alpha,
                  bptt_start_idx=bptt_start_idx, bptt_chunk_len=bptt_chunk_len,
                  grad_checkpoint=True,
                  grad_checkpoint_segment_len=grad_checkpoint_segment_len)
    y_b.pow(2).sum().backward()

    return y_a, y_b


def _assert_grads_close(model_a, model_b, atol=1e-5, rtol=1e-4):
    pairs_checked = 0
    for (na, pa), (nb, pb) in zip(model_a.named_parameters(),
                                   model_b.named_parameters()):
        assert na == nb, f"param name mismatch: {na} vs {nb}"
        if pa.grad is None and pb.grad is None:
            continue
        if pa.grad is None or pb.grad is None:
            raise AssertionError(f"{na}: grad presence mismatch "
                                 f"({pa.grad is None} vs {pb.grad is None})")
        diff = (pa.grad - pb.grad).abs().max().item()
        assert torch.allclose(pa.grad, pb.grad, atol=atol, rtol=rtol), (
            f"{na}: grad max diff = {diff:.2e} "
            f"(atol={atol}, rtol={rtol})"
        )
        pairs_checked += 1
    assert pairs_checked > 0, "no parameters had gradients to compare"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_grad_equivalence_single_mode():
    cfg, a, b = _build_paired_models_single()
    B, T, C = 2, 24, cfg.task.input_size
    torch.manual_seed(1); x = torch.randn(B, T, C)
    alpha = torch.full((T, C), 0.4); alpha[0].zero_()

    y_a, y_b = _run_pair(a, b, x, alpha, readout_idx=slice(0, T),
                          bptt_start_idx=4, bptt_chunk_len=8)
    assert torch.equal(y_a, y_b), \
        f"forward outputs differ: max diff = {(y_a - y_b).abs().max().item():.2e}"
    _assert_grads_close(a, b)


def test_grad_equivalence_k_batched_no_skip():
    # All-per-neuron variants. Avoids KnownIssues #6 (the _install_vec_mask
    # grad-hook does not fire under torch.utils.checkpoint when use_reentrant=
    # False; mask is all-ones for per-neuron variants so hook is a no-op).
    abls = ["srnn-e-only-per-neuron", "srnn-sfa-e-only-per-neuron"]
    cfg, a, b = _build_paired_models_batched(abls)
    B, T, C = 2, 24, cfg.task.input_size
    torch.manual_seed(2); x = torch.randn(B, T, C)
    alpha = torch.full((T, C), 0.5); alpha[0].zero_()

    y_a, y_b = _run_pair(a, b, x, alpha, readout_idx=slice(0, T),
                          bptt_start_idx=4, bptt_chunk_len=8)
    assert torch.equal(y_a, y_b)
    _assert_grads_close(a, b)


def test_grad_equivalence_with_skip():
    # All-per-neuron variants matching the cl250 production config (one skip,
    # one not). See note in test_grad_equivalence_k_batched_no_skip.
    abls = ["srnn-e-only-per-neuron", "srnn-e-only-skip-per-neuron"]
    cfg, a, b = _build_paired_models_batched(abls)
    B, T, C = 2, 24, cfg.task.input_size
    torch.manual_seed(3); x = torch.randn(B, T, C)
    alpha = torch.full((T, C), 0.6); alpha[0].zero_()

    y_a, y_b = _run_pair(a, b, x, alpha, readout_idx=slice(0, T),
                          bptt_start_idx=4, bptt_chunk_len=8)
    assert torch.equal(y_a, y_b)
    _assert_grads_close(a, b)


def test_grad_equivalence_segment_len_smaller_than_chunk():
    """grad_checkpoint_segment_len < bptt_chunk_len: more checkpoint
    boundaries inside each chunk. Result must still match."""
    cfg, a, b = _build_paired_models_single()
    B, T, C = 2, 24, cfg.task.input_size
    torch.manual_seed(4); x = torch.randn(B, T, C)
    alpha = torch.full((T, C), 0.4); alpha[0].zero_()

    y_a, y_b = _run_pair(a, b, x, alpha, readout_idx=slice(0, T),
                          bptt_start_idx=4, bptt_chunk_len=16,
                          grad_checkpoint_segment_len=4)
    assert torch.equal(y_a, y_b)
    _assert_grads_close(a, b)


def test_grad_equivalence_no_warmup():
    """bptt_start_idx=0: the warmup region is empty, so the closed-loop
    branch goes straight into the grad/checkpoint loop from t=0."""
    cfg, a, b = _build_paired_models_single()
    B, T, C = 2, 16, cfg.task.input_size
    torch.manual_seed(5); x = torch.randn(B, T, C)
    alpha = torch.full((T, C), 0.4); alpha[0].zero_()

    y_a, y_b = _run_pair(a, b, x, alpha, readout_idx=slice(0, T),
                          bptt_start_idx=0, bptt_chunk_len=8)
    assert torch.equal(y_a, y_b)
    _assert_grads_close(a, b)


def test_grad_equivalence_no_chunk_len():
    """bptt_chunk_len=None: the entire grad region is one segment, optionally
    one big checkpoint. Verify that path too."""
    cfg, a, b = _build_paired_models_single()
    B, T, C = 2, 16, cfg.task.input_size
    torch.manual_seed(6); x = torch.randn(B, T, C)
    alpha = torch.full((T, C), 0.4); alpha[0].zero_()

    y_a, y_b = _run_pair(a, b, x, alpha, readout_idx=slice(0, T),
                          bptt_start_idx=4, bptt_chunk_len=None)
    assert torch.equal(y_a, y_b)
    _assert_grads_close(a, b)


def test_grad_equivalence_int_readout_idx():
    """Last-step readout (int) instead of slice: the inner unroll is
    identical, only the final select differs."""
    cfg, a, b = _build_paired_models_single()
    B, T, C = 2, 20, cfg.task.input_size
    torch.manual_seed(7); x = torch.randn(B, T, C)
    alpha = torch.full((T, C), 0.3); alpha[0].zero_()

    y_a, y_b = _run_pair(a, b, x, alpha, readout_idx=T - 1,
                          bptt_start_idx=4, bptt_chunk_len=8)
    assert torch.equal(y_a, y_b)
    _assert_grads_close(a, b)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    fns = [v for k, v in globals().items()
           if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {fn.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"  ERROR {fn.__name__}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
