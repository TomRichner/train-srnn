"""Passing the hoisted effective weight into the cell gives the same outputs and gradients
as letting the cell rebuild it every step.

"""
from __future__ import annotations

import copy
import sys

import torch
from omegaconf import DictConfig

from train_srnn.config import compose_config

from train_srnn.models.factory import build_model


def _make_cfg(num_units: int = 16, n_features: int = 4,
              no_sfa: bool = True) -> DictConfig:
    """Small autoregressive SRNN config: input_size == output_size == 4."""
    overrides = [
        "model=srnn", "task=cheetah100", "seed=0",
        f"model.num_units={num_units}",
        f"task.input_size={n_features}", f"task.output_size={n_features}",
        "model.solver=rk4", "task.h=0.005", "task.ode_unfolds=1",
    ]
    if no_sfa:
        overrides += ["model.n_a_E=0", "model.n_a_I=0", "model.n_b_E=0", "model.n_b_I=0"]
    return compose_config(overrides)


def _zero_grads(mod):
    for p in mod.parameters():
        if p.grad is not None:
            p.grad.zero_()


def _grads(mod):
    return {n: p.grad.detach().clone()
            for n, p in mod.named_parameters() if p.grad is not None}


def _assert_grads_byte_identical(grads_a, grads_b):
    assert grads_a.keys() == grads_b.keys(), (
        f"grad name mismatch: "
        f"{sorted(grads_a.keys())} vs {sorted(grads_b.keys())}"
    )
    for name in grads_a:
        ga, gb = grads_a[name], grads_b[name]
        assert ga.shape == gb.shape, f"{name}: shape {ga.shape} vs {gb.shape}"
        assert torch.equal(ga, gb), (
            f"{name}: grads differ. "
            f"max abs diff = {(ga - gb).abs().max().item():.3e}"
        )


def _assert_grads_close(grads_a, grads_b, atol=1e-6, rtol=1e-5):
    """Multi-step variant: per-step backward accumulates dL/dW_raw via repeated
    in-place ``.grad +=`` adds, whereas the hoisted path flows a single
    backward through the shared W_eff tensor (a single sum). The math is
    identical, but float-add ordering differs by a couple of ULPs at fp32.
    """
    assert grads_a.keys() == grads_b.keys()
    for name in grads_a:
        ga, gb = grads_a[name], grads_b[name]
        assert ga.shape == gb.shape, f"{name}: shape {ga.shape} vs {gb.shape}"
        assert torch.allclose(ga, gb, atol=atol, rtol=rtol), (
            f"{name}: grad max diff = {(ga - gb).abs().max().item():.3e} "
            f"(atol={atol}, rtol={rtol})"
        )


# Cell-level equivalence

def _run_cell_pair(cell_a, cell_b, inputs, state):
    """Forward+backward through cell_a with internal W_eff vs cell_b with
    caller-hoisted W_eff. Returns (out_a, out_b, grads_a, grads_b)."""
    _zero_grads(cell_a); _zero_grads(cell_b)

    # Path A: cell rebuilds W_eff internally (legacy signature).
    out_a, state_a = cell_a(inputs, state)
    (out_a.pow(2).sum() + state_a.pow(2).sum()).backward()
    grads_a = _grads(cell_a)

    # Path B: caller hoists W_eff and passes in.
    W_eff = cell_b._effective_W()
    out_b, state_b = cell_b(inputs, state, W_eff=W_eff)
    (out_b.pow(2).sum() + state_b.pow(2).sum()).backward()
    grads_b = _grads(cell_b)

    return out_a, state_a, out_b, state_b, grads_a, grads_b


def test_single_cell_forward_backward_equivalence():
    """SRNNCell: hoisted W_eff vs internal — byte-identical fp32."""
    torch.manual_seed(0)
    cfg = _make_cfg()
    model_a = build_model(cfg).eval()  # eval to disable any dropout/etc
    model_b = copy.deepcopy(model_a)
    cell_a, cell_b = model_a.cell, model_b.cell

    B = 4
    torch.manual_seed(1)
    inputs = torch.randn(B, cfg.task.input_size)
    state = torch.randn(1, B, cell_a.state_size)

    out_a, state_a, out_b, state_b, ga, gb = _run_cell_pair(
        cell_a, cell_b, inputs, state,
    )
    assert torch.equal(out_a, out_b), (
        f"output diverges: max diff = {(out_a - out_b).abs().max():.3e}"
    )
    assert torch.equal(state_a, state_b), (
        f"state diverges: max diff = {(state_a - state_b).abs().max():.3e}"
    )
    _assert_grads_byte_identical(ga, gb)


def test_batched_cell_forward_backward_equivalence():
    """SRNNCell: hoisted W_eff vs internal — byte-identical fp32."""
    torch.manual_seed(0)
    cfg = _make_cfg()
    abls = ["srnn-e-only-per-neuron", "srnn-e-only-skip-per-neuron"]
    model_a = build_model(cfg, abls).eval()
    model_b = copy.deepcopy(model_a)
    cell_a, cell_b = model_a.cell, model_b.cell

    K, B = cell_a.K, 4
    torch.manual_seed(2)
    inputs = torch.randn(B, cfg.task.input_size)
    state = torch.randn(K, B, cell_a.state_size)

    out_a, state_a, out_b, state_b, ga, gb = _run_cell_pair(
        cell_a, cell_b, inputs, state,
    )
    assert torch.equal(out_a, out_b), (
        f"output diverges: max diff = {(out_a - out_b).abs().max():.3e}"
    )
    assert torch.equal(state_a, state_b), (
        f"state diverges: max diff = {(state_a - state_b).abs().max():.3e}"
    )
    _assert_grads_byte_identical(ga, gb)


# Multi-step equivalence (mirrors the per-timestep BPTT loop)

def test_single_cell_multistep_equivalence():
    """T-step unroll: hoist W_eff once vs rebuild every step. Outputs are
    byte-identical; grads match within fp32 ULP (different float-add order
    when accumulating dL/dW_raw across timesteps — see _assert_grads_close)."""
    torch.manual_seed(0)
    cfg = _make_cfg()
    model_a = build_model(cfg).eval()
    model_b = copy.deepcopy(model_a)
    cell_a, cell_b = model_a.cell, model_b.cell

    B, T = 4, 12
    torch.manual_seed(3)
    xs = torch.randn(T, B, cfg.task.input_size)
    s0 = torch.randn(1, B, cell_a.state_size)

    _zero_grads(cell_a); _zero_grads(cell_b)

    # Path A: rebuild W_eff every step (legacy).
    state = s0.clone()
    outs_a = []
    for t in range(T):
        out, state = cell_a(xs[t], state)
        outs_a.append(out)
    Y_a = torch.stack(outs_a, dim=0)
    (Y_a.pow(2).sum() + state.pow(2).sum()).backward()
    grads_a = _grads(cell_a)

    # Path B: hoist once outside the loop.
    state = s0.clone()
    W_eff = cell_b._effective_W()
    outs_b = []
    for t in range(T):
        out, state = cell_b(xs[t], state, W_eff=W_eff)
        outs_b.append(out)
    Y_b = torch.stack(outs_b, dim=0)
    (Y_b.pow(2).sum() + state.pow(2).sum()).backward()
    grads_b = _grads(cell_b)

    assert torch.equal(Y_a, Y_b), (
        f"multi-step output diverges: max diff = "
        f"{(Y_a - Y_b).abs().max().item():.3e}"
    )
    _assert_grads_close(grads_a, grads_b)


def test_batched_cell_multistep_equivalence():
    torch.manual_seed(0)
    cfg = _make_cfg()
    abls = ["srnn-e-only-per-neuron", "srnn-e-only-skip-per-neuron"]
    model_a = build_model(cfg, abls).eval()
    model_b = copy.deepcopy(model_a)
    cell_a, cell_b = model_a.cell, model_b.cell

    K, B, T = cell_a.K, 4, 12
    torch.manual_seed(4)
    xs = torch.randn(T, B, cfg.task.input_size)
    s0 = torch.randn(K, B, cell_a.state_size)

    _zero_grads(cell_a); _zero_grads(cell_b)

    state = s0.clone()
    outs_a = []
    for t in range(T):
        out, state = cell_a(xs[t], state)
        outs_a.append(out)
    Y_a = torch.stack(outs_a, dim=0)
    (Y_a.pow(2).sum() + state.pow(2).sum()).backward()
    grads_a = _grads(cell_a)

    state = s0.clone()
    W_eff = cell_b._effective_W()
    outs_b = []
    for t in range(T):
        out, state = cell_b(xs[t], state, W_eff=W_eff)
        outs_b.append(out)
    Y_b = torch.stack(outs_b, dim=0)
    (Y_b.pow(2).sum() + state.pow(2).sum()).backward()
    grads_b = _grads(cell_b)

    assert torch.equal(Y_a, Y_b), (
        f"multi-step output diverges: max diff = "
        f"{(Y_a - Y_b).abs().max().item():.3e}"
    )
    _assert_grads_close(grads_a, grads_b)



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
