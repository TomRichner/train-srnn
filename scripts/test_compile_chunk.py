"""Equivalence + traceability tests for the chunk-compile spike refactor.

The continuous trainer's chunk forwards (``_forward_chunk_pure_tf`` and
``_forward_chunk_closed_loop``) were refactored to:

  - Pre-allocate ``hidden_seq`` (and ``x_in_seq`` for closed-loop) up front
    instead of on the first iteration; this removes the ``if hidden_seq is
    None`` branch that Dynamo would otherwise specialize on.
  - Drop the per-step ``mark_cudagraph_step()`` call from inside the loop
    (the trainer marks once per chunk now).
  - Read output dim from ``cell.num_units`` (newly added attribute on
    SRNNCell and BatchedSRNNCell, alias of LSTMCellWrapper.num_units).

The eager-path math is unchanged, so paired runs against a from-scratch
reference unroll must match byte-for-byte at fp32. ``torch.compile`` in
default mode (no CUDA graphs) is also exercised on CPU to catch Dynamo
trace failures and verify no spurious recompiles fire on repeated calls
with identical shapes.

Reduce-overhead / CUDA-graph behavior is **not** testable on local Mac
(no CUDA). That ships in Phase B (cloud dispatch); see CompileChunkOutline.md.

Run: PYTHONPATH=. python scripts/test_compile_chunk.py
or:  pytest scripts/test_compile_chunk.py -v
"""
from __future__ import annotations

import io
import logging
import sys

import torch
from omegaconf import OmegaConf

from train_srnn.models.factory import build_batched_model, build_model
from train_srnn.training.continuous import (
    _attach_loss_and_metric,
    _forward_chunk_closed_loop,
    _forward_chunk_closed_loop_with_loss,
    _forward_chunk_pure_tf,
    _forward_chunk_pure_tf_with_loss,
    _readout_chunk,
)


# ---------------------------------------------------------------------------
# Fixtures (mirror scripts/test_effective_w_hoist.py)
# ---------------------------------------------------------------------------

def _make_seeg_cfg(num_units: int = 12, n_features: int = 4) -> OmegaConf:
    return OmegaConf.create({
        "seed": 0,
        "size": num_units,
        "model": {
            "type": "srnn",
            "name": "srnn-test",
            "num_units": num_units,
            "frac_E": 0.75,
            "alpha": 1.0 / 3.0,
            "level_of_chaos": 1.0,
            "solver": "rk4",
            "h": 0.005,
            "ode_unfolds": 1,
        },
        "task": {
            "task_type": "regression",
            "input_size": n_features,
            "output_size": n_features,
        },
    })


def _zero_grads(*mods):
    for m in mods:
        for p in m.parameters():
            if p.grad is not None:
                p.grad.zero_()


def _grads(model):
    return {n: p.grad.detach().clone()
            for n, p in model.named_parameters() if p.grad is not None}


def _assert_grads_byte_identical(grads_a, grads_b):
    assert grads_a.keys() == grads_b.keys(), (
        f"grad name mismatch: {sorted(grads_a.keys())} vs {sorted(grads_b.keys())}"
    )
    for name in grads_a:
        ga, gb = grads_a[name], grads_b[name]
        assert ga.shape == gb.shape, f"{name}: {ga.shape} vs {gb.shape}"
        assert torch.equal(ga, gb), (
            f"{name}: max abs diff = {(ga - gb).abs().max().item():.3e}"
        )


# ---------------------------------------------------------------------------
# Reference implementations (independent, in-test, eager unrolls)
# ---------------------------------------------------------------------------

def _ref_pure_tf(model, cell, chunk_x, state):
    """Plain teacher-forced unroll with append+stack — the simplest possible
    reference. Production ``_forward_chunk_pure_tf`` must match this byte-
    for-byte at fp32 (the math is identical; only buffer-allocation strategy
    differs).
    """
    T = chunk_x.shape[1]
    W_eff = cell._effective_W() if hasattr(cell, "_effective_W") else None
    hs = []
    for t in range(T):
        if W_eff is not None:
            h_t, state = cell(chunk_x[:, t, :], state, W_eff=W_eff)
        else:
            h_t, state = cell(chunk_x[:, t, :], state)
        state = state.clone()
        hs.append(h_t)
    hidden_seq = torch.stack(hs, dim=-2)
    if hidden_seq.dim() == 4:
        x_in_seq = chunk_x.unsqueeze(0).expand(hidden_seq.shape[0], -1, -1, -1)
    else:
        x_in_seq = chunk_x
    # Reference readout: per-step _readout_one + stack.
    ys = []
    for t in range(T):
        if hidden_seq.dim() == 4:
            h_t = hidden_seq[:, :, t, :]
            x_t = x_in_seq[:, :, t, :] if x_in_seq.dim() == 4 else x_in_seq[:, t, :]
        else:
            h_t = hidden_seq[:, t, :]
            x_t = x_in_seq[:, t, :]
        ys.append(model._readout_one(h_t, x_t))
    logits = torch.stack(ys, dim=-2)
    return logits, state


def _ref_closed_loop(model, cell, chunk_x, state, y_prev, alpha_chunk):
    """Reference closed-loop unroll: identical math to the production
    function, with append+stack accumulators."""
    T = chunk_x.shape[1]
    W_eff = cell._effective_W() if hasattr(cell, "_effective_W") else None
    hs, xs = [], []
    for t in range(T):
        alpha_t = alpha_chunk[:, t, :]
        x_real_t = chunk_x[:, t, :]
        x_in_t = (1.0 - alpha_t) * x_real_t + alpha_t * y_prev
        if W_eff is not None:
            h_t, state = cell(x_in_t, state, W_eff=W_eff)
        else:
            h_t, state = cell(x_in_t, state)
        state = state.clone()
        y_prev = model._readout_one(h_t, x_in_t)
        hs.append(h_t)
        xs.append(x_in_t)
    hidden_seq = torch.stack(hs, dim=-2)
    x_in_seq = torch.stack(xs, dim=-2)
    ys = []
    for t in range(T):
        if hidden_seq.dim() == 4:
            h_t = hidden_seq[:, :, t, :]
            x_t = x_in_seq[:, :, t, :]
        else:
            h_t = hidden_seq[:, t, :]
            x_t = x_in_seq[:, t, :]
        ys.append(model._readout_one(h_t, x_t))
    logits = torch.stack(ys, dim=-2)
    return logits, state, y_prev


# ---------------------------------------------------------------------------
# Refactor equivalence: pure-TF
# ---------------------------------------------------------------------------

def _build_pair(batched: bool):
    """Return two byte-identical fp32 models (for paired forward+backward)."""
    import copy
    cfg = _make_seeg_cfg()
    if batched:
        abls = ["srnn-e-only-per-neuron", "srnn-e-only-skip-per-neuron"]
        cfg.batched_ablations = abls
        m_a = build_batched_model(cfg, abls).eval()
    else:
        m_a = build_model(cfg).eval()
    m_b = copy.deepcopy(m_a)
    return cfg, m_a, m_b


def _make_inputs(cfg, model, B: int, T: int, *, with_alpha: bool):
    """Random fp32 chunk_x, state, y_prev, alpha for paired tests."""
    K = getattr(model.cell, "K", None)
    C = cfg.task.input_size
    torch.manual_seed(7)
    chunk_x = torch.randn(B, T, C)
    if K is not None:
        state = torch.randn(K, B, model.cell.state_size)
    else:
        state = torch.randn(B, model.cell.state_size)
    if with_alpha:
        if K is not None:
            y_prev = torch.randn(K, B, C)
        else:
            y_prev = torch.randn(B, C)
        # alpha in [0, 1], slot 0 zeroed (matches production sampler).
        alpha = torch.rand(B, T, C)
        alpha[:, 0, :] = 0.0
    else:
        y_prev = None
        alpha = None
    return chunk_x, state, y_prev, alpha


def _run_pure_tf_pair(model_a, model_b, chunk_x, state):
    _zero_grads(model_a, model_b)
    logits_a, _ = _ref_pure_tf(model_a, model_a.cell, chunk_x, state.clone())
    logits_a.pow(2).sum().backward()
    grads_a = _grads(model_a)

    logits_b, _ = _forward_chunk_pure_tf(model_b, model_b.cell, chunk_x, state.clone())
    logits_b.pow(2).sum().backward()
    grads_b = _grads(model_b)
    return logits_a, logits_b, grads_a, grads_b


def test_pure_tf_refactor_equivalence_single():
    """SRNNCell pure-TF: production fn vs reference unroll — byte-identical."""
    cfg, m_a, m_b = _build_pair(batched=False)
    chunk_x, state, _, _ = _make_inputs(cfg, m_a, B=4, T=8, with_alpha=False)
    la, lb, ga, gb = _run_pure_tf_pair(m_a, m_b, chunk_x, state)
    assert torch.equal(la, lb), (
        f"logits diverge: max diff = {(la - lb).abs().max().item():.3e}"
    )
    _assert_grads_byte_identical(ga, gb)


def test_pure_tf_refactor_equivalence_batched():
    """BatchedSRNNCell pure-TF: production fn vs reference — byte-identical."""
    cfg, m_a, m_b = _build_pair(batched=True)
    chunk_x, state, _, _ = _make_inputs(cfg, m_a, B=4, T=8, with_alpha=False)
    la, lb, ga, gb = _run_pure_tf_pair(m_a, m_b, chunk_x, state)
    assert torch.equal(la, lb), (
        f"logits diverge: max diff = {(la - lb).abs().max().item():.3e}"
    )
    _assert_grads_byte_identical(ga, gb)


# ---------------------------------------------------------------------------
# Refactor equivalence: closed-loop
# ---------------------------------------------------------------------------

def _run_cl_pair(model_a, model_b, chunk_x, state, y_prev, alpha):
    _zero_grads(model_a, model_b)
    la, _, _ = _ref_closed_loop(
        model_a, model_a.cell, chunk_x, state.clone(),
        y_prev.clone(), alpha,
    )
    la.pow(2).sum().backward()
    grads_a = _grads(model_a)

    lb, _, _ = _forward_chunk_closed_loop(
        model_b, model_b.cell, chunk_x, state.clone(),
        y_prev.clone(), alpha,
    )
    lb.pow(2).sum().backward()
    grads_b = _grads(model_b)
    return la, lb, grads_a, grads_b


def test_closed_loop_refactor_equivalence_single():
    cfg, m_a, m_b = _build_pair(batched=False)
    chunk_x, state, y_prev, alpha = _make_inputs(cfg, m_a, B=4, T=8, with_alpha=True)
    la, lb, ga, gb = _run_cl_pair(m_a, m_b, chunk_x, state, y_prev, alpha)
    assert torch.equal(la, lb), (
        f"closed-loop logits diverge: max diff = "
        f"{(la - lb).abs().max().item():.3e}"
    )
    _assert_grads_byte_identical(ga, gb)


def test_closed_loop_refactor_equivalence_batched():
    cfg, m_a, m_b = _build_pair(batched=True)
    chunk_x, state, y_prev, alpha = _make_inputs(cfg, m_a, B=4, T=8, with_alpha=True)
    la, lb, ga, gb = _run_cl_pair(m_a, m_b, chunk_x, state, y_prev, alpha)
    assert torch.equal(la, lb), (
        f"closed-loop logits diverge: max diff = "
        f"{(la - lb).abs().max().item():.3e}"
    )
    _assert_grads_byte_identical(ga, gb)


# ---------------------------------------------------------------------------
# Readout helper equivalence
# ---------------------------------------------------------------------------

def test_readout_chunk_refactor_equivalence_single():
    """_readout_chunk pre-alloc == stack reference, single-cell shapes."""
    cfg, m_a, _ = _build_pair(batched=False)
    B, T, N, C = 4, 8, m_a.cell.num_units, cfg.task.input_size
    torch.manual_seed(11)
    hidden_seq = torch.randn(B, T, N)
    x_in_seq = torch.randn(B, T, C)
    out_prod = _readout_chunk(m_a, hidden_seq, x_in_seq)
    # Reference: stack + readout per step.
    ys = [m_a._readout_one(hidden_seq[:, t, :], x_in_seq[:, t, :])
          for t in range(T)]
    out_ref = torch.stack(ys, dim=-2)
    assert torch.equal(out_prod, out_ref), (
        f"max diff = {(out_prod - out_ref).abs().max().item():.3e}"
    )


def test_readout_chunk_refactor_equivalence_batched():
    """_readout_chunk pre-alloc == stack reference, K-batched shapes."""
    cfg, m_a, _ = _build_pair(batched=True)
    K = m_a.cell.K
    B, T, N, C = 4, 8, m_a.cell.num_units, cfg.task.input_size
    torch.manual_seed(13)
    hidden_seq = torch.randn(K, B, T, N)
    x_in_seq = torch.randn(B, T, C)  # broadcast: (B, T, C) is supported
    out_prod = _readout_chunk(m_a, hidden_seq, x_in_seq)
    ys = [m_a._readout_one(hidden_seq[:, :, t, :], x_in_seq[:, t, :])
          for t in range(T)]
    out_ref = torch.stack(ys, dim=-2)
    assert torch.equal(out_prod, out_ref), (
        f"max diff = {(out_prod - out_ref).abs().max().item():.3e}"
    )


# ---------------------------------------------------------------------------
# torch.compile traceability on CPU (default mode only)
# ---------------------------------------------------------------------------

def _has_compile() -> bool:
    """torch.compile is available since PyTorch 2.0; we require ≥ 2.2."""
    return hasattr(torch, "compile")


def test_chunk_compile_traceable_default_mode_cpu():
    """torch.compile(_forward_chunk_pure_tf) traces and runs on CPU; output
    matches the eager call."""
    if not _has_compile():
        print("[skip] torch.compile not available")
        return
    import torch._dynamo as dynamo
    dynamo.reset()
    cfg, m_a, m_b = _build_pair(batched=False)
    chunk_x, state, _, _ = _make_inputs(cfg, m_a, B=2, T=6, with_alpha=False)

    # Eager
    with torch.no_grad():
        eager_out, eager_state = _forward_chunk_pure_tf(
            m_a, m_a.cell, chunk_x, state.clone(),
        )
    # Compiled
    compiled_fn = torch.compile(_forward_chunk_pure_tf)
    with torch.no_grad():
        compiled_out, compiled_state = compiled_fn(
            m_b, m_b.cell, chunk_x, state.clone(),
        )

    # Default-mode compile uses Inductor codegen but no CUDA graphs; output
    # should match within fp32 ULP (kernel-fusion can reorder adds).
    assert torch.allclose(eager_out, compiled_out, atol=1e-6, rtol=1e-5), (
        f"compiled output drifted: max diff = "
        f"{(eager_out - compiled_out).abs().max().item():.3e}"
    )
    assert torch.allclose(eager_state, compiled_state, atol=1e-6, rtol=1e-5)


def test_chunk_compile_closed_loop_traceable_cpu():
    if not _has_compile():
        print("[skip] torch.compile not available")
        return
    import torch._dynamo as dynamo
    dynamo.reset()
    cfg, m_a, m_b = _build_pair(batched=False)
    chunk_x, state, y_prev, alpha = _make_inputs(cfg, m_a, B=2, T=6, with_alpha=True)

    with torch.no_grad():
        eager_out, _, _ = _forward_chunk_closed_loop(
            m_a, m_a.cell, chunk_x, state.clone(),
            y_prev.clone(), alpha,
        )
    compiled_fn = torch.compile(_forward_chunk_closed_loop)
    with torch.no_grad():
        compiled_out, _, _ = compiled_fn(
            m_b, m_b.cell, chunk_x, state.clone(),
            y_prev.clone(), alpha,
        )
    assert torch.allclose(eager_out, compiled_out, atol=1e-6, rtol=1e-5), (
        f"closed-loop compiled output drifted: max diff = "
        f"{(eager_out - compiled_out).abs().max().item():.3e}"
    )


def test_chunk_compile_no_recompiles_cpu():
    """Same compile, called 3× with identical shapes — no recompiles.

    Captures Dynamo recompile log lines. If the production refactor still
    has shape-dependent specialization (e.g. a ``None`` first-iter branch),
    call 2 will recompile.
    """
    if not _has_compile():
        print("[skip] torch.compile not available")
        return
    import torch._dynamo as dynamo
    dynamo.reset()

    cfg, m_a, _ = _build_pair(batched=False)
    chunk_x, state, _, _ = _make_inputs(cfg, m_a, B=2, T=6, with_alpha=False)

    # Capture the dynamo logger; recompile messages go through 'torch._dynamo'.
    dyn_log = logging.getLogger("torch._dynamo")
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setLevel(logging.INFO)
    dyn_log.addHandler(handler)
    prev_level = dyn_log.level
    dyn_log.setLevel(logging.INFO)
    try:
        torch._logging.set_logs(recompiles=True)
        compiled_fn = torch.compile(_forward_chunk_pure_tf)
        with torch.no_grad():
            for _ in range(3):
                compiled_fn(m_a, m_a.cell, chunk_x, state.clone())
    finally:
        dyn_log.removeHandler(handler)
        dyn_log.setLevel(prev_level)
        # Reset log filters back to defaults so we don't leak across tests.
        torch._logging.set_logs()

    log_text = buf.getvalue().lower()
    # First call always traces ("Recompiling" only fires from call 2 onward).
    assert "recompil" not in log_text, (
        f"Dynamo recompiled on identical-shape calls 2/3:\n{buf.getvalue()}"
    )


# ---------------------------------------------------------------------------
# Loss-in-chunk wrapper equivalence
# ---------------------------------------------------------------------------

def _ref_with_loss_pure_tf(model, cell, chunk_x, chunk_y, state, criterion,
                            *, K, task_type):
    """Reference: split flow — eager forward + standalone loss/metric.
    The production _forward_chunk_pure_tf_with_loss must be byte-identical
    to this on every output."""
    logits, state = _forward_chunk_pure_tf(model, cell, chunk_x, state)
    loss, pkl, pkm = _attach_loss_and_metric(
        logits, chunk_y, criterion, K=K, task_type=task_type,
    )
    return loss, pkl, pkm, state


def test_pure_tf_with_loss_equivalence_single():
    """SRNNCell with-loss wrapper byte-identical to split forward + attach."""
    import copy
    cfg, m_a, m_b = _build_pair(batched=False)
    chunk_x, state, _, _ = _make_inputs(cfg, m_a, B=4, T=8, with_alpha=False)
    chunk_y = torch.randn(*chunk_x.shape)
    criterion = torch.nn.MSELoss()

    _zero_grads(m_a, m_b)

    loss_a, pkl_a, pkm_a, state_a = _ref_with_loss_pure_tf(
        m_a, m_a.cell, chunk_x, chunk_y, state.clone(), criterion,
        K=None, task_type="regression",
    )
    loss_a.backward()
    grads_a = _grads(m_a)

    loss_b, pkl_b, pkm_b, state_b = _forward_chunk_pure_tf_with_loss(
        m_b, m_b.cell, chunk_x, chunk_y, state.clone(), criterion,
        K=None, task_type="regression",
    )
    loss_b.backward()
    grads_b = _grads(m_b)

    assert torch.equal(loss_a, loss_b), (
        f"loss diverges: {(loss_a - loss_b).abs().max().item():.3e}"
    )
    assert torch.equal(pkl_a, pkl_b)
    assert torch.equal(pkm_a, pkm_b)
    assert torch.equal(state_a, state_b)
    _assert_grads_byte_identical(grads_a, grads_b)


def test_pure_tf_with_loss_equivalence_batched():
    """K=2 BatchedSRNNCell with-loss wrapper byte-identical to split."""
    cfg, m_a, m_b = _build_pair(batched=True)
    chunk_x, state, _, _ = _make_inputs(cfg, m_a, B=4, T=8, with_alpha=False)
    chunk_y = torch.randn(*chunk_x.shape)
    criterion = torch.nn.MSELoss()
    K = m_a.cell.K

    _zero_grads(m_a, m_b)

    loss_a, pkl_a, pkm_a, state_a = _ref_with_loss_pure_tf(
        m_a, m_a.cell, chunk_x, chunk_y, state.clone(), criterion,
        K=K, task_type="regression",
    )
    loss_a.backward()
    grads_a = _grads(m_a)

    loss_b, pkl_b, pkm_b, state_b = _forward_chunk_pure_tf_with_loss(
        m_b, m_b.cell, chunk_x, chunk_y, state.clone(), criterion,
        K=K, task_type="regression",
    )
    loss_b.backward()
    grads_b = _grads(m_b)

    assert torch.equal(loss_a, loss_b)
    assert torch.equal(pkl_a, pkl_b)
    assert torch.equal(pkm_a, pkm_b)
    assert pkl_b.shape == (K,)
    assert pkm_b.shape == (K,)
    assert torch.equal(state_a, state_b)
    _assert_grads_byte_identical(grads_a, grads_b)


def test_closed_loop_with_loss_equivalence_batched():
    """K=2 closed-loop with-loss wrapper byte-identical to split."""
    cfg, m_a, m_b = _build_pair(batched=True)
    chunk_x, state, y_prev, alpha = _make_inputs(cfg, m_a, B=4, T=8, with_alpha=True)
    chunk_y = torch.randn(*chunk_x.shape)
    criterion = torch.nn.MSELoss()
    K = m_a.cell.K

    _zero_grads(m_a, m_b)

    # Reference: split closed-loop fwd + attach
    logits_a, sa, ya = _forward_chunk_closed_loop(
        m_a, m_a.cell, chunk_x, state.clone(), y_prev.clone(), alpha,
    )
    loss_a, pkl_a, pkm_a = _attach_loss_and_metric(
        logits_a, chunk_y, criterion, K=K, task_type="regression",
    )
    loss_a.backward()
    grads_a = _grads(m_a)

    loss_b, pkl_b, pkm_b, sb, yb = _forward_chunk_closed_loop_with_loss(
        m_b, m_b.cell, chunk_x, chunk_y, state.clone(),
        y_prev.clone(), alpha, criterion,
        K=K, task_type="regression",
    )
    loss_b.backward()
    grads_b = _grads(m_b)

    assert torch.equal(loss_a, loss_b)
    assert torch.equal(pkl_a, pkl_b)
    assert torch.equal(pkm_a, pkm_b)
    assert torch.equal(sa, sb)
    assert torch.equal(ya, yb)
    _assert_grads_byte_identical(grads_a, grads_b)


def test_with_loss_compile_traceable_cpu():
    """torch.compile(_forward_chunk_pure_tf_with_loss) traces and runs on CPU.
    Tests the actual T2 #3 compile target, including criterion + tensor-only
    loss/metric. Output matches eager within fp32 ULP."""
    if not _has_compile():
        print("[skip] torch.compile not available")
        return
    import torch._dynamo as dynamo
    dynamo.reset()

    cfg, m_a, m_b = _build_pair(batched=False)
    chunk_x, state, _, _ = _make_inputs(cfg, m_a, B=2, T=6, with_alpha=False)
    chunk_y = torch.randn(*chunk_x.shape)
    criterion = torch.nn.MSELoss()

    with torch.no_grad():
        eager_loss, eager_pkl, eager_pkm, eager_state = _forward_chunk_pure_tf_with_loss(
            m_a, m_a.cell, chunk_x, chunk_y, state.clone(), criterion,
            K=None, task_type="regression",
        )
    compiled_fn = torch.compile(_forward_chunk_pure_tf_with_loss)
    with torch.no_grad():
        c_loss, c_pkl, c_pkm, c_state = compiled_fn(
            m_b, m_b.cell, chunk_x, chunk_y, state.clone(), criterion,
            K=None, task_type="regression",
        )

    assert torch.allclose(eager_loss, c_loss, atol=1e-6, rtol=1e-5), (
        f"compiled loss drifted: {(eager_loss - c_loss).abs().max().item():.3e}"
    )
    assert torch.allclose(eager_pkl, c_pkl, atol=1e-6, rtol=1e-5)
    assert torch.allclose(eager_pkm, c_pkm, atol=1e-6, rtol=1e-5)
    assert torch.allclose(eager_state, c_state, atol=1e-6, rtol=1e-5)


# ---------------------------------------------------------------------------

def main() -> int:
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
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
