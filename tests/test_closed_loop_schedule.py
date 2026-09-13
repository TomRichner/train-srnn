"""Tests for train_srnn.training.closed_loop schedule sampling.

Run as: python scripts/test_closed_loop_schedule.py
or:     pytest scripts/test_closed_loop_schedule.py -v
"""
from __future__ import annotations

import math

import torch

from train_srnn.training.closed_loop import (
    ClosedLoopConfig,
    effective_alpha_baseline,
    sample_alpha_schedule,
    summarize_alpha,
)


def _gen(seed: int = 0) -> torch.Generator:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    return g


def test_disabled_returns_none():
    cfg = ClosedLoopConfig(enabled=False)
    assert sample_alpha_schedule(cfg, T=10, C=4, generator=_gen()) is None


def test_pure_tf_batch_frac_one_always_returns_none():
    cfg = ClosedLoopConfig(enabled=True, teacher_forcing_batch_frac=1.0,
                           alpha_baseline=0.5)
    for s in range(20):
        assert sample_alpha_schedule(cfg, T=10, C=4, generator=_gen(s)) is None


def test_pure_tf_batch_frac_zero_never_returns_none():
    cfg = ClosedLoopConfig(enabled=True, teacher_forcing_batch_frac=0.0,
                           alpha_baseline=0.5)
    for s in range(20):
        a = sample_alpha_schedule(cfg, T=10, C=4, generator=_gen(s))
        assert a is not None


def test_t_zero_always_zero():
    cfg = ClosedLoopConfig(enabled=True, teacher_forcing_batch_frac=0.0,
                           alpha_baseline=0.9, alpha_baseline_jitter=0.05,
                           alpha_rnd_density=0.5, alpha_rnd_sigma=0.2,
                           t_warm=0)
    for s in range(20):
        a = sample_alpha_schedule(cfg, T=16, C=8, generator=_gen(s))
        assert a is not None
        assert torch.all(a[0] == 0.0), f"seed={s}: alpha[0] = {a[0]}"


def test_no_warmup_no_rnd_uniform_baseline_after_t0():
    cfg = ClosedLoopConfig(enabled=True, teacher_forcing_batch_frac=0.0,
                           alpha_baseline=0.3, t_warm=0)
    a = sample_alpha_schedule(cfg, T=10, C=4, generator=_gen())
    assert a is not None
    # t >= 1 should be uniformly 0.3 across t and c
    assert torch.allclose(a[1:], torch.full_like(a[1:], 0.3))


def test_envelope_reaches_one_at_t_warm():
    cfg = ClosedLoopConfig(enabled=True, teacher_forcing_batch_frac=0.0,
                           alpha_baseline=1.0, t_warm=8)
    a = sample_alpha_schedule(cfg, T=20, C=2, generator=_gen())
    assert a is not None
    # alpha_raw == 1.0 (baseline=1, no rnd), so alpha[t, c] == envelope(t)
    # envelope(t_warm) = 0.5*(1 - cos(pi)) = 1
    assert torch.allclose(a[8], torch.ones_like(a[8]), atol=1e-6)
    # envelope(t > t_warm) clamped to 1
    assert torch.allclose(a[15], torch.ones_like(a[15]), atol=1e-6)
    # envelope(t_warm/2) = 0.5*(1 - cos(pi/2)) = 0.5
    assert torch.allclose(a[4], torch.full_like(a[4], 0.5), atol=1e-6)


def test_alpha_in_unit_interval():
    cfg = ClosedLoopConfig(enabled=True, teacher_forcing_batch_frac=0.0,
                           alpha_baseline=0.7, alpha_baseline_jitter=0.2,
                           alpha_rnd_density=0.5, alpha_rnd_sigma=0.5,
                           t_warm=4)
    for s in range(50):
        a = sample_alpha_schedule(cfg, T=20, C=10, generator=_gen(s))
        assert a is not None
        assert torch.all(a >= 0.0) and torch.all(a <= 1.0)


def test_determinism_same_seed():
    cfg = ClosedLoopConfig(enabled=True, teacher_forcing_batch_frac=0.0,
                           alpha_baseline=0.5, alpha_rnd_density=0.3,
                           alpha_rnd_sigma=0.1, t_warm=4)
    a1 = sample_alpha_schedule(cfg, T=12, C=6, generator=_gen(123))
    a2 = sample_alpha_schedule(cfg, T=12, C=6, generator=_gen(123))
    assert torch.equal(a1, a2)


def test_determinism_different_seed_differs():
    cfg = ClosedLoopConfig(enabled=True, teacher_forcing_batch_frac=0.0,
                           alpha_baseline=0.5, alpha_rnd_density=0.5,
                           alpha_rnd_sigma=0.2, t_warm=4)
    a1 = sample_alpha_schedule(cfg, T=12, C=6, generator=_gen(1))
    a2 = sample_alpha_schedule(cfg, T=12, C=6, generator=_gen(2))
    assert not torch.equal(a1, a2)


def test_rnd_density_zero_gives_uniform_per_batch():
    cfg = ClosedLoopConfig(enabled=True, teacher_forcing_batch_frac=0.0,
                           alpha_baseline=0.4, alpha_rnd_density=0.0,
                           alpha_rnd_sigma=0.0, t_warm=0)
    a = sample_alpha_schedule(cfg, T=8, C=5, generator=_gen())
    assert a is not None
    # With density=0, all channels share the same alpha
    body = a[1:]
    for t in range(body.shape[0]):
        assert torch.allclose(body[t], torch.full_like(body[t], body[t, 0]))


def test_rnd_density_one_perturbs_all_channels():
    cfg = ClosedLoopConfig(enabled=True, teacher_forcing_batch_frac=0.0,
                           alpha_baseline=0.5, alpha_rnd_density=1.0,
                           alpha_rnd_sigma=0.2, t_warm=0)
    # With t_warm=0, envelope=1 for all t>=0. But alpha[0,:]=0 forced.
    # For t>=1, alpha[t,c] = clip(0.5 + N(0,0.2), 0, 1). Should differ across c.
    a = sample_alpha_schedule(cfg, T=4, C=20, generator=_gen())
    assert a is not None
    # Check there's actual variation across channels at t=1
    assert a[1].std().item() > 0.05


def test_summarize_alpha():
    # None => pure-TF marker
    s = summarize_alpha(None)
    assert s["is_pure_tf"] == 1.0
    assert s["alpha_mean"] == 0.0

    # Constant non-zero alpha
    a = torch.full((10, 4), 0.3)
    a[0].zero_()
    s = summarize_alpha(a)
    assert s["is_pure_tf"] == 0.0
    assert math.isclose(s["alpha_mean"], 0.3, abs_tol=1e-6)
    assert math.isclose(s["alpha_max"], 0.3, abs_tol=1e-6)
    assert math.isclose(s["alpha_active_frac"], 1.0, abs_tol=1e-6)


def test_effective_alpha_baseline_default_constant():
    """alpha_baseline_start=None -> always returns alpha_baseline."""
    cfg = ClosedLoopConfig(enabled=True, alpha_baseline=0.4)
    for e in range(10):
        assert effective_alpha_baseline(cfg, epoch=e, total_epochs=10) == 0.4


def test_effective_alpha_baseline_linear_ramp():
    cfg = ClosedLoopConfig(enabled=True, alpha_baseline=0.2,
                           alpha_baseline_start=0.0)
    # 5 epochs: e/(N-1) at e=0,1,2,3,4 -> 0, 0.25, 0.5, 0.75, 1.0
    expected = [0.0, 0.05, 0.1, 0.15, 0.2]
    for e, exp in enumerate(expected):
        got = effective_alpha_baseline(cfg, epoch=e, total_epochs=5)
        assert math.isclose(got, exp, abs_tol=1e-9), (e, got, exp)


def test_effective_alpha_baseline_single_epoch():
    """total_epochs <= 1 -> always returns alpha_baseline (avoid div-by-zero)."""
    cfg = ClosedLoopConfig(enabled=True, alpha_baseline=0.2,
                           alpha_baseline_start=0.0)
    assert effective_alpha_baseline(cfg, epoch=0, total_epochs=1) == 0.2


def test_effective_alpha_baseline_clamped_above_total():
    """epochs >= total_epochs-1 should clamp to alpha_baseline (final value)."""
    cfg = ClosedLoopConfig(enabled=True, alpha_baseline=0.2,
                           alpha_baseline_start=0.0)
    assert effective_alpha_baseline(cfg, epoch=10, total_epochs=5) == 0.2


def test_device_cpu():
    cfg = ClosedLoopConfig(enabled=True, teacher_forcing_batch_frac=0.0,
                           alpha_baseline=0.5, t_warm=2)
    a = sample_alpha_schedule(cfg, T=8, C=4, device="cpu", generator=_gen())
    assert a is not None
    assert a.device.type == "cpu"


# ---------------------------------------------------------------------------
# Run as a script: collect and execute every test_*.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
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
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
