"""STD strength matching (std_match / variant tokens) against the MATLAB decision note."""
import math

import pytest
import torch

from train_srnn.config import compose_config
from train_srnn.models import srnn_cell as S
from train_srnn.models.factory import build_model


def _cfg(**kw):
    c = S.SRNNConfig(num_units=20, n_a_E=3, n_a_I=3, n_b_E=2, n_b_I=2)
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def test_matching_numbers_match_the_matlab_note():
    # FractionalReservoir docs/notes/STD_strength_matching_2026-09-13.md, r_ref = 0.25, rho = 0.125
    c = _cfg()
    assert S.std_steady_single(c) == pytest.approx(1 / 3)
    assert S.std_route_scale(c, 2) == pytest.approx(3.0)
    usage = S.std_init_taus(_cfg(std_match="usage"), 2, "rel")
    assert usage == pytest.approx([0.68301, 1.36603], abs=1e-5)
    assert S.std_init_taus(_cfg(std_match="usage"), 2, "rec") == [2.0, 4.0]
    assert S.std_init_taus(_cfg(std_match="strong"), 1, "rel") == pytest.approx([0.0625])
    assert S.std_init_taus(_cfg(std_match="strong"), 2, "rel") == [0.25, 0.5]   # dual unchanged


def _build(variants):
    cfg = compose_config(["task=synthetic", "model=srnn", "model.num_units=20", "device=cpu",
                          f"model.variants=[{','.join(variants)}]", "model.variant_seeds=[1]"])
    return build_model(cfg).cell


def test_tokens_reach_the_cell():
    cell = _build(["srnn-sfa3-std2", "srnn-sfa3-std2-std-geo", "srnn-sfa3-std2-std-scale",
                   "srnn-sfa3-std2-std-usage", "srnn-sfa1-std1-std-strong", "srnn-no-adapt-w-matched"])
    assert cell.std_E_exponent.flatten().tolist() == [1.0, 0.5, 1.0, 1.0, 1.0, 1.0]
    assert cell.std_I_route_scale.flatten().tolist() == pytest.approx([1, 1, 3, 1, 1, 1])
    rel = cell._tau_b("E", "rel")                                   # (K, n_E, M)
    assert rel[3, 0].tolist() == pytest.approx([0.68301, 1.36603], abs=1e-4)
    assert rel[4, 0, 0].item() == pytest.approx(0.0625, rel=1e-4)
    assert rel[0, 0].tolist() == pytest.approx([0.25, 0.5], rel=1e-5)
    gains = cell.log_W_raw_gain.exp().tolist()
    assert gains[5] == pytest.approx(1 / 3) and gains[:5] == [1.0] * 5


def test_geo_and_scale_match_single_timescale_steady_state():
    """At a constant rate every matched dual variant has the single variant's steady output."""
    cell = _build(["srnn-sfa1-std1", "srnn-sfa3-std2-std-geo", "srnn-sfa3-std2"])
    r = 0.25
    for side in ("E", "I"):
        n = cell.n_E if side == "E" else cell.n_I
        rec, rel = cell._tau_b(side, "rec"), cell._tau_b(side, "rel")
        b_ss = 1.0 / (1.0 + r * rec / rel)                          # (K, n, M)
        if side == "E":
            b_E = b_ss.unsqueeze(1)
        else:
            b_I = b_ss.unsqueeze(1)
    full = cell._b_full(b_E, b_I)[:, 0, 0]
    assert full[0].item() == pytest.approx(1 / 3, rel=1e-5)
    assert full[1].item() == pytest.approx(1 / 3, rel=1e-5)       # geometric mean
    assert full[2].item() == pytest.approx(1 / 9, rel=1e-5)       # unmatched product


def test_conflicting_matching_tokens_rejected():
    from train_srnn.models import variants as V
    with pytest.raises(ValueError):
        V.parse_name("srnn-std-geo-std-scale") and V.make_variant(
            "srnn-std-geo-std-scale", {k: getattr(S.SRNNConfig(), k) for k in V.FLAGS}, 0)
