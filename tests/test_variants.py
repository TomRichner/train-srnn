"""Variant-name grammar: parsing, canonical names, seed crossing."""
import pytest

from train_srnn.models import variants as V

BASE = dict(dales=True, n_a_E=3, n_a_I=3, n_b_E=1, n_b_I=1,
            per_neuron=False, echo=False, skip=False)


def test_full_model_has_no_tokens():
    v = V.make_variant("srnn", BASE, default_seed=1)
    assert v.tokens == () and v.name == "srnn" and v.seed == 1
    assert (v.dales, v.n_a_E, v.n_a_I, v.n_b_E, v.n_b_I) == (True, 3, 3, 1, 1)


@pytest.mark.parametrize("name, flags", [
    ("srnn-no-adapt", dict(n_a_E=0, n_a_I=0, n_b_E=0, n_b_I=0)),
    ("srnn-e-only", dict(n_a_E=3, n_a_I=0, n_b_E=1, n_b_I=0)),
    ("srnn-sfa-e-only-skip", dict(n_a_E=3, n_a_I=0, n_b_E=0, n_b_I=0, skip=True)),
    ("srnn-std-e-only-per-neuron", dict(n_a_E=0, n_a_I=0, n_b_E=1, n_b_I=0, per_neuron=True)),
    ("srnn-no-adapt-no-dales-skip", dict(n_a_E=0, n_a_I=0, n_b_E=0, n_b_I=0, dales=False, skip=True)),
    ("srnn-echo", dict(echo=True)),
])
def test_tokens_set_flags(name, flags):
    v = V.make_variant(name, BASE, default_seed=0)
    for k, want in {**BASE, **flags}.items():
        assert getattr(v, k) == want, k
    assert v.name == name


def test_case_and_order_are_canonicalised():
    assert V.make_variant("srnn-E-only", BASE, 0).name == "srnn-e-only"
    assert V.make_variant("srnn-skip-no-dales", BASE, 0).name == "srnn-no-dales-skip"
    assert V.make_variant("srnn-e-only-sfa-only", BASE, 0).name == "srnn-sfa-e-only"


def test_explicit_seed_suffix():
    v = V.make_variant("srnn-skip-seed7", BASE, default_seed=1)
    assert v.seed == 7 and v.name == "srnn-skip-seed7" and v.base_name == "srnn-skip"


def test_base_flags_feed_through():
    v = V.make_variant("srnn-e-only", {**BASE, "n_a_E": 2}, 0)
    assert v.n_a_E == 2 and v.n_a_I == 0


@pytest.mark.parametrize("bad", ["lstm", "srnn-bogus", "srnn-skip-skip", "srnn-seedx"])
def test_bad_names_raise(bad):
    with pytest.raises(ValueError):
        V.make_variant(bad, BASE, 0)


def test_expand_is_variant_major_and_pairs_seeds():
    vs = V.expand(["srnn-skip", "srnn-no-adapt-skip"], [1, 2], BASE, default_seed=9)
    assert [v.name for v in vs] == ["srnn-skip-seed1", "srnn-skip-seed2",
                                    "srnn-no-adapt-skip-seed1", "srnn-no-adapt-skip-seed2"]
    assert [v.seed for v in vs] == [1, 2, 1, 2]
    plain = V.expand(["srnn", "srnn-skip"], None, BASE, default_seed=9)
    assert [v.seed for v in plain] == [9, 9]
    with pytest.raises(ValueError):
        V.expand(["srnn-seed3"], [1], BASE, 0)


@pytest.mark.parametrize("condition,a,b", [("sfa1-std1", 1, 1), ("sfa3-std2", 3, 2),
                                           ("sfa3-std1", 3, 1), ("sfa1-std2", 1, 2)])
def test_manuscript_condition_counts(condition, a, b):
    v = V.make_variant(f"srnn-{condition}-skip-no-dales", BASE, 2)
    assert (v.n_a_E, v.n_a_I, v.n_b_E, v.n_b_I) == (a, a, b, b)
    assert v.skip and not v.dales


@pytest.mark.parametrize("tokens", ["sfa1-std1-sfa3-std2", "sfa1-std1-no-adapt",
                                    "sfa3-std2-sfa-only", "sfa1-std1-std-only",
                                    "sfa3-std1-sfa1-std2", "sfa1-std2-e-only"])
def test_conflicting_manuscript_conditions_rejected(tokens):
    with pytest.raises(ValueError):
        V.make_variant(f"srnn-{tokens}", BASE, 0)
