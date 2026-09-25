"""scripts/report_timewarp.py statistics: exact sign-flip test, paired ratios, factorial effects."""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import report_timewarp as R  # noqa: E402


def test_sign_flip_is_exact():
    assert R.sign_flip_p([1.0] * 15) == pytest.approx(2 / 2 ** 15)
    assert R.sign_flip_p([1.0, -1.0]) == 1.0
    # Sums over the 8 sign patterns: +-2.5, +-3.5, +-0.5, +-1.5; four reach |2.5|.
    assert R.sign_flip_p([2.0, 1.0, -0.5]) == pytest.approx(4 / 8)


def test_paired_ratio_and_wins():
    a = {1: 1.0, 2: 2.0, 3: 4.0}
    b = {1: 0.5, 2: 1.0, 3: 8.0}
    r = R.paired(a, b)
    assert r["b_better"] == 2 and r["seeds"] == 3
    assert r["ratio_geomean"] == pytest.approx(math.exp((2 * math.log(0.5) + math.log(2)) / 3))


def test_factorial_recovers_multiplicative_effects():
    seeds = range(1, 7)
    # SFA3 halves the error, STD2 multiplies it by 1.5, no interaction.
    final = {"sfa1-std1": {s: 1.0 + 0.1 * s for s in seeds}}
    final["sfa3-std1"] = {s: 0.5 * v for s, v in final["sfa1-std1"].items()}
    final["sfa1-std2"] = {s: 1.5 * v for s, v in final["sfa1-std1"].items()}
    final["sfa3-std2"] = {s: 0.75 * v for s, v in final["sfa1-std1"].items()}
    f = R.factorial(final)
    assert f["sfa3_vs_sfa1"]["ratio"] == pytest.approx(0.5)
    assert f["std2_vs_std1"]["ratio"] == pytest.approx(1.5)
    assert f["interaction"]["ratio"] == pytest.approx(1.0)
    assert R.factorial({"sfa1-std1": {1: 1.0}}) is None
