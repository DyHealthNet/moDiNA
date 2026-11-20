"""Sign convention of `raw-E` for binary-continuous associations.

NApy (napypi 1.1.9) computes its t-test as `group0 - group1`, the opposite of the usual
`group1 - group0` convention, so both its `t` and its `cohens_d` come back negated.
`context_net_inference.napy_bi_cont` negates them back at the boundary.

These tests pin the *outcome* (does raw-E point the same way as the data?) rather than the
workaround, so they fail loudly in BOTH directions: if the negation is dropped, and equally if
napy ever fixes its convention upstream and the negation starts double-flipping.
"""

import numpy as np
import pandas as pd
import pytest

import modina

N_SAMPLES = 2000
META = pd.DataFrame({"label": ["bi1", "cont1"], "type": ["binary", "continuous"]})


def _two_variable_context(effect, seed):
    """A binary and a continuous variable whose true association has the sign of `effect`."""
    rng = np.random.default_rng(seed)
    b = rng.integers(0, 2, N_SAMPLES).astype(float)
    c = rng.normal(0, 1, N_SAMPLES) + effect * b
    return pd.DataFrame({"bi1": b, "cont1": c})


def _raw_e(context, test_type):
    scores = modina.compute_context_scores(context_data=context, meta_file=META,
                                           test_type=test_type, correction="bh", num_workers=1)
    row = scores.iloc[0]
    return float(row["raw-E"]), row["test_type"]


@pytest.mark.parametrize("effect", [2.0, -2.0, 0.8, -0.8])
@pytest.mark.parametrize("test_type", ["parametric", "nonparametric"])
def test_raw_e_sign_matches_the_data(effect, test_type):
    """raw-E must point the same way as the plain correlation of the two columns.

    Group 1 higher than group 0 => positive raw-E, for both the parametric (ttest, via Cohen's d)
    and nonparametric (mwu, rank-biserial) paths.
    """
    context = _two_variable_context(effect, seed=abs(int(effect * 10)))
    raw_e, used_test = _raw_e(context, test_type)
    data_r = np.corrcoef(context["bi1"], context["cont1"])[0, 1]

    assert used_test == ("ttest" if test_type == "parametric" else "mwu")
    assert np.sign(raw_e) == np.sign(data_r), (
        f"{used_test}: raw-E={raw_e:+.4f} disagrees in sign with the data's r={data_r:+.4f}. "
        f"If napypi changed its group ordering, the negation in napy_bi_cont is now "
        f"double-flipping and must be removed.")
    assert abs(raw_e) > 0.1, "effect planted here should be far from zero"


def test_parametric_and_nonparametric_agree_in_direction():
    """The two paths must not disagree with each other -- the symptom that exposed the bug."""
    context = _two_variable_context(1.5, seed=7)
    par, _ = _raw_e(context, "parametric")
    nonpar, _ = _raw_e(context, "nonparametric")
    assert np.sign(par) == np.sign(nonpar), (
        f"parametric raw-E={par:+.4f} and nonparametric raw-E={nonpar:+.4f} point opposite ways")


def test_between_context_difference_keeps_direction():
    """A consistent sign flip does NOT cancel in a between-context difference -- it flips the
    difference too. This is what silently reversed the signed effect sizes (rescaled-E, RE / VE scores)."""
    strong = _two_variable_context(2.0, seed=1)   # strong positive association
    weak = _two_variable_context(0.2, seed=2)     # weak positive association
    e_strong, _ = _raw_e(strong, "parametric")
    e_weak, _ = _raw_e(weak, "parametric")
    assert e_strong > e_weak > 0, (
        f"expected 0 < weak ({e_weak:+.4f}) < strong ({e_strong:+.4f}); a sign inversion makes "
        f"the difference point the wrong way between contexts")
