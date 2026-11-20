"""Per-pair sample counts ('n') from NApy.

1. Cohen's d -> point-biserial r must use the binary group sizes of the specific pair after
   pairwise removal of missing values, not the group sizes of the whole binary column.
2. Pairs tested on fewer than n_min_valid samples are dropped from that context and, through
   reconcile_flagged_variables, from the other context as well.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import modina
from modina.statistics_utils import reconcile_flagged_variables

NAN = -99.0


def _skewed_missingness_context(seed=0, n=400):
    """Binary 50/50; the continuous variable is missing for ~80% of group 1 only."""
    rng = np.random.default_rng(seed)
    b = np.repeat([0.0, 1.0], n // 2)
    c = rng.normal(0, 1, n) + 0.8 * b
    missing = (b == 1) & (rng.random(n) < 0.8)
    c[missing] = NAN
    return pd.DataFrame({"bi1": b, "cont1": c})


def test_ttest_r_uses_pairwise_group_sizes():
    context = _skewed_missingness_context()
    meta = pd.DataFrame({"label": ["bi1", "cont1"], "type": ["binary", "continuous"]})
    scores = modina.compute_context_scores(context_data=context, meta_file=meta, test_type="parametric",
                                           nan_value=NAN, n_min_valid=10)
    row = scores.iloc[0]
    assert row["test_type"] == "ttest"

    complete = context[context["cont1"] != NAN]
    assert row["n"] == len(complete)

    expected_r, _ = stats.pointbiserialr(complete["bi1"], complete["cont1"])
    # moDiNA's d->r formula uses n^2/(n1*n2) rather than the exact n(n-2)/(n1*n2), hence the tolerance.
    # The old column-wide group sizes would be off by ~30% here.
    assert row["raw-E"] == pytest.approx(expected_r, rel=1e-2)


def _context(n_observed_c2, seed):
    rng = np.random.default_rng(seed)
    n = 200
    df = pd.DataFrame({"c1": rng.normal(size=n), "c2": rng.normal(size=n), "c3": rng.normal(size=n)})
    df.loc[n_observed_c2:, "c2"] = NAN
    return df


META3 = pd.DataFrame({"label": ["c1", "c2", "c3"], "type": ["continuous"] * 3})


def test_low_n_pairs_dropped_in_both_contexts():
    ctx1 = _context(n_observed_c2=200, seed=1)
    ctx2 = _context(n_observed_c2=8, seed=2)

    s1 = modina.compute_context_scores(context_data=ctx1, meta_file=META3, nan_value=NAN, n_min_valid=10)
    s2 = modina.compute_context_scores(context_data=ctx2, meta_file=META3, nan_value=NAN, n_min_valid=10)

    assert len(s1) == 3
    assert set(zip(s2["label1"], s2["label2"])) == {("c1", "c3")}
    na = pd.DataFrame(s2.attrs["scores_na"])
    assert set(na["reason"]) == {"low_n"}
    assert (na["n"] == 8).all()

    r1, r2, _, _, _ = reconcile_flagged_variables(s1, s2)
    assert list(zip(r1["label1"], r1["label2"])) == [("c1", "c3")]
    assert list(zip(r2["label1"], r2["label2"])) == [("c1", "c3")]


def test_n_min_valid_zero_keeps_pairs():
    ctx2 = _context(n_observed_c2=8, seed=2)
    s2 = modina.compute_context_scores(context_data=ctx2, meta_file=META3, nan_value=NAN, n_min_valid=0)
    assert len(s2) == 3
    assert sorted(s2["n"].tolist()) == [8, 8, 200]
