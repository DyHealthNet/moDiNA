"""correction_scope: 'global' pools all edges into one FDR family, 'local' corrects per statistical
test type (e.g. pearson / ttest / chi2); for STC nodes, per test used to compare the node."""

import numpy as np
import pandas as pd
import pytest

import modina
from modina.statistics_utils import fdr_correction

NAN = -99.0


def _context(seed, n=200, p_cont=30):
    rng = np.random.default_rng(seed)
    data = {f"c{i}": rng.normal(size=n) for i in range(p_cont)}
    data["b1"] = rng.integers(0, 2, n).astype(float)
    data["b2"] = rng.integers(0, 2, n).astype(float)
    data["c0"] = data["c0"] + 1.0 * data["b1"]
    df = pd.DataFrame(data)
    meta = pd.DataFrame({"label": df.columns,
                         "type": ["continuous"] * p_cont + ["binary", "binary"]})
    return df, meta


def test_fdr_groups_equal_separate_runs():
    p = np.array([0.001, 0.01, 0.04, 0.2, 0.03, np.nan, 0.5])
    groups = np.array(["a", "a", "a", "a", "b", "b", "b"])
    out = fdr_correction(p, groups=groups)
    assert np.allclose(out[:4], fdr_correction(p[:4]))
    assert np.allclose(out[4:], fdr_correction(p[4:]), equal_nan=True)
    assert np.allclose(fdr_correction(p, groups=None), fdr_correction(p), equal_nan=True)


def test_invalid_scope_raises():
    ctx, meta = _context(0)
    with pytest.raises(ValueError):
        modina.compute_context_scores(ctx, meta, correction_scope="regional")


def _capture_groups(monkeypatch, module):
    calls = []
    original = module.fdr_correction

    def spy(pvalues, method="bh", groups=None):
        calls.append(None if groups is None else np.asarray(groups))
        return original(pvalues, method=method, groups=groups)

    monkeypatch.setattr(module, "fdr_correction", spy)
    return calls


def test_local_scope_edges_group_by_test_type(monkeypatch):
    import modina.context_net_inference as cni
    ctx, meta = _context(1)
    calls = _capture_groups(monkeypatch, cni)

    g = modina.compute_context_scores(ctx, meta, test_type="parametric", correction_scope="global")
    assert calls[-1] is None
    l = modina.compute_context_scores(ctx, meta, test_type="parametric", correction_scope="local")
    # one family per test type: pearson (cont-cont), ttest (cont-binary), chi2 (binary-binary)
    assert set(calls[-1]) == {"pearson", "ttest", "chi2"}
    assert g[["label1", "label2"]].equals(l[["label1", "label2"]])
    # The single binary-binary chi2 test is its own family -> not penalised by the ~450 other tests
    chi2 = (g["test_type"] == "chi2").to_numpy()
    assert chi2.sum() == 1
    assert l.loc[chi2, "adj-P"].iloc[0] <= g.loc[chi2, "adj-P"].iloc[0]
    # the unadjusted p-values do not depend on the correction scope
    assert np.allclose(g["raw-P"], l["raw-P"])
    assert (g["adj-P"] >= g["raw-P"]).all() and (l["adj-P"] >= l["raw-P"]).all()


def test_local_scope_nodes_group_by_test_type(monkeypatch):
    import modina.diff_net_construction as dnc
    ctx1, meta = _context(2)
    ctx2, _ = _context(3)
    calls = _capture_groups(monkeypatch, dnc)
    dnc.stat_test_centrality(ctx1, ctx2, meta, test_type="parametric", nan_value=NAN, correction_scope="local")
    assert set(calls[-1]) == {"ttest", "chi2"}
    dnc.stat_test_centrality(ctx1, ctx2, meta, test_type="nonparametric", nan_value=NAN, correction_scope="local")
    assert set(calls[-1]) == {"mwu", "chi2"}
    dnc.stat_test_centrality(ctx1, ctx2, meta, test_type="parametric", nan_value=NAN, correction_scope="global")
    assert calls[-1] is None


def test_local_scope_end_to_end_and_config():
    ctx1, meta = _context(2)
    ctx2, _ = _context(3)
    _, edges, nodes, config = modina.diffnet_analysis(
        ctx1, ctx2, meta, edge_metric="diff-LP", node_metric="STC-P",
        correction_scope="local", nan_value=NAN)
    assert config["correction_scope"] == "local"
    assert nodes["STC-P"].between(0, 1).all()
