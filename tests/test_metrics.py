"""Edge metrics, node metrics (WDC / PRC / STC) and the edge-statistic ranking algorithms
(Mean/MedianEdgeRank, Mean/MedianProductRank)."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import modina
from modina.diff_net_construction import compute_diff_edges, compute_diff_nodes, stat_test_centrality
from modina.metrics import (EDGE_METRICS, NODE_METRICS, RANKING_ALGORITHMS, ALG_REQUIRES, effect_rescaling)
from modina.ranking import compute_ranking, productrank
from modina.statistics_utils import fdr_correction

NAN = -99.0


def _scores(adj_p, raw_e, test_type='pearson', n=100):
    """Score table with the given adjusted p-values ('adj-P'). The unadjusted p-values ('raw-P') are set to
    adj_p / 2, so that a metric reading the wrong column fails the tests."""
    labels = [('a', 'b'), ('a', 'c'), ('b', 'c'), ('c', 'd')][:len(adj_p)]
    adj_p = np.asarray(adj_p, dtype=float)
    return pd.DataFrame({'label1': [l[0] for l in labels], 'label2': [l[1] for l in labels],
                         'test_type': test_type, 'raw-P': adj_p / 2, 'adj-P': adj_p, 'raw-E': raw_e, 'n': n})


# Edge 0: same p, sign flip (r = +0.3 vs -0.3); edge 1: strong vs weak; edge 2: extreme p vs moderate p.
S1 = _scores([0.01, 1e-6, 1e-20], [0.3, 0.6, 0.5])
S2 = _scores([0.01, 0.2, 1e-3], [-0.3, 0.1, 0.5])


def _edges(metric, s1=S1, s2=S2):
    return compute_diff_edges(s1.copy(), s2.copy(), metric)[0]


# ---------------------------------------------------------------------------------------- edges

@pytest.mark.parametrize('metric', list(EDGE_METRICS))
def test_edge_metric_columns_are_absolute(metric):
    e = _edges(metric)
    assert list(e.columns[:4]) == ['label1', 'label2', 'test_type', metric]
    assert not any(c.endswith('_signed') for c in e.columns)
    assert (e[metric] >= 0).all()


@pytest.mark.parametrize('metric, parts', [('sum-LP-VE', ('diff-LP', 'diff-VE')), ('sum-LP-RE', ('diff-LP', 'diff-RE'))])
def test_sum_metrics_are_sums_of_parts(metric, parts):
    e = _edges(metric)
    assert np.allclose(e[metric], _edges(parts[0])[parts[0]] + _edges(parts[1])[parts[1]])


def test_sign_flip_is_detected_by_effect_metrics_only():
    assert _edges('diff-LP')['diff-LP'].iloc[0] == pytest.approx(0.0)
    assert _edges('diff-VE')['diff-VE'].iloc[0] == pytest.approx(0.18)
    assert _edges('diff-LPxVE')['diff-LPxVE'].iloc[0] == pytest.approx(2 * 2 * 0.09)


def test_diff_p_uses_one_minus_p():
    e = _edges('diff-P')
    assert np.allclose(e['diff-P'], (S1['adj-P'] - S2['adj-P']).abs())


@pytest.mark.parametrize('metric_pvalue', ['adj-P', 'raw-P'])
def test_metric_pvalue_selects_p_column(metric_pvalue):
    e = compute_diff_edges(S1.copy(), S2.copy(), 'diff-LP', metric_pvalue=metric_pvalue)[0]
    assert np.allclose(e['diff-LP'], (np.log10(S1[metric_pvalue]) - np.log10(S2[metric_pvalue])).abs())
    # both p-values of both contexts are kept in the output, whichever one the metric uses
    for col in ('raw-P', 'adj-P'):
        assert np.allclose(e[f'{col}_context1'], S1[col]) and np.allclose(e[f'{col}_context2'], S2[col])


@pytest.mark.parametrize('metric', ['diff-P', 'diff-LPxRE', 'diff-Z'])
def test_raw_p_metrics_differ_from_adjusted(metric):
    adj = compute_diff_edges(S1.copy(), S2.copy(), metric)[0][metric]
    raw = compute_diff_edges(S1.copy(), S2.copy(), metric, metric_pvalue='raw-P')[0][metric]
    assert not np.allclose(adj, raw)


def test_invalid_metric_pvalue_raises():
    with pytest.raises(ValueError, match='metric_pvalue'):
        compute_diff_edges(S1.copy(), S2.copy(), 'diff-LP', metric_pvalue='log-P')


def test_scores_without_adj_p_are_rejected():
    # score tables of older moDiNA versions stored the adjusted p-value as 'raw-P' and have no 'adj-P'
    old1, old2 = S1.drop(columns='adj-P'), S2.drop(columns='adj-P')
    for metric_pvalue in ('adj-P', 'raw-P'):
        with pytest.raises(ValueError, match='older moDiNA version'):
            compute_diff_edges(old1.copy(), old2.copy(), 'diff-LP', metric_pvalue=metric_pvalue)


def test_lp_ignores_stale_log_p_column():
    s1, s2 = S1.copy(), S2.copy()
    s1['log-P'] = 1000.0
    s2['log-P'] = -1000.0
    for metric in ('diff-LP', 'diff-LPxRE'):
        assert np.allclose(_edges(metric, s1, s2)[metric], _edges(metric)[metric])


def _pearson_scores(r, n):
    """Score table of one edge (a, b) with the two-sided Pearson p-value of r at sample size n."""
    t = r * np.sqrt((n - 2) / (1 - r ** 2))
    return _scores([2 * stats.t.sf(abs(t), n - 2)], [r], n=n)


def test_diff_z_definition():
    s1, s2 = S1.assign(n=[100, 80, 60]), S2.assign(n=[90, 70, 50])
    z1 = stats.norm.isf(s1['adj-P'] / 2) * np.sign(s1['raw-E'])
    z2 = stats.norm.isf(s2['adj-P'] / 2) * np.sign(s2['raw-E'])
    expected = (z1 / np.sqrt(s1['n']) - z2 / np.sqrt(s2['n'])).abs() / np.sqrt(1 / s1['n'] + 1 / s2['n'])
    assert np.allclose(_edges('diff-Z', s1, s2)['diff-Z'], expected)


def test_diff_z_ignores_sample_size_for_equal_associations():
    # The same strong correlation in a large and a small context: diff-LP is large, diff-Z close to 0.
    s1, s2 = _pearson_scores(0.7, 300), _pearson_scores(0.7, 100)
    assert _edges('diff-LP', s1, s2)['diff-LP'].iloc[0] > 20
    assert _edges('diff-Z', s1, s2)['diff-Z'].iloc[0] < 0.5


def test_diff_z_does_not_inflate_differences_between_strong_associations():
    # The same change of r (0.09) between strong and between weak associations at equal n: diff-LP differs by an
    # order of magnitude, diff-Z only moderately (Fisher-z scale).
    n = 125
    strong = _edges('diff-LP', _pearson_scores(0.82, n), _pearson_scores(0.73, n))['diff-LP'].iloc[0]
    weak = _edges('diff-LP', _pearson_scores(0.30, n), _pearson_scores(0.21, n))['diff-LP'].iloc[0]
    assert strong / weak > 5
    strong_z = _edges('diff-Z', _pearson_scores(0.82, n), _pearson_scores(0.73, n))['diff-Z'].iloc[0]
    weak_z = _edges('diff-Z', _pearson_scores(0.30, n), _pearson_scores(0.21, n))['diff-Z'].iloc[0]
    assert strong_z / weak_z < 3


def test_diff_z_detects_sign_flip():
    # Edge 0: same p = 0.01 and n = 100, r = +0.3 vs -0.3. diff-LP misses it, diff-Z = 2 z / sqrt(2).
    assert _edges('diff-LP')['diff-LP'].iloc[0] == pytest.approx(0.0)
    assert _edges('diff-Z')['diff-Z'].iloc[0] == pytest.approx(2 * stats.norm.isf(0.01 / 2) / np.sqrt(2))


@pytest.mark.parametrize('test_type', ['kruskal', 'anova', 'chi2'])
def test_diff_z_treats_unsigned_effects_as_unsigned(test_type):
    # A slightly negative (bias-corrected) η² is no direction: the same p in both contexts gives diff-Z = 0.
    s1 = _scores([0.5], [-0.008], test_type=test_type)
    s2 = _scores([0.5], [0.002], test_type=test_type)
    assert _edges('diff-Z', s1, s2)['diff-Z'].iloc[0] == pytest.approx(0.0)


def test_diff_z_needs_sample_size():
    with pytest.raises(ValueError, match="'n'"):
        _edges('diff-Z', S1.drop(columns='n'), S2.drop(columns='n'))


@pytest.mark.parametrize('metric', ['diff-LPxRE-cap', 'diff-LPxVE-cap', 'STC-LPxRE-cap'])
def test_removed_capped_metrics_raise(metric):
    with pytest.raises(ValueError, match='Invalid'):
        if metric.startswith('diff-'):
            _edges(metric)
        else:
            ctx = pd.DataFrame(columns=list('abcd'))
            compute_diff_nodes(S1.copy(), S2.copy(), ctx, ctx, metric)


@pytest.mark.parametrize('metric', ['diff-L-P', 'diff-E', 'diff-T-E', 'diff-L-PE', 'int-IS-E', 'sum-diff-PE'])
def test_removed_edge_metrics_raise(metric):
    with pytest.raises(ValueError, match='diff-LPxRE'):
        _edges(metric)


# ---------------------------------------------------------------------------------------- nodes

def _context(seed, shift=0.0, n=120):
    rng = np.random.default_rng(seed)
    d = {f'c{i}': rng.normal(size=n) for i in range(5)}
    d['c0'] = d['c0'] + shift                           # continuous node shifted between contexts
    d['b1'] = (rng.random(n) < 0.5 + shift / 4).astype(float)  # binary node shifted between contexts
    d['b2'] = rng.integers(0, 2, n).astype(float)
    d['o1'] = rng.integers(0, 4, n).astype(float)
    d['n1'] = rng.integers(0, 3, n).astype(float)
    df = pd.DataFrame(d)
    meta = pd.DataFrame({'label': df.columns, 'type': ['continuous'] * 5 + ['binary', 'binary', 'ordinal', 'nominal']})
    return df, meta


def test_wdc_matches_hand_computation():
    s1 = _scores([0.1, 0.1, 0.1, 0.1], [0.5, -0.4, 0.3, 0.2])
    s2 = _scores([0.1, 0.1, 0.1, 0.1], [0.1, 0.4, 0.0, 0.2])
    ctx = pd.DataFrame(columns=list('abcd'))
    nodes = compute_diff_nodes(s1, s2, ctx, ctx, 'WDC-VE')
    # |VE| per node: context 1 a=.41 b=.34 c=.29 d=.04, context 2 a=.17 b=.01 c=.20 d=.04
    w1 = pd.Series({'a': 0.25 + 0.16, 'b': 0.25 + 0.09, 'c': 0.16 + 0.09 + 0.04, 'd': 0.04})
    w2 = pd.Series({'a': 0.01 + 0.16, 'b': 0.01, 'c': 0.16 + 0.04, 'd': 0.04})
    expected = (w1 / w1.max() - w2 / w2.max()).abs()
    assert np.allclose(nodes['WDC-VE'].reindex(expected.index), expected)


@pytest.mark.parametrize('metric', [m for m in NODE_METRICS if not m.startswith('STC')])
def test_wdc_prc_metrics_run(metric):
    s1 = _scores([0.01, 1e-6, 1e-20, 0.3], [0.3, 0.6, 0.5, 0.1])
    s2 = _scores([0.01, 0.2, 1e-3, 0.04], [-0.3, 0.1, 0.5, 0.2])
    ctx = pd.DataFrame(columns=list('abcd'))
    nodes = compute_diff_nodes(s1, s2, ctx, ctx, metric)
    assert list(nodes.columns) == [metric]
    assert nodes[metric].between(0, 1).all()


@pytest.mark.parametrize('test_type', ['parametric', 'nonparametric'])
def test_stc_scores(test_type):
    ctx1, meta = _context(1)
    ctx2, _ = _context(2, shift=1.0)
    stc = stat_test_centrality(ctx1, ctx2, meta, test_type=test_type, nan_value=NAN)
    p = stc['adj-P'].fillna(1.0)
    assert np.allclose(stc['STC-P'], 1 - p)
    assert np.allclose(stc['STC-LP'], -np.log10(p))
    assert stc['STC-VE'].between(0, 1).all()
    assert stc['STC-RE'].between(0, 1.96 + 1e-9).all()
    assert np.allclose(stc['STC-LPxRE'], stc['STC-LP'] * stc['STC-RE'])
    assert np.allclose(stc['STC-LPxVE'], stc['STC-LP'] * stc['STC-VE'])
    # within each test family the largest |effect| gets the maximal rescaled value
    for _, fam in stc.groupby('test_type'):
        if len(fam) > 1:
            assert fam.loc[fam['test_E'].abs().idxmax(), 'STC-RE'] == pytest.approx(1.96, abs=1e-2)
    # the shifted continuous node is the most differential continuous node, with a positive effect
    # (Context 2 higher) under both tests
    cont = stc.loc[[f'c{i}' for i in range(5)]]
    assert cont['STC-LPxVE'].idxmax() == 'c0'
    assert stc.loc['c0', 'test_E'] > 0


def test_stc_p_matches_scipy():
    ctx1, meta = _context(1)
    ctx2, _ = _context(2, shift=1.0)
    stc = stat_test_centrality(ctx1, ctx2, meta, test_type='parametric', nan_value=NAN)
    cont = [f'c{i}' for i in range(5)]
    p_cont = [stats.ttest_ind(ctx1[c], ctx2[c]).pvalue for c in cont]
    p_ord = [stats.mannwhitneyu(ctx1['o1'], ctx2['o1']).pvalue]
    p_chi = [stats.chi2_contingency(pd.crosstab(np.r_[np.zeros(len(ctx1)), np.ones(len(ctx2))],
                                                 pd.concat([ctx1[c], ctx2[c]]).to_numpy()), correction=False).pvalue
             for c in ('b1', 'b2', 'n1')]
    labels = cont + ['o1', 'b1', 'b2', 'n1']
    adjusted = pd.Series(fdr_correction(np.array(p_cont + p_ord + p_chi)), index=labels)
    assert np.allclose(stc.loc[labels, 'STC-P'], 1 - adjusted, atol=1e-3)
    # metric_pvalue='raw-P' uses the unadjusted node p-values instead
    stc_raw = stat_test_centrality(ctx1, ctx2, meta, test_type='parametric', nan_value=NAN, metric_pvalue='raw-P')
    assert np.allclose(stc_raw.loc[labels, 'STC-P'], 1 - np.array(p_cont + p_ord + p_chi), atol=1e-3)
    # both p-values are stored whichever one the scores use
    for out in (stc, stc_raw):
        assert np.allclose(out.loc[labels, 'raw-P'], p_cont + p_ord + p_chi, atol=1e-3)
        assert np.allclose(out.loc[labels, 'adj-P'], adjusted, atol=1e-3)


@pytest.mark.parametrize('metric_pvalue', ['adj-P', 'raw-P'])
def test_stc_node_output_keeps_both_pvalues(metric_pvalue):
    ctx1, meta = _context(1)
    ctx2, _ = _context(2, shift=1.0)
    pairs = [(a, b) for i, a in enumerate(ctx1.columns) for b in ctx1.columns[i + 1:]]
    s = pd.DataFrame({'label1': [a for a, _ in pairs], 'label2': [b for _, b in pairs], 'test_type': 'pearson',
                      'raw-P': 0.05, 'adj-P': 0.1, 'raw-E': 0.1, 'n': 100})
    nodes = compute_diff_nodes(s, s, ctx1, ctx2, 'STC-LP', meta_file=meta, test_type='parametric', nan_value=NAN,
                               metric_pvalue=metric_pvalue)
    assert list(nodes.columns) == ['STC-LP', 'raw-P', 'adj-P']
    assert (nodes['adj-P'] >= nodes['raw-P']).all()
    assert np.allclose(nodes['STC-LP'], -np.log10(nodes[metric_pvalue]))


def test_stc_requires_meta_file():
    ctx1, _ = _context(1)
    s = _scores([0.1], [0.1])
    with pytest.raises(ValueError, match='meta_file'):
        compute_diff_nodes(s, s, ctx1, ctx1, 'STC-LP')


@pytest.mark.parametrize('metric', ['STC', 'DC-P', 'DC-E', 'WDC-E', 'WDC-L-P', 'PRC-E', 'PRC-L-P'])
def test_removed_node_metrics_raise(metric):
    s = _scores([0.1], [0.1])
    ctx = pd.DataFrame(columns=list('ab'))
    with pytest.raises(ValueError, match='STC-LPxRE'):
        compute_diff_nodes(s, s, ctx, ctx, metric)


# ---------------------------------------------------------------------------------------- ranking

def test_productrank_hand_computed():
    nodes_diff = pd.DataFrame({'WDC-P': [3.0, 2.0, 2.0, 1.0]}, index=list('abcd'))
    stats_ = pd.DataFrame({'edge-mean': [0.1, 0.4, 0.4], 'edge-median': [0.5, 0.2, 0.1]}, index=list('abd'))   # 'c' has no edges
    pr = productrank(nodes_diff, 'WDC-P', stats_, statistic='edge-mean')
    # node ranks: a=1, b=c=2.5, d=4; edge ranks: b=d=1.5, a=3, c=4 (no edges -> last)
    assert pr['rank_node'].to_dict() == {'a': 1.0, 'b': 2.5, 'c': 2.5, 'd': 4.0}
    assert pr['rank_edge'].to_dict() == {'a': 3.0, 'b': 1.5, 'c': 4.0, 'd': 1.5}
    assert pr['score'].to_dict() == {'a': 3.0, 'b': 3.75, 'c': 10.0, 'd': 6.0}
    # median edge ranks: a=1, b=2, d=3, c=4 (no edges -> last)
    pr = productrank(nodes_diff, 'WDC-P', stats_, statistic='edge-median')
    assert pr['rank_edge'].to_dict() == {'a': 1.0, 'b': 2.0, 'c': 4.0, 'd': 3.0}
    assert pr['score'].to_dict() == {'a': 1.0, 'b': 5.0, 'c': 10.0, 'd': 12.0}


# Edges a-b 0.9, a-c 0.1, a-d 0.2, b-c 0.4: mean and median of a node's incident edges disagree for 'a'
_STAT_EDGES = pd.DataFrame({'label1': list('aaab'), 'label2': list('bcdc'), 'test_type': 'pearson',
                            'diff-LP': [0.9, 0.1, 0.2, 0.4]})


@pytest.mark.parametrize('alg, expected', [
    ('MeanEdgeRank', {'a': 0.4, 'b': 0.65, 'c': 0.25, 'd': 0.2}),
    ('MedianEdgeRank', {'a': 0.2, 'b': 0.65, 'c': 0.25, 'd': 0.2}),
])
def test_edge_statistic_rankings(alg, expected):
    ranking = compute_ranking(None, _STAT_EDGES, alg)
    assert ranking.set_index('node')['score'].to_dict() == pytest.approx(expected)
    assert ranking['score'].is_monotonic_decreasing
    assert ranking['node'].iloc[0] == 'b'
    with pytest.raises(ValueError, match='edges_diff'):
        compute_ranking(pd.DataFrame({'WDC-P': [1.0]}, index=['a']), None, alg)


@pytest.mark.parametrize('alg, statistic', [('MeanProductRank', 'edge-mean'), ('MedianProductRank', 'edge-median')])
def test_productrank_ranking_is_ascending(alg, statistic):
    edges = _edges('diff-LP', s1=_scores([0.01, 1e-6, 1e-20, 0.3], [0.3, 0.6, 0.5, 0.1]),
                   s2=_scores([0.01, 0.2, 1e-3, 0.04], [-0.3, 0.1, 0.5, 0.2]))
    nodes = pd.DataFrame({'WDC-P': [0.9, 0.1, 0.5, 0.0]}, index=list('abcd'))
    ranking = compute_ranking(nodes, edges, alg)
    assert ranking['rank'].iloc[0] == 1
    assert ranking['score'].is_monotonic_increasing
    assert {'rank_node', 'rank_edge', 'WDC-P', statistic} <= set(ranking.columns)
    with pytest.raises(ValueError, match='nodes_diff'):
        compute_ranking(None, edges, alg)


def test_median_product_rank_uses_median_edge_rank():
    nodes = pd.DataFrame({'WDC-P': [0.1, 0.2, 0.3, 0.4]}, index=list('abcd'))
    mean = compute_ranking(nodes, _STAT_EDGES, 'MeanProductRank').set_index('node')['rank_edge']
    median = compute_ranking(nodes, _STAT_EDGES, 'MedianProductRank').set_index('node')['rank_edge']
    # edge ranks follow the MeanEdgeRank / MedianEdgeRank scores (descending, ties averaged)
    assert mean.to_dict() == {'b': 1.0, 'a': 2.0, 'c': 3.0, 'd': 4.0}
    assert median.to_dict() == {'b': 1.0, 'c': 2.0, 'a': 3.5, 'd': 3.5}


@pytest.mark.parametrize('metric, rescaling', [
    ('diff-P', None), ('diff-LP', None), ('diff-RE', 'RE'), ('diff-LPxRE', 'RE'),
    ('sum-LP-RE', 'RE'), ('diff-VE', 'VE'), ('diff-LPxVE', 'VE'), ('sum-LP-VE', 'VE'), ('diff-Z', None),
    ('STC-P', None), ('WDC-LP', None), ('PRC-RE', 'RE'), ('STC-LPxRE', 'RE'), ('WDC-VE', 'VE'), ('STC-LPxVE', 'VE'),
])
def test_effect_rescaling(metric, rescaling):
    assert effect_rescaling(metric) == rescaling


@pytest.mark.parametrize('node_metric, edge_metric, allowed', [
    ('STC-LPxRE', 'diff-LPxRE', True), ('STC-VE', 'sum-LP-VE', True), ('STC-P', 'diff-LPxVE', True),
    ('WDC-RE', 'diff-LP', True), ('STC-LPxRE', 'diff-LPxVE', False), ('PRC-VE', 'sum-LP-RE', False),
    ('STC-LPxRE', 'diff-Z', True), ('STC-VE', 'diff-Z', True),
])
def test_rescaling_consistency_for_combined_algorithms(node_metric, edge_metric, allowed):
    edges = _edges(edge_metric)
    nodes = pd.DataFrame({node_metric: [0.9, 0.1, 0.5]}, index=list('abc'))
    for alg in ('PageRank+', 'MeanProductRank', 'MedianProductRank'):
        if allowed:
            compute_ranking(nodes, edges, alg)
        else:
            with pytest.raises(ValueError, match='rescaling'):
                compute_ranking(nodes, edges, alg)
    # algorithms using only one of the two metrics are not restricted
    compute_ranking(nodes, edges, 'nodeRank')
    compute_ranking(nodes, edges, 'PageRank')


def test_invalid_ranking_alg_raises():
    with pytest.raises(ValueError, match='MedianProductRank'):
        compute_ranking(None, _edges('diff-LP'), 'direct_node')
    for removed in ('productRank', 'absDimontRank', 'DimontRank'):
        with pytest.raises(ValueError, match='Invalid ranking algorithm'):
            compute_ranking(None, _edges('diff-LP'), removed)


# ---------------------------------------------------------------------------------------- end-to-end

@pytest.fixture(scope='module')
def network():
    ctx1, meta = _context(1, n=80)
    ctx2, _ = _context(2, shift=1.0, n=80)
    s1 = modina.compute_context_scores(ctx1, meta, nan_value=NAN)
    s2 = modina.compute_context_scores(ctx2, meta, nan_value=NAN)
    return ctx1, ctx2, meta, s1, s2


@pytest.fixture(scope='module')
def all_nodes(network):
    ctx1, ctx2, meta, s1, s2 = network
    return {m: modina.compute_diff_nodes(s1, s2, ctx1, ctx2, m, meta_file=meta, nan_value=NAN)
            for m in NODE_METRICS}


@pytest.mark.parametrize('edge_metric', list(EDGE_METRICS))
def test_all_metric_and_algorithm_combinations(network, all_nodes, edge_metric):
    _, _, meta, s1, s2 = network
    edges, stats_ = modina.compute_diff_edges(s1, s2, edge_metric)
    for node_metric, nodes in all_nodes.items():
        mixed = {effect_rescaling(node_metric), effect_rescaling(edge_metric)} == {'RE', 'VE'}
        for alg in RANKING_ALGORITHMS:
            if mixed and all(ALG_REQUIRES[alg]):
                with pytest.raises(ValueError, match='rescaling'):
                    compute_ranking(nodes, edges, alg, meta_file=meta, edge_node_stats=stats_)
                continue
            ranking = compute_ranking(nodes, edges, alg, meta_file=meta, edge_node_stats=stats_)
            assert ranking['rank'].min() == 1
            assert ranking['node'].is_unique


def test_diffnet_analysis_validates_before_computing():
    ctx1, meta = _context(1)
    with pytest.raises(ValueError, match='Invalid edge metric'):
        modina.diffnet_analysis(ctx1, ctx1, meta, edge_metric='diff-LPxRE-cap', node_metric='STC-P')
    with pytest.raises(ValueError):
        modina.diffnet_analysis(ctx1, ctx1, meta, edge_metric='int-IS-E', node_metric='STC-P')


def test_diffnet_analysis_rejects_mixed_rescaling_before_computing():
    ctx1, meta = _context(1)
    with pytest.raises(ValueError, match='rescaling'):
        modina.diffnet_analysis(ctx1, ctx1, meta, edge_metric='diff-LPxVE', node_metric='STC-LPxRE', ranking_alg='PageRank+')


def test_diffnet_analysis_records_config():
    ctx1, meta = _context(1)
    ctx2, _ = _context(2, shift=1.0)
    ranking, _, nodes, config = modina.diffnet_analysis(
        ctx1, ctx2, meta, edge_metric='diff-LPxRE', node_metric='STC-LPxRE',
        ranking_alg='MeanProductRank', nan_value=NAN)
    assert config['edge_metric'] == 'diff-LPxRE' and config['node_metric'] == 'STC-LPxRE'
    assert 'lp_cap' not in config
    assert list(nodes.columns) == ['STC-LPxRE', 'raw-P', 'adj-P']
    assert ranking['rank'].iloc[0] == 1
