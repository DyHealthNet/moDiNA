"""Hubs in `modina.simulate_copula`.

Planted hubs: one node with k differential edges to random partners, all changed in one context.
Module hubs: in one context every background correlation of the hub is scaled by `module_hub_scale`
(under the module background: the hub's module loading shrinks).
"""

import random

import numpy as np
import pandas as pd
import pytest

import modina

RANGES = dict(corr_low=0.1, corr_high=0.4, shift_low=0.2, shift_high=0.8)


def _simulate(seed=0, **kwargs):
    random.seed(seed)
    np.random.seed(seed)
    args = dict(n_bi=30, n_cont=30, n_cat=30, n_samples_1=200, n_samples_2=200, background_seed=seed, **RANGES)
    args.update(kwargs)
    return modina.simulate_copula(**args)


# ---------------------------------------------------------------------------------------------- planted hubs

def test_planted_hub_edges_change_in_one_context():
    *_, (shift_nodes, corr_nodes, _), effects = _simulate(
        n_hub_cont=2, n_hub_bi=1, n_hub_cat=1, hub_degree_low=3, hub_degree_high=6,
        n_shift_cont=2, n_corr_cont_cont=2, n_corr_bi_cat=1)
    c1, c2, base, nodes = effects['corr_matrix1'], effects['corr_matrix2'], effects['corr_base'], effects['node_order']
    hubs, partners = effects['hub_nodes'], effects['hub_partners']
    assert [h[:2] for h in hubs] == ['co', 'co', 'bi', 'or']
    other = set(shift_nodes) | {n for pair in corr_nodes for n in pair}
    all_partners = [p for hub in hubs for p in partners[hub]]
    assert len(all_partners) == len(set(all_partners))                 # partners are exclusive
    assert not (set(hubs) | set(all_partners)) & other                 # and disjoint from other effects
    for hub in hubs:
        h, idx = nodes.index(hub), [nodes.index(p) for p in partners[hub]]
        assert 3 <= len(idx) <= 6
        changed, untouched = (c1, c2) if effects['hub_context'][hub] == 1 else (c2, c1)
        np.testing.assert_allclose(untouched[h], base[h])              # the other context keeps the background
        rest = np.setdiff1d(np.arange(len(nodes)), idx + [h])
        np.testing.assert_allclose(changed[h, rest], base[h, rest])   # only the k partner cells change
        delta = np.abs(changed[h, idx] - base[h, idx])
        assert np.all((delta >= 0.1 - 1e-12) & (delta <= 0.4 + 1e-12))
        for p, d in zip(partners[hub], delta):
            assert effects['hub_edge_magnitude'][frozenset((hub, p))] == pytest.approx(d)
            assert effects['node_degree'][p] == 1
        assert effects['node_degree'][hub] == len(idx)
        assert effects['node_sum_magnitude'][hub] == pytest.approx(delta.sum())
    for matrix in (c1, c2):
        assert np.linalg.eigvalsh(matrix).min() > 0


def test_planted_hub_ground_truth_files(tmp_path):
    _simulate(path=str(tmp_path), n_hub_bi=1, hub_degree_low=4, hub_degree_high=4, n_shift_cat=1)
    nodes = pd.read_csv(tmp_path / 'ground_truth_nodes.txt', sep=', ', engine='python')
    assert nodes['description'].value_counts().to_dict() == {'diff. corr. planted hub partner': 4,
                                                             'diff. corr. planted hub': 1, 'mean shift': 1}
    assert nodes.set_index('node').loc['bi1', 'n_edges_tweaked'] == 4
    edges = pd.read_csv(tmp_path / 'ground_truth_edges.txt', sep=', ', engine='python')
    assert (edges['description'] == 'diff. corr. planted hub').sum() == 4
    assert edges['corr'].between(0.1, 0.4).all()


def test_too_strong_hubs_fail_loudly():
    with pytest.raises(ValueError, match='positive definite draw'):
        _simulate(n_hub_cont=1, hub_degree_low=40, hub_degree_high=40, corr_low=0.5, corr_high=0.6)


# ----------------------------------------------------------------------------------------------- module hubs

def test_module_hub_row_is_scaled_in_one_context_only():
    *_, (shift_nodes, corr_nodes, _), effects = _simulate(
        n_module_hub_cont=2, n_module_hub_bi=1, n_module_hub_cat=1, module_hub_scale=0.5,
        n_shift_cont=2, n_corr_cont_cont=2, n_corr_bi_cat=1)
    c1, c2, base, nodes = effects['corr_matrix1'], effects['corr_matrix2'], effects['corr_base'], effects['node_order']
    hubs = effects['module_hub_nodes']
    assert [h[:2] for h in hubs] == ['co', 'co', 'bi', 'or']
    planted = {nodes.index(n) for pair in corr_nodes for n in pair}
    for hub in hubs:
        h = nodes.index(hub)
        scaled, other = (c1, c2) if effects['module_hub_context'][hub] == 1 else (c2, c1)
        # all partners except other hubs (a hub-hub cell is also scaled by the other hub's rewiring)
        off = ~np.isin(np.arange(len(nodes)), [nodes.index(x) for x in hubs])
        np.testing.assert_allclose(other[h, off], base[h, off])
        np.testing.assert_allclose(scaled[h, off], 0.5 * base[h, off])
        assert scaled[h, h] == other[h, h] == 1.0
    diff = np.abs(c1 - c2) > 0
    hub_idx = [nodes.index(h) for h in hubs]
    diff[hub_idx, :] = diff[:, hub_idx] = False
    rows, cols = np.nonzero(diff)
    assert set(rows) | set(cols) <= planted
    for matrix in (c1, c2):
        assert np.linalg.eigvalsh(matrix).min() > 0
    assert not (set(shift_nodes) | {n for pair in corr_nodes for n in pair}) & set(hubs)


def test_module_hub_ground_truth(tmp_path):
    *_, (shift_nodes, corr_nodes, shift_corr_nodes), effects = _simulate(
        n_module_hub_cont=2, module_hub_scale=0.0, n_corr_cont_cont=1)
    c1, c2, nodes = effects['corr_matrix1'], effects['corr_matrix2'], effects['node_order']
    hub_edges = effects['module_hub_edge_magnitude']
    for hub in effects['module_hub_nodes']:
        h = nodes.index(hub)
        delta = np.abs(c1[h] - c2[h])
        expected = {frozenset((hub, nodes[j])) for j in np.flatnonzero(delta >= 0.1) if j != h}
        assert {e for e in hub_edges if hub in e} == expected
        assert len(expected) > 0
        assert effects['node_degree'][hub] == len(expected)
        assert effects['node_sum_magnitude'][hub] == pytest.approx(sum(hub_edges[e] for e in expected))

    # partners: the other end of every hub edge, except other hubs; their stats count those edges
    hubs = set(effects['module_hub_nodes'])
    partners = effects['module_hub_partners']
    for hub in hubs:
        assert set(partners[hub]) == {n for e in hub_edges if hub in e for n in e} - hubs
    pair_nodes = {n for pair in corr_nodes for n in pair}
    for p in {p for ps in partners.values() for p in ps}:
        n_hub_edges = sum(1 for e in hub_edges if p in e)
        assert effects['node_degree'][p] == n_hub_edges + (p in pair_nodes)

    all_partners = {p for ps in partners.values() for p in ps}
    modina.save_gt((shift_nodes, corr_nodes, shift_corr_nodes), tmp_path / 'nodes.txt', mode='node',
                   node_stats={n: (effects['node_degree'][n], effects['node_sum_magnitude'][n])
                               for n in hubs | all_partners},
                   module_hub_nodes=effects['module_hub_nodes'], module_hub_partners=partners)
    modina.save_gt((shift_nodes, corr_nodes, shift_corr_nodes), tmp_path / 'edges.txt', mode='edge',
                   edge_magnitude=effects['edge_magnitude'], module_hub_edge_magnitude=hub_edges)
    gt_nodes = pd.read_csv(tmp_path / 'nodes.txt', sep=', ', engine='python')
    assert list(gt_nodes.loc[gt_nodes['description'] == 'diff. corr. hub', 'node']) == effects['module_hub_nodes']
    written = gt_nodes.loc[gt_nodes['description'] == 'diff. corr. hub partner', 'node']
    assert written.is_unique and set(written) == all_partners
    assert (gt_nodes['description'] == 'diff. corr. pair').sum() == 2
    gt_edges = pd.read_csv(tmp_path / 'edges.txt', sep=', ', engine='python')
    assert (gt_edges['description'] == 'diff. corr. hub').sum() == len(hub_edges)
    assert (gt_edges['description'] == 'diff. corr. pair').sum() == 1


# ------------------------------------------------------------------------------------------- shared behaviour

def test_hubs_leave_all_other_effects_unchanged():
    """Planted hubs, module hubs and no hubs share the pairs, the shift nodes and their magnitudes."""
    common = dict(seed=5, n_shift_cont=2, n_shift_bi=2, n_corr_cont_cont=2, n_both_bi_cat=1)
    runs = [_simulate(**common), _simulate(**common, n_hub_cont=1, n_hub_bi=1),
            _simulate(**common, n_module_hub_cont=1, n_module_hub_bi=1)]
    for run in runs[1:]:
        assert run[3] == runs[0][3]
        assert run[4]['edge_magnitude'] == runs[0][4]['edge_magnitude']
        assert run[4]['node_shift_magnitude'] == runs[0][4]['node_shift_magnitude']
    # both hub types take the same hub nodes
    assert runs[1][4]['hub_nodes'] == runs[2][4]['module_hub_nodes']


def test_no_hubs_draws_nothing():
    a = _simulate(seed=3, n_corr_cont_cont=2, n_shift_bi=1)
    b = _simulate(seed=3, n_corr_cont_cont=2, n_shift_bi=1, hub_degree_low=2, module_hub_scale=0.3)
    pd.testing.assert_frame_equal(a[0], b[0])
    pd.testing.assert_frame_equal(a[1], b[1])
    assert a[4]['hub_nodes'] == [] and a[4]['module_hub_nodes'] == []


@pytest.mark.parametrize('kwargs, match', [
    (dict(module_hub_scale=1.0), 'module_hub_scale'),
    (dict(module_hub_scale=-0.1), 'module_hub_scale'),
    (dict(n_hub_cont=-1), 'n_hub'),
    (dict(n_module_hub_bi=-1), 'n_hub'),
    (dict(hub_degree_low=0), 'hub_degree_low'),
    (dict(hub_degree_low=5, hub_degree_high=4), 'hub_degree_low'),
    (dict(n_module_hub_cont=31), 'continuous abnormal nodes'),
    (dict(n_hub_cont=10, hub_degree_high=9), 'partners'),   # 10 x (1 + 9) > 90 nodes
])
def test_hub_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        _simulate(**kwargs)
