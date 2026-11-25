"""Tests for `modina.simulate_copula`/`modina.save_gt` after promoting the background-enabled
simulator (see `simulation_doc/Simulation_TODO.md` Tier 1.1) and replacing the fixed `shift`/`corr`
+ jitter mechanism with required `corr_low`/`corr_high`/`shift_low`/`shift_high` ranges.

Covers: the new parameters being required and validated, magnitudes actually being drawn from the
given range and recorded for ranking (`edge_magnitude`, the new `node_shift_magnitude`), the
already-working background construction (untouched by this change, exercised here for the first
time since no test file previously existed), and that `_jittered_effect`/`_EFFECT_JITTER` are
actually gone rather than just unused.
"""

import random

import numpy as np
import pandas as pd
import pytest

import modina
from modina import context_simulation


def test_missing_range_params_raise_type_error():
    with pytest.raises(TypeError):
        modina.simulate_copula(n_cont=5, n_bi=0, n_cat=0)


def test_corr_range_validation():
    with pytest.raises(ValueError, match='corr_low'):
        modina.simulate_copula(corr_low=0.5, corr_high=0.3, shift_low=0.1, shift_high=0.5)
    with pytest.raises(ValueError, match='corr_low'):
        modina.simulate_copula(corr_low=0.1, corr_high=0.99, shift_low=0.1, shift_high=0.5)


def test_shift_range_validation():
    with pytest.raises(ValueError, match='shift_low'):
        modina.simulate_copula(corr_low=0.1, corr_high=0.5, shift_low=0.8, shift_high=0.2)


def test_magnitudes_drawn_from_the_given_range():
    random.seed(0)
    np.random.seed(0)
    _, _, _, (shift_nodes, corr_nodes, shift_corr_nodes), effects = modina.simulate_copula(
        n_cont=20, n_bi=0, n_cat=0, n_samples_1=50, n_samples_2=50,
        n_corr_cont_cont=5, n_shift_cont=5,
        corr_low=0.1, corr_high=0.5, shift_low=0.2, shift_high=0.8)

    assert len(effects['edge_magnitude']) == 5
    for magnitude in effects['edge_magnitude'].values():
        assert 0.1 <= magnitude <= 0.5

    shifted = set(shift_nodes) | {n for pair in shift_corr_nodes for n in pair}
    assert len(shifted) == 5
    for node in shifted:
        assert 0.2 <= effects['node_shift_magnitude'][node] <= 0.8
    # A correlation-only node (not itself shifted) has no shift magnitude of its own.
    corr_only = {n for pair in corr_nodes for n in pair} - shifted
    for node in corr_only:
        assert effects['node_shift_magnitude'].get(node, 0.0) == 0.0


def test_edge_magnitudes_span_a_wide_range_for_ranking(tmp_path):
    random.seed(1)
    np.random.seed(1)
    modina.simulate_copula(
        path=str(tmp_path), name1='c1', name2='c2',
        n_cont=20, n_bi=0, n_cat=0, n_samples_1=50, n_samples_2=50,
        n_corr_cont_cont=8,
        corr_low=0.05, corr_high=0.9, shift_low=0.1, shift_high=0.2)

    edges = pd.read_csv(tmp_path / 'ground_truth_edges.txt', skipinitialspace=True)
    assert list(edges.columns) == ['edge', 'description', 'corr']
    # A single fixed value +/- 0.1 jitter (the old mechanism) could never span this wide a range.
    assert edges['corr'].max() - edges['corr'].min() > 0.1


def test_shift_column_end_to_end(tmp_path):
    random.seed(2)
    np.random.seed(2)
    _, _, _, (shift_nodes, corr_nodes, shift_corr_nodes), effects = modina.simulate_copula(
        path=str(tmp_path), name1='c1', name2='c2',
        n_cont=20, n_bi=0, n_cat=0, n_samples_1=50, n_samples_2=50,
        n_corr_cont_cont=3, n_shift_cont=3,
        corr_low=0.2, corr_high=0.4, shift_low=0.3, shift_high=0.6)

    nodes = pd.read_csv(tmp_path / 'ground_truth_nodes.txt', skipinitialspace=True)
    # Pinned as an exact list, like the edge file above: these headers are the interface the
    # downstream evaluation reads by name, so a rename must fail here rather than in a Nextflow run.
    assert list(nodes.columns) == ['node', 'description', 'n_edges_tweaked', 'sum_corr', 'shift']

    shifted = set(shift_nodes) | {n for pair in shift_corr_nodes for n in pair}
    corr_only = {n for pair in corr_nodes for n in pair} - shifted
    for _, row in nodes.iterrows():
        if row['node'] in shifted:
            assert row['shift'] == pytest.approx(effects['node_shift_magnitude'][row['node']])
        elif row['node'] in corr_only:
            assert row['shift'] == 0.0


def test_jittered_effect_removed():
    assert not hasattr(context_simulation, '_jittered_effect')
    assert not hasattr(context_simulation, '_EFFECT_JITTER')


def test_background_smoke():
    random.seed(3)
    np.random.seed(3)
    _, _, _, (shift_nodes, corr_nodes, shift_corr_nodes), effects = modina.simulate_copula(
        n_cont=30, n_bi=0, n_cat=0, n_samples_1=50, n_samples_2=50,
        n_corr_cont_cont=2,
        corr_low=0.1, corr_high=0.3, shift_low=0.1, shift_high=0.2,
        background_model='scale_free', background_strength=0.2, background_density=2,
        background_seed=0)

    corr_base = effects['corr_base']
    assert np.allclose(np.diag(corr_base), 1.0)
    assert np.linalg.eigvalsh(corr_base).min() > 0

    planted = {n for pair in corr_nodes for n in pair} | set(shift_nodes)
    node_order = effects['node_order']
    unplanted_idx = [i for i, node in enumerate(node_order) if node not in planted]
    assert len(unplanted_idx) >= 2
    i, j = unplanted_idx[0], unplanted_idx[1]
    assert effects['corr_matrix1'][i, j] == effects['corr_matrix2'][i, j]


def test_module_background():
    random.seed(3)
    np.random.seed(3)
    _, _, _, (shift_nodes, corr_nodes, _), effects = modina.simulate_copula(
        n_cont=40, n_bi=0, n_cat=0, n_samples_1=50, n_samples_2=50,
        n_corr_cont_cont=2,
        corr_low=0.1, corr_high=0.4, shift_low=0.1, shift_high=0.2,
        background_model='modules', module_size=10, module_correlation=0.2, background_seed=0)

    corr_base = effects['corr_base']
    assert np.allclose(np.diag(corr_base), 1.0)
    assert np.linalg.eigvalsh(corr_base).min() > 1 - context_simulation._MODULE_LOADING[1] ** 2 - 1e-9
    # four modules of ten variables: 4 * 45 within-module pairs, all nonzero; between-module
    # correlations are weaker by the factor module_correlation
    assert len(effects['background_edges']) == 4 * 45
    within = np.array([abs(corr_base[a, b]) for a, b in
                       ((effects['node_order'].index(x), effects['node_order'].index(y))
                        for x, y in map(tuple, effects['background_edges']))])
    assert within.min() >= context_simulation._MODULE_LOADING[0] ** 2 - 1e-12
    assert np.abs(corr_base[~np.eye(40, dtype=bool)]).max() <= context_simulation._MODULE_LOADING[1] ** 2 + 1e-12

    planted = {n for pair in corr_nodes for n in pair} | set(shift_nodes)
    unplanted_idx = [i for i, node in enumerate(effects['node_order']) if node not in planted]
    i, j = unplanted_idx[0], unplanted_idx[1]
    assert effects['corr_matrix1'][i, j] == effects['corr_matrix2'][i, j]


def test_module_background_is_the_default_and_scales_with_the_variables():
    random.seed(4)
    np.random.seed(4)
    for n_cont, fraction, n_modules in ((40, 0.1, 10), (40, 0.25, 4), (80, 0.1, 10)):
        _, _, _, _, effects = modina.simulate_copula(
            n_cont=n_cont, n_bi=0, n_cat=0, n_samples_1=20, n_samples_2=20,
            corr_low=0.1, corr_high=0.3, shift_low=0.1, shift_high=0.2,
            module_fraction=fraction, background_seed=0)
        size = round(fraction * n_cont)
        # equal-size modules: n_modules * size*(size-1)/2 within-module pairs
        assert len(effects['background_edges']) == n_modules * size * (size - 1) // 2
    # an explicit module_size overrides module_fraction
    _, _, _, _, effects = modina.simulate_copula(
        n_cont=40, n_bi=0, n_cat=0, n_samples_1=20, n_samples_2=20,
        corr_low=0.1, corr_high=0.3, shift_low=0.1, shift_high=0.2,
        module_size=20, module_fraction=0.1, background_seed=0)
    assert len(effects['background_edges']) == 2 * 190


def test_module_background_rejects_bad_parameters():
    with pytest.raises(ValueError, match='module_correlation'):
        context_simulation._build_module_background(20, 5, 1.0, 0)
    with pytest.raises(ValueError, match='module_size'):
        context_simulation._build_module_background(20, 1, 0.2, 0)
    with pytest.raises(ValueError, match='module_fraction'):
        modina.simulate_copula(n_cont=5, n_bi=0, n_cat=0, corr_low=0.1, corr_high=0.2, shift_low=0.1,
                               shift_high=0.2, module_fraction=0.0)
    with pytest.raises(ValueError, match='background_model'):
        modina.simulate_copula(n_cont=5, n_bi=0, n_cat=0, corr_low=0.1, corr_high=0.2, shift_low=0.1,
                               shift_high=0.2, background_model='dense')


def test_infeasible_background_raises_naming_corr_high(monkeypatch):
    # The background/hub logic that decides whether a config is ACTUALLY infeasible is Jannik's,
    # unchanged here and not re-validated by this test. What this change touched is the message --
    # it used to name `corr`, now it must name `corr_high` -- so the exact-eigenvalue-check branch
    # is forced deterministically rather than gambled on via a statistically infeasible config.
    monkeypatch.setattr(np.linalg, 'eigvalsh', lambda m: np.array([-1.0]))
    with pytest.raises(ValueError, match='corr_high'):
        modina.simulate_copula(
            n_cont=5, n_bi=0, n_cat=0, n_samples_1=20, n_samples_2=20,
            n_corr_cont_cont=1,
            corr_low=0.3, corr_high=0.3, shift_low=0.1, shift_high=0.1)



def _simulate_with_missing(missing_low=0.0, missing_high=0.0, missing_differential=0.0):
    random.seed(7)
    np.random.seed(7)
    return modina.simulate_copula(
        n_cont=20, n_bi=20, n_cat=20, n_samples_1=400, n_samples_2=300,
        n_shift_cont=2, n_corr_bi_cont=2,
        corr_low=0.1, corr_high=0.4, shift_low=0.2, shift_high=0.8, background_seed=1,
        missing_low=missing_low, missing_high=missing_high, missing_differential=missing_differential)


def test_missingness_only_removes_values():
    c1, c2, meta, gt, effects = _simulate_with_missing()
    m1, m2, meta_m, gt_m, effects_m = _simulate_with_missing(0.1, 0.3)
    assert effects['missing_rate_1'] == {} and not c1.isna().any().any()
    # same ground truth and same complete data: masking draws from its own stream, after the sampling
    assert gt_m == gt and meta_m.equals(meta)
    for full, masked in ((c1, m1), (c2, m2)):
        observed = masked.notna()
        assert (masked[observed] == full[observed]).sum().sum() == observed.sum().sum()
    # shared rates in [missing_low, missing_high], realised close to the draw
    rates = pd.Series(effects_m['missing_rate_1'])
    assert effects_m['missing_rate_2'] == effects_m['missing_rate_1'] and effects_m['missing_differential_nodes'] == []
    assert set(rates.index) == set(m1.columns) and rates.between(0.1, 0.3).all()
    for masked in (m1, m2):
        assert (masked.isna().mean() - rates[masked.columns]).abs().max() < 0.1


def test_missingness_differential_levels_are_paired():
    levels = {f: _simulate_with_missing(0.1, 0.3, f) for f in (0.0, 0.2, 1.0)}
    shared, mixed, per_context = (levels[f] for f in (0.0, 0.2, 1.0))
    # context 1 identical across the levels
    assert shared[0].equals(mixed[0]) and shared[0].equals(per_context[0])
    nodes = {f: set(level[4]['missing_differential_nodes']) for f, level in levels.items()}
    assert len(nodes[0.2]) == round(0.2 * 60) and nodes[0.2] <= nodes[1.0] and len(nodes[1.0]) == 60
    # context 2: variables with a shared rate are masked identically in every level
    same = [c for c in shared[1].columns if c not in nodes[0.2]]
    assert shared[1][same].equals(mixed[1][same])
    r1, r2 = pd.Series(per_context[4]['missing_rate_1']), pd.Series(per_context[4]['missing_rate_2'])
    assert r2.between(0.1, 0.3).all() and (r1 != r2).all()
    r2_mixed = pd.Series(mixed[4]['missing_rate_2'])
    assert (r2_mixed[sorted(nodes[0.2])] == r2[sorted(nodes[0.2])]).all()


def test_missingness_rate_validation():
    for low, high, differential in ((0.3, 0.1, 0.0), (-0.1, 0.2, 0.0), (0.1, 1.0, 0.0), (0.1, 0.3, 1.5)):
        with pytest.raises(ValueError, match='missing'):
            _simulate_with_missing(low, high, differential)
