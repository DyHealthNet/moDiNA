"""Mixed-strength variant of the copula-based context simulator.

`context_simulation.simulate_copula` plants every differential edge/node in a run at one target
magnitude (jittered +/- 0.1 around a single `shift`/`corr` scalar), so one run samples the
detection-vs-effect-size curve at exactly one point -- see `simulation_doc/Simulation_TODO.md`,
Tier 1.3 ("Every differential edge has the identical magnitude"). `simulate_copula_mixed` plants
a *mix* of weak/moderate/strong effects in a single run instead: each injected edge/node
independently draws which band it belongs to, then its magnitude uniformly from that band's
(low, high) range, and the drawn band is recorded alongside the usual ground truth.

This module is a self-contained sibling of `context_simulation.py`, not a modification of it: it
reuses only `_simu_gaussian`, `save_gt` and two constants from there, and reimplements its own
version of `_set_corr` (`_set_corr_mixed`) so that `context_simulation.py` needs no changes at
all, and the existing `simulate_copula` stays exactly as it is.
"""

import logging
import os
import random

import numpy as np
import pandas as pd
import scipy as sc

from modina.context_simulation import _HUB_SUMSQ_CEILING, _MIN_MINORITY_COUNT, _simu_gaussian, save_gt

BAND_NAMES = ('weak', 'moderate', 'strong')


def _validate_bands(bands, param_name):
    if set(bands) != set(BAND_NAMES):
        raise ValueError(f"{param_name} must have exactly the keys {BAND_NAMES}, got {sorted(bands)}.")
    for name in BAND_NAMES:
        low, high = bands[name]
        if not 0.0 <= low <= high:
            raise ValueError(f"{param_name}['{name}'] must satisfy 0 <= low <= high, got ({low}, {high}).")
    return bands


def _draw_banded_effect(bands, weights=None, high=None):
    """Pick a band per `weights` (default: uniform over BAND_NAMES), then draw the magnitude
    uniformly from that band's (low, high), optionally capped at `high`.

    :param bands: Dict {'weak': (low, high), 'moderate': (low, high), 'strong': (low, high)}.
    :param weights: Optional dict of relative weights per band name. Default: uniform.
    :param high: Optional hard ceiling on the drawn magnitude (e.g. the 0.95 correlation ceiling).
    :return: (magnitude, band_name)
    """
    w = [weights.get(name, 1.0) for name in BAND_NAMES] if weights else None
    band = random.choices(BAND_NAMES, weights=w, k=1)[0]
    low, band_high = bands[band]
    magnitude = np.random.uniform(low, band_high)
    if high is not None:
        magnitude = min(magnitude, high)
    return magnitude, band


# Simulate mixed data using a gaussian copula, with weak/moderate/strong effects mixed into one run
def simulate_copula_mixed(path=None, name1='context1', name2='context2',
                          n_bi=50, n_cont=50, n_cat=50, n_samples_1=500, n_samples_2=500,
                          n_shift_cont=0, n_shift_bi=0, n_shift_cat=0,
                          n_corr_cont_cont=0, n_corr_bi_bi=0, n_corr_cat_cat=0, n_corr_bi_cont=0, n_corr_bi_cat=0, n_corr_cont_cat=0,
                          n_both_cont_cont=0, n_both_bi_bi=0, n_both_cat_cat=0, n_both_bi_cont=0, n_both_bi_cat=0, n_both_cont_cat=0,
                          corr_bands=None, shift_bands=None, band_weights=None, hub_reuse_prob=0.0,
                          binary_p_low=0.3, binary_p_high=0.7, ordinal_concentration=20.0):
    """
    Simulate two contexts with binary, continuous and ordinal nodes using a Gaussian copula, as
    `context_simulation.simulate_copula` does, but planting a *mix* of weak/moderate/strong
    differential effects in the same run instead of one uniform magnitude.

    All parameters shared with `simulate_copula` have the same meaning (see its docstring).
    Parameters specific to this function:

    :param corr_bands: Required. Dict {'weak': (low, high), 'moderate': (low, high), 'strong':
                       (low, high)} -- the correlation-magnitude range each band draws from.
                       Each injected correlation/both edge independently draws which band it
                       belongs to (see `band_weights`), then its magnitude uniformly from that
                       band's range.
    :param shift_bands: Required. Same shape as `corr_bands`, for mean-shift magnitude.
    :param band_weights: Optional dict of relative weights per band name, e.g. {'weak': 1,
                         'moderate': 1, 'strong': 1} (the default: every band equally likely).
                         Independent of the n_corr_*/n_shift_*/n_both_* counts, which still
                         control how many edges of each *type pair* are injected -- only which
                         band an individual edge falls into is governed by this.
    :return: A tuple containing the two simulated contexts, a meta file, a list of ground truth
             nodes and a dict of per-edge/per-node injected effect sizes.
             - context1: pd.DataFrame of the first simulated context.
             - context2: pd.DataFrame of the second simulated context.
             - meta: pd.DataFrame containing the data type for each simulated variable.
             - ground_truth: A tuple containing three lists of ground truth nodes: (shift_nodes, corr_nodes, shift_corr_nodes).
             - effects: dict with 'edge_magnitude', 'edge_band' (dicts keyed by
               frozenset({node1, node2})), 'node_degree', 'node_sum_magnitude' (both keyed by
               node, correlation-edge derived, as in `simulate_copula`), and
               'node_shift_magnitude'/'node_shift_band' (dicts keyed by node, for nodes that
               received their own mean-shift draw). Feeds the extra columns written to
               `ground_truth_edges.txt`/`ground_truth_nodes.txt` plus the
               `ground_truth_edge_bands.csv`/`ground_truth_node_bands.csv` sidecars when `path`
               is given.
    """
    if corr_bands is None or shift_bands is None:
        raise ValueError('simulate_copula_mixed requires both corr_bands and shift_bands; use '
                         'context_simulation.simulate_copula for a single fixed magnitude.')
    corr_bands = _validate_bands(corr_bands, 'corr_bands')
    shift_bands = _validate_bands(shift_bands, 'shift_bands')

    if n_bi <= 0 and n_cont <= 0 and n_cat <= 0:
        raise ValueError('Either n_bi, n_cont, or n_cat needs to be larger than zero.')
    if not 0.0 <= binary_p_low <= binary_p_high <= 1.0:
        raise ValueError(f'binary_p_low and binary_p_high must satisfy 0 <= low <= high <= 1, '
                         f'got low={binary_p_low}, high={binary_p_high}.')
    if ordinal_concentration <= 0:
        raise ValueError(f'ordinal_concentration must be larger than zero, got {ordinal_concentration}.')

    n_min_samples = min(n_samples_1, n_samples_2)
    n_bi_effects = (n_shift_bi + n_corr_bi_bi + n_both_bi_bi + n_corr_bi_cont + n_both_bi_cont
                    + n_corr_bi_cat + n_both_bi_cat)
    if n_bi_effects > 0 and binary_p_low * n_min_samples < _MIN_MINORITY_COUNT:
        logging.warning(
            f'binary_p_low={binary_p_low} with {n_min_samples} samples in the smaller context leaves '
            f'only about {binary_p_low * n_min_samples:.0f} samples in a binary node\'s rarer level, '
            f'below the ~{_MIN_MINORITY_COUNT} needed for a stable association estimate. Binary ground '
            f'truth effects will be noise-limited at this sample size whatever the base rates are, so '
            f'raising binary_p_low will not help. Increase n_samples_1/n_samples_2 instead.')

    n_cat_effects = (n_shift_cat + n_corr_cat_cat + n_both_cat_cat + n_corr_bi_cat + n_both_bi_cat
                     + n_corr_cont_cat + n_both_cont_cat)
    if n_cat_effects > 0 and n_min_samples / 5 < _MIN_MINORITY_COUNT:
        logging.warning(
            f'{n_min_samples} samples in the smaller context spread over 5 ordinal categories leaves '
            f'about {n_min_samples / 5:.0f} samples per category even when perfectly balanced, below '
            f'the ~{_MIN_MINORITY_COUNT} needed for a stable association estimate. Ordinal ground truth '
            f'effects will be noise-limited at this sample size regardless of ordinal_concentration.')
    if n_shift_cont + n_corr_cont_cont*2 + n_both_cont_cont*2 + n_corr_bi_cont + n_both_bi_cont + n_corr_cont_cat + n_both_cont_cat > n_cont:
        raise ValueError('The number of continuous abnormal nodes is larger than the total number of continuous nodes.')
    if n_shift_bi + n_corr_bi_bi*2 + n_both_bi_bi*2 + n_corr_bi_cont + n_both_bi_cont + n_corr_bi_cat + n_both_bi_cat > n_bi:
        raise ValueError('The number of binary abnormal nodes is larger than the total number of binary nodes.')
    if n_shift_cat + n_corr_cat_cat*2 + n_both_cat_cat*2 + n_corr_bi_cat + n_both_bi_cat + n_corr_cont_cat + n_both_cont_cat > n_cat:
        raise ValueError('The number of categorical abnormal nodes is larger than the total number of categorical nodes.')

    # Prepare dataframes
    cont_cols = [f"cont{i+1}" for i in range(n_cont)]
    context1_cont = pd.DataFrame(np.nan, index=range(n_samples_1), columns=cont_cols)
    context2_cont = pd.DataFrame(np.nan, index=range(n_samples_2), columns=cont_cols)

    bi_cols = [f"bi{i+1}" for i in range(n_bi)]
    context1_bi = pd.DataFrame(np.nan, index=range(n_samples_1), columns=bi_cols)
    context2_bi = pd.DataFrame(np.nan, index=range(n_samples_2), columns=bi_cols)

    cat_cols = [f"ord{i+1}" for i in range(n_cat)]
    context1_cat = pd.DataFrame(np.nan, index=range(n_samples_1), columns=cat_cols)
    context2_cat = pd.DataFrame(np.nan, index=range(n_samples_2), columns=cat_cols)

    # Create meta file
    all_cols = cont_cols + bi_cols + cat_cols
    meta = pd.DataFrame({
        "label": all_cols,
        "type": ["continuous"] * n_cont + ["binary"] * n_bi + ["ordinal"] * n_cat
    })

    # Initialize lists for ground truth nodes
    shift_nodes = []
    corr_nodes = []
    shift_corr_nodes = []

    normal_nodes_cont = list(context1_cont.columns) if n_cont > 0 else []
    normal_nodes_bi = list(context1_bi.columns) if n_bi > 0 else []
    normal_nodes_cat = list(context1_cat.columns) if n_cat > 0 else []
    nodes = normal_nodes_cont + normal_nodes_bi + normal_nodes_cat

    # Create correlation matrix
    n_vars = n_bi + n_cont + n_cat
    corr1 = np.eye(n_vars)
    corr2 = np.eye(n_vars)

    # Bookkeeping shared across every _set_corr_mixed call below, mirroring
    # context_simulation._set_corr's hub-reuse bookkeeping, plus the new per-edge band.
    node_degree = {}
    node_sumsq = {}
    node_sum_magnitude = {}
    node_first_partner = {}
    hub_eligible = set()
    edge_magnitude = {}
    edge_band = {}

    def _inject(**pools):
        node_pair, magnitude, band, c1, c2, *_ = _set_corr_mixed(
            nodes=nodes, corr_bands=corr_bands, band_weights=band_weights,
            corr_matrix1=corr1, corr_matrix2=corr2,
            node_degree=node_degree, node_sumsq=node_sumsq, node_sum_magnitude=node_sum_magnitude,
            node_first_partner=node_first_partner, hub_eligible=hub_eligible,
            hub_reuse_prob=hub_reuse_prob, **pools)
        edge_magnitude[frozenset(node_pair)] = magnitude
        edge_band[frozenset(node_pair)] = band
        return node_pair, c1, c2

    # Introduce fixed correlations in context 1 (leave context 2 uncorrelated)
    for _ in range(n_corr_cont_cont):
        node_pair, corr1, corr2 = _inject(normal_nodes_cont=normal_nodes_cont)
        corr_nodes.append(node_pair)

    for _ in range(n_corr_bi_bi):
        node_pair, corr1, corr2 = _inject(normal_nodes_bi=normal_nodes_bi)
        corr_nodes.append(node_pair)

    for _ in range(n_corr_cat_cat):
        node_pair, corr1, corr2 = _inject(normal_nodes_cat=normal_nodes_cat)
        corr_nodes.append(node_pair)

    for _ in range(n_corr_bi_cont):
        node_pair, corr1, corr2 = _inject(normal_nodes_bi=normal_nodes_bi, normal_nodes_cont=normal_nodes_cont)
        corr_nodes.append(node_pair)

    for _ in range(n_corr_bi_cat):
        node_pair, corr1, corr2 = _inject(normal_nodes_bi=normal_nodes_bi, normal_nodes_cat=normal_nodes_cat)
        corr_nodes.append(node_pair)

    for _ in range(n_corr_cont_cat):
        node_pair, corr1, corr2 = _inject(normal_nodes_cont=normal_nodes_cont, normal_nodes_cat=normal_nodes_cat)
        corr_nodes.append(node_pair)

    for _ in range(n_both_cont_cont):
        node_pair, corr1, corr2 = _inject(normal_nodes_cont=normal_nodes_cont)
        shift_corr_nodes.append(node_pair)

    for _ in range(n_both_bi_bi):
        node_pair, corr1, corr2 = _inject(normal_nodes_bi=normal_nodes_bi)
        shift_corr_nodes.append(node_pair)

    for _ in range(n_both_cat_cat):
        node_pair, corr1, corr2 = _inject(normal_nodes_cat=normal_nodes_cat)
        shift_corr_nodes.append(node_pair)

    for _ in range(n_both_bi_cont):
        node_pair, corr1, corr2 = _inject(normal_nodes_bi=normal_nodes_bi, normal_nodes_cont=normal_nodes_cont)
        shift_corr_nodes.append(node_pair)

    for _ in range(n_both_bi_cat):
        node_pair, corr1, corr2 = _inject(normal_nodes_bi=normal_nodes_bi, normal_nodes_cat=normal_nodes_cat)
        shift_corr_nodes.append(node_pair)

    for _ in range(n_both_cont_cat):
        node_pair, corr1, corr2 = _inject(normal_nodes_cont=normal_nodes_cont, normal_nodes_cat=normal_nodes_cat)
        shift_corr_nodes.append(node_pair)

    # Select nodes for mean shifts. Selection is deterministic (first available node in order),
    # so repeated simulations with the same settings always tweak exactly the same variables.
    # Nodes are drawn from the front of each type pool, after correlation pairs above consumed theirs.
    for _ in range(n_shift_cont):
        assert normal_nodes_cont, 'Introducing correlations was unsuccessful.'
        node = normal_nodes_cont.pop(0)
        shift_nodes.append(node)

    for _ in range(n_shift_bi):
        assert normal_nodes_bi, 'Introducing correlations was unsuccessful.'
        node = normal_nodes_bi.pop(0)
        shift_nodes.append(node)

    for _ in range(n_shift_cat):
        assert normal_nodes_cat, 'Introducing correlations was unsuccessful.'
        node = normal_nodes_cat.pop(0)
        shift_nodes.append(node)

    mean_vector1 = np.zeros(n_vars)
    mean_vector2 = np.zeros(n_vars)
    node_shift_magnitude = {}
    node_shift_band = {}
    for node in nodes:
        if node in shift_nodes or any(node in pair for pair in shift_corr_nodes):
            sign = random.choice([1, -1])
            context_idx = random.choice([1, 2])
            magnitude, band = _draw_banded_effect(shift_bands, band_weights)
            node_shift_magnitude[node] = magnitude
            node_shift_band[node] = band
            if context_idx == 1:
                mean_vector1[nodes.index(node)] = sign * magnitude
            else:
                mean_vector2[nodes.index(node)] = sign * magnitude

    # Gaussian copula
    u1 = _simu_gaussian(n=n_vars, m=n_samples_1, corr_matrix=corr1, mean_vector=mean_vector1)
    u2 = _simu_gaussian(n=n_vars, m=n_samples_2, corr_matrix=corr2, mean_vector=mean_vector2)

    # Transform to marginal distributions using the inverse CDF
    for i, node in enumerate(nodes):
        if i < n_cont:
            # Continuous node
            mean = 0.0
            std = 0.5
            context1_cont[node] = sc.stats.norm.ppf(u1[i, :], loc=mean, scale=std)
            context2_cont[node] = sc.stats.norm.ppf(u2[i, :], loc=mean, scale=std)

        elif n_cont <= i < n_cont + n_bi:
            # Binary node
            p = np.random.uniform(binary_p_low, binary_p_high)
            context1_bi[node] = sc.stats.bernoulli.ppf(u1[i, :], p=p).astype(int)
            context2_bi[node] = sc.stats.bernoulli.ppf(u2[i, :], p=p).astype(int)

        else:
            # Ordinal node
            n_categories = 5  # fixed number of categories
            p = np.random.dirichlet(np.full(n_categories, ordinal_concentration), size=1).flatten()
            cdf = np.cumsum(p)
            context1_cat[node] = np.searchsorted(cdf, u1[i, :])
            context2_cat[node] = np.searchsorted(cdf, u2[i, :])

    # Combine continuous and binary data
    context1 = context1_cont.join(context1_bi)
    context2 = context2_cont.join(context2_bi)

    context1 = context1.join(context1_cat)
    context2 = context2.join(context2_cat)

    context1 = context1.sort_index(axis=1)
    context2 = context2.sort_index(axis=1)

    # Per-node ground-truth stats (n_edges_tweaked, sum_jittered_magnitude), for every node that is
    # a ground-truth node at all.
    gt_nodes = set(shift_nodes)
    for pair in corr_nodes + shift_corr_nodes:
        gt_nodes.update(pair)
    node_stats = {node: (node_degree.get(node, 0), node_sum_magnitude.get(node, 0.0)) for node in gt_nodes}
    effects = {'edge_magnitude': edge_magnitude, 'edge_band': edge_band,
              'node_degree': node_degree, 'node_sum_magnitude': node_sum_magnitude,
              'node_shift_magnitude': node_shift_magnitude, 'node_shift_band': node_shift_band}

    # Save simulated contexts and ground truth nodes
    if path:
        context1.to_csv(os.path.join(path, f'{name1}.csv'))
        context2.to_csv(os.path.join(path, f'{name2}.csv'))
        meta.to_csv(os.path.join(path, 'meta.csv'), index=False)
        # The two files below keep the exact historical two-plus-existing-jitter-column format
        # that context_simulation.save_gt already produces, so every R evaluation script keeps
        # parsing them unmodified.
        save_gt((shift_nodes, corr_nodes, shift_corr_nodes), os.path.join(path, 'ground_truth_nodes.txt'), mode='node', node_stats=node_stats, node_shift_magnitude=node_shift_magnitude)
        save_gt((shift_nodes, corr_nodes, shift_corr_nodes), os.path.join(path, 'ground_truth_edges.txt'), mode='edge', edge_magnitude=edge_magnitude)

        # Band sidecars -- kept as separate files, rather than extra columns on the files above,
        # so the shared ground-truth format (and every R reader of it) is never at risk.
        edge_band_rows = [
            {'edge': '_'.join(sorted(pair)), 'band': edge_band.get(frozenset(pair)),
             'magnitude': edge_magnitude.get(frozenset(pair))}
            for pair in corr_nodes + shift_corr_nodes
        ]
        pd.DataFrame(edge_band_rows, columns=['edge', 'band', 'magnitude']).to_csv(
            os.path.join(path, 'ground_truth_edge_bands.csv'), index=False)

        node_band_rows = [
            {'node': node, 'band': node_shift_band[node], 'magnitude': node_shift_magnitude[node]}
            for node in node_shift_band
        ]
        pd.DataFrame(node_band_rows, columns=['node', 'band', 'magnitude']).to_csv(
            os.path.join(path, 'ground_truth_node_bands.csv'), index=False)

    return context1, context2, meta, (shift_nodes, corr_nodes, shift_corr_nodes), effects


# Helper function to set correlation in copula-based simulation, drawing the magnitude from
# `corr_bands` instead of jittering around a single scalar. Node-pool-selection and hub-reuse
# logic is otherwise identical to context_simulation._set_corr.
def _set_corr_mixed(nodes, corr_bands, corr_matrix1, corr_matrix2, normal_nodes_bi=None, normal_nodes_cont=None, normal_nodes_cat=None,
                    node_degree=None, node_sumsq=None, node_sum_magnitude=None, node_first_partner=None, hub_eligible=None,
                    hub_reuse_prob=0.0, band_weights=None):
    node_degree = {} if node_degree is None else node_degree
    node_sumsq = {} if node_sumsq is None else node_sumsq
    node_sum_magnitude = {} if node_sum_magnitude is None else node_sum_magnitude
    node_first_partner = {} if node_first_partner is None else node_first_partner
    hub_eligible = set() if hub_eligible is None else hub_eligible

    # Node selection is deterministic (first available node in order) for the fresh case, so
    # repeated simulations with the same settings still introduce the correlation on exactly the
    # same variable pairs when hub_reuse_prob=0. Each branch below just picks which pool(s) feed
    # the pair's two slots; the actual draw (fresh vs. hub reuse) happens after, uniformly.
    if normal_nodes_bi is not None and normal_nodes_cont is not None and normal_nodes_cat is None:
        pool1, prefix1 = normal_nodes_cont, 'cont'
        pool2, prefix2 = normal_nodes_bi, 'bi'
    elif normal_nodes_bi is not None and normal_nodes_cat is not None and normal_nodes_cont is None:
        pool1, prefix1 = normal_nodes_bi, 'bi'
        pool2, prefix2 = normal_nodes_cat, 'ord'
    elif normal_nodes_cont is not None and normal_nodes_cat is not None and normal_nodes_bi is None:
        pool1, prefix1 = normal_nodes_cont, 'cont'
        pool2, prefix2 = normal_nodes_cat, 'ord'
    elif normal_nodes_bi is not None and normal_nodes_cont is None and normal_nodes_cat is None:
        pool1, prefix1 = normal_nodes_bi, 'bi'
        pool2, prefix2 = normal_nodes_bi, 'bi'
    elif normal_nodes_cont is not None and normal_nodes_bi is None and normal_nodes_cat is None:
        pool1, prefix1 = normal_nodes_cont, 'cont'
        pool2, prefix2 = normal_nodes_cont, 'cont'
    elif normal_nodes_cat is not None and normal_nodes_bi is None and normal_nodes_cont is None:
        pool1, prefix1 = normal_nodes_cat, 'ord'
        pool2, prefix2 = normal_nodes_cat, 'ord'
    else:
        raise ValueError('At least one of normal_nodes_bi, normal_nodes_cont, or normal_nodes_cat must be provided.')

    # Decide, before drawing anything, whether this edge attempts a hub reuse and if so which of
    # the pair's two slots attempts it -- never both (see context_simulation._set_corr for the
    # positive-definiteness argument this relies on).
    magnitude, band = _draw_banded_effect(corr_bands, band_weights, high=0.95)
    reused_node = None
    if hub_reuse_prob > 0 and random.random() < hub_reuse_prob:
        reuse_slot = random.choice([1, 2])
        prefix = prefix1 if reuse_slot == 1 else prefix2
        candidates = [n for n in hub_eligible if n.startswith(prefix)]
        if candidates:
            candidate = random.choice(candidates)
            if node_sumsq.get(candidate, 0.0) + magnitude ** 2 < _HUB_SUMSQ_CEILING:
                reused_node = candidate
            # else: committing this edge would push the hub past the validity ceiling -- abandon
            # the reuse (no shrinking, no search for a different hub) and draw a fresh pair below.
        # else: no eligible node of this type has been placed yet -- draw a fresh pair below.

    if reused_node is not None and reuse_slot == 1:
        node1, node2 = reused_node, pool2.pop(0)
    elif reused_node is not None and reuse_slot == 2:
        node1, node2 = pool1.pop(0), reused_node
    elif pool1 is pool2:
        node1, node2 = pool1.pop(0), pool1.pop(0)
    else:
        node1, node2 = pool1.pop(0), pool2.pop(0)

    # Update per-node bookkeeping -- degree, sum(magnitude**2) and plain sum(magnitude) -- for both
    # endpoints regardless of whether this edge is a fresh pair or a hub reuse.
    for node, partner in ((node1, node2), (node2, node1)):
        if node not in node_degree:
            node_first_partner[node] = partner
        node_degree[node] = node_degree.get(node, 0) + 1
        node_sumsq[node] = node_sumsq.get(node, 0.0) + magnitude ** 2
        node_sum_magnitude[node] = node_sum_magnitude.get(node, 0.0) + magnitude

    if reused_node is not None:
        fresh_partner = node2 if reused_node == node1 else node1
        # The hub itself stays eligible (it may be reused again later, subject to the ceiling check
        # next time); its brand-new partner becomes a permanent leaf. The hub's *original* partner
        # is frozen too, but only the first time the hub is reused (degree just went 1 -> 2) --
        # on every later reuse it is already frozen, so this is a no-op.
        hub_eligible.discard(fresh_partner)
        if node_degree[reused_node] == 2:
            hub_eligible.discard(node_first_partner[reused_node])
    else:
        # An ordinary fresh pair: both endpoints remain eligible to become a hub later.
        hub_eligible.add(node1)
        hub_eligible.add(node2)

    idx1 = nodes.index(node1)
    idx2 = nodes.index(node2)

    # Choose direction of correlation change
    sign = random.choice([1, -1])
    which = random.choice([1, 2])

    if which == 1:
        corr_matrix1[idx1, idx2] = corr_matrix1[idx2, idx1] = magnitude * sign
    else:
        corr_matrix2[idx1, idx2] = corr_matrix2[idx2, idx1] = magnitude * sign

    return (node1, node2), magnitude, band, corr_matrix1, corr_matrix2, normal_nodes_bi, normal_nodes_cont, normal_nodes_cat
