from typing import Optional
import logging
import numpy as np
import random
import pandas as pd
import scipy as sc
import os


# Minimum number of samples expected in a discrete node's rarest level before its association
# estimates become noise-dominated. Below this, tightening the base-rate bounds does not help:
# with 36 samples per context, ~7% of binary ground truth edges are undetectable even at a
# perfectly balanced p=0.5, versus ~7.5% at binary_p_low=0.3.
_MIN_MINORITY_COUNT = 20

# Half-width of the uniform noise added to each individual shift/corr effect size. Injecting the
# exact same magnitude on every ground truth node/edge makes them artificially uniform; jittering
# each one independently around the requested value gives a more realistic spread of effect sizes
# while keeping the average at `shift`/`corr`.
_EFFECT_JITTER = 0.1

# Ceiling on a hub node's running sum(magnitude**2) across its correlation edges. A node that is
# reused as a "hub" -- correlated to several independent leaves that are never themselves reused --
# forms a star, and a star's correlation matrix is positive definite iff that sum stays below 1
# (the Schur-complement condition for a hub with pairwise-independent leaves). Staying a bit under
# the true ceiling avoids a matrix that is technically valid but singular in floating point.
_HUB_SUMSQ_CEILING = 0.95


def _jittered_effect(value, low=0.0, high=None, delta=_EFFECT_JITTER):
    """Draw an effect magnitude uniformly from [value - delta, value + delta], clipped to [low, high].

    `value` is treated as a magnitude (direction is applied separately via a random sign), so the
    result is floored at `low` by default to avoid accidentally flipping the intended direction.
    """
    jittered = np.random.uniform(value - delta, value + delta)
    if low is not None:
        jittered = max(jittered, low)
    if high is not None:
        jittered = min(jittered, high)
    return jittered


# Simulate mixed data using a gaussian copula 
def simulate_copula(path=None, name1='context1', name2='context2',
                    n_bi=50, n_cont=50, n_cat=50, n_samples_1=500, n_samples_2=500,
                    n_shift_cont=0, n_shift_bi=0, n_shift_cat=0,
                    n_corr_cont_cont=0, n_corr_bi_bi=0, n_corr_cat_cat=0, n_corr_bi_cont=0, n_corr_bi_cat=0, n_corr_cont_cat=0, 
                    n_both_cont_cont=0, n_both_bi_bi=0, n_both_cat_cat=0, n_both_bi_cont=0, n_both_bi_cat=0, n_both_cont_cat=0,
                    shift=0.5, corr=0.7, hub_reuse_prob=0.0,
                    binary_p_low=0.3, binary_p_high=0.7, ordinal_concentration=20.0):
    """
    Simulate two contexts with binary and continuous nodes using a Gaussian copula.
    
    :param path: Path to save the simulated contexts, the meta file and the ground truth information. If None, files are not saved.
    :param name1: Name of the first context.
    :param name2: Name of the second context.
    :param n_bi: Number of binary nodes to simulate.
    :param n_cont: Number of continuous nodes to simulate.
    :param n_cat: Number of categorical nodes to simulate.
    :param n_samples_1: Number of samples for the first context.
    :param n_samples_2: Number of samples for the second context.
    :param n_shift_cont: Number of continuous nodes with an artificially introduced mean shift.
    :param n_shift_bi: Number of binary nodes with an artificially introduced mean shift.
    :param n_shift_cat: Number of categorical nodes with an artificially introduced mean shift.
    :param n_corr_cont_cont: Number of continuous node pairs with an artifically introduced correlation difference.
    :param n_corr_bi_bi: Number of binary node pairs with an artificially introduced correlation difference.
    :param n_corr_cat_cat: Number of categorical node pairs with an artificially introduced correlation difference.
    :param n_corr_bi_cat: Number of binary-categorical node pairs with an artificially introduced correlation difference.
    :param n_corr_cont_cat: Number of continuous-categorical node pairs with an artificially introduced correlation difference.
    :param n_corr_bi_cont: Number of mixed node pairs with an artificially introduced correlation difference.
    :param n_both_cont_cont: Number of continuous node pairs with both an aritificially introduced mean shift and correlation difference.
    :param n_both_bi_bi: Number of binary node pairs with both an artificially introduced mean shift and correlation difference.
    :param n_both_cat_cat: Number of categorical node pairs with both an artificially introduced mean shift and correlation difference.
    :param n_both_bi_cat: Number of binary-categorical node pairs with both an artificially introduced mean shift and correlation difference.
    :param n_both_cont_cat: Number of continuous-categorical node pairs with both an artificially introduced mean shift and correlation difference.
    :param n_both_bi_cont: Number of mixed node pairs with both an artificially introduced mean shift and correlation difference.
    :param shift: Target magnitude of the mean shift. Each shifted node draws its actual magnitude
                 uniformly from [shift - 0.1, shift + 0.1] (floored at 0) so ground truth effects
                 vary in strength rather than all being identical.
    :param corr: Target magnitude of the correlation difference (measured as correlation coefficient
                 between 0 and 1). Each correlated pair draws its actual magnitude uniformly from
                 [corr - 0.1, corr + 0.1] (clipped to [0, 0.95]) so ground truth effects vary in
                 strength rather than all being identical.
    :param hub_reuse_prob: Probability that a new correlation/both pair reuses an already-placed
                           node as one endpoint ("hub") instead of drawing two fresh nodes. Real
                           differential networks concentrate rewiring on a few hub nodes rather than
                           a perfect matching. When it fires, exactly one of the pair's two slots is
                           filled by a node that already has at least one edge; the other slot is
                           always a fresh node, and that fresh node -- plus the hub's original first
                           partner, the first time the hub is reused -- is permanently barred from
                           ever being reused itself. This keeps every hub isolated from every other
                           hub, which is what makes the one cheap check below exact: a reuse is only
                           committed if the hub's running sum of its edges' magnitude**2 stays below
                           `_HUB_SUMSQ_CEILING` (0.95); otherwise it is abandoned and a fresh pair is
                           drawn instead, silently, with no shrinking or retry. Default 0.0
                           reproduces the historical one-edge-per-node behaviour exactly.
    :param binary_p_low: Lower bound of the base rate drawn per binary node. How strongly two binary
                         variables can correlate is capped by how balanced they are, so a rare node
                         attenuates the correlation difference actually realised on its edges. Widen
                         this range for more marginal variety, at the cost of less comparable effect
                         sizes across edges. Use 0.0/1.0 for the historical behaviour.
    :param binary_p_high: Upper bound of the base rate drawn per binary node.
    :param ordinal_concentration: Symmetric Dirichlet concentration for the category probabilities of
                                  each ordinal node. Large values keep the categories near-balanced;
                                  1.0 (the historical behaviour) is uniform over all splits and
                                  routinely leaves categories nearly empty.
    :return: A tuple containing the two simulated contexts, a meta file, a list of ground truth nodes
             and a dict of per-edge/per-node injected effect sizes.
             - context1: pd.DataFrame of the first simulated context.
             - context2: pd.DataFrame of the second simulated context.
             - meta: pd.DataFrame containing the data type for each simulated variable.
             - ground_truth: A tuple containing three lists of ground truth nodes: (shift_nodes, corr_nodes, shift_corr_nodes).
             - effects: dict with 'edge_magnitude' (dict keyed by frozenset({node1, node2}) -> the
               jittered correlation magnitude actually injected on that edge), 'node_degree' (dict
               node -> number of correlation edges touching it) and 'node_sum_magnitude' (dict node
               -> sum of jittered magnitudes over those edges). Feeds the extra columns written to
               `ground_truth_edges.txt`/`ground_truth_nodes.txt` when `path` is given, and lets a
               caller that writes those files itself (as the Nextflow CLI wrapper does) do the same.
    """
    if n_bi <= 0 and n_cont <= 0 and n_cat <= 0:
        raise ValueError('Either n_bi, n_cont, or n_cat needs to be larger than zero.')
    if not 0.0 <= binary_p_low <= binary_p_high <= 1.0:
        raise ValueError(f'binary_p_low and binary_p_high must satisfy 0 <= low <= high <= 1, '
                         f'got low={binary_p_low}, high={binary_p_high}.')
    if ordinal_concentration <= 0:
        raise ValueError(f'ordinal_concentration must be larger than zero, got {ordinal_concentration}.')

    # Bounding the base rates caps how much of the injected correlation the discretisation throws
    # away, but it cannot rescue a sample size too small to estimate the association at all. Warn
    # when that regime is entered, since raising the bounds is then not the fix.
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

    # Bookkeeping shared across every _set_corr call below, so a node can be recognised as an
    # already-placed "hub" candidate regardless of which pair-type loop first drew it. See
    # `_set_corr` for how these are used and updated; `edge_magnitude`/`node_degree`/
    # `node_sum_magnitude` also feed the extra ground-truth columns written further down.
    node_degree = {}
    node_sumsq = {}
    node_sum_magnitude = {}
    node_first_partner = {}
    hub_eligible = set()
    edge_magnitude = {}

    def _inject(**pools):
        node_pair, magnitude, c1, c2, *_ = _set_corr(
            nodes=nodes, corr_param=corr, corr_matrix1=corr1, corr_matrix2=corr2,
            node_degree=node_degree, node_sumsq=node_sumsq, node_sum_magnitude=node_sum_magnitude,
            node_first_partner=node_first_partner, hub_eligible=hub_eligible,
            hub_reuse_prob=hub_reuse_prob, **pools)
        edge_magnitude[frozenset(node_pair)] = magnitude
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
    for node in nodes:
        if node in shift_nodes or any(node in pair for pair in shift_corr_nodes):
            sign = random.choice([1, -1])
            context_idx = random.choice([1, 2])
            magnitude = _jittered_effect(shift)
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
            # Binary node. The base rate is drawn from [binary_p_low, binary_p_high] rather than
            # from the full unit interval: the maximum attainable phi between two binary variables
            # is bounded by their marginals, so a node that is positive in only a few percent of
            # samples attenuates the correlation difference its edges actually carry, even though
            # the ground truth records the same `corr` for every edge.
            p = np.random.uniform(binary_p_low, binary_p_high)
            context1_bi[node] = sc.stats.bernoulli.ppf(u1[i, :], p=p).astype(int)
            context2_bi[node] = sc.stats.bernoulli.ppf(u2[i, :], p=p).astype(int)

        else:
            # Categorical node
            #n_categories = np.random.randint(3, 10) # Randomly choose number of categories between 3 and 10
            n_categories = 5 # fixed number of categories
            # Symmetric Dirichlet: ordinal_concentration=1 is uniform over every possible split and
            # routinely leaves categories nearly empty, which attenuates the realised association the
            # same way an extreme binary base rate does. Larger values keep the categories balanced.
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
    # a ground-truth node at all -- a mean-shift-only node has no entry in node_degree/
    # node_sum_magnitude, so it correctly falls back to (0, 0.0) via save_gt's own .get() default.
    gt_nodes = set(shift_nodes)
    for pair in corr_nodes + shift_corr_nodes:
        gt_nodes.update(pair)
    node_stats = {node: (node_degree.get(node, 0), node_sum_magnitude.get(node, 0.0)) for node in gt_nodes}
    effects = {'edge_magnitude': edge_magnitude, 'node_degree': node_degree, 'node_sum_magnitude': node_sum_magnitude}

    # Save simulated contexts and ground truth nodes
    if path:
        context1.to_csv(os.path.join(path, f'{name1}.csv'))
        context2.to_csv(os.path.join(path, f'{name2}.csv'))
        meta.to_csv(os.path.join(path, 'meta.csv'), index=False)
        save_gt((shift_nodes, corr_nodes, shift_corr_nodes), os.path.join(path, 'ground_truth_nodes.txt'), mode='node', node_stats=node_stats)
        save_gt((shift_nodes, corr_nodes, shift_corr_nodes), os.path.join(path, 'ground_truth_edges.txt'), mode='edge', edge_magnitude=edge_magnitude)

    return context1, context2, meta, (shift_nodes, corr_nodes, shift_corr_nodes), effects


# Helper function to set correlation in copula-based simulation
def _set_corr(nodes, corr_param, corr_matrix1, corr_matrix2, normal_nodes_bi=None, normal_nodes_cont=None, normal_nodes_cat=None,
              node_degree=None, node_sumsq=None, node_sum_magnitude=None, node_first_partner=None, hub_eligible=None,
              hub_reuse_prob=0.0):
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
    # the pair's two slots attempts it -- never both. That restriction is what keeps a hub's own
    # sum(magnitude**2) check below exact rather than approximate: it guarantees a hub's edges only
    # ever land on leaves that never grow a second edge of their own, so every hub is isolated from
    # every other hub (see the module docstring / Simulation_TODO.md Tier 3.1 discussion -- a plain
    # chain of ordinary-looking edges can be invalid even when no single node looks overloaded).
    magnitude = _jittered_effect(corr_param, high=0.95)
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
    # endpoints regardless of whether this edge is a fresh pair or a hub reuse. This is what the
    # ground-truth `n_edges_tweaked`/`sum_jittered_magnitude` columns are built from, and it is also
    # what the next call's hub-eligibility decision reads.
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

    return (node1, node2), magnitude, corr_matrix1, corr_matrix2, normal_nodes_bi, normal_nodes_cont, normal_nodes_cat


# Adapted from pycop package
def _simu_gaussian(n: int, m: int, corr_matrix: np.ndarray, mean_vector: Optional[np.ndarray]=None):
    """ 
    # Gaussian Copula simulations with a given correlation matrix

    :param n: number of simulated variables
    :param m: sample size
    :param corr_matrix: correlation matrix
    :return: simulated samples from a Gaussian copula

    """

    if not all(isinstance(v, int) for v in [n, m]):
        raise TypeError("The 'n' and 'm' arguments must both be integer types.")
    if not isinstance(corr_matrix, np.ndarray):
        raise TypeError("The 'corr_matrix' argument must be a numpy array.")
    if not isinstance(mean_vector, np.ndarray):
        mean_vector = np.zeros(n)

    # Generate n independent standard Gaussian random variables V = (v1 ,..., vn):
    v = [np.random.normal(0, 1, m) for i in range(0, n)]

    # Compute the lower triangular Cholesky factorization of the correlation matrix:
    l = sc.linalg.cholesky(corr_matrix, lower=True)

    # The mean shift is added AFTER mixing, so E[y] = mean_vector exactly. Cov(y) = L L^T =
    # corr_matrix either way, so the correlation structure is unaffected by this choice.
    # Shifting v instead (E[y] = L @ mean_vector) pushes each node's shift through the same
    # matrix that manufactures the correlations, which shrinks the node's own shift by L_ii
    # and leaks it into every correlated partner.
    y = np.dot(l, v) + mean_vector[:, np.newaxis]
    u = sc.stats.norm.cdf(y, 0, 1)

    return u


# Save ground truth nodes to file
def save_gt(groundtruths, path, mode='node', edge_magnitude=None, node_stats=None):
    """
    Write ground truth nodes or edges to a two- (or, with the optional data below, wider-) column file.

    :param groundtruths: (shift_nodes, corr_nodes, shift_corr_nodes) as returned by `simulate_copula`.
    :param path: File to write.
    :param mode: 'node' writes one row per (node, description); 'edge' writes one row per
                (edge, description), edge = '_'.join(sorted(pair)).
    :param edge_magnitude: Optional, mode='edge' only. Dict keyed by frozenset({node1, node2}) ->
                           the jittered correlation magnitude injected on that edge. When given, an
                           extra `jittered_magnitude` column is appended. Omit to reproduce the
                           historical two-column file exactly (used unchanged by
                           `context_simulation_advanced.save_gt_advanced`).
    :param node_stats: Optional, mode='node' only. Dict node -> (n_edges_tweaked,
                       sum_jittered_magnitude). When given, two extra columns are appended; a node
                       absent from the dict (e.g. mean-shift-only) is written as (0, 0.0). Omit to
                       reproduce the historical two-column file exactly.
    """
    shift = groundtruths[0]
    corr = groundtruths[1]
    shift_corr = groundtruths[2]

    if mode == 'node':
        with open(path, 'w') as f:
            header = 'node, description'
            if node_stats is not None:
                header += ', n_edges_tweaked, sum_jittered_magnitude'
            f.write(header + '\n')

            def _write_node(node, description):
                line = f'{node}, {description}'
                if node_stats is not None:
                    n_edges, sum_magnitude = node_stats.get(node, (0, 0.0))
                    line += f', {n_edges}, {sum_magnitude}'
                f.write(line + '\n')

            for node in shift:
                _write_node(node, 'mean shift')
            for pair in corr:
                _write_node(pair[0], 'diff. corr.')
                _write_node(pair[1], 'diff. corr.')
            for pair in shift_corr:
                _write_node(pair[0], 'mean shift + diff. corr.')
                _write_node(pair[1], 'mean shift + diff. corr.')

    if mode == 'edge':
        with open(path, 'w') as f:
            header = 'edge, description'
            if edge_magnitude is not None:
                header += ', jittered_magnitude'
            f.write(header + '\n')

            def _write_edge(pair, description):
                edge = '_'.join(sorted(pair))
                line = f'{edge}, {description}'
                if edge_magnitude is not None:
                    line += f', {edge_magnitude.get(frozenset(pair))}'
                f.write(line + '\n')

            for pair in corr:
                _write_edge(pair, 'diff. corr.')
            for pair in shift_corr:
                _write_edge(pair, 'mean shift + diff. corr.')

