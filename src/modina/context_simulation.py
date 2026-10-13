from collections import Counter
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
# perfectly balanced p=0.5, versus ~7.5% at _BINARY_P_LOW=0.3.
_MIN_MINORITY_COUNT = 20

# Marginal shapes. These are fixed rather than exposed because they are not a factor the benchmark
# varies: they only exist to keep the discretisation from eating the planted effects. The maximum
# attainable association between two discretised variables is capped by their marginals, so a binary
# base rate far from 0.5, or an ordinal category left nearly empty, attenuates every edge touching
# that node regardless of the magnitude recorded in the ground truth. Binary base rates are drawn
# uniformly from [_BINARY_P_LOW, _BINARY_P_HIGH]; ordinal category probabilities from a symmetric
# Dirichlet with concentration _ORDINAL_CONCENTRATION (1.0 would be uniform over all splits and
# routinely leaves categories nearly empty; 20 keeps them near-balanced).
_BINARY_P_LOW = 0.3
_BINARY_P_HIGH = 0.7
_ORDINAL_CONCENTRATION = 20.0
_N_ORDINAL_CATEGORIES = 5

# Hard clip applied when a planted magnitude is added on top of a background value. Only reachable
# with a strong background and a large `corr_high` simultaneously; it exists so the result is never
# an out-of-range correlation, but hitting it means the requested effect was not delivered in full,
# so it is warned about rather than passed over silently.
_MAX_ABS_CORR = 0.99


# Magnitude written into every off-diagonal cell of the background precision matrix Omega. Its
# absolute value is arbitrary: `background_strength` is reached by solving for the diagonal
# loading, which can compensate for any choice here, so exposing both would be two dials for one
# effect. Only the random +/- sign per edge matters structurally.
_BACKGROUND_OMEGA_WEIGHT = 0.5

# Bracket and iteration count for the bisection that turns `background_strength` (a target on the
# OUTPUT correlation matrix) into a diagonal loading (an input to Omega). Strength is monotonically
# decreasing in the loading, so bisection is exact to ~1e-14 after this many halvings.
_BACKGROUND_LOAD_BRACKET = (1e-4, 1e3)
_BACKGROUND_LOAD_ITERS = 60


def _background_edges(n_vars, density, rng):
    """Draw the sparse scale-free wiring that the background is built from.

    Returns a list of (i, j) index pairs with i < j. `density` is the number of connections each
    newly added node makes.
    """
    if density < 1:
        raise ValueError(f'background_density must be at least 1, got {density}.')
    if density >= n_vars:
        raise ValueError(f'background_density ({density}) must be smaller than the number of '
                         f'variables ({n_vars}).')

    # Barabasi-Albert preferential attachment: each new node attaches to `density` existing
    # nodes chosen with probability proportional to their current degree. The rich-get-richer
    # feedback is what produces hubs -- they emerge from the growth rule, not from a parameter.
    edges = [(i, j) for i in range(density) for j in range(i + 1, density)]
    repeated = [end for edge in edges for end in edge] or list(range(density))
    for src in range(density, n_vars):
        targets = []
        while len(targets) < density:
            candidate = int(rng.choice(repeated))
            if candidate not in targets:
                targets.append(candidate)
        edges.extend((min(src, t), max(src, t)) for t in targets)
        repeated.extend(targets + [src] * density)
    return edges


def _background_sigma(n_vars, edges, signs, load):
    """Assemble Omega at the given diagonal loading and return the implied correlation matrix.

    The diagonal is set to each row's absolute off-diagonal sum plus `load`, which makes Omega
    strictly diagonally dominant. By Gershgorin's circle theorem every eigenvalue then lies in
    [load, 2*rowsum + load], so Omega is positive definite by construction for any load > 0 -- no
    check, no repair. Inverting a positive definite matrix yields a positive definite one, so
    Sigma_base is valid for free. That guarantee covers the BACKGROUND only; it says nothing about
    what happens once differential edges are written into Sigma afterwards.
    """
    omega = np.zeros((n_vars, n_vars))
    for (a, b), sign in zip(edges, signs):
        omega[a, b] = omega[b, a] = _BACKGROUND_OMEGA_WEIGHT * sign
    np.fill_diagonal(omega, np.abs(omega).sum(axis=1) + load)
    cov = np.linalg.inv(omega)
    scale = np.sqrt(np.diag(cov))
    return cov / np.outer(scale, scale)


def _build_background(n_vars, strength, density, seed):
    """Build a dense background correlation matrix from a sparse scale-free network.

    Returns (sigma_base, edges, lambda_min).

    `strength` is a target on the OUTPUT (mean |r| over directly-wired pairs), not an input, because
    the entries of Omega have no interpretable scale on their own: what a given Omega entry produces
    in Sigma depends on the density of the network and on the degree of the two nodes it joins. The
    diagonal loading that hits the target is found by bisection, which is valid because strength is
    strictly decreasing in the loading.
    """
    if not 0.0 < strength < 1.0:
        raise ValueError(f'background_strength must lie strictly between 0 and 1, got {strength}.')

    # A dedicated Generator, never the global `random`/`np.random` streams the rest of the
    # simulation draws from. Changing the background parameters therefore does not shift any
    # downstream draw: the same seed plants the same edges with the same magnitudes either way.
    rng = np.random.default_rng(seed)
    edges = _background_edges(n_vars, density, rng)
    signs = rng.choice([-1.0, 1.0], size=len(edges))
    # Preferential attachment makes the earliest-added nodes the hubs, but node index is not
    # neutral: it runs continuous, then binary, then ordinal, and the first nodes of each type are
    # the ones planted with ground truth. Relabel the grown network at random so that being a
    # background hub is independent of both type and ground truth status.
    perm = rng.permutation(n_vars)
    edges = [(int(min(perm[a], perm[b])), int(max(perm[a], perm[b]))) for a, b in edges]

    def wired_mean(load):
        sigma = _background_sigma(n_vars, edges, signs, load)
        return float(np.mean([abs(sigma[a, b]) for a, b in edges]))

    # Strength is bounded above by the wiring: denser wiring forces a larger diagonal, which
    # dilutes every correlation. Requesting more than the ceiling is unsatisfiable at any loading,
    # and a plain bisection would silently return the bracket edge instead of saying so.
    low, high = _BACKGROUND_LOAD_BRACKET
    ceiling = wired_mean(low)
    if strength >= ceiling:
        raise ValueError(
            f'background_strength={strength} is unreachable at background_density={density} with '
            f'{n_vars} variables: the highest attainable mean |r| on wired pairs is {ceiling:.3f}. '
            f'Lower background_strength, or lower background_density (a sparser network sustains '
            f'stronger correlations).')

    for _ in range(_BACKGROUND_LOAD_ITERS):
        mid = (low + high) / 2
        if wired_mean(mid) > strength:
            low = mid
        else:
            high = mid
    sigma_base = _background_sigma(n_vars, edges, signs, mid)
    return sigma_base, edges, float(np.linalg.eigvalsh(sigma_base).min())


def _plan_hub_reuse(edge_types, hub_reuse_prob):
    """Decide up front which planted edges reuse an existing node as a hub, and of which type.

    `edge_types` lists the (type, type) of every planted edge in planting order. Returns a list of
    the same length holding the node-type prefix whose existing node the edge should attach to, or
    None for an ordinary fresh pair.

    Every reuse collapses one would-be fresh endpoint onto an existing node, so it removes one
    ground truth node of the hub's type. Deciding per edge by coin flip makes that count random,
    and with it the number of ground truth nodes per type. Instead, each pair type reuses exactly
    round(hub_reuse_prob * n_edges) of its edges, at random positions, and a cross-type reuse goes
    to whichever of its two types has collapsed the smaller share of its endpoints so far. Every
    type therefore loses the same share of its endpoints to hubs; with equal counts per type, the
    ground truth node counts per type agree to within one.
    """
    endpoints = Counter(t for pair in edge_types for t in pair)
    reuses = Counter()
    plan = [None] * len(edge_types)
    positions = {}
    for i, pair in enumerate(edge_types):
        positions.setdefault(pair, []).append(i)
    # Same-type pairs first: their hub type is forced, so the cross types can balance around them.
    for pair in sorted(positions, key=lambda pair: pair[0] != pair[1]):
        idx = positions[pair]
        for i in sorted(random.sample(idx, int(hub_reuse_prob * len(idx) + 0.5))):
            a, b = pair
            share_a, share_b = reuses[a] / endpoints[a], reuses[b] / endpoints[b]
            hub_type = a if share_a < share_b or (share_a == share_b and random.random() < 0.5) else b
            plan[i] = hub_type
            reuses[hub_type] += 1
    return plan


# Simulate mixed data using a gaussian copula 
def simulate_copula(path=None, name1='context1', name2='context2',
                    n_bi=50, n_cont=50, n_cat=50, n_samples_1=500, n_samples_2=500,
                    n_shift_cont=0, n_shift_bi=0, n_shift_cat=0,
                    n_corr_cont_cont=0, n_corr_bi_bi=0, n_corr_cat_cat=0, n_corr_bi_cont=0, n_corr_bi_cat=0, n_corr_cont_cat=0, 
                    n_both_cont_cont=0, n_both_bi_bi=0, n_both_cat_cat=0, n_both_bi_cont=0, n_both_bi_cat=0, n_both_cont_cat=0,
                    *, corr_low, corr_high, shift_low, shift_high,
                    hub_reuse_prob=0.0,
                    background_strength=0.2, background_density=2, background_seed=None):
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
    :param corr_low: Lower bound of the correlation-difference magnitude range. Each correlated pair
                     independently draws its actual magnitude uniformly from
                     [corr_low, corr_high], so ground truth effects span a continuum -- rankable by
                     realized magnitude -- rather than clustering at one fixed value. Required.
    :param corr_high: Upper bound of the correlation-difference magnitude range. Must satisfy
                      `corr_low <= corr_high <= 0.95` (0.95 is the historical ceiling on a single
                      edge's magnitude). Required.
    :param shift_low: Lower bound of the mean-shift magnitude range. Each shifted node independently
                      draws its actual magnitude uniformly from [shift_low, shift_high], for the same
                      reason as `corr_low`/`corr_high`. Required.
    :param shift_high: Upper bound of the mean-shift magnitude range. Must satisfy
                       `shift_low <= shift_high`. Required.
    :param hub_reuse_prob: Fraction of each pair type's correlation/both pairs that reuse an
                           already-placed node as one endpoint ("hub") instead of drawing two fresh
                           nodes. Real differential networks concentrate rewiring on a few hub nodes
                           rather than a perfect matching. The count per pair type is fixed at
                           round(hub_reuse_prob * n_pairs) rather than drawn per edge, and cross-type
                           reuses are spread so every node type loses the same share of its endpoints
                           to hubs (see `_plan_hub_reuse`); this keeps the number of ground truth
                           nodes per type consistent across replicates and types. A reused hub is
                           picked at random among the eligible nodes of its type whose running
                           sum(magnitude**2) stays below the background's lambda_min**2. A reuse
                           only ever fills one slot of a pair; the other is always a fresh node, and
                           that fresh node -- plus the hub's original first partner, the first time
                           the hub is reused -- is permanently barred from ever being reused itself.
                           This keeps every hub isolated from every other hub, which is what makes
                           the sum(magnitude**2) check exact. If no eligible hub exists yet, the
                           reuse is deferred to the next fresh pair of the same pair type. Default
                           0.0 gives one edge per node.
    The background correlation structure is always a sparse scale-free network (preferential
    attachment, producing a few high-degree hubs and a heavy-tailed degree distribution), turned
    into a dense correlation matrix. It is built ONCE and shared by both contexts, so every pair
    that is not planted has bit-identical correlation in the two contexts and the negatives stay
    exactly null.
    :param background_strength: Target mean |r| over the directly-wired pairs of the background.
                                This is a property of the resulting correlation matrix, not a value
                                written anywhere: the diagonal loading that attains it is solved for
                                by bisection. It is bounded above by the topology (denser wiring
                                forces weaker correlations); an unreachable request raises with the
                                attainable ceiling.
    :param background_density: Number of connections each node makes when it is added to the
                               network. Sets both the edge count and how large the hubs grow. Higher density lowers the reachable strength.
    :param background_seed: Seed for the background's own random generator. Kept separate from the
                            global streams the rest of the simulation draws from, so changing the
                            background does not change which pairs get planted or with what
                            magnitudes.
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
               Also carries 'node_shift_magnitude' (dict node -> its drawn mean-shift magnitude, 0.0
               for nodes with no mean-shift draw of their own), 'corr_base' (the shared background
               correlation matrix), 'background_edges' (the directly-wired pairs, as node-name
               frozensets) and 'background_lambda_min' (the perturbation budget the planting step was
               checked against).
    """
    if n_bi <= 0 and n_cont <= 0 and n_cat <= 0:
        raise ValueError('Either n_bi, n_cont, or n_cat needs to be larger than zero.')
    if not 0.0 <= hub_reuse_prob <= 1.0:
        raise ValueError(f'hub_reuse_prob must lie between 0 and 1, got {hub_reuse_prob}.')
    if not 0.0 <= corr_low <= corr_high <= 0.95:
        raise ValueError(f'corr_low and corr_high must satisfy 0 <= corr_low <= corr_high <= 0.95, '
                         f'got corr_low={corr_low}, corr_high={corr_high}.')
    if not 0.0 <= shift_low <= shift_high:
        raise ValueError(f'shift_low and shift_high must satisfy 0 <= shift_low <= shift_high, '
                         f'got shift_low={shift_low}, shift_high={shift_high}.')

    # Bounding the base rates caps how much of the injected correlation the discretisation throws
    # away, but it cannot rescue a sample size too small to estimate the association at all. Warn
    # when that regime is entered, since raising the bounds is then not the fix.
    n_min_samples = min(n_samples_1, n_samples_2)
    n_bi_effects = (n_shift_bi + n_corr_bi_bi + n_both_bi_bi + n_corr_bi_cont + n_both_bi_cont
                    + n_corr_bi_cat + n_both_bi_cat)
    if n_bi_effects > 0 and _BINARY_P_LOW * n_min_samples < _MIN_MINORITY_COUNT:
        logging.warning(
            f'A binary base rate as low as {_BINARY_P_LOW} with {n_min_samples} samples in the smaller '
            f'context leaves only about {_BINARY_P_LOW * n_min_samples:.0f} samples in a binary node\'s '
            f'rarer level, below the ~{_MIN_MINORITY_COUNT} needed for a stable association estimate. '
            f'Binary ground truth effects will be noise-limited at this sample size. Increase '
            f'n_samples_1/n_samples_2.')

    n_cat_effects = (n_shift_cat + n_corr_cat_cat + n_both_cat_cat + n_corr_bi_cat + n_both_bi_cat
                     + n_corr_cont_cat + n_both_cont_cat)
    if n_cat_effects > 0 and n_min_samples / _N_ORDINAL_CATEGORIES < _MIN_MINORITY_COUNT:
        logging.warning(
            f'{n_min_samples} samples in the smaller context spread over {_N_ORDINAL_CATEGORIES} ordinal '
            f'categories leaves about {n_min_samples / _N_ORDINAL_CATEGORIES:.0f} samples per category '
            f'even when perfectly balanced, below the ~{_MIN_MINORITY_COUNT} needed for a stable '
            f'association estimate. Ordinal ground truth effects will be noise-limited at this sample '
            f'size.')
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

    # The background is built once and copied into both contexts, rather than drawn twice. That is
    # what keeps the negatives clean: an unplanted pair holds bit-identical values in corr1 and
    # corr2, so its true between-context difference is exactly zero rather than merely small.
    sigma_base, background_pairs, background_lambda_min = _build_background(
        n_vars, background_strength, background_density, background_seed)
    corr1 = sigma_base.copy()
    corr2 = sigma_base.copy()

    # Budget for the planting step. Every differential edge nudges one cell of Sigma; Weyl's
    # inequality bounds the damage by the spectral norm of the whole perturbation, and because
    # hub_reuse_prob only ever produces disjoint stars that norm is exactly
    # sqrt(max over nodes of sum(magnitude**2)) -- the quantity `node_sumsq` already tracks. Keeping
    # every node's sum(magnitude**2) below lambda_min**2 is therefore a sufficient condition for
    # both context matrices to stay positive definite.
    #
    # Note this ceiling is consulted ONLY when deciding whether a node may take on another edge, so
    # it constrains how much a hub accumulates -- it does not bound the very first edge on a node.
    # A single edge larger than lambda_min can break validity on its own, which is what the check
    # just below warns about and the eigenvalue test before sampling catches for certain.
    hub_sumsq_ceiling = background_lambda_min ** 2

    # The largest magnitude any single edge can draw. If even one edge exceeds the budget, no hub
    # ceiling can help -- warn here, where the fix (lower `corr_high`, or weaken the background) is
    # obvious, rather than leaving only the eigenvalue failure several steps later. Warned rather
    # than raised because Weyl is a worst case: a perturbation rarely points along the narrowest
    # direction, so configurations moderately over this line usually still produce a valid matrix.
    if corr_high >= background_lambda_min:
        logging.warning(
            f'corr_high={corr_high} can draw magnitudes up to {corr_high:.3f}, which exceeds the '
            f'budget left by the background (lambda_min={background_lambda_min:.4f}). The '
            f'correlation matrices may not be positive definite. Lower `corr_high`, or lower '
            f'`background_strength` to leave more budget.')

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

    pools = {'cont': normal_nodes_cont, 'bi': normal_nodes_bi, 'ord': normal_nodes_cat}
    pool_args = {'cont': 'normal_nodes_cont', 'bi': 'normal_nodes_bi', 'ord': 'normal_nodes_cat'}

    # Every differential edge in planting order: all correlation-only pairs, then all
    # correlation-plus-shift pairs, each in the pair-type order below.
    pair_types = [('cont', 'cont'), ('bi', 'bi'), ('ord', 'ord'), ('cont', 'bi'), ('bi', 'ord'), ('cont', 'ord')]
    n_corr = (n_corr_cont_cont, n_corr_bi_bi, n_corr_cat_cat, n_corr_bi_cont, n_corr_bi_cat, n_corr_cont_cat)
    n_both = (n_both_cont_cont, n_both_bi_bi, n_both_cat_cat, n_both_bi_cont, n_both_bi_cat, n_both_cont_cat)
    plan = ([(pair, corr_nodes) for pair, n in zip(pair_types, n_corr) for _ in range(n)]
            + [(pair, shift_corr_nodes) for pair, n in zip(pair_types, n_both) for _ in range(n)])
    reuse_plan = _plan_hub_reuse([pair for pair, _ in plan], hub_reuse_prob)

    # A planned reuse can only fail when no node of the hub type is eligible yet (e.g. the very
    # first pair of its type) or none has budget left; it is then carried to the next fresh pair
    # of the same pair type, so the planned count is still met whenever possible.
    deferred = {}
    for (pair, ground_truth), reuse_type in zip(plan, reuse_plan):
        if reuse_type is None and deferred.get(pair):
            reuse_type = deferred[pair].pop()
        node_pair, magnitude, reused = _set_corr(
            nodes=nodes, corr_low=corr_low, corr_high=corr_high, corr_matrix1=corr1, corr_matrix2=corr2,
            node_degree=node_degree, node_sumsq=node_sumsq, node_sum_magnitude=node_sum_magnitude,
            node_first_partner=node_first_partner, hub_eligible=hub_eligible,
            reuse_type=reuse_type, hub_sumsq_ceiling=hub_sumsq_ceiling,
            **{pool_args[t]: pools[t] for t in pair})
        if reuse_type is not None and not reused:
            deferred.setdefault(pair, []).append(reuse_type)
        edge_magnitude[frozenset(node_pair)] = magnitude
        ground_truth.append(node_pair)

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
    for node in nodes:
        if node in shift_nodes or any(node in pair for pair in shift_corr_nodes):
            sign = random.choice([1, -1])
            context_idx = random.choice([1, 2])
            magnitude = np.random.uniform(shift_low, shift_high)
            node_shift_magnitude[node] = magnitude
            if context_idx == 1:
                mean_vector1[nodes.index(node)] = sign * magnitude
            else:
                mean_vector2[nodes.index(node)] = sign * magnitude

    # The background construction guarantees Sigma_base is valid, but that guarantee does not
    # survive writing differential edges into it. The hub ceiling above is a sufficient condition
    # and should already have prevented any violation; this is the exact test, and it is cheap
    # (a few ms even at several hundred variables) relative to the sampling that follows. It fails
    # loudly rather than repairing, because the usual repair (projecting to the nearest valid
    # matrix) perturbs cells that were never planted and would silently corrupt the ground truth.
    for label, matrix in ((name1, corr1), (name2, corr2)):
        smallest = float(np.linalg.eigvalsh(matrix).min())
        if smallest <= 0:
            raise ValueError(
                f'The correlation matrix for {label} is not positive definite (smallest eigenvalue '
                f'{smallest:.4g}) after planting the differential edges, so no data can have this '
                f'structure. The background left a budget of lambda_min='
                f'{background_lambda_min:.4f}; a node carrying degree d edges of size m spends '
                f'sqrt(d)*m of it. Reduce `corr_high`, reduce `hub_reuse_prob` so fewer edges pile '
                f'onto one node, or lower `background_strength` to free up budget.')

    # Gaussian copula
    u1 = _simu_gaussian(n=n_vars, m=n_samples_1, corr_matrix=corr1, mean_vector=mean_vector1)
    u2 = _simu_gaussian(n=n_vars, m=n_samples_2, corr_matrix=corr2, mean_vector=mean_vector2)

    # Transform to marginal distributions using the inverse CDF
    marginals = {}
    for i, node in enumerate(nodes):
        if i < n_cont:
            # Continuous node
            mean = 0.0
            std = 0.5
            context1_cont[node] = sc.stats.norm.ppf(u1[i, :], loc=mean, scale=std)
            context2_cont[node] = sc.stats.norm.ppf(u2[i, :], loc=mean, scale=std)
            marginals[node] = {'kind': 'continuous', 'loc': mean, 'scale': std}

        elif n_cont <= i < n_cont + n_bi:
            # Binary node. The base rate is bounded away from 0 and 1 (see _BINARY_P_LOW).
            p = np.random.uniform(_BINARY_P_LOW, _BINARY_P_HIGH)
            context1_bi[node] = sc.stats.bernoulli.ppf(u1[i, :], p=p).astype(int)
            context2_bi[node] = sc.stats.bernoulli.ppf(u2[i, :], p=p).astype(int)
            marginals[node] = {'kind': 'binary', 'p': float(p)}

        else:
            # Categorical node. Near-balanced categories (see _ORDINAL_CONCENTRATION).
            p = np.random.dirichlet(np.full(_N_ORDINAL_CATEGORIES, _ORDINAL_CONCENTRATION), size=1).flatten()
            cdf = np.cumsum(p)
            context1_cat[node] = np.searchsorted(cdf, u1[i, :])
            context2_cat[node] = np.searchsorted(cdf, u2[i, :])
            marginals[node] = {'kind': 'ordinal', 'cdf': cdf.tolist()}

    # Combine continuous and binary data
    context1 = context1_cont.join(context1_bi)
    context2 = context2_cont.join(context2_bi)

    context1 = context1.join(context1_cat)
    context2 = context2.join(context2_cat)

    context1 = context1.sort_index(axis=1)
    context2 = context2.sort_index(axis=1)

    # Per-node ground-truth stats (n_edges_tweaked, sum_corr), for every node that is
    # a ground-truth node at all -- a mean-shift-only node has no entry in node_degree/
    # node_sum_magnitude, so it correctly falls back to (0, 0.0) via save_gt's own .get() default.
    gt_nodes = set(shift_nodes)
    for pair in corr_nodes + shift_corr_nodes:
        gt_nodes.update(pair)
    node_stats = {node: (node_degree.get(node, 0), node_sum_magnitude.get(node, 0.0)) for node in gt_nodes}
    effects = {'edge_magnitude': edge_magnitude, 'node_degree': node_degree, 'node_sum_magnitude': node_sum_magnitude,
               'node_shift_magnitude': node_shift_magnitude,
               'corr_matrix1': corr1, 'corr_matrix2': corr2,
               'mean_vector1': mean_vector1, 'mean_vector2': mean_vector2,
               'node_order': list(nodes), 'marginals': marginals,
               'corr_base': sigma_base,
               'background_edges': [frozenset((nodes[a], nodes[b])) for a, b in background_pairs],
               'background_lambda_min': background_lambda_min}

    # Save simulated contexts and ground truth nodes
    if path:
        context1.to_csv(os.path.join(path, f'{name1}.csv'))
        context2.to_csv(os.path.join(path, f'{name2}.csv'))
        meta.to_csv(os.path.join(path, 'meta.csv'), index=False)
        save_gt((shift_nodes, corr_nodes, shift_corr_nodes), os.path.join(path, 'ground_truth_nodes.txt'), mode='node', node_stats=node_stats, node_shift_magnitude=node_shift_magnitude)
        save_gt((shift_nodes, corr_nodes, shift_corr_nodes), os.path.join(path, 'ground_truth_edges.txt'), mode='edge', edge_magnitude=edge_magnitude)

    return context1, context2, meta, (shift_nodes, corr_nodes, shift_corr_nodes), effects


# Helper function to set correlation in copula-based simulation
def _set_corr(nodes, corr_low, corr_high, corr_matrix1, corr_matrix2, normal_nodes_bi=None, normal_nodes_cont=None, normal_nodes_cat=None,
              node_degree=None, node_sumsq=None, node_sum_magnitude=None, node_first_partner=None, hub_eligible=None,
              reuse_type=None, *, hub_sumsq_ceiling):
    node_degree = {} if node_degree is None else node_degree
    node_sumsq = {} if node_sumsq is None else node_sumsq
    node_sum_magnitude = {} if node_sum_magnitude is None else node_sum_magnitude
    node_first_partner = {} if node_first_partner is None else node_first_partner
    hub_eligible = set() if hub_eligible is None else hub_eligible

    # Node selection is deterministic (first available node in order) for the fresh case, so
    # repeated simulations with the same settings still introduce the correlation on exactly the
    # same variable pairs when no reuse is planned. Each branch below just picks which pool(s) feed
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

    # Whether this edge reuses a hub, and of which type, is decided by the caller (see
    # `_plan_hub_reuse`). A reuse only ever fills one of the pair's two slots -- never both. That
    # restriction is what keeps a hub's own sum(magnitude**2) check below exact rather than
    # approximate: it guarantees a hub's edges only ever land on leaves that never grow a second
    # edge of their own, so every hub is isolated from every other hub (see the module docstring /
    # Simulation_TODO.md Tier 3.1 discussion -- a plain chain of ordinary-looking edges can be
    # invalid even when no single node looks overloaded).
    magnitude = np.random.uniform(corr_low, corr_high)
    reused_node = None
    if reuse_type is not None:
        if reuse_type not in (prefix1, prefix2):
            raise ValueError(f'reuse_type {reuse_type!r} matches neither endpoint type '
                             f'({prefix1!r}, {prefix2!r}).')
        reuse_slot = 1 if reuse_type == prefix1 else 2
        # `sorted` is load-bearing, not cosmetic: hub_eligible is a set, and iterating a set of
        # strings follows hash order, which Python randomises per process (PEP 456). Without a
        # fixed order, random.choice below picks a different hub in every process even from an
        # identical RNG state -- seeding `random` and `np.random` is not enough -- so the planted
        # network, and therefore the simulated data, silently differs between otherwise-identical
        # runs. Guarded by tests/test_hub_reuse_reproducibility.py.
        # Only hubs that can take this edge without passing the validity ceiling are candidates.
        candidates = sorted(n for n in hub_eligible
                            if n.startswith(reuse_type) and node_sumsq[n] + magnitude ** 2 < hub_sumsq_ceiling)
        if candidates:
            reused_node = random.choice(candidates)
        # else: no eligible hub of this type -- draw a fresh pair below; the caller defers the reuse.

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
    # ground-truth `n_edges_tweaked`/`sum_corr` columns are built from, and it is also
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

    # Add to whatever is already there rather than overwriting it. With no background the cell
    # holds 0 and the two are identical (0 + m*s == m*s), so this is a no-op for the historical
    # behaviour. With a background the distinction matters twice over: adding makes the realised
    # between-context difference exactly `magnitude` regardless of the background value underneath,
    # and it moves the cell by exactly `magnitude` instead of by |magnitude*sign - background|,
    # which can be nearly twice as far and eats the positive-definiteness budget accordingly.
    target = corr_matrix1 if which == 1 else corr_matrix2
    updated = np.clip(target[idx1, idx2] + magnitude * sign, -_MAX_ABS_CORR, _MAX_ABS_CORR)
    target[idx1, idx2] = target[idx2, idx1] = updated

    return (node1, node2), magnitude, reused_node is not None


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
def save_gt(groundtruths, path, mode='node', edge_magnitude=None, node_stats=None, node_shift_magnitude=None):
    """
    Write ground truth nodes or edges to a two- (or, with the optional data below, wider-) column file.

    :param groundtruths: (shift_nodes, corr_nodes, shift_corr_nodes) as returned by `simulate_copula`.
    :param path: File to write.
    :param mode: 'node' writes one row per (node, description); 'edge' writes one row per
                (edge, description), edge = '_'.join(sorted(pair)).
    :param edge_magnitude: Optional, mode='edge' only. Dict keyed by frozenset({node1, node2}) ->
                           the correlation magnitude injected on that edge. When given, an extra
                           `corr` column is appended. Omit to reproduce the historical two-column
                           file exactly.
    :param node_stats: Optional, mode='node' only. Dict node -> (n_edges_tweaked, sum_corr). When
                       given, two extra columns are appended; a node absent from the dict (e.g.
                       mean-shift-only) is written as (0, 0.0). Omit to reproduce the historical
                       two-column file exactly.
    :param node_shift_magnitude: Optional, mode='node' only. Dict node -> mean-shift magnitude
                                 injected on that node (e.g. `effects['node_shift_magnitude']`).
                                 When given, an extra `shift` column is appended; a node with no
                                 mean-shift draw of its own (e.g. correlation-only) is written as
                                 0.0. Omit to reproduce the file exactly as it was before this
                                 column existed.

    The magnitude columns are named `shift` and `corr` after the quantity they hold, matching the
    `shift_low`/`shift_high` and `corr_low`/`corr_high` parameters the values are drawn from.
    """
    shift = groundtruths[0]
    corr = groundtruths[1]
    shift_corr = groundtruths[2]

    if mode == 'node':
        with open(path, 'w') as f:
            header = 'node, description'
            if node_stats is not None:
                header += ', n_edges_tweaked, sum_corr'
            if node_shift_magnitude is not None:
                header += ', shift'
            f.write(header + '\n')

            def _write_node(node, description):
                line = f'{node}, {description}'
                if node_stats is not None:
                    n_edges, sum_magnitude = node_stats.get(node, (0, 0.0))
                    line += f', {n_edges}, {sum_magnitude}'
                if node_shift_magnitude is not None:
                    line += f', {node_shift_magnitude.get(node, 0.0)}'
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
                header += ', corr'
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

