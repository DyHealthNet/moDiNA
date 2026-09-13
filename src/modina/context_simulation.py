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

# Hard clip applied when a planted magnitude is added on top of a background value. Only reachable
# with a strong background and a large `corr` simultaneously; it exists so the result is never an
# out-of-range correlation, but hitting it means the requested effect was not delivered in full, so
# it is warned about rather than passed over silently.
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


def _background_edges(n_vars, topology, density, rng):
    """Draw the sparse wiring that the background is built from.

    Returns a list of (i, j) index pairs with i < j. `density` is the number of connections each
    newly added node makes; both generators produce the same edge count so the two topologies are
    comparable at equal density.
    """
    if density < 1:
        raise ValueError(f'background_density must be at least 1, got {density}.')
    if density >= n_vars:
        raise ValueError(f'background_density ({density}) must be smaller than the number of '
                         f'variables ({n_vars}).')

    if topology == 'scale_free':
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

    if topology == 'erdos_renyi':
        # Uniformly random wiring at the same edge count: no hubs, degrees concentrated around the
        # mean. Useful as a null topology to check that a result is driven by hub structure rather
        # than by density alone.
        all_pairs = [(i, j) for i in range(n_vars) for j in range(i + 1, n_vars)]
        n_edges = min(len(all_pairs), n_vars * density - density * (density + 1) // 2)
        picked = rng.choice(len(all_pairs), size=n_edges, replace=False)
        return [all_pairs[i] for i in sorted(picked)]

    raise ValueError(f"background_topology must be None, 'scale_free' or 'erdos_renyi', "
                     f"got {topology!r}.")


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


def _build_background(n_vars, topology, strength, density, seed):
    """Build a dense background correlation matrix from a sparse network.

    Returns (sigma_base, edges, lambda_min). With `topology=None` this is the identity, an empty
    edge list and lambda_min 1.0 -- i.e. exactly the historical "no background" behaviour, and the
    rest of the simulation is unchanged.

    `strength` is a target on the OUTPUT (mean |r| over directly-wired pairs), not an input, because
    the entries of Omega have no interpretable scale on their own: what a given Omega entry produces
    in Sigma depends on the density of the network and on the degree of the two nodes it joins. The
    diagonal loading that hits the target is found by bisection, which is valid because strength is
    strictly decreasing in the loading.
    """
    if topology is None:
        return np.eye(n_vars), [], 1.0
    if not 0.0 < strength < 1.0:
        raise ValueError(f'background_strength must lie strictly between 0 and 1, got {strength}.')

    # A dedicated Generator, never the global `random`/`np.random` streams the rest of the
    # simulation draws from. Turning the background on therefore does not shift any downstream
    # draw: the same seed plants the same edges with the same magnitudes either way.
    rng = np.random.default_rng(seed)
    edges = _background_edges(n_vars, topology, density, rng)
    signs = rng.choice([-1.0, 1.0], size=len(edges))

    def wired_mean(load):
        sigma = _background_sigma(n_vars, edges, signs, load)
        return float(np.mean([abs(sigma[a, b]) for a, b in edges]))

    # Strength is bounded above by the topology: denser wiring forces a larger diagonal, which
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


# Simulate mixed data using a gaussian copula 
def simulate_copula(path=None, name1='context1', name2='context2',
                    n_bi=50, n_cont=50, n_cat=50, n_samples_1=500, n_samples_2=500,
                    n_shift_cont=0, n_shift_bi=0, n_shift_cat=0,
                    n_corr_cont_cont=0, n_corr_bi_bi=0, n_corr_cat_cat=0, n_corr_bi_cont=0, n_corr_bi_cat=0, n_corr_cont_cat=0, 
                    n_both_cont_cont=0, n_both_bi_bi=0, n_both_cat_cat=0, n_both_bi_cont=0, n_both_bi_cat=0, n_both_cont_cat=0,
                    shift=0.5, corr=0.7, hub_reuse_prob=0.0,
                    background_topology=None, background_strength=0.2, background_density=2,
                    background_seed=None,
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
    :param background_topology: Wiring used to build the background correlation structure, or None
                                (default) for the historical behaviour in which every pair that was
                                not deliberately planted is exactly independent. 'scale_free' grows
                                the network by preferential attachment, producing a few high-degree
                                hubs and a heavy-tailed degree distribution; 'erdos_renyi' wires the
                                same number of edges uniformly at random, producing no hubs. The
                                background is built ONCE and shared by both contexts, so every pair
                                that is not planted has bit-identical correlation in the two
                                contexts and the negatives stay exactly null.
    :param background_strength: Target mean |r| over the directly-wired pairs of the background.
                                This is a property of the resulting correlation matrix, not a value
                                written anywhere: the diagonal loading that attains it is solved for
                                by bisection. It is bounded above by the topology (denser wiring
                                forces weaker correlations); an unreachable request raises with the
                                attainable ceiling. Ignored when background_topology is None.
    :param background_density: Number of connections each node makes when it is added to the
                               network. Sets both the edge count and, for 'scale_free', how large
                               the hubs grow. Higher density lowers the reachable strength.
    :param background_seed: Seed for the background's own random generator. Kept separate from the
                            global streams the rest of the simulation draws from, so switching the
                            background on or off does not change which pairs get planted or with
                            what magnitudes.
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
               Also carries 'corr_base' (the shared background correlation matrix), 'background_edges'
               (the directly-wired pairs, as node-name frozensets) and 'background_lambda_min' (the
               perturbation budget the planting step was checked against).
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

    # The background is built once and copied into both contexts, rather than drawn twice. That is
    # what keeps the negatives clean: an unplanted pair holds bit-identical values in corr1 and
    # corr2, so its true between-context difference is exactly zero rather than merely small.
    sigma_base, background_pairs, background_lambda_min = _build_background(
        n_vars, background_topology, background_strength, background_density, background_seed)
    corr1 = sigma_base.copy()
    corr2 = sigma_base.copy()

    # Budget for the planting step. Every differential edge nudges one cell of Sigma; Weyl's
    # inequality bounds the damage by the spectral norm of the whole perturbation, and because
    # hub_reuse_prob only ever produces disjoint stars that norm is exactly
    # sqrt(max over nodes of sum(magnitude**2)) -- the quantity `node_sumsq` already tracks. Keeping
    # every node's sum(magnitude**2) below lambda_min**2 is therefore a sufficient condition for
    # both context matrices to stay positive definite. With the background off lambda_min is 1 and
    # the rule reduces to sum(magnitude**2) < 1, the exact Schur-complement condition the hard-coded
    # _HUB_SUMSQ_CEILING was approximating; the historical constant is kept verbatim in that case so
    # behaviour without a background is unchanged.
    #
    # Note this ceiling is consulted ONLY when deciding whether a node may take on another edge, so
    # it constrains how much a hub accumulates -- it does not bound the very first edge on a node.
    # A single edge larger than lambda_min can break validity on its own, which is what the check
    # just below warns about and the eigenvalue test before sampling catches for certain.
    hub_sumsq_ceiling = (_HUB_SUMSQ_CEILING if background_topology is None
                         else background_lambda_min ** 2)

    # The largest magnitude any single edge can draw. If even one edge exceeds the budget, no hub
    # ceiling can help -- warn here, where the fix (lower `corr`, or weaken the background) is
    # obvious, rather than leaving only the eigenvalue failure several steps later. Warned rather
    # than raised because Weyl is a worst case: a perturbation rarely points along the narrowest
    # direction, so configurations moderately over this line usually still produce a valid matrix.
    _max_magnitude = min(corr + _EFFECT_JITTER, 0.95)
    if background_topology is not None and _max_magnitude >= background_lambda_min:
        logging.warning(
            f'corr={corr} can draw magnitudes up to {_max_magnitude:.3f}, which exceeds the '
            f'budget left by the background (lambda_min={background_lambda_min:.4f}). The '
            f'correlation matrices may not be positive definite. Lower `corr`, or lower '
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

    def _inject(**pools):
        node_pair, magnitude, c1, c2, *_ = _set_corr(
            nodes=nodes, corr_param=corr, corr_matrix1=corr1, corr_matrix2=corr2,
            node_degree=node_degree, node_sumsq=node_sumsq, node_sum_magnitude=node_sum_magnitude,
            node_first_partner=node_first_partner, hub_eligible=hub_eligible,
            hub_reuse_prob=hub_reuse_prob, hub_sumsq_ceiling=hub_sumsq_ceiling, **pools)
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
                f'sqrt(d)*m of it. Reduce `corr`, reduce `hub_reuse_prob` so fewer edges pile onto '
                f'one node, or lower `background_strength` to free up budget.')

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
            # Binary node. The base rate is drawn from [binary_p_low, binary_p_high] rather than
            # from the full unit interval: the maximum attainable phi between two binary variables
            # is bounded by their marginals, so a node that is positive in only a few percent of
            # samples attenuates the correlation difference its edges actually carry, even though
            # the ground truth records the same `corr` for every edge.
            p = np.random.uniform(binary_p_low, binary_p_high)
            context1_bi[node] = sc.stats.bernoulli.ppf(u1[i, :], p=p).astype(int)
            context2_bi[node] = sc.stats.bernoulli.ppf(u2[i, :], p=p).astype(int)
            marginals[node] = {'kind': 'binary', 'p': float(p)}

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
            marginals[node] = {'kind': 'ordinal', 'cdf': cdf.tolist()}

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
    effects = {'edge_magnitude': edge_magnitude, 'node_degree': node_degree, 'node_sum_magnitude': node_sum_magnitude,
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
        save_gt((shift_nodes, corr_nodes, shift_corr_nodes), os.path.join(path, 'ground_truth_nodes.txt'), mode='node', node_stats=node_stats)
        save_gt((shift_nodes, corr_nodes, shift_corr_nodes), os.path.join(path, 'ground_truth_edges.txt'), mode='edge', edge_magnitude=edge_magnitude)

    return context1, context2, meta, (shift_nodes, corr_nodes, shift_corr_nodes), effects


# Helper function to set correlation in copula-based simulation
def _set_corr(nodes, corr_param, corr_matrix1, corr_matrix2, normal_nodes_bi=None, normal_nodes_cont=None, normal_nodes_cat=None,
              node_degree=None, node_sumsq=None, node_sum_magnitude=None, node_first_partner=None, hub_eligible=None,
              hub_reuse_prob=0.0, hub_sumsq_ceiling=_HUB_SUMSQ_CEILING):
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
            if node_sumsq.get(candidate, 0.0) + magnitude ** 2 < hub_sumsq_ceiling:
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

    # Add to whatever is already there rather than overwriting it. With no background the cell
    # holds 0 and the two are identical (0 + m*s == m*s), so this is a no-op for the historical
    # behaviour. With a background the distinction matters twice over: adding makes the realised
    # between-context difference exactly `magnitude` regardless of the background value underneath,
    # and it moves the cell by exactly `magnitude` instead of by |magnitude*sign - background|,
    # which can be nearly twice as far and eats the positive-definiteness budget accordingly.
    target = corr_matrix1 if which == 1 else corr_matrix2
    updated = np.clip(target[idx1, idx2] + magnitude * sign, -_MAX_ABS_CORR, _MAX_ABS_CORR)
    target[idx1, idx2] = target[idx2, idx1] = updated

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
def save_gt(groundtruths, path, mode='node', edge_magnitude=None, node_stats=None, node_shift_magnitude=None):
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
    :param node_shift_magnitude: Optional, mode='node' only. Dict node -> mean-shift magnitude
                                 injected on that node (e.g. `simulate_copula_mixed`'s
                                 `effects['node_shift_magnitude']`). When given, an extra
                                 `shift_magnitude` column is appended; a node with no mean-shift
                                 draw of its own (e.g. correlation-only) is written as 0.0. Omit to
                                 reproduce the file exactly as it was before this column existed.
    """
    shift = groundtruths[0]
    corr = groundtruths[1]
    shift_corr = groundtruths[2]

    if mode == 'node':
        with open(path, 'w') as f:
            header = 'node, description'
            if node_stats is not None:
                header += ', n_edges_tweaked, sum_jittered_magnitude'
            if node_shift_magnitude is not None:
                header += ', shift_magnitude'
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

