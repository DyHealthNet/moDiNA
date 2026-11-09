from typing import Optional


# Score types computed per context (see statistics_utils.add_score_columns). Every edge and node
# metric is built from one of these, so the same score can be compared across metric families:
#   P     = 1 - p             (linear significance; p is the multiple-testing adjusted p-value)
#   LP    = -log10(p)         (log significance)
#   RE    = rank-based probit rescaled effect size (rescaled-E)
#   VE    = variance-explained effect size (transformed-E, sign(r) * r² or η²)
#   LPxRE = LP * RE,  LPxVE = LP * VE
SCORE_TYPES = ('P', 'LP', 'RE', 'VE', 'LPxRE', 'LPxVE')

# Edge metrics, mapped to the score types whose absolute between-context differences are summed.
EDGE_METRICS = {
    **{f'diff-{s}': (s,) for s in SCORE_TYPES},
    'sum-LP-VE': ('LP', 'VE'),
    'sum-LP-RE': ('LP', 'RE'),
}

# Node metrics: weighted degree centrality (WDC), PageRank centrality (PRC) and statistical test
# centrality (STC), each with every score type.
NODE_FAMILIES = ('WDC', 'PRC', 'STC')
NODE_METRICS = tuple(f'{f}-{s}' for f in NODE_FAMILIES for s in SCORE_TYPES)

RANKING_ALGORITHMS = ('PageRank+', 'PageRank', 'MeanEdgeRank', 'MedianEdgeRank', 'nodeRank', 'MeanProductRank',
                      'MedianProductRank')

# (needs edges_diff, needs nodes_diff) per ranking algorithm
ALG_REQUIRES = {
    'PageRank+': (True, True),
    'PageRank': (True, False),
    'MeanEdgeRank': (True, False),
    'MedianEdgeRank': (True, False),
    'nodeRank': (False, True),
    'MeanProductRank': (True, True),
    'MedianProductRank': (True, True),
}

# Per-node edge statistic (column of edge_node_statistics) used by the edge-statistic based algorithms
ALG_EDGE_STATISTIC = {
    'MeanEdgeRank': 'edge-mean',
    'MedianEdgeRank': 'edge-median',
    'MeanProductRank': 'edge-mean',
    'MedianProductRank': 'edge-median',
}

# Ranking algorithms whose score is better when lower (sorted and ranked ascending).
ASCENDING_ALGORITHMS = frozenset({'MeanProductRank', 'MedianProductRank'})


def metric_score(metric: str) -> str:
    """Score type a node metric is built from, e.g. 'PRC-LPxRE' -> 'LPxRE'."""
    return metric.split('-', 1)[1]


def check_edge_metric(edge_metric: str) -> str:
    if edge_metric not in EDGE_METRICS:
        raise ValueError(f"Invalid edge metric '{edge_metric}'. Choose from: {', '.join(EDGE_METRICS)}.")
    return edge_metric


def check_node_metric(node_metric: str) -> str:
    if node_metric not in NODE_METRICS:
        raise ValueError(f"Invalid node metric '{node_metric}'. Choose from: {', '.join(NODE_METRICS)}.")
    return node_metric


def check_ranking_alg(ranking_alg: str) -> str:
    if ranking_alg not in RANKING_ALGORITHMS:
        raise ValueError(f"Invalid ranking algorithm '{ranking_alg}'. Choose from: {', '.join(RANKING_ALGORITHMS)}.")
    return ranking_alg


def effect_rescaling(metric: Optional[str]) -> Optional[str]:
    """Effect-size rescaling a metric uses: 'RE' (rank-based), 'VE' (variance-explained) or None (P, LP)."""
    if metric is None:
        return None
    components = EDGE_METRICS[metric] if metric in EDGE_METRICS else (metric_score(metric),)
    for rescaling in ('RE', 'VE'):
        if any(rescaling in c for c in components):
            return rescaling
    return None


def check_rescaling_consistency(node_metric: Optional[str], edge_metric: Optional[str], ranking_alg: Optional[str] = None) -> None:
    """Node and edge metrics combined in one ranking must not mix the RE and VE effect-size rescalings."""
    if ranking_alg is not None and not all(ALG_REQUIRES.get(ranking_alg, (False, False))):
        return
    node_rescaling, edge_rescaling = effect_rescaling(node_metric), effect_rescaling(edge_metric)
    if node_rescaling is not None and edge_rescaling is not None and node_rescaling != edge_rescaling:
        raise ValueError(f"The node metric '{node_metric}' ({node_rescaling}) and the edge metric '{edge_metric}' ({edge_rescaling}) "
                         f"use different effect-size rescalings and cannot be combined. Use the same rescaling (RE or VE) for both.")
