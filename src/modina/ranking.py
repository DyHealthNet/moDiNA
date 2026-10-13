import networkx as nx
import pandas as pd
import logging
import os
from typing import Tuple, Optional

from modina.metrics import (EDGE_METRICS, NODE_METRICS, ALG_REQUIRES, ALG_EDGE_STATISTIC, ASCENDING_ALGORITHMS,
                            check_ranking_alg, check_rescaling_consistency)


# Compute ranking
def compute_ranking(nodes_diff: Optional[pd.DataFrame | pd.Series], edges_diff: Optional[pd.DataFrame | pd.Series], ranking_alg: str,
                    path: Optional[str] = None, meta_file: Optional[pd.DataFrame] = None,
                    edge_node_stats: Optional[pd.DataFrame] = None) -> pd.DataFrame | pd.Series:
    """
    Compute a ranking based on the specified ranking algorithm.

    The output is additionally enriched with the node-metric value (when a node metric was employed)
    and with per-node statistics over the incident edges (when an edge metric was employed).
    'MeanEdgeRank' / 'MedianEdgeRank' score each node by the mean / median edge metric over its
    incident edges ('edge-mean' / 'edge-median'). 'MeanProductRank' / 'MedianProductRank' rank nodes by
    the product of their rank by the node metric and their rank by that mean / median incident edge
    metric; their score is therefore better when lower, while 'rank' is 1 for the top node for every
    algorithm. Algorithms combining node and edge metrics ('PageRank+', 'MeanProductRank',
    'MedianProductRank') require both to use the same effect-size rescaling (RE or VE); P- and LP-based
    metrics can be combined with either.

    :param nodes_diff: Differential node scores.
    :param edges_diff: Differential edge scores.
    :param ranking_alg: Ranking algorithm to compute. Options are 'PageRank+', 'PageRank', 'MeanEdgeRank', 'MedianEdgeRank', 'nodeRank', 'MeanProductRank' and 'MedianProductRank'.
    :param meta_file: Metadata file containing a 'label' and 'type' column to specify the data type of each variable.
    :param path: Optional path to save the ranking as a CSV file.
    :param edge_node_stats: Optional precomputed per-node edge statistics (see edge_node_statistics). If not
                            provided but an edge metric was employed, the statistics are computed on the fly.
    :return: A DataFrame containing the ranking, enriched with the node-metric value and per-node edge statistics.
    """
    check_ranking_alg(ranking_alg)

    node_metric = _find_metric(nodes_diff, NODE_METRICS, fallback=0) if nodes_diff is not None else None
    edge_metric = _find_metric(edges_diff, EDGE_METRICS, fallback=3) if edges_diff is not None else None

    needs_edges, needs_nodes = ALG_REQUIRES[ranking_alg]
    if (needs_edges and edges_diff is None) or (needs_nodes and nodes_diff is None):
        required = [name for name, needed in (('nodes_diff', needs_nodes), ('edges_diff', needs_edges)) if needed]
        raise ValueError(f"To compute '{ranking_alg}', please provide {' and '.join(repr(r) for r in required)}.")
    if needs_edges and needs_nodes:
        check_rescaling_consistency(node_metric, edge_metric)

    # Per-node statistics over the incident edges (used by the edge-statistic algorithms and to enrich the output)
    if edge_metric is not None and edge_node_stats is None:
        from modina.diff_net_construction import edge_node_statistics
        edge_node_stats = edge_node_statistics(edges_diff, edge_metric)

    extra = None

    # Personalized PageRank
    if ranking_alg == 'PageRank+':
        ranking_scores = pagerank(nodes_diff=nodes_diff, edges_diff=edges_diff,
                                  node_metric=node_metric, edge_metric=edge_metric,
                                  personalization=True)

    # PageRank
    elif ranking_alg == 'PageRank':
        ranking_scores = pagerank(edges_diff=edges_diff, edge_metric=edge_metric,
                                personalization=False)

    # Mean / median edge metric over each node's incident edges
    elif ranking_alg in ('MeanEdgeRank', 'MedianEdgeRank'):
        ranking_scores = edge_node_stats[ALG_EDGE_STATISTIC[ranking_alg]].to_dict()

    # Direct ranking based on node metric
    elif ranking_alg == 'nodeRank':
        ranking_scores = nodes_diff[node_metric].to_dict()

    # Product of the node-metric rank and the mean / median edge-metric rank
    else:
        extra = productrank(nodes_diff=nodes_diff, node_metric=node_metric, edge_node_stats=edge_node_stats,
                            statistic=ALG_EDGE_STATISTIC[ranking_alg])
        ranking_scores = extra['score'].to_dict()

    # Construct ranking dataframe
    ascending = ranking_alg in ASCENDING_ALGORITHMS
    ranking_df = pd.DataFrame({'node': list(ranking_scores.keys()), 'score': list(ranking_scores.values())})
    ranking_df = ranking_df.sort_values('score', ascending=ascending).reset_index(drop=True)
    ranking_df['rank'] = ranking_df['score'].rank(method='min', ascending=ascending).astype(int)
    ranking_df = ranking_df[['node', 'rank', 'score']]

    if extra is not None:
        ranking_df = ranking_df.merge(extra.drop(columns='score'), how='left', left_on='node', right_index=True)

    if meta_file is None:
        logging.warning('No meta_file was provided to specify node types.')
    else:
        meta_file = meta_file.set_index('label')
        ranking_df['type'] = ranking_df['node'].map(meta_file['type'])

    # Enrich the node-indexed ranking with the node-metric value and per-node edge statistics.
    if node_metric is not None:
        ranking_df[node_metric] = ranking_df['node'].map(nodes_diff[node_metric])

    if edge_metric is not None:
        ranking_df = ranking_df.merge(edge_node_stats, how='left', left_on='node', right_index=True)

    if path is not None:
        ranking_df.to_csv(path, index=False)

    return ranking_df


# Name of the metric column: the first column that is a known metric, else the column at the
# position the differential-network functions write it to (with a warning).
def _find_metric(df, known, fallback):
    for col in df.columns:
        if col in known:
            return col
    col = df.columns[fallback]
    logging.warning(f"No known metric column found; using column '{col}' as the metric.")
    return col


# PageRank algorithm
def pagerank(edges_diff, edge_metric, nodes_diff=None, node_metric=None, personalization=True):
    edges_diff = edges_diff.copy()
    edges_diff['weight'] = edges_diff[edge_metric]

    # Create network
    network = nx.from_pandas_edgelist(edges_diff, 'label1', 'label2', 'weight')

    if personalization:
        # Add node metric
        if nodes_diff is None or node_metric is None:
            raise ValueError('When personalization should be used, nodes_diff and node_metric must be provided.')

        nodes_diff = nodes_diff.copy()
        nodes_diff['weight'] = nodes_diff[node_metric]

        # Apply PageRank with personalization
        if (nodes_diff['weight'].nunique() == 1):
            ranking = nx.pagerank(network)
        else:
            personalization = nodes_diff['weight'].to_dict()
            ranking = nx.pagerank(network, personalization=personalization)
    else:
        # Apply PageRank without personalization
        ranking = nx.pagerank(network)

    return ranking


# MeanProductRank / MedianProductRank: rank nodes by the node metric and by the mean / median of their
# incident edge metric (statistic 'edge-mean' / 'edge-median'; both descending, ties get the average
# rank) and score each node by the product of both ranks (lower = better). Nodes without edges or
# without a node score are ranked last in that component.
def productrank(nodes_diff, node_metric, edge_node_stats, statistic='edge-mean'):
    nodes = nodes_diff.index.union(edge_node_stats.index)
    rank_node = nodes_diff[node_metric].reindex(nodes).rank(method='average', ascending=False, na_option='bottom')
    rank_edge = edge_node_stats[statistic].reindex(nodes).rank(method='average', ascending=False, na_option='bottom')
    return pd.DataFrame({'score': rank_node * rank_edge, 'rank_node': rank_node, 'rank_edge': rank_edge})
