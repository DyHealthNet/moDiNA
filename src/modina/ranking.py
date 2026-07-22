from collections import defaultdict
import networkx as nx
import numpy as np
import pandas as pd
import logging
import os
from typing import Tuple, Optional


# Compute ranking
def compute_ranking(nodes_diff: Optional[pd.DataFrame | pd.Series], edges_diff: Optional[pd.DataFrame | pd.Series], ranking_alg: str,
                    path: Optional[str] = None, meta_file: Optional[pd.DataFrame] = None,
                    edge_node_stats: Optional[pd.DataFrame] = None) -> pd.DataFrame | pd.Series:
    """
    Compute a ranking based on the specified ranking algorithm.

    For the node-indexed rankings ('PageRank+', 'PageRank', 'absDimontRank', 'DimontRank', 'nodeRank') the
    output is additionally enriched with the node-metric value (when a node metric was employed) and with
    per-node statistics over the incident edges (when an edge metric was employed).

    :param nodes_diff: Differential node scores.
    :param edges_diff: Differential edge scores.
    :param ranking_alg: Ranking algorithm to compute. Options are 'PageRank+', 'PageRank', 'absDimontRank', 'DimontRank' and 'nodeRank'.
    :param meta_file: Metadata file containing a 'label' and 'type' column to specify the data type of each variable.
    :param path: Optional path to save the ranking as a CSV file.
    :param edge_node_stats: Optional precomputed per-node edge statistics (see edge_node_statistics). If not
                            provided but an edge metric was employed, the statistics are computed on the fly.
    :return: A DataFrame containing the ranking, enriched with the node-metric value and per-node edge statistics.
    """
    if nodes_diff is not None:
        node_metric = nodes_diff.columns[0]
    else:
        node_metric = None

    if edges_diff is not None:
        edge_metric = edges_diff.columns[3]
    else:
        edge_metric = None

    # Personalized PageRank
    if ranking_alg == 'PageRank+':
        if nodes_diff is None or edges_diff is None:
            raise ValueError("To compute 'PageRank+', please provide both 'nodes_diff' and 'edges_diff'.")

        ranking_scores = pagerank(nodes_diff=nodes_diff, edges_diff=edges_diff,
                                  node_metric=node_metric, edge_metric=edge_metric,
                                  personalization=True)

    # PageRank
    elif ranking_alg == 'PageRank':
        if edges_diff is None:
            raise ValueError("To compute 'PageRank', please provide 'edges_diff'.")

        ranking_scores = pagerank(edges_diff=edges_diff, edge_metric=edge_metric,
                                personalization=False)

    # DimontRank with absolute difference
    elif ranking_alg == 'absDimontRank':
        if edges_diff is None:
            raise ValueError("To compute 'absDimontRank', please provide 'edges_diff'.")

        ranking_scores = dimontrank(edges_diff=edges_diff, edge_metric=edge_metric, mode='abs')

    # DimontRank with signed difference
    elif ranking_alg == 'DimontRank':
        if edges_diff is None:
            raise ValueError("To compute 'DimontRank', please provide 'edges_diff'.")

        ranking_scores = dimontrank(edges_diff=edges_diff, edge_metric=edge_metric, mode='signed')

    # Direct ranking based on node metric
    elif ranking_alg == 'nodeRank':
        if nodes_diff is None:
            raise ValueError("To compute 'nodeRank', please provide 'nodes_diff'.")

    else:
        raise ValueError(f"Invalid ranking algorithm {ranking_alg}. "
                         "Choose from: 'PageRank+', 'PageRank', 'absDimontRank', 'DimontRank' or 'nodeRank'.")

    # Construct ranking dataframe
    if ranking_alg == 'nodeRank':
        assert type(node_metric) == str, "node_metric must be a string."
        assert nodes_diff is not None, "nodes_diff must be provided for nodeRank."

        ranking_df = nodes_diff[[node_metric]].copy()
        ranking_df = ranking_df.rename(columns={node_metric: 'score'}).reset_index(names='node')
        ranking_df = ranking_df.sort_values('score', ascending=False).reset_index(drop=True)
        ranking_df['rank'] = ranking_df['score'].rank(method='min', ascending=False).astype(int)
        ranking_df = ranking_df[['node', 'rank', 'score']]

        if meta_file is None:
            logging.warning('No meta_file was provided to specify node types.')
        else:
            meta_file = meta_file.set_index('label')
            ranking_df['type'] = ranking_df['node'].map(meta_file['type'])

    else:
        ranking_df = pd.DataFrame({'node': list(ranking_scores.keys()), 'score': list(ranking_scores.values())})
        ranking_df = ranking_df.sort_values('score', ascending=False).reset_index(drop=True)
        ranking_df['rank'] = ranking_df['score'].rank(method='min', ascending=False).astype(int)
        ranking_df = ranking_df[['node', 'rank', 'score']]
        
        if meta_file is None:
            logging.warning('No meta_file was provided to specify node types.')
        else:
            meta_file = meta_file.set_index('label')
            ranking_df['type'] = ranking_df['node'].map(meta_file['type'])

    # Enrich the node-indexed ranking with the node-metric value and per-node edge statistics.
    if node_metric is not None and nodes_diff is not None:
        ranking_df[node_metric] = ranking_df['node'].map(nodes_diff[node_metric])

    if edge_metric is not None and edges_diff is not None:
        if edge_node_stats is None:
            from modina.diff_net_construction import edge_node_statistics
            edge_node_stats = edge_node_statistics(edges_diff, edge_metric)
        ranking_df = ranking_df.merge(edge_node_stats, how='left', left_on='node', right_index=True)

    if path is not None:
        ranking_df.to_csv(path, index=False)

    return ranking_df


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


# DimontRank algorithm
def dimontrank(edges_diff, edge_metric, mode='abs'):
    if mode == 'signed':
        # Check for valid edge metric
        if edge_metric not in ['diff-P', 'int-IS-E', 'diff-E', 'sum-diff-PE',
                               'sum-diff-L-PE', 'diff-L-PE', 'diff-L-P']:
            raise ValueError(f"DimontRank can only be applied with edge metrics 'diff-P', 'int-IS-E', 'diff-E', 'sum-diff-PE', 'sum-diff-L-PE', 'diff-L-PE', or 'diff-L-P'. But '{edge_metric}' was provided.")
        edge_metric = edge_metric + '_signed'

    sums = defaultdict(float)
    counts = defaultdict(int)
    for n1, n2, w in zip(edges_diff['label1'], edges_diff['label2'], edges_diff[edge_metric]):
        sums[n1] += w
        sums[n2] += w
        counts[n1] += 1
        counts[n2] += 1

    return {n: (np.abs(sums[n] / counts[n]) if counts[n] != 0 else 0) for n in sums}
