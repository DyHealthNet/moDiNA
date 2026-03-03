from collections import defaultdict
import networkx as nx
import numpy as np
import pandas as pd
import logging
import os
from typing import Tuple, Optional


# Compute ranking
def compute_ranking(nodes_diff: Optional[pd.DataFrame], edges_diff: Optional[pd.DataFrame], ranking_alg: str,
                    path: Optional[str] = None, meta_file: Optional[pd.DataFrame] = None) -> Tuple[list, dict]:
    """
    Compute a ranking based on the specified ranking algorithm.

    :param nodes_diff: Differential node scores.
    :param edges_diff: Differential edge scores.
    :param ranking_alg: Ranking algorithm to compute. Options are 'PageRank+', 'PageRank', 'absDimontRank', 'DimontRank', 'direct_node' and 'direct_edge'.
    :param meta_file: Metadata file containing a 'label' and 'type' column to specify the data type of each variable.
    :param path: Optional path to save the ranking as a CSV file.
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
        ranks = pd.Series(ranking_scores).sort_values(ascending=False).index.tolist()

    # PageRank
    elif ranking_alg == 'PageRank':
        if edges_diff is None:
            raise ValueError("To compute 'PageRank', please provide 'edges_diff'.")

        ranking_scores = pagerank(edges_diff=edges_diff, edge_metric=edge_metric,
                                personalization=False)
        ranks = pd.Series(ranking_scores).sort_values(ascending=False).index.tolist()

    # DimontRank with absolute difference
    elif ranking_alg == 'absDimontRank':
        if edges_diff is None:
            raise ValueError("To compute 'absDimontRank', please provide 'edges_diff'.")

        ranking_scores = dimontrank(edges_diff=edges_diff, edge_metric=edge_metric, mode='abs')
        ranks = pd.Series(ranking_scores).sort_values(ascending=False).index.tolist()

    # DimontRank with signed difference
    elif ranking_alg == 'DimontRank':
        if edges_diff is None:
            raise ValueError("To compute 'DimontRank', please provide 'edges_diff'.")

        ranking_scores = dimontrank(edges_diff=edges_diff, edge_metric=edge_metric, mode='signed')
        ranks = pd.Series(ranking_scores).sort_values(ascending=False).index.tolist()

    # Direct ranking based on node metric
    elif ranking_alg == 'direct_node':
        if nodes_diff is None:
            raise ValueError("To compute 'direct_node', please provide 'nodes_diff'.")

        ranking_scores = nodes_diff[node_metric]
        ranks = pd.Series(ranking_scores).sort_values(ascending=False).index.tolist()

    elif ranking_alg == 'direct_edge':
        if edges_diff is None or edge_metric is None:
            raise ValueError("To compute 'direct_edge', please provide 'edges_diff'.")

        ranking_scores = edges_diff[['label1', 'label2', edge_metric]].copy()
        ranking_scores = ranking_scores.sort_values(by=edge_metric, ascending=False).reset_index(drop=True)
        #ranks = ranking_scores[['label1', 'label2']].values.tolist()
        ranks = (
            ranking_scores[['label1', 'label2']]
            .apply(lambda row: '_'.join(sorted(row)), axis=1)
            .tolist()
        )

    else:
        raise ValueError(f"Invalid ranking algorithm {ranking_alg}. "
                         "Choose from: 'PageRank+', 'PageRank', 'absDimontRank', 'DimontRank', 'direct_node' or 'direct_edge'.")

    rank_cont = []
    rank_cat = []
    rank_bi = []

    if ranking_alg != 'direct_edge':
        if meta_file is None:
            logging.warning('No meta_file was provided. Rankings per data type cannot be computed.')

        else:
            meta_file = meta_file.set_index('label')

            for node in ranks:
                node_type = meta_file.at[node, 'type']

                if node_type == 'continuous':
                    rank_cont.append(node)
                elif node_type in ('ordinal', 'nominal'):
                    rank_cat.append(node)
                elif node_type == 'binary':
                    rank_bi.append(node)
                else:
                    raise ValueError(f"Invalid node type '{node_type}' for node '{node}' in meta_file.")

    else:
        logging.info('Ranking per type is not available for direct_edge ranking algorithm.')

    if path is not None:
        ranking_df = pd.DataFrame({"node": ranks, "rank": range(1, len(ranks) + 1)})
        ranking_df = ranking_df.sort_values("node")
        ranking_df.to_csv(path, index=False)

        if ranking_alg != 'direct_edge' and meta_file is not None:
            rank_cont_df = pd.DataFrame({"node": rank_cont, "rank": range(1, len(rank_cont) + 1)})
            rank_cat_df = pd.DataFrame({"node": rank_cat, "rank": range(1, len(rank_cat) + 1)})
            rank_bi_df = pd.DataFrame({"node": rank_bi, "rank": range(1, len(rank_bi) + 1)})

            rank_cont_path = os.path.splitext(path)[0] + '_continuous.csv'
            rank_cat_path = os.path.splitext(path)[0] + '_categorical.csv'
            rank_bi_path = os.path.splitext(path)[0] + '_binary.csv'

            rank_cont_df.to_csv(rank_cont_path, index=False)
            rank_cat_df.to_csv(rank_cat_path, index=False)
            rank_bi_df.to_csv(rank_bi_path, index=False)

    return ranks, {'cont': rank_cont, 'cat': rank_cat, 'bi': rank_bi}


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
        if edge_metric not in ['pre-P', 'pre-E', 'pre-PE', 'pre-LS', 'int-IS']:
            raise ValueError(f"DimontRank can only be applied with edge metrics 'pre-P', 'pre-E', 'pre-PE', 'pre-LS', or 'int-IS'. But '{edge_metric}' was provided.")
        edge_metric = edge_metric + '_signed'

    sums = defaultdict(float)
    counts = defaultdict(int)
    for n1, n2, w in zip(edges_diff['label1'], edges_diff['label2'], edges_diff[edge_metric]):
        sums[n1] += w
        sums[n2] += w
        counts[n1] += 1
        counts[n2] += 1

    return {n: (np.abs(sums[n] / counts[n]) if counts[n] != 0 else 0) for n in sums}
