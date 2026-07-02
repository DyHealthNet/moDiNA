from modina.statistics_utils import probit_rescaling

import logging
import math
import os
from typing import Tuple, Optional
import pandas as pd
import numpy as np

logging.basicConfig(level = logging.INFO)


def _num_target_edges(filter_method: Optional[str], filter_param: float, n_nodes: int) -> int:
    """
    Translate a filtering method ('degree' or 'density') and its parameter into the
    target number of edges to retain for a network with n_nodes nodes.
    """
    if filter_method == 'degree':
        degree = filter_param

        if degree < 1 or degree >= n_nodes:
            raise ValueError(f"For 'degree' filtering, 'filter_param' must be between 1 and {n_nodes - 1}.")

        return math.ceil(degree * n_nodes / 2)

    elif filter_method == 'density':
        density = filter_param

        if density <= 0.0 or density > 1.0:
            raise ValueError("For 'density' filtering, 'filter_param' must be between 0 and 1.")

        possible_edges = n_nodes * (n_nodes - 1) / 2
        return math.ceil(density * possible_edges)

    else:
        raise ValueError(f"Invalid filtering method '{filter_method}'. Choose from: 'degree' or 'density'")


def _edge_keep_mask(scores: pd.DataFrame, filter_metric: str, n_target: int) -> pd.Series:
    """
    Compute a boolean 'keep' mask for a single network that retains the n_target strongest
    edges according to filter_metric. For 'raw-P' the smallest p-values are kept; for
    'rescaled-E' the largest absolute effect sizes are kept. Ties at the threshold are kept,
    so the mask may retain slightly more than n_target edges.
    """
    if filter_metric == 'raw-P':
        threshold = scores[filter_metric].sort_values(ascending=True).iloc[n_target - 1]
        return scores[filter_metric] <= threshold
    elif filter_metric == 'rescaled-E':
        threshold = np.abs(scores[filter_metric]).sort_values(ascending=False).iloc[n_target - 1]
        return np.abs(scores[filter_metric]) >= threshold
    else:
        raise ValueError(f"Invalid filter metric '{filter_metric}'. Choose from: 'raw-P' or 'rescaled-E'.")


# Edge filtering (two context-specific networks)
def filter(scores1: pd.DataFrame, scores2: pd.DataFrame, context1: pd.DataFrame, context2: pd.DataFrame,
           filter_method: Optional[str] = None, filter_param: float = 0.0,
           filter_metric: Optional[str] = None, filter_rule: Optional[str]=None,
           path: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Filter association scores and context data based on the specified filtering configurations.

    :param scores1: Statistical association scores of Context 1.
    :param scores2: Statistical association scores of Context 2.
    :param context1: The first context for the differential network analysis.
    :param context2: The second context for the differential network analysis.
    :param filter_method: Method used for filtering. Defaults to None.
    :param filter_param: Parameter for the specified filtering method. Defaults to 0.0.
    :param filter_metric: Edge metric used for filtering. Options include 'raw-P' and 'rescaled-E'. Defaults to None.
    :param filter_rule: Rule to integrate the networks during filtering. Defaults to None.
    :param path: Optional path to save the filtered scores and context data as CSV files. Defaults to None.
    :return: A tuple containing the filtered scores and context data.
    """

    if not scores1['label1'].equals(scores2['label1']) and not scores1['label2'].equals(scores2['label2']):
        raise ValueError('scores1 and scores2 need to have the same structure and order of edges.')

    # Check input parameters
    if filter_method is None:
        raise ValueError("Please provide a 'filter_method'.")

    if filter_metric is None:
        raise ValueError("Please provide a 'filter_metric'.")

    if filter_rule is None:
        raise ValueError("Please provide a 'filter_rule'.")

    if filter_param is None:
        raise ValueError("Please provide a 'filter_param'.")

    # Rescaling
    if filter_metric == 'rescaled-E':
        scores1, scores2 = probit_rescaling(scores1=scores1, scores2=scores2, metric='rescaled-E')

    # Compute number of edges to retain according to the specified method
    n_nodes = context1.shape[1]
    n_edges_before = scores1.shape[0]
    n_filtered_edges = _num_target_edges(filter_method, filter_param, n_nodes)

    logging.info(f"Filtering edges using method '{filter_method}' with parameter {filter_param}.")
    logging.info(f"Number of edges to retain after filtering: {n_filtered_edges}.")

    # Compute per-context keep masks
    mask1 = _edge_keep_mask(scores1, filter_metric, n_filtered_edges)
    mask2 = _edge_keep_mask(scores2, filter_metric, n_filtered_edges)

    # Apply the filtering threshold to scores and raw data if provided
    if filter_rule == 'union':
        mask = mask1 | mask2

        # Apply mask
        scores1_filtered = scores1[mask].copy()
        scores2_filtered = scores2[mask].copy()

    elif filter_rule == 'zero':
        # Apply mask
        filtered1 = scores1[mask1].copy()
        filtered2 = scores2[mask2].copy()
        filtered1 = filtered1.set_index(['label1', 'label2'])
        filtered2 = filtered2.set_index(['label1', 'label2'])

        # Unify indices and set missing values
        indices = filtered1.index.union(filtered2.index)
        scores1_filtered = filtered1.reindex(indices)
        scores2_filtered = filtered2.reindex(indices)

        fill_values = {
            'raw-P': 1.0,
            'raw-E': 0.0,
            'rescaled-E': 0.0,
            'log-P': 0.0,
            'inv-P': 0.0
        }

        for metric, value in fill_values.items():
            if metric in scores1_filtered.columns:
                scores1_filtered[metric] = scores1_filtered[metric].fillna(value)
            if metric in scores2_filtered.columns:
                scores2_filtered[metric] = scores2_filtered[metric].fillna(value)

        if 'test_type' in scores1_filtered.columns and 'test_type' in scores2_filtered.columns:
            merged_test_type = scores1_filtered['test_type'].combine_first(scores2_filtered['test_type'])
            scores1_filtered['test_type'] = merged_test_type
            scores2_filtered['test_type'] = merged_test_type

        scores1_filtered = scores1_filtered.reset_index()
        scores2_filtered = scores2_filtered.reset_index()

    else:
        raise ValueError(f"Invalid filtering rule '{filter_rule}'.")

    # Filter context data to only include nodes present in the filtered scores
    filtered_nodes = np.concatenate((scores1_filtered['label1'].values,
                                     scores1_filtered['label2'].values))
    filtered_nodes = pd.unique(filtered_nodes)
    context1_filtered = context1[filtered_nodes].copy()
    context2_filtered = context2[filtered_nodes].copy()

    n_edges_after = scores1_filtered.shape[0]

    logging.info(f'Reduced the number of edges from {n_edges_before} to {n_edges_after}.')

    if path is not None:
        scores1_filtered.to_csv(os.path.join(path, 'scores1_filtered.csv'), index=False)
        scores2_filtered.to_csv(os.path.join(path, 'scores2_filtered.csv'), index=False)
        context1_filtered.to_csv(os.path.join(path, 'context1_filtered.csv'), index=False)
        context2_filtered.to_csv(os.path.join(path, 'context2_filtered.csv'), index=False)

    return scores1_filtered, scores2_filtered, context1_filtered, context2_filtered


# Edge filtering (single context-specific network)
def filter_single(scores: pd.DataFrame, context: pd.DataFrame,
                  filter_method: Optional[str] = None, filter_param: float = 0.0,
                  filter_metric: Optional[str] = None,
                  path: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filter the association scores and context data of a single context network based on the
    specified filtering configurations. Unlike :func:`filter`, this operates on one network only
    and therefore does not take a 'filter_rule' (there is nothing to integrate across contexts).

    :param scores: Statistical association scores of the context.
    :param context: Observed data of the context (rows: samples, columns: variables).
    :param filter_method: Method used for filtering ('degree' or 'density'). Defaults to None.
    :param filter_param: Parameter for the specified filtering method. Defaults to 0.0.
    :param filter_metric: Edge metric used for filtering. Options are 'raw-P' and 'rescaled-E'. Defaults to None.
    :param path: Optional path to save the filtered scores and context data as CSV files. Defaults to None.
    :return: A tuple (scores_filtered, context_filtered) with the filtered scores and context data.
    """
    if filter_method is None:
        raise ValueError("Please provide a 'filter_method'.")

    if filter_metric is None:
        raise ValueError("Please provide a 'filter_metric'.")

    if filter_param is None:
        raise ValueError("Please provide a 'filter_param'.")

    # 'rescaled-E' is produced by a joint probit rescaling over both contexts and cannot be
    # derived from a single network. Require it to be already present in the scores.
    if filter_metric == 'rescaled-E' and 'rescaled-E' not in scores.columns:
        raise ValueError("Filtering a single network on 'rescaled-E' requires the 'rescaled-E' column to "
                         "already be present, as it is computed by a joint rescaling over both contexts. "
                         "Please filter on 'raw-P' instead, or provide already-rescaled scores.")

    n_nodes = context.shape[1]
    n_edges_before = scores.shape[0]
    n_filtered_edges = _num_target_edges(filter_method, filter_param, n_nodes)

    logging.info(f"Filtering edges of a single network using method '{filter_method}' with parameter {filter_param}.")
    logging.info(f"Number of edges to retain after filtering: {n_filtered_edges}.")

    mask = _edge_keep_mask(scores, filter_metric, n_filtered_edges)
    scores_filtered = scores[mask].copy()

    # Filter context data to only include nodes present in the filtered scores
    filtered_nodes = np.concatenate((scores_filtered['label1'].values,
                                     scores_filtered['label2'].values))
    filtered_nodes = pd.unique(filtered_nodes)
    context_filtered = context[filtered_nodes].copy()

    n_edges_after = scores_filtered.shape[0]
    logging.info(f'Reduced the number of edges from {n_edges_before} to {n_edges_after}.')

    if path is not None:
        scores_filtered.to_csv(os.path.join(path, 'scores_filtered.csv'), index=False)
        context_filtered.to_csv(os.path.join(path, 'context_filtered.csv'), index=False)

    return scores_filtered, context_filtered


# Edge filtering (differential network)
def filter_differential(edges_diff: pd.DataFrame, edge_metric: str,
                        filter_method: Optional[str] = None, filter_param: float = 0.0,
                        path: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filter the differential network by retaining the strongest edges according to the
    already-computed edge metric. Because the differential network is a single graph, there is
    no per-context integration ('filter_rule') and no p-value-vs-effect-size choice
    ('filter_metric'): filtering always operates on the provided edge_metric.

    :param edges_diff: Differential edge scores with columns 'label1', 'label2', 'test_type',
                       edge_metric and edge_metric + '_signed'.
    :param edge_metric: Name of the (absolute) differential edge-metric column to filter on.
    :param filter_method: Method used for filtering ('degree' or 'density'). Defaults to None.
    :param filter_param: Parameter for the specified filtering method. Defaults to 0.0.
    :param path: Optional path to save the filtered edges and per-node statistics as CSV files. Defaults to None.
    :return: A tuple (edges_filtered, edge_node_stats) with the filtered differential edges and the
             per-node statistics recomputed over the retained edges.
    """
    if filter_method is None:
        raise ValueError("Please provide a 'filter_method'.")

    if edge_metric is None:
        raise ValueError("Please provide the 'edge_metric' to filter on.")

    if edge_metric not in edges_diff.columns:
        raise ValueError(f"Edge metric '{edge_metric}' not found in the differential network columns.")

    n_nodes = pd.unique(edges_diff[['label1', 'label2']].values.ravel()).size
    n_edges_before = edges_diff.shape[0]
    n_filtered_edges = min(_num_target_edges(filter_method, filter_param, n_nodes), n_edges_before)

    logging.info(f"Filtering differential edges on '{edge_metric}' using method '{filter_method}' "
                 f"with parameter {filter_param}.")
    logging.info(f"Number of edges to retain after filtering: {n_filtered_edges}.")

    # Keep the strongest edges by absolute edge metric (the column is already non-negative;
    # abs() is defensive). Preserve the original edge order after selection.
    order = edges_diff[edge_metric].abs().sort_values(ascending=False)
    keep_idx = order.iloc[:n_filtered_edges].index
    edges_filtered = edges_diff.loc[keep_idx].sort_index().reset_index(drop=True)

    n_edges_after = edges_filtered.shape[0]
    logging.info(f'Reduced the number of differential edges from {n_edges_before} to {n_edges_after}.')

    # Recompute per-node statistics over the retained edges. Local import mirrors ranking.py and
    # avoids a top-level import cycle with diff_net_construction.
    from modina.diff_net_construction import edge_node_statistics
    edge_node_stats = edge_node_statistics(edges_filtered, edge_metric)

    if path is not None:
        edges_filtered.to_csv(os.path.join(path, 'diff_edges_filtered.csv'))
        edge_node_stats.to_csv(os.path.join(path, 'diff_edges_filtered_node_stats.csv'))

    return edges_filtered, edge_node_stats
