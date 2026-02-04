from modina.statistics_utils import *

import os
from typing import Optional, Tuple
import networkx as nx
import numpy as np
import pandas as pd
import igraph as ig
import scipy.stats as sc


# Differential network computation
def compute_diff_network(scores1: pd.DataFrame, scores2: pd.DataFrame, context1: pd.DataFrame, context2: pd.DataFrame,
                         edge_metric: Optional[str] = None, node_metric: Optional[str] = None,
                         stc_test: str = 'mwu', max_path_length: int = 2, correction: str = 'bh',
                         path: Optional[str] = None, format: str = 'csv') -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Computation of a differential network defined by a node metric and an edge metric.
    
    :param scores1: Statistical association scores of Context 1, rescaled and potentially filtered.
    :param scores2: Statistical association scores of Context 2, rescaled and potentially filtered.
    :param context1: Observed data of Context 1, potentially filtered.
    :param context2: Observed data of Context 2, potentially filtered.
    :param edge_metric: Edge metric used to construct the differential network.
    :param node_metric: Node metric used to construct the differential network.
    :param stc_test: Statistical test to use for significance testing in STC node metric. Defaults to 'mwu'.
    :param max_path_length: Maximum length of paths to consider in the computation of integrated interaction scores. Defaults to 2.
    :param correction: Correction method for multiple testing. Defaults to 'bh'.
    :param path: Optional path to save the differential scores as CSV files. Defaults to None.
    :param format: File format to save the differential network. Options are 'csv' and 'graphml'. Defaults to 'csv'.
    :return: A tuple (edges_diff, nodes_diff) containing the computed differential edges and nodes.
    """
    if edge_metric is None and node_metric is None:
        raise ValueError('Please provide at least one of edge_metric or node_metric to compute the differential network.')
    edges_diff = None
    nodes_diff = None

    # Rescaling
    if not 'pre-E' in scores1.columns or not 'pre-E' in scores2.columns:
        scores1, scores2 = pre_rescaling(scores1=scores1, scores2=scores2, metric='pre-E') 
    if not 'pre-P' in scores1.columns or not 'pre-P' in scores2.columns:
        scores1, scores2 = pre_rescaling(scores1=scores1, scores2=scores2, metric='pre-P')

    # Edges
    if edge_metric is not None:
        edges_diff = _compute_diff_edges(scores1=scores1, scores2=scores2, edge_metric=edge_metric, max_path_length=max_path_length)

    # Nodes
    if node_metric is not None:
        nodes_diff = _compute_diff_nodes(context1=context1, context2=context2, scores1=scores1, scores2=scores2,
                                         node_metric=node_metric, correction=correction, stc_test=stc_test)

    if path is not None:
        if format == 'csv':
            if edges_diff is not None:
                file_path_edges = os.path.join(path, f'diff_edges.csv')
                edges_diff.to_csv(file_path_edges)
            if nodes_diff is not None:
                file_path_nodes = os.path.join(path, f'diff_nodes.csv')
                nodes_diff.to_csv(file_path_nodes)

        elif format == 'graphml':
            if edges_diff is None:
                raise ValueError("To save the differential network in 'graphml' format, please provide an 'edge_metric'.")
            file_path = os.path.join(path, f'diff_net.graphml')
            diff_net = nx.from_pandas_edgelist(edges_diff, 'label1', 'label2', edge_metric)

            # Add node weights
            if nodes_diff is not None:
                nx.set_node_attributes(diff_net, nodes_diff[node_metric].to_dict(), node_metric)

            nx.write_graphml(diff_net, file_path)

        else:
            raise ValueError(f"Invalid format {format}. Choose from 'csv' or 'graphml'.")

    if edge_metric is not None and edges_diff is not None:
        if 'post' in edge_metric:
            edges_diff = edges_diff[['label1', 'label2', 'test_type', edge_metric]]
        else:
            edge_metric_signed = edge_metric + '_signed'
            edges_diff = edges_diff[['label1', 'label2', 'test_type', edge_metric, edge_metric_signed]]
    
    if node_metric is not None and nodes_diff is not None:
        nodes_diff = nodes_diff[[node_metric]]

    return edges_diff, nodes_diff 


# Adjusted DrDimont implementation to compute integrated interaction scores
def calculate_interaction_score(data, max_path_length=3, metric='pre-E'):
    if max_path_length >= 5:
        raise ValueError('The maximum path length considered in interaction scores has to be smaller than 5.')
    data_extended = data.copy()

    # Create network
    # Nodes
    graph = ig.Graph()
    nodes = pd.unique(data[['label1', 'label2']].values.ravel())
    graph.add_vertices(nodes)
    # Edges
    edges = list(zip(data['label1'], data['label2']))
    graph.add_edges(edges)
    # Edge weights
    graph.es[metric] = data[metric].tolist()

    # Loop through all edges
    for idx, row in data.iterrows():
        # Extract edge
        edge = (row['label1'], row['label2'])

        # Initialize variables
        sums_of_weight_products = [0] * max_path_length
        num_paths = [0] * max_path_length

        # Find all paths between the two extracted nodes and loop through them
        simple_paths = graph.get_all_simple_paths(edge[0], edge[1], maxlen=max_path_length)
        for path in simple_paths:
            # Get path length
            path_length = len(path) - 1
            # Add the product of the weights along the path to the corresponding sum 
            sums_of_weight_products[path_length - 1] += np.prod([graph.es.find(_source=path[i], _target=path[i+1])[metric] for i in range(path_length)])
            # Increase the count of paths of this length
            num_paths[path_length - 1] += 1

        sums_of_weight_products = np.array(sums_of_weight_products, dtype=float)
        num_paths = np.array(num_paths, dtype=float)

        # Normalize the sums by the number of paths of the corresponding length
        normalized_sums = np.divide(sums_of_weight_products, num_paths, out=np.zeros_like(num_paths), where=num_paths != 0)

        # Sum up the normalized sums and save as interaction score
        edge_score = np.sum(normalized_sums)
        data_extended.loc[idx, 'int-IS'] = edge_score

    return data_extended


# Calculate differential (weighted) degree centralities
def calculate_degree_centrality(nodes_diff, scores1, scores2, metric='pre-P', weighted=False):
    if weighted:
        if metric == 'pre-P':
            scores1['pre-P_inverted'] = 1 - scores1[metric]
            scores2['pre-P_inverted'] = 1 - scores2[metric]
            metric = 'pre-P_inverted'
            method = 'WDC-P'
        elif metric == 'pre-E':
            method = 'WDC-E'
        else:
            raise ValueError(f"Invalid metric '{metric}' for weighted degree centrality.")
    else:
        if metric == 'pre-P':
            method = 'DC-P'
        elif metric == 'pre-E':
            method = 'DC-E'
        else:
            raise ValueError(f"Invalid metric '{metric}' for degree centrality.")

    nodes_diff = nodes_diff.copy()
    nodes = nodes_diff.index
    degree_centrality = pd.DataFrame()
    degree_centrality['labels'] = nodes
    degree_centrality.set_index('labels', inplace=True)
    degree_centrality['context_a'] = np.nan
    degree_centrality['context_b'] = np.nan

    # TODO: optimize implementation according to DimontRank implementation using defaultdict
    if weighted:
        # Sum up all weights of adjacent edges
        for node in nodes:
            sum1 = 0
            sum2 = 0
            for i in scores1.index:
                if (scores1.loc[i, 'label1'] == node) or (scores1.loc[i, 'label2'] == node):
                    sum1 += scores1.loc[i, metric]

            for i in scores2.index:
                if (scores2.loc[i, 'label1'] == node) or (scores2.loc[i, 'label2'] == node):
                    sum2 += scores2.loc[i, metric]

            degree_centrality.loc[node, 'context_a'] = sum1
            degree_centrality.loc[node, 'context_b'] = sum2

    else:
        for node in nodes:
            count1 = 0
            count2 = 0

            for i in scores1.index:
                if (scores1.loc[i, 'label1'] == node) or (scores1.loc[i, 'label2'] == node):
                    if metric == 'pre-P':
                        if scores1.loc[i, metric] != 1:
                            count1 += 1
                    elif metric == 'pre-E':
                        if scores1.loc[i, metric] != 0:
                            count1 += 1
                    else:
                        raise ValueError(f"Invalid metric '{metric}' for degree centrality.")

            for i in scores2.index:
                if (scores2.loc[i, 'label1'] == node) or (scores2.loc[i, 'label2'] == node):
                    if metric == 'pre-P':
                        if scores2.loc[i, metric] != 1:
                            count2 += 1
                    elif metric == 'pre-E':
                        if scores2.loc[i, metric] != 0:
                            count2 += 1
                    else:
                        raise ValueError(f"Invalid metric '{metric}' for degree centrality.")

            degree_centrality.loc[node, 'context_a'] = count1
            degree_centrality.loc[node, 'context_b'] = count2

    # Normalize by max
    max1 = max(degree_centrality['context_a'])
    max2 = max(degree_centrality['context_b'])

    if max1 == 0.:
        max1 = 1.
    if max2 == 0.:
        max2 = 1.

    # (Absolute) difference
    #method_signed = method + '_signed'
    #nodes_diff[method_signed] = (degree_centrality['context_a'] / max1) - (degree_centrality['context_b'] / max2)
    nodes_diff[method] = abs((degree_centrality['context_a'] / max1) - (degree_centrality['context_b'] / max2))

    return nodes_diff


# Compute differential PageRank centrality
def calculate_pagerank_centrality(nodes_diff, scores1, scores2, metric='pre-E'):
    scores1 = scores1.copy()
    scores2 = scores2.copy()

    # Inverse weights in the case of p-values and set edge 'weight' column
    if metric == 'pre-P':
        method = 'PRC-P'
        scores1['weight'] = 1 - scores1[metric]
        scores2['weight'] = 1 - scores2[metric]
    elif metric == 'pre-E':
        method = 'PRC-E'
        scores1['weight'] = scores1[metric]
        scores2['weight'] = scores2[metric]
    else:
        raise ValueError(f"Invalid metric '{metric}' for PageRank centrality.")

    # Create network and apply pagerank algorithm
    network1 = nx.from_pandas_edgelist(scores1, 'label1', 'label2', 'weight')
    network2 = nx.from_pandas_edgelist(scores2, 'label1', 'label2', 'weight')
    ranking1 = nx.pagerank(network1)
    ranking2 = nx.pagerank(network2)

    # Normalize by max
    max1 = max(ranking1.values())
    max2 = max(ranking2.values())

    nodes_diff[method] = nodes_diff.index.map(
        lambda node: abs((ranking1.get(node, 0) / max1) - (ranking2.get(node, 0) / max2))
    )

    return nodes_diff


# Differential edge computation
def _compute_diff_edges(scores1: pd.DataFrame, scores2: pd.DataFrame, edge_metric: str , max_path_length: int = 2) -> pd.DataFrame:
    """
    Compute differential edge scores based on the specified edge metric.

    :param scores1: Statistical association scores of Context 1, rescaled and potentially filtered.
    :param scores2: Statistical association scores of Context 2, rescaled and potentially filtered.
    :param edge_metric: Edge metric to compute the differential edge scores.
    :param max_path_length: Maximum length of paths to consider in the computation of integrated interaction scores. Defaults to 2.
    :return: A DataFrame containing the computed differential edge scores.
    """

    edges_diff = None

    # Pre-rescaled effect size (pre-E) or rescaled multiple-testing adjusted p-value (pre-P)
    if edge_metric == 'pre-P' or edge_metric == 'pre-E':
        pass
    
    # Post-rescaled p-value (post-P)
    elif edge_metric == 'post-P':
        # Compute differences in edge metrics first
        edges_diff = _subtract_edges(scores1, scores2, metrics=['raw-P'], included_cols=['test_type'])
        # Min-Max rescaling
        edges_diff = post_rescaling(diff_scores=edges_diff, metric=edge_metric)

    # Post-rescaled effect size (post-E)
    elif edge_metric == 'post-E':
        # Compute differences in edge metrics first
        edges_diff = _subtract_edges(scores1, scores2, metrics=['raw-E'], included_cols=['test_type'])
        # Min-Max rescaling
        edges_diff = post_rescaling(diff_scores=edges_diff, metric=edge_metric)

    # Pre-rescaled combined score (pre-CS)
    elif edge_metric == 'pre-CS':
        # Compute combined score from rescaled effect size and p-value
        scores1[edge_metric] = scores1['pre-E'] - scores1['pre-P']
        scores2[edge_metric] = scores2['pre-E'] - scores2['pre-P']

    # Post-rescaled combined score (post-CS)
    elif edge_metric == 'post-CS':
        # Compute combined score from raw effect size and p-value
        scores1['raw-CS'] = scores1['raw-E'] - scores1['raw-P']
        scores2['raw-CS'] = scores2['raw-E'] - scores2['raw-P']

        # Compute differences in edge metrics first
        edges_diff = _subtract_edges(scores1, scores2, metrics=['raw-CS'], included_cols=['test_type'])
        # Min-Max rescaling
        edges_diff = post_rescaling(diff_scores=edges_diff, metric=edge_metric)

    # Sum of pre-P and pre-E (pre-PE)
    elif edge_metric == 'pre-PE':
        # Compute 'pre-P' and 'pre-E'
        edges_diff = _subtract_edges(scores1, scores2, metrics=['pre-P', 'pre-E'], included_cols=['test_type'])
        # Sum the two scores
        edges_diff[edge_metric] = edges_diff['pre-P'] + edges_diff['pre-E']

        # Compute the signed version
        edges_diff['pre-PE_signed'] = (scores1['pre-P'] - scores2['pre-P']) + (scores1['pre-E'] - scores2['pre-E'])

    # Sum of post-P and post-E (post-PE)
    elif edge_metric == 'post-PE':
        # Compute differences in raw association scores
        edges_diff = _subtract_edges(scores1, scores2, metrics=['raw-P', 'raw-E'], included_cols=['test_type'])
        # Rescale difference
        edges_diff = post_rescaling(diff_scores=edges_diff, metric='post-E')
        edges_diff = post_rescaling(diff_scores=edges_diff, metric='post-P')
        # Sum the two scores
        edges_diff[edge_metric] = edges_diff['post-P'] + edges_diff['post-E']

    # Integrated Interaction Score (int-IS)
    elif edge_metric == 'int-IS':
        # Compute interaction score using DrDimont method
        scores1 = calculate_interaction_score(scores1, metric='pre-E', max_path_length=max_path_length)
        scores2 = calculate_interaction_score(scores2, metric='pre-E', max_path_length=max_path_length)
    
    # Log-transformed p-value and pre-rescaled effect size combined score (pre-LS)
    elif edge_metric == 'pre-LS':
        # Replace zero values by small epsilon (1/10 of the minimum non-zero value)
        p_vals_combined = np.concatenate([scores1[scores1['pre-P'] > 0]['pre-P'].to_numpy(), 
                                            scores2[scores2['pre-P'] > 0]['pre-P'].to_numpy()])
        min_non_zero = p_vals_combined.min()
        epsilon = min_non_zero / 10.0

        p_vals1 = scores1['pre-P'].to_numpy()
        p_vals2 = scores2['pre-P'].to_numpy()
        p_vals1 = np.where(p_vals1 == 0, epsilon, p_vals1)
        p_vals2 = np.where(p_vals2 == 0, epsilon, p_vals2)

        # - log10(pre-P) * pre-E
        values1 = - np.log10(p_vals1) * scores1['pre-E']
        values2 = - np.log10(p_vals2) * scores2['pre-E']

        # Replace -0.0 with +0.0
        values1 = np.where(values1 == -0.0, 0.0, values1)
        values2 = np.where(values2 == -0.0, 0.0, values2)

        scores1[edge_metric] = values1
        scores2[edge_metric] = values2

    # Post rescaled absolute difference in (log-transformed raw p-value multiplied by raw effect size) (post-LS)
    elif edge_metric == 'post-LS':
        # Replace zero values by small epsilon (1/10 of the minimum non-zero value)
        p_vals_combined = np.concatenate([scores1[scores1['raw-P'] > 0]['raw-P'].to_numpy(), 
                                            scores2[scores2['raw-P'] > 0]['raw-P'].to_numpy()])
        min_non_zero = p_vals_combined.min()
        epsilon = min_non_zero / 10.0

        p_vals1 = scores1['raw-P'].to_numpy()
        p_vals2 = scores2['raw-P'].to_numpy()
        p_vals1 = np.where(p_vals1 == 0, epsilon, p_vals1)
        p_vals2 = np.where(p_vals2 == 0, epsilon, p_vals2)

        # - log10(raw-P) * raw-E
        values1 = - np.log10(p_vals1) * scores1['raw-E']
        values2 = - np.log10(p_vals2) * scores2['raw-E']

        # Replace -0.0 with +0.0
        values1 = np.where(values1 == -0.0, 0.0, values1)
        values2 = np.where(values2 == -0.0, 0.0, values2)

        scores1['raw-LS'] = values1
        scores2['raw-LS'] = values2

        # Compute differences in edge metrics first
        edges_diff = _subtract_edges(scores1, scores2,
                                    metrics=['raw-LS'], included_cols=['test_type'])
        # Min-Max rescaling
        edges_diff = post_rescaling(diff_scores=edges_diff, metric=edge_metric)

    else:
        raise ValueError(f"Invalid edge metric '{edge_metric}'. Choose from: 'pre-P', 'post-P', 'pre-E', 'post-E', 'pre-PE', 'post-PE', 'pre-CS', 'post-CS', 'pre-LS' or 'post-LS', int-IS'.")

    if edges_diff is None:
        # Compute difference in edge scores
        edges_diff = _subtract_edges(scores1, scores2, metrics=[edge_metric], included_cols=('test_type',))
    
    return edges_diff


# Differential node computation
def _compute_diff_nodes(scores1: pd.DataFrame, scores2: pd.DataFrame, context1: pd.DataFrame, context2: pd.DataFrame,
                        node_metric: str, correction: str = 'bh', stc_test: str = 'mwu') -> pd.DataFrame:
    """
    Compute differential node scores based on the specified node metric.

    :param scores1: Statistical association scores of Context 1, rescaled and potentially filtered.
    :param scores2: Statistical association scores of Context 2, rescaled and potentially filtered.
    :param context1: Observed data of Context 1, potentially filtered.
    :param context2: Observed data of Context 2, potentially filtered.
    :param node_metric: Node metric to compute the differential node scores.
    :param correction: Correction method for multiple testing. Only needed if node_metric is 'STC'. Defaults to 'bh'.
    :param stc_test: Statistical test to use for significance testing in STC node metric. Defaults to 'mwu'.
    :return: A DataFrame containing the computed differential node scores.
    """

    assert context1.columns.equals(context2.columns), 'Context a and b need to have the same structure.'

    nodes_diff = None

    # Statistical test centrality (STC)
    if node_metric == 'STC':
        nodes_diff = _subtract_nodes(context1=context1, context2=context2, test=True, test_type=stc_test, correction=correction)

    # Degree centrality based on pre-P (DC-P)
    elif node_metric == 'DC-P':
        nodes_diff = _subtract_nodes(context1=context1, context2=context2, test=False)
        nodes_diff = calculate_degree_centrality(nodes_diff=nodes_diff, weighted=False,
                                                 scores1=scores1, scores2=scores2,
                                                 metric='pre-P')

    # Degree centrality based on pre-E (DC-E)
    elif node_metric == 'DC-E':
        nodes_diff = _subtract_nodes(context1=context1, context2=context2, test=False)
        nodes_diff = calculate_degree_centrality(nodes_diff=nodes_diff, weighted=False,
                                                 scores1=scores1, scores2=scores2,
                                                 metric='pre-E')

    # Weighted degree centrality based on pre-P (WDC-P)
    elif node_metric == 'WDC-P':
        nodes_diff = _subtract_nodes(context1=context1, context2=context2, test=False)
        nodes_diff = calculate_degree_centrality(nodes_diff=nodes_diff, weighted=True,
                                                 scores1=scores1, scores2=scores2,
                                                 metric='pre-P')

    # Weighted degree centrality based on pre-E (WDC-E)
    elif node_metric == 'WDC-E':
        nodes_diff = _subtract_nodes(context1=context1, context2=context2, test=False)
        nodes_diff = calculate_degree_centrality(nodes_diff=nodes_diff, weighted=True,
                                                 scores1=scores1, scores2=scores2,
                                                 metric='pre-E')

    # PageRank centrality based on pre-E (PRC-E)
    elif node_metric == 'PRC-E':
        nodes_diff = _subtract_nodes(context1=context1, context2=context2, test=False)
        nodes_diff = calculate_pagerank_centrality(nodes_diff=nodes_diff, metric='pre-E',
                                                   scores1=scores1, scores2=scores2)
    
    # PageRank centrality based on pre-P (PRC-P)
    elif node_metric == 'PRC-P':
        nodes_diff = _subtract_nodes(context1=context1, context2=context2, test=False)
        nodes_diff = calculate_pagerank_centrality(nodes_diff=nodes_diff, metric='pre-P',
                                                   scores1=scores1, scores2=scores2)

    else:
        raise ValueError(f"Invalid node metric '{node_metric}'. Choose from: 'STC', 'DC-P', 'DC-E', 'WDC-P', 'WDC-E', 'PRC-P' or 'PRC-E'.")

    return nodes_diff


# Compute absolute differences in edge scores between two contexts
def _subtract_edges(scores1, scores2, metrics, included_cols=None):
    if not scores1['label1'].equals(scores2['label1']) or not scores1['label2'].equals(scores2['label2']):
        raise ValueError('Context a and b need to have the same structure.')

    edges_diff = scores1[['label1', 'label2']].copy()

    if included_cols is not None:
        for column in included_cols:
            edges_diff[column] = scores1[column].copy()

    for met in metrics:
        signed_metric = met + '_signed'
        edges_diff[signed_metric] = scores1[met] - scores2[met]
        edges_diff[met] = abs(scores1[met] - scores2[met])

    return edges_diff


# Compute absolute mean difference and statistical significance for each node between two contexts
def _subtract_nodes(context1, context2, test=True, test_type='ttest', correction='bh'):
    if not context1.columns.equals(context2.columns):
        raise ValueError('Context a and b need to have the same structure.')

    nodes = context1.columns

    # Absolute mean difference
    nodes_diff = pd.DataFrame(abs(context1.mean(axis=0) - context2.mean(axis=0)),
                              columns=['diff_mean'], index=nodes)

    # Perform statistical test if specified
    if test is True:
        for node in nodes:
            if test_type == 'ttest':
                result = sc.ttest_ind(context1[node], context2[node], nan_policy='omit')
            elif test_type == 'mwu':
                result = sc.mannwhitneyu(context1[node], context2[node], nan_policy='omit')
            else:
                raise ValueError(f"Invalid test_type '{test_type}'. Choose from 'ttest' or 'mwu'.")

            nodes_diff.loc[node, 'test_p'] = result.pvalue

        nodes_diff['test_p'] = nodes_diff['test_p'].fillna(1.0)
        nodes_diff['STC'] = sc.false_discovery_control(nodes_diff['test_p'], method=correction)

    return nodes_diff
