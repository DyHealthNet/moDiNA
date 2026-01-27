import numpy as np
import pandas as pd
import scipy.stats as sc
import networkx as nx
from collections import defaultdict
import igraph as ig


# --- Differential network computation ---
# Compute absolute differences in edge scores between two contexts
def subtract_edges(scores1, scores2, metrics, included_cols=None):
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
def subtract_nodes(context1, context2, test=True, test_type='ttest', correction='bh'):
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


# --- Node and edge metrics ---
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


# Pre-rescaling
def pre_rescaling(scores1, scores2, metric):
    scores1 = scores1.copy()
    scores2 = scores2.copy()

    # Get raw metrics
    if metric == 'pre-E':
        metric_raw = 'raw-E'
    elif metric == 'pre-P':
        metric_raw = 'raw-P'
    else:
        raise ValueError(f"Invalid metric {metric}. Only 'pre-E' and 'pre-P' are supported.")
    
    # Consider both contexts at the same time to make them comparable
    scores1[metric] = np.nan
    scores2[metric] = np.nan

    # Perform rescaling for every test type separately
    if not scores1['test_type'].equals(scores2['test_type']):
        raise ValueError("scores1 and scores2 must have identical 'test_type' columns.")
    test_types = np.unique(scores1['test_type'])
    for test in test_types:
        scores1_filtered = scores1[scores1['test_type'] == test]
        scores2_filtered = scores2[scores2['test_type'] == test]
        values = np.concatenate([scores1_filtered[metric_raw].to_numpy(), scores2_filtered[metric_raw].to_numpy()])

        # Min-Max normalization
        min_val = np.min(values)
        max_val = np.max(values)

        if min_val == max_val:
            rescaled1 = 0
            rescaled2 = 0
        else:
            rescaled1 = (scores1_filtered[metric_raw] - min_val) / (max_val - min_val)
            rescaled2 = (scores2_filtered[metric_raw] - min_val) / (max_val - min_val)

        scores1.loc[scores1['test_type'] == test, metric] = rescaled1
        scores2.loc[scores2['test_type'] == test, metric] = rescaled2

    return scores1, scores2
    

# Post-rescaling
def post_rescaling(diff_scores, metric):
    diff_scores = diff_scores.copy()

    if metric == 'post-LS':
        metric_raw = 'raw-LS'
    elif metric == 'post-E':
        metric_raw = 'raw-E'
    elif metric == 'post-P':
        metric_raw = 'raw-P'
    else:
        raise ValueError(f"Invalid metric '{metric}'. Only 'post-E', 'post-P' and 'post-LS' are supported.")
    
    diff_scores[metric] = np.nan

    # Perform rescaling for every test type separately
    test_types = np.unique(diff_scores['test_type'])
    for test in test_types:
        diff_scores_filtered = diff_scores[diff_scores['test_type'] == test]
        values = diff_scores_filtered[metric_raw].to_numpy()

        # Min-Max normalization
        min_val = np.min(values)
        max_val = np.max(values)

        if min_val == max_val:
            rescaled = 0
        else:
            rescaled = (diff_scores_filtered[metric_raw] - min_val) / (max_val - min_val)

        diff_scores.loc[diff_scores['test_type'] == test, metric] = rescaled

    return diff_scores


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
        simple_paths = graph.get_all_simple_paths(edge[0], edge[1], cutoff=max_path_length)
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


# --- Ranking algorithms ---
# PageRank algorithm
def pagerank(edges_diff, edge_metric, nodes_diff=None, node_metric=None, invert=False, personalization=True):
    edges_diff = edges_diff.copy()
    edges_diff['weight'] = edges_diff[edge_metric]

    # Create network
    network = nx.from_pandas_edgelist(edges_diff, 'label1', 'label2', 'weight')

    if personalization:
        # Add node metric
        if nodes_diff is None or node_metric is None:
            raise ValueError('When personalization should be used, nodes_diff and node_metric must be provided.')
        
        nodes_diff = nodes_diff.copy()
        if invert is True:
            nodes_diff['weight'] = 1 - nodes_diff[node_metric]
        else:
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
        edge_metric = edge_metric + '_signed'

    sums = defaultdict(float)
    counts = defaultdict(int)
    for n1, n2, w in zip(edges_diff['label1'], edges_diff['label2'], edges_diff[edge_metric]):
        sums[n1] += w
        sums[n2] += w
        counts[n1] += 1
        counts[n2] += 1

    return {n: (np.abs(sums[n] / counts[n]) if counts[n] != 0 else 0) for n in sums}
    