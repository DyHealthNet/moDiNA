from modina.statistics_utils import std_rescaling, probit_rescaling, _df_to_numpy, _separate_types
from modina.context_net_inference import _order_categories

import os
from typing import Optional, Tuple
import networkx as nx
import numpy as np
import pandas as pd
import igraph as ig
import napypi as napy
import logging



# Differential network computation
def compute_diff_network(scores1: pd.DataFrame, scores2: pd.DataFrame, context1: pd.DataFrame, context2: pd.DataFrame,
                         edge_metric: Optional[str] = None, node_metric: Optional[str] = None,
                         max_path_length: int = 2, correction: str = 'bh', num_workers: int = 1,
                         path: Optional[str] = None, format: str = 'csv',
                         meta_file: Optional[pd.DataFrame] = None, test_type: str = 'nonparametric', nan_value: Optional[int] = None) -> Tuple[pd.Series | pd.DataFrame | None, pd.Series | pd.DataFrame | None]:
    """
    Computation of a differential network defined by a node metric and an edge metric.
    
    :param scores1: Statistical association scores of Context 1, rescaled and potentially filtered.
    :param scores2: Statistical association scores of Context 2, rescaled and potentially filtered.
    :param context1: Observed data of Context 1, potentially filtered.
    :param context2: Observed data of Context 2, potentially filtered.
    :param edge_metric: Edge metric used to construct the differential network.
    :param node_metric: Node metric used to construct the differential network.
    :param max_path_length: Maximum length of paths to consider in the computation of integrated interaction scores. Defaults to 2.
    :param correction: Correction method for multiple testing. Defaults to 'bh'.
    :param num_workers: Number of workers for parallel computation of STC. Defaults to 1.
    :param path: Optional path to save the differential scores as CSV files. Defaults to None.
    :param format: File format to save the differential network. Options are 'csv' and 'graphml'. Defaults to 'csv'.
    :param meta_file: Meta file containing the node types. Only needed if node_metric is 'STC'. Defaults to None.
    :param test_type: Test type to use for continuous nodes in STC metric. Defaults to 'nonparametric'.
    :param nan_value: Numerical value used for NaN values in the context data. If None, an error will be raised if such values are present. Defaults to None.
    :return: A tuple (edges_diff, nodes_diff) containing the computed differential edges and nodes.
    """
    if edge_metric is None and node_metric is None:
        raise ValueError('Please provide at least one of edge_metric or node_metric to compute the differential network.')
    edges_diff = None
    nodes_diff = None

    # Check for variables with only one observed category
    assert context1.columns.equals(context2.columns), 'Context data should contain the same columns.'
    vars = context1.columns
    for var in vars:
        if pd.concat([context1[var], context2[var]]).nunique() <= 1:
            logging.warning(f'Variable "{var}" has only one observed category in both contexts. It is recommended to remove this variable in future analyses,'
                            f'as it does not provide meaningful information for differential network analysis.')

    # Edges
    if edge_metric is not None:
        edges_diff = compute_diff_edges(scores1=scores1, scores2=scores2, edge_metric=edge_metric, max_path_length=max_path_length)

    # Nodes
    if node_metric is not None:
        nodes_diff = compute_diff_nodes(context1=context1, context2=context2, scores1=scores1, scores2=scores2,
                                         node_metric=node_metric, correction=correction, meta_file=meta_file, test_type=test_type, nan_value=nan_value, num_workers=num_workers)

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

    return edges_diff, nodes_diff 


# Adjusted DrDimont implementation to compute integrated interaction scores
def interaction_score(data, max_path_length=3, metric='std-E'):
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
        data_extended.loc[idx, 'raw-IS'] = edge_score

    return data_extended


# Calculate differential (weighted) degree centralities
def degree_centrality(nodes_diff, scores1, scores2, metric='DC-P'):
    if 'W' in metric:
        if metric == 'WDC-P':
            scores1['inverted-P'] = 1 - scores1['raw-P']
            scores2['inverted-P'] = 1 - scores2['raw-P']
            met = 'inverted-P'
        elif metric == 'WDC-E-std':
            met = 'std-E'
        elif metric == 'WDC-E-probit':
            met = 'probit-E'
        else:
            raise ValueError(f"Invalid metric '{metric}' for weighted degree centrality.")
    else:
        if metric == 'DC-P':
            met = 'raw-P'
        elif metric == 'DC-E-std':
            met = 'std-E'
        elif metric == 'DC-E-probit':
            met = 'probit-E'
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
    if 'W' in metric:
        # Sum up absolute weights of incident edges (abs handles signed metrics like std-E/probit-E)
        for node in nodes:
            sum1 = 0
            sum2 = 0
            for i in scores1.index:
                if (scores1.loc[i, 'label1'] == node) or (scores1.loc[i, 'label2'] == node):
                    sum1 += abs(scores1.loc[i, met])

            for i in scores2.index:
                if (scores2.loc[i, 'label1'] == node) or (scores2.loc[i, 'label2'] == node):
                    sum2 += abs(scores2.loc[i, met])

            degree_centrality.loc[node, 'context_a'] = sum1
            degree_centrality.loc[node, 'context_b'] = sum2

    else:
        for node in nodes:
            count1 = 0
            count2 = 0

            for i in scores1.index:
                if (scores1.loc[i, 'label1'] == node) or (scores1.loc[i, 'label2'] == node):
                    if met == 'raw-P':
                        if scores1.loc[i, met] != 1:
                            count1 += 1
                    elif met in ('std-E', 'probit-E'):
                        if scores1.loc[i, met] != 0:
                            count1 += 1
                    else:
                        raise ValueError(f"Invalid metric '{metric}' for degree centrality.")

            for i in scores2.index:
                if (scores2.loc[i, 'label1'] == node) or (scores2.loc[i, 'label2'] == node):
                    if met == 'raw-P':
                        if scores2.loc[i, met] != 1:
                            count2 += 1
                    elif met in ('std-E', 'probit-E'):
                        if scores2.loc[i, met] != 0:
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

    # Absolute difference
    nodes_diff[metric] = abs((degree_centrality['context_a'] / max1) - (degree_centrality['context_b'] / max2))

    return nodes_diff


# Compute differential PageRank centrality
def pagerank_centrality(nodes_diff, scores1, scores2, metric='PRC-P'):
    scores1 = scores1.copy()
    scores2 = scores2.copy()

    # Inverse weights in the case of p-values and set edge 'weight' column
    if metric == 'PRC-P':
        scores1['weight'] = 1 - scores1['raw-P']
        scores2['weight'] = 1 - scores2['raw-P']
    elif metric == 'PRC-E-std':
        scores1['weight'] = np.abs(scores1['std-E'])
        scores2['weight'] = np.abs(scores2['std-E'])
    elif metric == 'PRC-E-probit':
        scores1['weight'] = np.abs(scores1['probit-E'])
        scores2['weight'] = np.abs(scores2['probit-E'])
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

    nodes_diff[metric] = nodes_diff.index.map(
        lambda node: abs((ranking1.get(node, 0) / max1) - (ranking2.get(node, 0) / max2))
    )

    return nodes_diff


# Compute absolute mean difference and statistical significance for each node between two contexts
def stat_test_centrality(context1, context2, meta_file, test_type='nonparametric', correction='bh', nan_value: Optional[int] = None, num_workers: int = 1):
    if not context1.columns.equals(context2.columns):
        raise ValueError('Context a and b need to have the same structure.')
    
    print("Computing statistical test centrality (STC) for nodes...", flush=True)
    
    # Search for non-numeric and NaN values
    if context1.apply(lambda col: pd.to_numeric(col, errors="coerce").isna()).values.any() > 0 or context2.apply(lambda col: pd.to_numeric(col, errors="coerce").isna()).values.any() > 0:
        if nan_value is not None:
            logging.warning(f'The context data contains non-numeric or NaN values. These will be replaced by the specified nan_value {nan_value}.')

            context1 = context1.apply(pd.to_numeric, errors="coerce")
            context1 = context1.fillna(nan_value)

            context2 = context2.apply(pd.to_numeric, errors="coerce")
            context2 = context2.fillna(nan_value)
        
        else:
            raise ValueError('The context data contains non-numeric or NaN values. Please clean the data and/or specify a nan_value to replace these values.')
    
    else:
        if nan_value is None:
            # Find a value that does not exist in the data use as nan_value for napy
            existing = set(context1.stack().values) | set(context2.stack().values)
            nan_value = -999
            while True:
                if nan_value not in existing:
                    break
                else:
                    nan_value -= 1
            
            logging.warning(f'The context data does not contain any missing values. '
                            f'For statistical tests, the randomly generated value {nan_value} will be used as '
                            f'the NaN replacement, as this value does not occur in the data. '
                            f'If you want to specify a different value, please provide it via the \'nan_value\' argument.')

    # Initialize nodes_diff DataFrame
    nodes = context1.columns
    nodes_diff = pd.DataFrame(index=nodes)
    nodes_diff['test_p'] = 1.0
    nodes_diff['STC'] = 0.0

    # Initialize p-values
    p_nom = {}
    p_ord = {}
    p_bi = {}
    p_cont = {}

    # Separate data types
    ord1, nom1, cont1, bi1 = _separate_types(context1, meta_file)
    ord2, nom2, cont2, bi2 = _separate_types(context2, meta_file)

    assert nom1.columns.equals(nom2.columns) and ord1.columns.equals(ord2.columns) and cont1.columns.equals(cont2.columns) and bi1.columns.equals(bi2.columns), 'Context a and b need to have the same structure.'

    # Determine the return type for p-values based on the correction method
    if correction == 'bh':
        return_p = 'p_benjamini_hb'
    elif correction == 'by':
        return_p = 'p_benjamini_yek'
    else:
        raise ValueError(f"Invalid correction method '{correction}'. Choose from: 'bh' or 'yek'.")

    # nominal
    if nom1.shape[1] > 0:
        combined = pd.concat([nom1, nom2], axis=0)
        combined = _order_categories(combined)
        combined['context'] = [0] * len(nom1) + [1] * len(nom2)
        combined, colnames = _df_to_numpy(combined)
        
        result = napy.chi_squared(combined, axis=1, threads=num_workers, use_numba=False, return_types=[return_p], nan_value=nan_value)[return_p]
        result_df = pd.DataFrame(result, index=colnames, columns=colnames)
        p_nom = result_df['context'].drop('context')
        p_nom = p_nom.to_dict()

    # ordinal
    if ord1.shape[1] > 0:
        context_info = pd.DataFrame({'context': [0] * len(ord1) + [1] * len(ord2)})
        context_info, _ = _df_to_numpy(context_info)
        combined = pd.concat([ord1, ord2], axis=0)
        combined, colnames = _df_to_numpy(combined)
        
        result = napy.mwu(bin_data = context_info, cont_data = combined, axis = 1, threads = num_workers, return_types = [return_p], use_numba=False, nan_value=nan_value)[return_p]
        p_ord = dict(zip(colnames, result[0].tolist()))

    # binary
    if bi1.shape[1] > 0:
        combined = pd.concat([bi1, bi2], axis=0)
        combined = _order_categories(combined)
        combined['context'] = [0] * len(bi1) + [1] * len(bi2)
        combined, colnames = _df_to_numpy(combined)
        
        result = napy.chi_squared(combined, axis=1, threads=num_workers, use_numba=False, return_types=[return_p], nan_value=nan_value)[return_p]
        result_df = pd.DataFrame(result, index=colnames, columns=colnames)
        p_bi = result_df['context'].drop('context')
        p_bi = p_bi.to_dict()

    # continuous
    if cont1.shape[1] > 0:
        context_info = pd.DataFrame({'context': [0] * len(cont1) + [1] * len(cont2)})
        context_info, _ = _df_to_numpy(context_info)
        combined = pd.concat([cont1, cont2], axis=0)
        combined, colnames = _df_to_numpy(combined)

        if test_type == 'parametric':
            result = napy.ttest(bin_data = context_info, cont_data = combined, axis = 1, threads = num_workers, use_numba=False, return_types=[return_p], nan_value=nan_value)[return_p]
        elif test_type == 'nonparametric':
            result = napy.mwu(bin_data = context_info, cont_data = combined, axis = 1, threads = num_workers, use_numba=False, return_types = [return_p], nan_value=nan_value)[return_p]
        else:
            raise ValueError(f"Invalid test type '{test_type}' for continuous nodes. Choose from 'parametric' or 'nonparametric'.")
        
        p_cont = dict(zip(colnames, result[0].tolist()))

    # STC = 1 - p-value
    if p_ord:
        pvals = pd.Series(p_ord)
        nodes_diff.loc[pvals.index, 'test_p'] = pvals
        nodes_diff.loc[pvals.index, 'STC'] = 1 - pvals

    if p_nom:
        pvals = pd.Series(p_nom)
        nodes_diff.loc[pvals.index, 'test_p'] = pvals
        nodes_diff.loc[pvals.index, 'STC'] = 1 - pvals

    if p_bi:
        pvals = pd.Series(p_bi)
        nodes_diff.loc[pvals.index, 'test_p'] = pvals
        nodes_diff.loc[pvals.index, 'STC'] = 1 - pvals

    if p_cont:
        pvals = pd.Series(p_cont)
        nodes_diff.loc[pvals.index, 'test_p'] = pvals
        nodes_diff.loc[pvals.index, 'STC'] = 1 - pvals

    # In case a test failed and returned NaN, set p-value to 1.0
    nodes_diff['test_p'] = nodes_diff['test_p'].fillna(1.0)
    nodes_diff['STC'] = nodes_diff['STC'].fillna(0.0)

    return nodes_diff


# Differential edge computation
def compute_diff_edges(scores1: pd.DataFrame, scores2: pd.DataFrame, edge_metric: str , max_path_length: int = 2,
                       path: Optional[str] = None) -> pd.DataFrame | None | pd.Series:
    """
    Compute differential edge scores based on the specified edge metric.

    :param scores1: Statistical association scores of Context 1, rescaled and potentially filtered.
    :param scores2: Statistical association scores of Context 2, rescaled and potentially filtered.
    :param edge_metric: Edge metric to compute the differential edge scores.
    :param max_path_length: Maximum length of paths to consider in the computation of integrated interaction scores. Defaults to 2.
    :param path: Optional path to save the differential edge scores as a CSV file. Defaults to None.
    :return: A DataFrame containing the computed differential edge scores.
    """

    edges_diff = None

    # Multiple-testing adjusted p-value (diff-P)
    if edge_metric == 'diff-P':
        edges_diff = _subtract_edges(scores1, scores2, values='raw-P', metric=edge_metric)

    # Std-rescaled interaction score (std-int-IS)
    elif edge_metric == 'std-int-IS':
        if 'std-E' not in scores1.columns:
            scores1, scores2 = std_rescaling(scores1, scores2, metric='std-E')
        scores1 = interaction_score(scores1, metric='std-E', max_path_length=max_path_length)
        scores2 = interaction_score(scores2, metric='std-E', max_path_length=max_path_length)
        edges_diff = _subtract_edges(scores1, scores2, values='raw-IS', metric=edge_metric)

    # Probit-rescaled interaction score (probit-int-IS)
    elif edge_metric == 'probit-int-IS':
        if 'probit-E' not in scores1.columns:
            scores1, scores2 = probit_rescaling(scores1, scores2, metric='probit-E')
        scores1 = interaction_score(scores1, metric='probit-E', max_path_length=max_path_length)
        scores2 = interaction_score(scores2, metric='probit-E', max_path_length=max_path_length)
        edges_diff = _subtract_edges(scores1, scores2, values='raw-IS', metric=edge_metric)

    # Probit-rescaled effect size (probit-E)
    elif edge_metric == 'probit-E':
        if 'probit-E' not in scores1.columns:
            scores1, scores2 = probit_rescaling(scores1, scores2, metric='probit-E')
        edges_diff = _subtract_edges(scores1, scores2, values='probit-E', metric=edge_metric)

    # Sum of diff-P and probit-E (probit-PE)
    elif edge_metric == 'probit-PE':
        if 'probit-E' not in scores1.columns:
            scores1, scores2 = probit_rescaling(scores1, scores2, metric='probit-E')
        edges_diff = _subtract_edges(scores1, scores2, values='probit-E', metric='probit-E')
        edges_diff2 = _subtract_edges(scores1, scores2, values='raw-P', metric='diff-P')
        edges_diff[edge_metric] = edges_diff2['diff-P'] + edges_diff['probit-E']
        edges_diff['probit-PE_signed'] = (scores1['raw-P'] - scores2['raw-P']) + \
                                         (scores1['probit-E'] - scores2['probit-E'])

    # Log-transformed p-value and probit-rescaled effect size combined score (probit-LS)
    elif edge_metric == 'probit-LS':
        if 'probit-E' not in scores1.columns:
            scores1, scores2 = probit_rescaling(scores1, scores2, metric='probit-E')

        # Replace zero values by small epsilon
        p_vals_combined = np.concatenate([scores1[scores1['raw-P'] > 0]['raw-P'].to_numpy(),
                                          scores2[scores2['raw-P'] > 0]['raw-P'].to_numpy()])
        min_non_zero = p_vals_combined.min()
        epsilon = min_non_zero / 10.0

        p_vals1 = scores1['raw-P'].to_numpy()
        p_vals2 = scores2['raw-P'].to_numpy()
        p_vals1 = np.where(p_vals1 == 0, epsilon, p_vals1)
        p_vals2 = np.where(p_vals2 == 0, epsilon, p_vals2)

        # - log10(raw-P) * probit-E
        values1 = - np.log10(p_vals1) * scores1['probit-E']
        values2 = - np.log10(p_vals2) * scores2['probit-E']

        values1 = np.where(values1 == -0.0, 0.0, values1)
        values2 = np.where(values2 == -0.0, 0.0, values2)

        scores1['raw-probit-LS'] = values1
        scores2['raw-probit-LS'] = values2

        edges_diff = _subtract_edges(scores1, scores2, values='raw-probit-LS', metric=edge_metric)

    # Std-rescaled effect size (std-E)
    elif edge_metric == 'std-E':
        if 'std-E' not in scores1.columns:
            scores1, scores2 = std_rescaling(scores1, scores2, metric='std-E')
        edges_diff = _subtract_edges(scores1, scores2, values='std-E', metric=edge_metric)

    # Sum of diff-P and std-E (std-PE)
    elif edge_metric == 'std-PE':
        if 'std-E' not in scores1.columns:
            scores1, scores2 = std_rescaling(scores1, scores2, metric='std-E')
        edges_diff = _subtract_edges(scores1, scores2, values='std-E', metric='std-E')
        edges_diff2 = _subtract_edges(scores1, scores2, values='raw-P', metric='diff-P')
        edges_diff[edge_metric] = edges_diff2['diff-P'] + edges_diff['std-E']
        edges_diff['std-PE_signed'] = (scores1['raw-P'] - scores2['raw-P']) + \
                                      (scores1['std-E'] - scores2['std-E'])

    # Log-transformed p-value and std-rescaled effect size (std-LS)
    elif edge_metric == 'std-LS':
        if 'std-E' not in scores1.columns:
            scores1, scores2 = std_rescaling(scores1, scores2, metric='std-E')

        p_vals_combined = np.concatenate([scores1[scores1['raw-P'] > 0]['raw-P'].to_numpy(),
                                          scores2[scores2['raw-P'] > 0]['raw-P'].to_numpy()])
        epsilon = p_vals_combined.min() / 10.0

        p_vals1 = scores1['raw-P'].to_numpy()
        p_vals2 = scores2['raw-P'].to_numpy()
        p_vals1 = np.where(p_vals1 == 0, epsilon, p_vals1)
        p_vals2 = np.where(p_vals2 == 0, epsilon, p_vals2)

        values1 = -np.log10(p_vals1) * scores1['std-E']
        values2 = -np.log10(p_vals2) * scores2['std-E']
        values1 = np.where(values1 == -0.0, 0.0, values1)
        values2 = np.where(values2 == -0.0, 0.0, values2)

        scores1['raw-std-LS'] = values1
        scores2['raw-std-LS'] = values2
        edges_diff = _subtract_edges(scores1, scores2, values='raw-std-LS', metric=edge_metric)

    else:
        raise ValueError(f"Invalid edge metric '{edge_metric}'. Choose from: 'diff-P', 'std-int-IS', 'probit-int-IS', 'probit-E', 'probit-PE', 'probit-LS', 'std-E', 'std-PE', or 'std-LS'.")

    # Extract only the relevant columns (eliminate intermediary columns used for computation)
    edge_metric_signed = edge_metric + '_signed'
    edges_diff = edges_diff[['label1', 'label2', 'test_type', edge_metric, edge_metric_signed]]

    if path is not None:
        edges_diff.to_csv(path)
    
    return edges_diff


# Differential node computation
def compute_diff_nodes(scores1: pd.DataFrame, scores2: pd.DataFrame, context1: pd.DataFrame, context2: pd.DataFrame,
                        node_metric: str, correction: str = 'bh', meta_file: Optional[pd.DataFrame] = None, test_type: str = 'nonparametric', 
                        nan_value: Optional[int] = None, num_workers: int = 1,
                        path: Optional[str] = None) -> pd.DataFrame | None | pd.Series:
    """
    Compute differential node scores based on the specified node metric.

    :param scores1: Statistical association scores of Context 1, rescaled and potentially filtered.
    :param scores2: Statistical association scores of Context 2, rescaled and potentially filtered.
    :param context1: Observed data of Context 1, potentially filtered.
    :param context2: Observed data of Context 2, potentially filtered.
    :param node_metric: Node metric to compute the differential node scores.
    :param correction: Correction method for multiple testing. Only needed if node_metric is 'STC'. Defaults to 'bh'.
    :param meta_file: Meta file containing the node types. Only needed if node_metric is 'STC'. Defaults to None.
    :param test_type: Test type to compare continuous variables across contexts for the 'STC' node metric. Defaults to 'nonparametric'.
    :param nan_value: Numerical value used for NaN values in the context data. If None, an error will be raised if such values are present. Defaults to None.
    :param num_workers: Number of workers for parallel computation of STC. Only needed if node_metric is 'STC'. Defaults to 1.
    :param path: Optional path to save the differential node scores as a CSV file. Defaults to None.
    :return: A DataFrame containing the computed differential node scores.
    """
    assert context1.columns.equals(context2.columns), 'Context a and b need to have the same structure.'

    # Keep only the variables that are present in the scores dataframes
    vars = pd.concat([scores1['label1'], scores1['label2'], scores2['label1'], scores2['label2']]).unique()
    context1 = context1[context1.columns.intersection(vars)]
    context2 = context2[context2.columns.intersection(vars)]

    # Prepare nodes_diff DataFrame
    nodes = context1.columns
    nodes_diff = pd.DataFrame(index=nodes)

    # Statistical test centrality (STC)
    if node_metric == 'STC':
        if meta_file is None:
            raise ValueError("To compute the 'STC' node metric, please provide a 'meta_file' containing the node types.")
        nodes_diff = stat_test_centrality(context1=context1, context2=context2, correction=correction, meta_file=meta_file, test_type=test_type, nan_value=nan_value, num_workers=num_workers)

    # Degree centrality based on raw-P (DC-P)
    elif node_metric == 'DC-P':
        nodes_diff = degree_centrality(nodes_diff=nodes_diff, scores1=scores1, scores2=scores2, metric='DC-P')

    # Degree centrality based on std-E (DC-E-std)
    elif node_metric == 'DC-E-std':
        if 'std-E' not in scores1.columns:
            scores1, scores2 = std_rescaling(scores1, scores2, metric='std-E')
        nodes_diff = degree_centrality(nodes_diff=nodes_diff, scores1=scores1, scores2=scores2, metric='DC-E-std')

    # Degree centrality based on probit-E (DC-E-probit)
    elif node_metric == 'DC-E-probit':
        if 'probit-E' not in scores1.columns:
            scores1, scores2 = probit_rescaling(scores1, scores2, metric='probit-E')
        nodes_diff = degree_centrality(nodes_diff=nodes_diff, scores1=scores1, scores2=scores2, metric='DC-E-probit')

    # Weighted degree centrality based on raw-P (WDC-P)
    elif node_metric == 'WDC-P':
        nodes_diff = degree_centrality(nodes_diff=nodes_diff, scores1=scores1, scores2=scores2, metric='WDC-P')

    # Weighted degree centrality based on std-E (WDC-E-std)
    elif node_metric == 'WDC-E-std':
        if 'std-E' not in scores1.columns:
            scores1, scores2 = std_rescaling(scores1, scores2, metric='std-E')
        nodes_diff = degree_centrality(nodes_diff=nodes_diff, scores1=scores1, scores2=scores2, metric='WDC-E-std')

    # Weighted degree centrality based on probit-E (WDC-E-probit)
    elif node_metric == 'WDC-E-probit':
        if 'probit-E' not in scores1.columns:
            scores1, scores2 = probit_rescaling(scores1, scores2, metric='probit-E')
        nodes_diff = degree_centrality(nodes_diff=nodes_diff, scores1=scores1, scores2=scores2, metric='WDC-E-probit')

    # PageRank centrality based on std-E (PRC-E-std)
    elif node_metric == 'PRC-E-std':
        if 'std-E' not in scores1.columns:
            scores1, scores2 = std_rescaling(scores1, scores2, metric='std-E')
        nodes_diff = pagerank_centrality(nodes_diff=nodes_diff, scores1=scores1, scores2=scores2, metric='PRC-E-std')

    # PageRank centrality based on probit-E (PRC-E-probit)
    elif node_metric == 'PRC-E-probit':
        if 'probit-E' not in scores1.columns:
            scores1, scores2 = probit_rescaling(scores1, scores2, metric='probit-E')
        nodes_diff = pagerank_centrality(nodes_diff=nodes_diff, scores1=scores1, scores2=scores2, metric='PRC-E-probit')

    # PageRank centrality based on raw-P (PRC-P)
    elif node_metric == 'PRC-P':
        nodes_diff = pagerank_centrality(nodes_diff=nodes_diff, scores1=scores1, scores2=scores2, metric='PRC-P')

    else:
        raise ValueError(f"Invalid node metric '{node_metric}'. Choose from: 'STC', 'DC-P', 'DC-E-std', 'DC-E-probit', 'WDC-P', 'WDC-E-std', 'WDC-E-probit', 'PRC-P', 'PRC-E-std', or 'PRC-E-probit'.")
    
    # Extract only the relevant columns (eliminate intermediary columns used for computation)
    nodes_diff = nodes_diff[[node_metric]]

    if path is not None:
        nodes_diff.to_csv(path)

    return nodes_diff


# Compute absolute differences in edge scores between two contexts
def _subtract_edges(scores1, scores2, values, metric):
    if not scores1['label1'].equals(scores2['label1']) or not scores1['label2'].equals(scores2['label2']):
        raise ValueError('Context a and b need to have the same structure.')

    edges_diff = scores1[['label1', 'label2', 'test_type']].copy()

    signed_metric = metric + '_signed'
    edges_diff[signed_metric] = scores1[values] - scores2[values]
    edges_diff[metric] = abs(scores1[values] - scores2[values])

    return edges_diff

