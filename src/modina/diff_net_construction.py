from modina.statistics_utils import probit_rescaling, _df_to_numpy, _separate_types, add_pval_transforms, fdr_correction
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
                         meta_file: Optional[pd.DataFrame] = None, test_type: str = 'nonparametric', nan_value: Optional[int] = None,
                         name1: str = 'context1', name2: str = 'context2') -> Tuple[pd.Series | pd.DataFrame | None, pd.Series | pd.DataFrame | None]:
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
    :return: A tuple (edges_diff, nodes_diff, edge_node_stats) containing the computed differential edges,
             differential nodes, and per-node statistics over incident edges (None if no edge_metric).
    """
    if edge_metric is None and node_metric is None:
        raise ValueError('Please provide at least one of edge_metric or node_metric to compute the differential network.')
    edges_diff = None
    nodes_diff = None
    edge_node_stats = None

    # Check for variables with only one observed category
    assert context1.columns.equals(context2.columns), 'Context data should contain the same columns.'
    vars = context1.columns
    for var in vars:
        if pd.concat([context1[var], context2[var]]).nunique() <= 1:
            logging.warning(f'Variable "{var}" has only one observed category in both contexts. It is recommended to remove this variable in future analyses,'
                            f'as it does not provide meaningful information for differential network analysis.')

    # Edges
    if edge_metric is not None:
        edges_diff, edge_node_stats = compute_diff_edges(scores1=scores1, scores2=scores2, edge_metric=edge_metric, max_path_length=max_path_length, name1=name1, name2=name2)

    # Nodes
    if node_metric is not None:
        nodes_diff = compute_diff_nodes(context1=context1, context2=context2, scores1=scores1, scores2=scores2,
                                         node_metric=node_metric, correction=correction, meta_file=meta_file, test_type=test_type, nan_value=nan_value, num_workers=num_workers)

    if path is not None:
        if format == 'csv':
            if edges_diff is not None:
                file_path_edges = os.path.join(path, f'diff_edges.csv')
                edges_diff.to_csv(file_path_edges)
                edge_node_stats.to_csv(_edge_node_stats_path(file_path_edges))
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

    return edges_diff, nodes_diff, edge_node_stats


# Adjusted DrDimont implementation to compute integrated interaction scores
def interaction_score(data, max_path_length=3, metric='rescaled-E'):
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
            met = 'inv-P'
        elif metric == 'WDC-L-P':
            met = 'log-P'
        elif metric == 'WDC-E':
            met = 'rescaled-E'
        else:
            raise ValueError(f"Invalid metric '{metric}' for weighted degree centrality.")
    else:
        if metric == 'DC-P':
            met = 'raw-P'
        elif metric == 'DC-E':
            met = 'rescaled-E'
        else:
            raise ValueError(f"Invalid metric '{metric}' for degree centrality.")

    # Ensure the p-value transform columns are present (safety for direct-API calls)
    if met in ('inv-P', 'log-P') and met not in scores1.columns:
        scores1 = add_pval_transforms(scores1)
        scores2 = add_pval_transforms(scores2)

    nodes_diff = nodes_diff.copy()
    nodes = nodes_diff.index
    degree_centrality = pd.DataFrame()
    degree_centrality['labels'] = nodes
    degree_centrality.set_index('labels', inplace=True)
    degree_centrality['context_a'] = np.nan
    degree_centrality['context_b'] = np.nan

    # TODO: optimize implementation according to DimontRank implementation using defaultdict
    if 'W' in metric:
        # Sum up absolute weights of incident edges (abs handles signed probit-E)
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
                    elif met == 'rescaled-E':
                        if scores1.loc[i, met] != 0:
                            count1 += 1
                    else:
                        raise ValueError(f"Invalid metric '{metric}' for degree centrality.")

            for i in scores2.index:
                if (scores2.loc[i, 'label1'] == node) or (scores2.loc[i, 'label2'] == node):
                    if met == 'raw-P':
                        if scores2.loc[i, met] != 1:
                            count2 += 1
                    elif met == 'rescaled-E':
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

    # Set the edge 'weight' column as a significance/effect strength (higher = stronger)
    if metric in ('PRC-P', 'PRC-L-P'):
        col = 'inv-P' if metric == 'PRC-P' else 'log-P'
        if col not in scores1.columns:
            scores1 = add_pval_transforms(scores1)
            scores2 = add_pval_transforms(scores2)
        scores1['weight'] = scores1[col]
        scores2['weight'] = scores2[col]
    elif metric == 'PRC-E':
        scores1['weight'] = np.abs(scores1['rescaled-E'])
        scores2['weight'] = np.abs(scores2['rescaled-E'])
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

    # Request the unadjusted p-value from napy; a single FDR correction across all nodes (all
    # data types pooled into one family) is applied below, rather than napy's per-type correction.
    return_p = 'p_unadjusted'

    # nominal
    if nom1.shape[1] > 0:
        combined = pd.concat([nom1, nom2], axis=0)
        combined = _order_categories(combined, nan_value=nan_value)
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
        combined = _order_categories(combined, nan_value=nan_value)
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

    # Pool the unadjusted p-values from all data types into one family and apply a single
    # multiple-testing correction across all tested nodes. STC = 1 - adjusted p-value.
    # Nodes without a test keep the default test_p=1.0, STC=0.0 and are excluded from the family.
    pvals = pd.Series({**p_ord, **p_nom, **p_bi, **p_cont})
    if not pvals.empty:
        pvals[:] = fdr_correction(pvals.to_numpy(), method=correction)
        nodes_diff.loc[pvals.index, 'test_p'] = pvals
        nodes_diff.loc[pvals.index, 'STC'] = 1 - pvals

    # In case a test failed and returned NaN, set p-value to 1.0
    nodes_diff['test_p'] = nodes_diff['test_p'].fillna(1.0)
    nodes_diff['STC'] = nodes_diff['STC'].fillna(0.0)

    return nodes_diff


# Differential edge computation
def compute_diff_edges(scores1: pd.DataFrame, scores2: pd.DataFrame, edge_metric: str , max_path_length: int = 2,
                       path: Optional[str] = None, name1: str = 'context1', name2: str = 'context2') -> pd.DataFrame | None | pd.Series:
    """
    Compute differential edge scores based on the specified edge metric.

    :param scores1: Statistical association scores of Context 1, rescaled and potentially filtered.
    :param scores2: Statistical association scores of Context 2, rescaled and potentially filtered.
    :param edge_metric: Edge metric to compute the differential edge scores.
    :param max_path_length: Maximum length of paths to consider in the computation of integrated interaction scores. Defaults to 2.
    :param path: Optional path to save the differential edge scores as a CSV file. Defaults to None.
    :param name1: Name of Context 1, used to label the per-context ``raw-P``/``raw-E`` columns. Defaults to 'context1'.
    :param name2: Name of Context 2, used to label the per-context ``raw-P``/``raw-E`` columns. Defaults to 'context2'.
    :return: A tuple (edges_diff, edge_node_stats): the computed differential edge scores and a
             node-indexed DataFrame of per-node statistics over incident edges (see edge_node_statistics).
    """

    # Snapshot the per-context raw p-values and effect sizes before any metric-specific
    # reassignment of scores1/scores2 (e.g. int-IS-E). Merged back onto the result below
    # so downstream tools can show both contexts' statistics per edge.
    raw_cols1 = scores1[['label1', 'label2', 'raw-P', 'raw-E']].rename(
        columns={'raw-P': f'raw-P_{name1}', 'raw-E': f'raw-E_{name1}'})
    raw_cols2 = scores2[['label1', 'label2', 'raw-P', 'raw-E']].rename(
        columns={'raw-P': f'raw-P_{name2}', 'raw-E': f'raw-E_{name2}'})

    edges_diff = None

    # Multiple-testing adjusted p-value (diff-P)
    if edge_metric == 'diff-P':
        edges_diff = _subtract_edges(scores1, scores2, values='raw-P', metric=edge_metric)

    # Rescaled interaction score (int-IS-E)
    elif edge_metric == 'int-IS-E':
        if 'rescaled-E' not in scores1.columns:
            scores1, scores2 = probit_rescaling(scores1, scores2, metric='rescaled-E')
        scores1 = interaction_score(scores1, metric='rescaled-E', max_path_length=max_path_length)
        scores2 = interaction_score(scores2, metric='rescaled-E', max_path_length=max_path_length)
        edges_diff = _subtract_edges(scores1, scores2, values='raw-IS', metric=edge_metric)

    # Absolute difference of rescaled effect sizes (diff-E)
    elif edge_metric == 'diff-E':
        if 'rescaled-E' not in scores1.columns:
            scores1, scores2 = probit_rescaling(scores1, scores2, metric='rescaled-E')
        edges_diff = _subtract_edges(scores1, scores2, values='rescaled-E', metric=edge_metric)

    # Sum of diff-P and diff-E (sum-diff-PE) — naive raw-scale baseline (no normalization,
    # so diff-E dominates the larger range). The signed form mixes p-value and effect-size
    # directions; use 'sum-diff-L-PE' for the direction-coherent variant.
    elif edge_metric == 'sum-diff-PE':
        if 'rescaled-E' not in scores1.columns:
            scores1, scores2 = probit_rescaling(scores1, scores2, metric='rescaled-E')
        edges_diff = _subtract_edges(scores1, scores2, values='rescaled-E', metric='diff-E')
        edges_diff2 = _subtract_edges(scores1, scores2, values='raw-P', metric='diff-P')
        edges_diff[edge_metric] = edges_diff2['diff-P'] + edges_diff['diff-E']
        edges_diff['sum-diff-PE_signed'] = (scores1['raw-P'] - scores2['raw-P']) + \
                                           (scores1['rescaled-E'] - scores2['rescaled-E'])

    # Sum of diff-L-P and diff-E (sum-diff-L-PE) — raw-scale baseline. Uses log-P so the
    # signed form is direction-coherent: both terms increase when context 1 is stronger
    # (more significant and larger effect).
    elif edge_metric == 'sum-diff-L-PE':
        if 'rescaled-E' not in scores1.columns:
            scores1, scores2 = probit_rescaling(scores1, scores2, metric='rescaled-E')
        if 'log-P' not in scores1.columns:
            scores1 = add_pval_transforms(scores1)
            scores2 = add_pval_transforms(scores2)
        edges_diff = _subtract_edges(scores1, scores2, values='rescaled-E', metric='diff-E')
        edges_diff2 = _subtract_edges(scores1, scores2, values='log-P', metric='diff-L-P')
        edges_diff[edge_metric] = edges_diff2['diff-L-P'] + edges_diff['diff-E']
        edges_diff[edge_metric + '_signed'] = (scores1['log-P'] - scores2['log-P']) + \
                                              (scores1['rescaled-E'] - scores2['rescaled-E'])

    # Absolute difference of log-p × rescaled effect size (diff-L-PE, formerly LS-PE)
    elif edge_metric == 'diff-L-PE':
        if 'rescaled-E' not in scores1.columns:
            scores1, scores2 = probit_rescaling(scores1, scores2, metric='rescaled-E')
        if 'log-P' not in scores1.columns:
            scores1 = add_pval_transforms(scores1)
            scores2 = add_pval_transforms(scores2)

        # log-P (= -log10(raw-P)) * rescaled-E, per context
        values1 = (scores1['log-P'] * scores1['rescaled-E']).to_numpy()
        values2 = (scores2['log-P'] * scores2['rescaled-E']).to_numpy()
        values1 = np.where(values1 == -0.0, 0.0, values1)
        values2 = np.where(values2 == -0.0, 0.0, values2)
        scores1['raw-L-PE'] = values1
        scores2['raw-L-PE'] = values2

        edges_diff = _subtract_edges(scores1, scores2, values='raw-L-PE', metric=edge_metric)

    # Absolute difference of -log10(raw-P) (diff-L-P)
    elif edge_metric == 'diff-L-P':
        if 'log-P' not in scores1.columns:
            scores1 = add_pval_transforms(scores1)
            scores2 = add_pval_transforms(scores2)
        edges_diff = _subtract_edges(scores1, scores2, values='log-P', metric=edge_metric)

    else:
        raise ValueError(f"Invalid edge metric '{edge_metric}'. Choose from: 'diff-P', 'int-IS-E', 'diff-E', 'sum-diff-PE', 'sum-diff-L-PE', 'diff-L-PE', or 'diff-L-P'.")

    # Extract only the relevant columns (eliminate intermediary columns used for computation)
    edge_metric_signed = edge_metric + '_signed'
    edges_diff = edges_diff[['label1', 'label2', 'test_type', edge_metric, edge_metric_signed]]

    # Attach each context's raw p-value and effect size (matched on the variable pair).
    edges_diff = edges_diff.merge(raw_cols1, on=['label1', 'label2'], how='left')
    edges_diff = edges_diff.merge(raw_cols2, on=['label1', 'label2'], how='left')

    # Compute per-node statistics over the edges incident to each node. These are calculated
    # once here (in the edge-metric calculation) so that rankings using this edge metric can
    # retrieve them directly without recomputing them for every ranking configuration.
    edge_node_stats = edge_node_statistics(edges_diff, edge_metric)

    if path is not None:
        edges_diff.to_csv(path)
        stats_path = _edge_node_stats_path(path)
        edge_node_stats.to_csv(stats_path)

    return edges_diff, edge_node_stats


def _edge_node_stats_path(edges_path: str) -> str:
    """Derive the per-node edge-statistics file path from the edges file path."""
    root, ext = os.path.splitext(edges_path)
    return f'{root}_node_stats{ext or ".csv"}'


# Per-node statistics over incident edges
def edge_node_statistics(edges_diff: pd.DataFrame, edge_metric: str) -> pd.DataFrame:
    """
    Compute summary statistics of the (absolute) edge metric over the edges incident to each node.

    :param edges_diff: Differential edge scores with columns 'label1', 'label2' and edge_metric.
    :param edge_metric: Name of the (absolute) edge metric column.
    :return: A node-indexed DataFrame with columns 'edge-min', 'edge-max', 'edge-median',
             'edge-mean', 'edge-sd' and 'edge-percentile-mean'. sd is the sample standard deviation
             (ddof=1) and is NaN for nodes with a single incident edge. 'edge-percentile-mean' is the
             mean, over a node's incident edges, of each edge's percentile rank within the global
             distribution of all edges (in (0, 1]); it captures how high a node's edges sit relative
             to all edges in the network.
    """
    # Percentile rank of each (unique) edge within the global edge-metric distribution
    edges = edges_diff[['label1', 'label2', edge_metric]].copy()
    edges['__pct'] = edges[edge_metric].rank(pct=True)

    # Stack both endpoints so every incident edge contributes a row per node
    long = pd.concat([
        edges[['label1', edge_metric, '__pct']].rename(columns={'label1': 'node'}),
        edges[['label2', edge_metric, '__pct']].rename(columns={'label2': 'node'}),
    ], ignore_index=True)

    stats = long.groupby('node').agg(
        **{
            'edge-min': (edge_metric, 'min'),
            'edge-max': (edge_metric, 'max'),
            'edge-median': (edge_metric, 'median'),
            'edge-mean': (edge_metric, 'mean'),
            'edge-sd': (edge_metric, 'std'),
            'edge-percentile-mean': ('__pct', 'mean'),
        }
    )
    stats.index.name = 'node'
    return stats


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

    # Degree centrality based on rescaled-E (DC-E)
    elif node_metric == 'DC-E':
        if 'rescaled-E' not in scores1.columns:
            scores1, scores2 = probit_rescaling(scores1, scores2, metric='rescaled-E')
        nodes_diff = degree_centrality(nodes_diff=nodes_diff, scores1=scores1, scores2=scores2, metric='DC-E')

    # Weighted degree centrality based on inv-P = 1 - p (WDC-P)
    elif node_metric == 'WDC-P':
        nodes_diff = degree_centrality(nodes_diff=nodes_diff, scores1=scores1, scores2=scores2, metric='WDC-P')

    # Weighted degree centrality based on log-P = -log10(p) (WDC-L-P)
    elif node_metric == 'WDC-L-P':
        nodes_diff = degree_centrality(nodes_diff=nodes_diff, scores1=scores1, scores2=scores2, metric='WDC-L-P')

    # Weighted degree centrality based on rescaled-E (WDC-E)
    elif node_metric == 'WDC-E':
        if 'rescaled-E' not in scores1.columns:
            scores1, scores2 = probit_rescaling(scores1, scores2, metric='rescaled-E')
        nodes_diff = degree_centrality(nodes_diff=nodes_diff, scores1=scores1, scores2=scores2, metric='WDC-E')

    # PageRank centrality based on rescaled-E (PRC-E)
    elif node_metric == 'PRC-E':
        if 'rescaled-E' not in scores1.columns:
            scores1, scores2 = probit_rescaling(scores1, scores2, metric='rescaled-E')
        nodes_diff = pagerank_centrality(nodes_diff=nodes_diff, scores1=scores1, scores2=scores2, metric='PRC-E')

    # PageRank centrality based on inv-P = 1 - p (PRC-P)
    elif node_metric == 'PRC-P':
        nodes_diff = pagerank_centrality(nodes_diff=nodes_diff, scores1=scores1, scores2=scores2, metric='PRC-P')

    # PageRank centrality based on log-P = -log10(p) (PRC-L-P)
    elif node_metric == 'PRC-L-P':
        nodes_diff = pagerank_centrality(nodes_diff=nodes_diff, scores1=scores1, scores2=scores2, metric='PRC-L-P')

    else:
        raise ValueError(f"Invalid node metric '{node_metric}'. Choose from: 'STC', 'DC-P', 'DC-E', 'WDC-P', 'WDC-L-P', 'WDC-E', 'PRC-P', 'PRC-L-P', or 'PRC-E'.")
    
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

