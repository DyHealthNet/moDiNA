from modina.statistics_utils import (_df_to_numpy, _separate_types, _neglog10_p, _probit_magnitude, add_score_columns,
                                     fdr_correction, reconcile_flagged_variables, check_correction_scope)
from modina.context_net_inference import _order_categories, TEST_EFFECT_SIZES
from modina.metrics import (EDGE_METRICS, metric_score, check_edge_metric, check_node_metric, check_lp_cap,
                            check_cap_consistency)

import os
from typing import Optional, Tuple
import networkx as nx
import numpy as np
import pandas as pd
import napypi as napy
import logging



# Differential network computation
def compute_diff_network(scores1: pd.DataFrame, scores2: pd.DataFrame, context1: pd.DataFrame, context2: pd.DataFrame,
                         edge_metric: Optional[str] = None, node_metric: Optional[str] = None,
                         lp_cap: Optional[float] = None, correction: str = 'bh', correction_scope: str = 'global', num_workers: int = 1,
                         path: Optional[str] = None, format: str = 'csv',
                         meta_file: Optional[pd.DataFrame] = None, test_type: str = 'nonparametric', nan_value: Optional[int] = None,
                         name1: str = 'context1', name2: str = 'context2') -> Tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None]:
    """
    Computation of a differential network defined by a node metric and an edge metric.
    
    :param scores1: Statistical association scores of Context 1, rescaled and potentially filtered.
    :param scores2: Statistical association scores of Context 2, rescaled and potentially filtered.
    :param context1: Observed data of Context 1, potentially filtered.
    :param context2: Observed data of Context 2, potentially filtered.
    :param edge_metric: Edge metric used to construct the differential network.
    :param node_metric: Node metric used to construct the differential network.
    :param lp_cap: Cap for -log10(p) used by the capped product metrics ('<...>-LPxRE-cap', '<...>-LPxVE-cap'), which are only available when it is set. Other metrics are not affected. Defaults to None.
    :param correction: Correction method for multiple testing. Defaults to 'bh'.
    :param correction_scope: Scope of the multiple-testing correction for STC: 'global' (all nodes in one family) or 'local' (nodes corrected separately per test type used to compare them across contexts). Defaults to 'global'.
    :param num_workers: Number of workers for parallel computation of STC. Defaults to 1.
    :param path: Optional path to save the differential scores as CSV files. Defaults to None.
    :param format: File format to save the differential network. Options are 'csv' and 'graphml'. Defaults to 'csv'.
    :param meta_file: Meta file containing the node types. Only needed for the STC node metrics. Defaults to None.
    :param test_type: Test type to use for continuous nodes in the STC node metrics. Defaults to 'nonparametric'.
    :param nan_value: Numerical value used for NaN values in the context data. If None, an error will be raised if such values are present. Defaults to None.
    :return: A tuple (edges_diff, nodes_diff, edge_node_stats) containing the computed differential edges,
             differential nodes, and per-node statistics over incident edges (None if no edge_metric).
    """
    if edge_metric is None and node_metric is None:
        raise ValueError('Please provide at least one of edge_metric or node_metric to compute the differential network.')
    if edge_metric is not None:
        check_edge_metric(edge_metric)
    if node_metric is not None:
        check_node_metric(node_metric)
    lp_cap = check_lp_cap(lp_cap)
    check_cap_consistency(edge_metric, node_metric, lp_cap)
    edges_diff = None
    nodes_diff = None
    edge_node_stats = None

    assert context1.columns.equals(context2.columns), 'Context data should contain the same columns.'

    # Remove variables flagged (single observed category, or entirely missing) in only one of
    # the two contexts, so scores1/scores2 (and context1/context2) stay aligned on the same
    # node set for the edge/node computations below.
    scores1, scores2, context1, context2, removed_variables = reconcile_flagged_variables(scores1, scores2, context1, context2, path=path)

    # Edges
    if edge_metric is not None:
        edges_diff, edge_node_stats = compute_diff_edges(scores1=scores1, scores2=scores2, edge_metric=edge_metric, lp_cap=lp_cap, name1=name1, name2=name2)
        # scores1/scores2 are already reconciled above, so compute_diff_edges' own (redundant)
        # reconciliation pass finds nothing left to remove -- overwrite with the list found here,
        # which reflects what was actually removed for this differential network.
        edges_diff.attrs['removed_variables'] = removed_variables

    # Nodes
    if node_metric is not None:
        nodes_diff = compute_diff_nodes(context1=context1, context2=context2, scores1=scores1, scores2=scores2,
                                         node_metric=node_metric, lp_cap=lp_cap, correction=correction, correction_scope=correction_scope, meta_file=meta_file, test_type=test_type, nan_value=nan_value, num_workers=num_workers)
        nodes_diff.attrs['removed_variables'] = removed_variables

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


# Per-node weighted degree in one context: sum of |score| over the incident edges
def _weighted_degree(scores, col, nodes):
    w = scores[col].abs().to_numpy(dtype=float)
    long = pd.DataFrame({'node': np.concatenate([scores['label1'].to_numpy(), scores['label2'].to_numpy()]),
                         'w': np.concatenate([w, w])})
    return long.groupby('node')['w'].sum().reindex(nodes, fill_value=0.0)


# Calculate differential weighted degree centralities (WDC-<score>)
def degree_centrality(nodes_diff, scores1, scores2, metric='WDC-P', lp_cap: Optional[float] = None):
    if not metric.startswith('WDC-'):
        raise ValueError(f"Invalid metric '{metric}' for weighted degree centrality.")
    score = metric_score(metric)
    scores1, scores2 = add_score_columns(scores1, scores2, lp_cap=lp_cap)

    nodes_diff = nodes_diff.copy()
    degree1 = _weighted_degree(scores1, score, nodes_diff.index)
    degree2 = _weighted_degree(scores2, score, nodes_diff.index)

    # Normalize by max
    max1 = degree1.max() if len(degree1) and degree1.max() > 0 else 1.0
    max2 = degree2.max() if len(degree2) and degree2.max() > 0 else 1.0

    # Absolute difference
    nodes_diff[metric] = (degree1 / max1 - degree2 / max2).abs()

    return nodes_diff


# Compute differential PageRank centrality (PRC-<score>)
def pagerank_centrality(nodes_diff, scores1, scores2, metric='PRC-P', lp_cap: Optional[float] = None):
    if not metric.startswith('PRC-'):
        raise ValueError(f"Invalid metric '{metric}' for PageRank centrality.")
    score = metric_score(metric)
    scores1, scores2 = add_score_columns(scores1, scores2, lp_cap=lp_cap)

    # Edge weight = |score| (PageRank requires non-negative weights; higher = stronger)
    scores1['weight'] = scores1[score].abs()
    scores2['weight'] = scores2[score].abs()

    # Create network and apply pagerank algorithm
    network1 = nx.from_pandas_edgelist(scores1, 'label1', 'label2', 'weight')
    network2 = nx.from_pandas_edgelist(scores2, 'label1', 'label2', 'weight')
    ranking1 = nx.pagerank(network1)
    ranking2 = nx.pagerank(network2)

    # Normalize by max
    max1 = max(ranking1.values(), default=0.0) or 1.0
    max2 = max(ranking2.values(), default=0.0) or 1.0

    nodes_diff[metric] = nodes_diff.index.map(
        lambda node: abs((ranking1.get(node, 0) / max1) - (ranking2.get(node, 0) / max2))
    )

    return nodes_diff


# Statistical test centrality (STC-<score>): test for each node whether its distribution differs between
# the two contexts and derive all node scores from the test's adjusted p-value and effect size.
def stat_test_centrality(context1, context2, meta_file, test_type='nonparametric', correction='bh', nan_value: Optional[int] = None,
                         num_workers: int = 1, correction_scope: str = 'global', lp_cap: Optional[float] = None):
    check_correction_scope(correction_scope)
    lp_cap = check_lp_cap(lp_cap)
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
    nodes_diff['test_E'] = 0.0
    nodes_diff['test_type'] = None

    # Unadjusted p-values, effect sizes and the test used, per node
    p_vals = {}
    e_vals = {}
    node_test = {}

    # Separate data types
    ord1, nom1, cont1, bi1 = _separate_types(context1, meta_file)
    ord2, nom2, cont2, bi2 = _separate_types(context2, meta_file)

    assert nom1.columns.equals(nom2.columns) and ord1.columns.equals(ord2.columns) and cont1.columns.equals(cont2.columns) and bi1.columns.equals(bi2.columns), 'Context a and b need to have the same structure.'

    # Request the unadjusted p-value from napy; a single FDR correction across all nodes (all
    # data types pooled into one family) is applied below, rather than napy's per-type correction.
    return_p = 'p_unadjusted'

    def _collect(colnames, p, e, test):
        p_vals.update(zip(colnames, np.asarray(p, dtype=float).tolist()))
        e_vals.update(zip(colnames, np.asarray(e, dtype=float).tolist()))
        node_test.update(dict.fromkeys(colnames, test))

    # nominal and binary: chi-squared test of the variable against the context label (Cramér's V)
    for cat1, cat2 in ((nom1, nom2), (bi1, bi2)):
        if cat1.shape[1] > 0:
            return_e = TEST_EFFECT_SIZES['chi2']
            combined = pd.concat([cat1, cat2], axis=0)
            combined = _order_categories(combined, nan_value=nan_value)
            combined['context'] = [0] * len(cat1) + [1] * len(cat2)
            combined, colnames = _df_to_numpy(combined)

            result = napy.chi_squared(combined, axis=1, threads=num_workers, use_numba=False, return_types=[return_p, return_e], nan_value=nan_value)
            p = pd.DataFrame(result[return_p], index=colnames, columns=colnames)['context'].drop('context')
            e = pd.DataFrame(result[return_e], index=colnames, columns=colnames)['context'].drop('context')
            _collect(p.index, p.to_numpy(), e.reindex(p.index).to_numpy(), 'chi2')

    # ordinal: Mann-Whitney U test between the contexts (rank-biserial correlation)
    if ord1.shape[1] > 0:
        return_e = TEST_EFFECT_SIZES['mwu']
        context_info = pd.DataFrame({'context': [0] * len(ord1) + [1] * len(ord2)})
        context_info, _ = _df_to_numpy(context_info)
        combined = pd.concat([ord1, ord2], axis=0)
        combined, colnames = _df_to_numpy(combined)
        
        result = napy.mwu(bin_data = context_info, cont_data = combined, axis = 1, threads = num_workers, return_types = [return_p, return_e], use_numba=False, nan_value=nan_value)
        _collect(colnames, result[return_p][0], result[return_e][0], 'mwu')

    # continuous: t-test (Cohen's d, converted to point-biserial r) or Mann-Whitney U test (rank-biserial correlation)
    if cont1.shape[1] > 0:
        context_info = pd.DataFrame({'context': [0] * len(cont1) + [1] * len(cont2)})
        context_info, _ = _df_to_numpy(context_info)
        combined = pd.concat([cont1, cont2], axis=0)
        combined, colnames = _df_to_numpy(combined)

        if test_type == 'parametric':
            return_e = TEST_EFFECT_SIZES['ttest']
            result = napy.ttest(bin_data = context_info, cont_data = combined, axis = 1, threads = num_workers, use_numba=False, return_types=[return_p, return_e], nan_value=nan_value)
            # NApy returns Cohen's d negated (group0 - group1), see context_net_inference.py; flip it
            # to the usual convention and convert to point-biserial r with the per-node group sizes
            # (NApy's 'n' is the total number of non-missing values only).
            d = -np.asarray(result[return_e][0], dtype=float)
            n1 = (cont1 != nan_value).sum().to_numpy(dtype=float)
            n2 = (cont2 != nan_value).sum().to_numpy(dtype=float)
            with np.errstate(divide='ignore', invalid='ignore'):
                e = np.where((n1 > 0) & (n2 > 0), d / np.sqrt(d ** 2 + (n1 + n2) ** 2 / (n1 * n2)), np.nan)
            _collect(colnames, result[return_p][0], e, 'ttest')
        elif test_type == 'nonparametric':
            return_e = TEST_EFFECT_SIZES['mwu']
            result = napy.mwu(bin_data = context_info, cont_data = combined, axis = 1, threads = num_workers, use_numba=False, return_types = [return_p, return_e], nan_value=nan_value)
            _collect(colnames, result[return_p][0], result[return_e][0], 'mwu')
        else:
            raise ValueError(f"Invalid test type '{test_type}' for continuous nodes. Choose from 'parametric' or 'nonparametric'.")

    # Multiple-testing correction across the tested nodes.
    #   'global': unadjusted p-values from all data types pooled into one family.
    #   'local':  one family per test type used for the node (chi2 for nominal and binary nodes, mwu for
    #             ordinal nodes, ttest or mwu for continuous nodes depending on test_type).
    # Nodes without a test keep the default test_p=1.0, test_E=0.0 and are excluded from the family.
    pvals = pd.Series(p_vals, dtype=float)
    if not pvals.empty:
        node_groups = pvals.index.map(node_test).to_numpy() if correction_scope == 'local' else None
        pvals[:] = fdr_correction(pvals.to_numpy(), method=correction, groups=node_groups)
        nodes_diff.loc[pvals.index, 'test_p'] = pvals
        nodes_diff.loc[pvals.index, 'test_E'] = pd.Series(e_vals, dtype=float)
        nodes_diff.loc[pvals.index, 'test_type'] = pvals.index.map(node_test)

    # In case a test failed and returned NaN, set p-value to 1.0 and effect size to 0.0
    nodes_diff['test_p'] = nodes_diff['test_p'].fillna(1.0)
    nodes_diff['test_E'] = nodes_diff['test_E'].fillna(0.0)

    # Node scores, analogous to the per-context edge scores (see statistics_utils.add_score_columns).
    # The sign of the effect only encodes which context is higher, so the absolute effect is used.
    p = nodes_diff['test_p'].to_numpy(dtype=float)
    abs_e = nodes_diff['test_E'].abs().to_numpy(dtype=float)
    lp = _neglog10_p(p)
    # RE: rank-based probit rescaling of |effect| within each test family (as for edges per test type)
    re = np.zeros(len(nodes_diff))
    tests = nodes_diff['test_type'].to_numpy()
    for test in pd.unique(tests[pd.notna(tests)]):
        idx = tests == test
        re[idx] = _probit_magnitude(abs_e[idx])
    # VE: Cramér's V, rank-biserial and point-biserial correlations are all r-like -> r²
    ve = abs_e ** 2

    nodes_diff['STC-P'] = 1.0 - p
    nodes_diff['STC-LP'] = lp
    nodes_diff['STC-RE'] = re
    nodes_diff['STC-VE'] = ve
    nodes_diff['STC-LPxRE'] = lp * re
    nodes_diff['STC-LPxVE'] = lp * ve
    if lp_cap is not None:
        lp_capped = np.minimum(lp, lp_cap)
        nodes_diff['STC-LPxRE-cap'] = lp_capped * re
        nodes_diff['STC-LPxVE-cap'] = lp_capped * ve

    return nodes_diff


# Differential edge computation
def compute_diff_edges(scores1: pd.DataFrame, scores2: pd.DataFrame, edge_metric: str, lp_cap: Optional[float] = None,
                       path: Optional[str] = None, name1: str = 'context1', name2: str = 'context2') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute differential edge scores based on the specified edge metric.

    Every edge metric is built from per-context scores (see statistics_utils.add_score_columns):
    'diff-<score>' is the absolute between-context difference of one score (P, LP, RE, VE, LPxRE,
    LPxVE, LPxRE-cap or LPxVE-cap), and 'sum-LP-VE' / 'sum-LP-RE' add up the absolute differences of
    LP and VE / RE.

    :param scores1: Statistical association scores of Context 1, rescaled and potentially filtered.
    :param scores2: Statistical association scores of Context 2, rescaled and potentially filtered.
    :param edge_metric: Edge metric to compute the differential edge scores.
    :param lp_cap: Cap for -log10(p) used by the capped product metrics 'diff-LPxRE-cap' and 'diff-LPxVE-cap', which require it. Defaults to None.
    :param path: Optional path to save the differential edge scores as a CSV file. Defaults to None.
    :param name1: Name of Context 1, used to label the per-context ``raw-P``/``raw-E`` columns. Defaults to 'context1'.
    :param name2: Name of Context 2, used to label the per-context ``raw-P``/``raw-E`` columns. Defaults to 'context2'.
    :return: A tuple (edges_diff, edge_node_stats): the computed differential edge scores and a
             node-indexed DataFrame of per-node statistics over incident edges (see edge_node_statistics).
    """
    check_edge_metric(edge_metric)
    lp_cap = check_lp_cap(lp_cap)
    check_cap_consistency(edge_metric, None, lp_cap)

    # Remove variables flagged (single observed category, or entirely missing) in only one of
    # the two contexts, so scores1/scores2 are row-aligned for the subtraction below. This also
    # protects direct callers that skip compute_diff_network (e.g. a direct Python/API call, or
    # the Nextflow pipeline's differential_edge_inference.py, which reads scores1/scores2
    # straight from CSV).
    scores1, scores2, _, _, removed_variables = reconcile_flagged_variables(scores1, scores2)

    # Snapshot the per-context raw p-values and effect sizes. Merged back onto the result below
    # so downstream tools can show both contexts' statistics per edge.
    raw_cols1 = scores1[['label1', 'label2', 'raw-P', 'raw-E']].rename(
        columns={'raw-P': f'raw-P_{name1}', 'raw-E': f'raw-E_{name1}'})
    raw_cols2 = scores2[['label1', 'label2', 'raw-P', 'raw-E']].rename(
        columns={'raw-P': f'raw-P_{name2}', 'raw-E': f'raw-E_{name2}'})

    scores1, scores2 = add_score_columns(scores1, scores2, lp_cap=lp_cap)

    # Sum the absolute differences of the metric's score components
    edges_diff = None
    for score in EDGE_METRICS[edge_metric]:
        part = _subtract_edges(scores1, scores2, values=score, metric=score)
        if edges_diff is None:
            edges_diff = part[['label1', 'label2', 'test_type']].copy()
            edges_diff[edge_metric] = 0.0
        edges_diff[edge_metric] += part[score]

    # Extract only the relevant columns
    edges_diff = edges_diff[['label1', 'label2', 'test_type', edge_metric]]

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

    # Always attach the (possibly empty) removed-variables list, so a direct Python caller can
    # inspect it via edges_diff.attrs['removed_variables'] even without passing 'path'.
    edges_diff.attrs['removed_variables'] = removed_variables

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
                        path: Optional[str] = None, correction_scope: str = 'global', lp_cap: Optional[float] = None) -> pd.DataFrame:
    """
    Compute differential node scores based on the specified node metric.

    Node metrics are '<family>-<score>' with family WDC (weighted degree centrality), PRC (PageRank
    centrality) or STC (statistical test centrality) and score P, LP, RE, VE, LPxRE, LPxVE,
    LPxRE-cap or LPxVE-cap (see statistics_utils.add_score_columns). WDC and PRC use the score as
    edge weight in each context and take the absolute difference of the max-normalised
    centralities; STC tests each node between the contexts and derives the score from the test's
    adjusted p-value and effect size.

    :param scores1: Statistical association scores of Context 1, rescaled and potentially filtered.
    :param scores2: Statistical association scores of Context 2, rescaled and potentially filtered.
    :param context1: Observed data of Context 1, potentially filtered.
    :param context2: Observed data of Context 2, potentially filtered.
    :param node_metric: Node metric to compute the differential node scores.
    :param correction: Correction method for multiple testing. Only needed for the STC node metrics. Defaults to 'bh'.
    :param meta_file: Meta file containing the node types. Only needed for the STC node metrics. Defaults to None.
    :param test_type: Test type to compare continuous variables across contexts for the STC node metrics. Defaults to 'nonparametric'.
    :param nan_value: Numerical value used for NaN values in the context data. If None, an error will be raised if such values are present. Defaults to None.
    :param num_workers: Number of workers for parallel computation of STC. Only needed for the STC node metrics. Defaults to 1.
    :param path: Optional path to save the differential node scores as a CSV file. Defaults to None.
    :param correction_scope: Scope of the multiple-testing correction for STC: 'global' (all nodes in one family) or 'local' (nodes corrected separately per test type used to compare them across contexts). Defaults to 'global'.
    :param lp_cap: Cap for -log10(p) used by the capped product metrics ('<family>-LPxRE-cap', '<family>-LPxVE-cap'), which require it. Defaults to None.
    :return: A DataFrame containing the computed differential node scores.
    """
    check_node_metric(node_metric)
    lp_cap = check_lp_cap(lp_cap)
    check_cap_consistency(None, node_metric, lp_cap)
    assert context1.columns.equals(context2.columns), 'Context a and b need to have the same structure.'

    # Remove variables flagged (single observed category, or entirely missing) in only one of
    # the two contexts, so scores1/scores2 don't disagree on which nodes exist. This also
    # protects direct callers that skip compute_diff_network (e.g. a direct Python/API call, or
    # the Nextflow pipeline's differential_node_inference.py, which reads scores1/scores2
    # straight from CSV).
    scores1, scores2, context1, context2, removed_variables = reconcile_flagged_variables(scores1, scores2, context1, context2)

    # Keep only the variables that are present in the scores dataframes
    vars = pd.concat([scores1['label1'], scores1['label2'], scores2['label1'], scores2['label2']]).unique()
    context1 = context1[context1.columns.intersection(vars)]
    context2 = context2[context2.columns.intersection(vars)]

    # Prepare nodes_diff DataFrame
    nodes = context1.columns
    nodes_diff = pd.DataFrame(index=nodes)

    family = node_metric.split('-', 1)[0]

    # Statistical test centrality (STC-<score>)
    if family == 'STC':
        if meta_file is None:
            raise ValueError(f"To compute the '{node_metric}' node metric, please provide a 'meta_file' containing the node types.")
        nodes_diff = stat_test_centrality(context1=context1, context2=context2, correction=correction, correction_scope=correction_scope, meta_file=meta_file, test_type=test_type, nan_value=nan_value, num_workers=num_workers, lp_cap=lp_cap)

    # Weighted degree centrality (WDC-<score>)
    elif family == 'WDC':
        nodes_diff = degree_centrality(nodes_diff=nodes_diff, scores1=scores1, scores2=scores2, metric=node_metric, lp_cap=lp_cap)

    # PageRank centrality (PRC-<score>)
    else:
        nodes_diff = pagerank_centrality(nodes_diff=nodes_diff, scores1=scores1, scores2=scores2, metric=node_metric, lp_cap=lp_cap)

    # Extract only the relevant columns (eliminate intermediary columns used for computation)
    nodes_diff = nodes_diff[[node_metric]]

    if path is not None:
        nodes_diff.to_csv(path)

    # Always attach the (possibly empty) removed-variables list, so a direct Python caller can
    # inspect it via nodes_diff.attrs['removed_variables'] even without passing 'path'.
    nodes_diff.attrs['removed_variables'] = removed_variables

    return nodes_diff


# Compute absolute differences in edge scores between two contexts
def _subtract_edges(scores1, scores2, values, metric):
    if not scores1['label1'].equals(scores2['label1']) or not scores1['label2'].equals(scores2['label2']):
        raise ValueError('Context a and b need to have the same structure.')

    edges_diff = scores1[['label1', 'label2', 'test_type']].copy()
    edges_diff[metric] = abs(scores1[values] - scores2[values])

    return edges_diff

