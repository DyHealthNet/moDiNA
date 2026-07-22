from modina.context_net_inference import compute_context_scores
from modina.diff_net_construction import compute_diff_network
from modina.edge_filtering import filter, filter_differential
from modina.ranking import compute_ranking

import json
import logging
import os
from typing import Tuple, Optional
import pandas as pd


# Wrapper function to perform the whole moDiNA pipeline
def diffnet_analysis(context1: pd.DataFrame, context2: pd.DataFrame, meta_file: pd.DataFrame, edge_metric: Optional[str] = None, node_metric: Optional[str] = None, ranking_alg: str = 'PageRank+',
                     filter_method: Optional[str] = None, filter_param: float = 0.0, filter_metric: Optional[str] = None, filter_rule: Optional[str]=None, filter_target: Optional[str] = None, max_path_length: int=2,
                     test_type: str = 'nonparametric', nan_value: Optional[int] = None,
                     correction: str = 'bh', num_workers: int=1,
                     project_path: Optional[str] = None, name1: str = 'context1', name2: str = 'context2'):
    """
    Wrapper function to perform an end-to-end differential network analysis following the moDiNA pipeline.

    :param context1: Observed data of Context 1 (rows: samples, columns: variables).
    :param context2: Observed data of Context 2 (rows: samples, columns: variables).
    :param meta_file: Metadata file containing a 'label' and 'type' column to specify the data type of each variable.
    :param test_type: Type of statistical tests to use for association score calculation. Defaults to 'nonparametric'.
    :param nan_value: Numerical value used for NaN values in the context data. If None, an error will be raised if such values are present. Defaults to None.
    :param correction: Correction method for multiple testing. Defaults to 'bh'.
    :param num_workers: Number of workers for parallel processing. Defaults to 1.
    :param filter_method: Method used for filtering. Defaults to None.
    :param filter_param: Parameter for the specified filtering method. Defaults to 0.0.
    :param filter_metric: Edge metric used for filtering. Only used when filter_target is 'context-specific'. Defaults to None.
    :param filter_rule: Rule to integrate the networks during filtering. Only used when filter_target is 'context-specific'. Defaults to None.
    :param filter_target: Which network to filter. None (default) performs no filtering. 'context-specific' filters the two context networks before building the differential network (using filter_metric and filter_rule); 'differential' filters the differential network on the computed edge_metric (filter_metric and filter_rule are ignored). When filter_target is not None, filter_method must be provided.
    :param edge_metric: Edge metric used to construct the differential network.
    :param node_metric: Node metric used to construct the differential network.
    :param max_path_length: Maximum length of paths to consider in the computation of integrated interaction scores. Defaults to 2.
    :param ranking_alg: Ranking algorithm to compute. Options are 'PageRank+', 'PageRank', 'absDimontRank', 'DimontRank' and 'nodeRank'. Defaults to 'PageRank+'.
    :param name1: Name of Context 1. Used for saving files. Defaults to 'context1'.
    :param name2: Name of Context 2. Used for saving files. Defaults to 'context2'.
    :param project_path: Optional path to save results. Defaults to None.
    :return: A tuple (ranking, edges_diff, nodes_diff, config) containing the computed ranking, differential edges, differential nodes, and configuration parameters.
    """

    if project_path is not None:
        os.makedirs(project_path, exist_ok=True)
        config_path = os.path.join(project_path, 'config.json')
        ranking_path = os.path.join(project_path, f'{ranking_alg}.csv')
    else:
        config_path = None
        ranking_path = None

    # Score calculation
    logging.info('Computing association scores...')
    scores1 = compute_context_scores(context_data=context1, meta_file=meta_file, test_type=test_type, correction=correction,
                                     num_workers=num_workers, path=project_path, name=name1, nan_value=nan_value)
    scores2 = compute_context_scores(context_data=context2, meta_file=meta_file, test_type=test_type, correction=correction,
                                     num_workers=num_workers, path=project_path, name=name2, nan_value=nan_value)
    logging.info('Done.')

    if filter_target is not None:
        if filter_target not in ('context-specific', 'differential'):
            raise ValueError(f"Invalid filter_target '{filter_target}'. Choose from: 'context-specific', 'differential', or None (no filtering).")
        if filter_method is None:
            raise ValueError("Please provide a 'filter_method' when 'filter_target' is set.")

    # Context-specific filtering (before differential network construction)
    if filter_target == 'context-specific':
        logging.info('Edge filtering (context-specific networks)...')
        scores1_filtered, scores2_filtered, context1_filtered, context2_filtered = filter(context1=context1, context2=context2, scores1=scores1, scores2=scores2,
                                                                                            filter_method=filter_method, filter_param=filter_param, filter_metric=filter_metric, filter_rule=filter_rule)
        logging.info('Done.')
    else:
        if filter_target is None:
            logging.warning('The differential network will be computed based on unfiltered data. No filtering will be performed (filter_target is None).')
        scores1_filtered = scores1.copy()
        scores2_filtered = scores2.copy()
        context1_filtered = context1.copy()
        context2_filtered = context2.copy()

    # Differential network computation
    logging.info('Computing differential network...')
    edges_diff, nodes_diff, edge_node_stats = compute_diff_network(scores1=scores1_filtered, scores2=scores2_filtered,
                                                  context1=context1_filtered, context2=context2_filtered,
                                                  edge_metric=edge_metric, node_metric=node_metric,
                                                  max_path_length=max_path_length,
                                                  correction=correction,
                                                  path=project_path, format='csv',
                                                  meta_file=meta_file, test_type=test_type, nan_value=nan_value,
                                                  name1=name1, name2=name2)
    logging.info('Done.')

    # Differential filtering (after differential network construction)
    if filter_target == 'differential':
        if edge_metric is None:
            raise ValueError("Differential filtering (filter_target='differential') requires an 'edge_metric' to filter on.")
        if filter_metric is not None or filter_rule is not None:
            logging.warning("filter_metric and filter_rule are ignored when filter_target is 'differential'; "
                            "filtering always uses the computed edge_metric.")
        logging.info('Edge filtering (differential network)...')
        edges_diff, edge_node_stats = filter_differential(edges_diff=edges_diff, edge_metric=edge_metric,
                                                           filter_method=filter_method, filter_param=filter_param)
        # Overwrite the differential-network files saved by compute_diff_network with the filtered versions
        if project_path is not None:
            edges_path = os.path.join(project_path, 'diff_edges.csv')
            edges_diff.to_csv(edges_path)
            edge_node_stats.to_csv(os.path.join(project_path, 'diff_edges_node_stats.csv'))
        logging.info('Done.')

    # Ranking
    logging.info('Computing ranking...')
    ranking = compute_ranking(edges_diff=edges_diff, nodes_diff=nodes_diff, ranking_alg=ranking_alg,
                              path=ranking_path, meta_file=meta_file, edge_node_stats=edge_node_stats)
    logging.info('Done.')

    # Create config dict
    params = {
        'name1': name1,
        'name2': name2,
        'test_type': test_type,
        'correction':  correction,
        'filter_method': filter_method,
        'filter_metric': filter_metric,
        'filter_rule': filter_rule,
        'filter_param': filter_param,
        'filter_target': filter_target,
        'edge_metric': edge_metric,
        'node_metric': node_metric,
        'ranking_alg': ranking_alg
    }

    if edge_metric == 'int-IS-E':
        params['max_path_length'] = max_path_length

    if config_path is not None:
        with open(config_path, 'w') as f:
            json.dump(params, f, indent=4)

    return ranking, edges_diff, nodes_diff, params