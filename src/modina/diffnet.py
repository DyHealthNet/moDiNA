import os
import logging
import math
from typing import Optional, Tuple
import json

from modina.score_calculation import *
from modina.statistics_utils import *


# Wrapper function to perform the whole moDiNA pipeline
def diffnet_analysis(context1: pd.DataFrame, context2: pd.DataFrame, meta_file: pd.DataFrame, edge_metric: str = 'pre-LS', node_metric: str = 'STC', ranking_alg: str = 'PageRank+',
                     filter_method: Optional[str] = None, filter_param: float = 0.0, filter_metric: Optional[str] = None, filter_rule: Optional[str]=None,
                     stc_test: str = 'mwu', max_path_length: int=2, dc_metric: str = 'pre-P', 
                     cont_cont: str = 'spearman', bi_cont: str = 'mwu', cont_cat: str = 'kruskal',
                     correction: str = 'bh', nan_value: int = -89, num_workers: int=1,
                     project_path: Optional[str] = None, name1: str = 'context1', name2: str = 'context2') -> Tuple[list, pd.DataFrame, pd.DataFrame, dict]:
    """
    Wrapper function to perform an end-to-end differential network analysis following the moDiNA pipeline.
    
    :param context1: Observed data of Context 1 (rows: samples, columns: variables).
    :param context2: Observed data of Context 2 (rows: samples, columns: variables).
    :param meta_file: Metadata file containing a 'label' and 'type' column to specify the data type of each variable.
    :param cont_cont: Test for continuous-continuous association scores. Defaults to 'spearman'.
    :param bi_cont: Test for categorical-continuous association (binary) scores. Defaults to 'mwu'.
    :param cont_cat: Test for categorical-continuous association (multiple) scores. Defaults to 'kruskal'.
    :param correction: Correction method for multiple testing. Defaults to 'bh'.
    :param nan_value: Value to represent NaN in the data. Defaults to -89.
    :param num_workers: Number of workers for parallel processing. Defaults to 1.
    :param filter_method: Method used for filtering. Defaults to None.
    :param filter_param: Parameter for the specified filtering method. Defaults to 0.0.
    :param filter_metric: Edge metric used for filtering. Defaults to None.
    :param filter_rule: Rule to integrate the networks during filtering. Defaults to None.
    :param edge_metric: Edge metric used to construct the differential network. Defaults to 'pre-LS'.
    :param node_metric: Node metric used to construct the differential network. Defaults to 'STC'.
    :param stc_test: Statistical test to use for significance testing in STC node metric. Defaults to 'mwu'.
    :param max_path_length: Maximum length of paths to consider in the computation of integrated interaction scores. Defaults to 2.
    :param dc_metric: Edge metric used for differential degree centrality computation. Defaults to 'pre-P'.
    :param ranking_alg: Ranking algorithm to compute. Options are 'PageRank+', 'PageRank', 'absDimontRank', 'DimontRank', 'direct_node' and 'direct_edge'. Defaults to 'PageRank+'.
    :param name1: Name of Context 1. Used for saving files. Defaults to 'context1'.
    :param name2: Name of Context 2. Used for saving files. Defaults to 'context2'.
    :param project_path: Optional path to save results. Defaults to None.
    :return: A tuple (ranking, edges_diff, nodes_diff, config) containing the computed ranking, differential edges, differential nodes, and configuration parameters.
    """

    if project_path is not None:
        os.makedirs(project_path, exist_ok=True)
        config_path = os.path.join(project_path, 'config.json')
        scores1_path = os.path.join(project_path, f'{name1}_scores.csv')
        scores2_path = os.path.join(project_path, f'{name2}_scores.csv')
        ranking_path = os.path.join(project_path, 'ranking.csv')
    else:
        scores1_path = None
        config_path = None
        scores2_path = None
        ranking_path = None        

    # Score calculation
    logging.info('Computing association scores...')
    scores1 = compute_context_scores(context_data=context1, meta_file=meta_file, cont_cont=cont_cont,
                                     bi_cont=bi_cont, cont_cat=cont_cat, correction=correction, nan_value=nan_value,
                                     num_workers=num_workers, path=scores1_path)
    scores2 = compute_context_scores(context_data=context2, meta_file=meta_file, cont_cont=cont_cont,
                                     bi_cont=bi_cont, cont_cat=cont_cat, correction=correction, nan_value=nan_value,
                                     num_workers=num_workers, path=scores2_path)
    logging.info('Done.')

    # Filtering
    if any([filter_method, filter_metric, filter_rule, filter_param]):
        logging.info('Edge filtering...')
        scores1_filtered, scores2_filtered, context1_filtered, context2_filtered = filter(context1=context1, context2=context2, scores1=scores1, scores2=scores2,
                                                                                            filter_method=filter_method, filter_param=filter_param, filter_metric=filter_metric, filter_rule=filter_rule)
        logging.info('Done.')
    else:
        logging.warning('The differential network will be computed based on unfiltered data. No filter parameters were specified.')
        scores1_filtered = scores1.copy()
        scores2_filtered = scores2.copy()
        context1_filtered = context1.copy()
        context2_filtered = context2.copy()
    
    # Differential network computation
    logging.info('Computing differential network...')
    edges_diff, nodes_diff = compute_diff_network(scores1=scores1_filtered, scores2=scores2_filtered,
                                                  context1=context1_filtered, context2=context2_filtered,
                                                  edge_metric=edge_metric, node_metric=node_metric,
                                                  stc_test=stc_test, max_path_length=max_path_length,
                                                  correction=correction, dc_metric=dc_metric,
                                                  path=project_path, format='csv')
    logging.info('Done.')

    # Ranking
    logging.info('Computing ranking...')
    ranking = compute_ranking(edges_diff=edges_diff, nodes_diff=nodes_diff, ranking_alg=ranking_alg,
                              edge_metric=edge_metric, node_metric=node_metric,
                              path=ranking_path)
    logging.info('Done.')

    # Create config dict
    params = {
        'name1': name1,
        'name2': name2,
        'cont_cont': cont_cont,
        'bi_cont': bi_cont,
        'cont_cat': cont_cat,
        'correction':  correction,
        'filter_method': filter_method,
        'filter_metric': filter_metric,
        'filter_rule': filter_rule,
        'filter_param': filter_param,
        'edge_metric': edge_metric,
        'node_metric': node_metric,
        'ranking_alg': ranking_alg   
    }

    if node_metric == 'STC':
        params['stc_test'] = stc_test
    if edge_metric == 'int-IS':
        params['max_path_length'] = max_path_length
    if node_metric in ['DC-STC', 'DC']:
        params['dc_metric'] = dc_metric

    if config_path is not None:
        with open(config_path, 'w') as f:
            json.dump(params, f, indent=4)

    return ranking, edges_diff, nodes_diff, params


# Statistical association score computation
def compute_context_scores(context_data: pd.DataFrame, meta_file: pd.DataFrame, 
                           cont_cont: str = 'spearman', bi_cont: str = 'mwu', cont_cat: str = 'kruskal', 
                           correction: str = 'bh', nan_value: int = -89, num_workers: int = 1, 
                           path: Optional[str] = None) -> pd.DataFrame:
    """
    Compute association scores for a given context.
    
    :param context_data: The raw context data (rows: samples, columns: variables).
    :param meta_file: Metadata file containing a 'label' and 'type' column to specify the data type of each variable.
    :param cont_cont: Test for continuous-continuous association scores. Defaults to 'spearman'.
    :param bi_cont: Test for categorical-continuous association (binary) scores. Defaults to 'mwu'.
    :param cont_cat: Test for categorical-continuous association (multiple) scores. Defaults to 'kruskal'.
    :param correction: Correction method for multiple testing. Defaults to 'bh'.
    :param nan_value: Value to represent NaN in the data. Defaults to -89.
    :param num_workers: Number of workers for parallel processing. Defaults to 1.
    :param path: Optional path to save the computed scores as a CSV file. Defaults to None.
    :return: A pd.DataFrame containing the computed association scores.
    """
    # Check if input data is in a valid format
    _check_input_data(context=context_data, meta_file=meta_file)

    # Separate the data into categorical and continuous data
    cat, cont, bi = separate_types(context_data, meta_file)

    # Create dict for test types
    tests = {
        "cont_cont": cont_cont,
        "bi_cont": bi_cont,
        "cont_cat": cont_cat
    }

    # Calculate scores
    scores = calculate_association_scores(cat_data=cat, cont_data=cont, bi_data=bi, tests=tests, num_workers=num_workers, nan_value=nan_value)

    #TODO: make this simpler and change the testing process in score_calculation.py => not necessary to stick to the DyHealthNet code anymore!
    # Take the adjusted p-value and the corresponding effect size
    column_names = scores.iloc[:, 2:].columns
    if correction == 'bh':
        correction = 'benjamini_hb'

    p_adj = '_p_' + correction
    p_columns = [column for column in column_names if p_adj in column]
    e_columns = [column for column in column_names if '_e_' in column]

    scores_final = scores[['label1', 'label2']].copy()
    for i in scores.index:
        for p_type in p_columns:
            if pd.notna(scores.loc[i, p_type]):
                test = p_type.split('_')[0]
                e_type = [column for column in e_columns if test in column][0]

                scores_final.loc[i, 'raw-P'] = scores.loc[i, p_type]
                scores_final.loc[i, 'raw-E'] = scores.loc[i, e_type]
                scores_final.loc[i, 'test_type'] = test
                break
    
    scores_final = scores_final.drop_duplicates(subset=['label1', 'label2', 'test_type'], keep='first')
    scores_final = scores_final.sort_values(by=['label1', 'label2', 'test_type']).reset_index(drop=True)

    # Save scores
    if path is not None:
        scores_final.to_csv(path, index=False)
        
    return scores_final


# Edge filtering
# TODO: filtering on unscaled scores? absolute effect sizes? per effect size type? only p-values? 
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
    :param filter_metric: Edge metric used for filtering. Defaults to None.
    :param filter_rule: Rule to integrate the networks during filtering. Defaults to None.
    :param path: Optional path to save the filtered scores and context data as CSV files. Defaults to None.
    :return: A tuple containing the filtered scores and context data.
    """

    if not scores1['label1'].equals(scores2['label1']) and not scores1['label2'].equals(scores2['label2']):
        raise ValueError('scores1 and scores2 need to have the same structure and order of edges.')
    
    # Rescaling
    scores1, scores2 = pre_rescaling(scores1=scores1, scores2=scores2, metric='pre-P')
    scores1, scores2 = pre_rescaling(scores1=scores1, scores2=scores2, metric='pre-E')
    
    # Check input parameters
    if filter_method is None:
        raise ValueError("Please provide a 'filter_method'.")
    
    if filter_metric is None:
        raise ValueError("Please provide a 'filter_metric'.")
    
    if filter_rule is None:
        raise ValueError("Please provide a 'filter_rule'.")
    
    if filter_param is None:
        raise ValueError("Please provide a 'filter_param'.")

    # Set sorting order based on filter metric
    ascending = True if filter_metric == 'pre-P' else False

    # Compute filtering threshold according to the specified method
    threshold1 = None
    threshold2 = None
    n_nodes = context1.shape[1]
    n_edges_before = scores1.shape[0]

    if filter_method == 'threshold':
        threshold1 = threshold2 = filter_param

    elif filter_method == 'degree':
        degree = filter_param
        n_filtered_edges = math.ceil(degree * n_nodes / 2)

        threshold1 = scores1[filter_metric].sort_values(ascending=ascending).iloc[n_filtered_edges - 1]
        threshold2 = scores2[filter_metric].sort_values(ascending=ascending).iloc[n_filtered_edges - 1]

    elif filter_method == 'density':
        density = filter_param
        possible_edges = n_nodes * (n_nodes - 1) / 2
        n_filtered_edges = math.ceil(density * possible_edges)

        threshold1 = scores1[filter_metric].sort_values(ascending=ascending).iloc[n_filtered_edges - 1]
        threshold2 = scores2[filter_metric].sort_values(ascending=ascending).iloc[n_filtered_edges - 1]

    else:
        raise ValueError(f"Invalid filtering method '{filter_method}'. Choose from: 'threshold', 'degree' or 'density'")
    
    # Apply the filtering threshold to scores and raw data if provided
    if filter_rule == 'union':
        if ascending is True:
            mask = (scores1[filter_metric] <= threshold1) | (
                    scores2[filter_metric] <= threshold2)
        else:
            mask = (scores1[filter_metric] >= threshold1) | (
                    scores2[filter_metric] >= threshold2)
        
        # Apply mask
        scores1_filtered = scores1[mask].copy()
        scores2_filtered = scores2[mask].copy()

    elif filter_rule == 'zero':
        if ascending is True:
            mask1 = scores1[filter_metric] <= threshold1
            mask2 = scores2[filter_metric] <= threshold2
        else:
            mask1 = scores1[filter_metric] >= threshold1
            mask2 = scores2[filter_metric] >= threshold2
        
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
            'pre-P': 1.0,
            'raw-E': 0.0,
            'pre-E': 0.0,
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


# Differential network computation
def compute_diff_network(scores1: pd.DataFrame, scores2: pd.DataFrame, context1: pd.DataFrame, context2: pd.DataFrame,
                         edge_metric: str = 'pre-LS', node_metric: str = 'STC',
                         stc_test: str = 'mwu', max_path_length: int = 2, correction: str = 'bh', dc_metric: str = 'pre-P',
                         path: Optional[str] = None, format: str = 'csv') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Computation of a differential network defined by a node metric and an edge metric.
    
    :param scores1: Statistical association scores of Context 1, rescaled and potentially filtered.
    :param scores2: Statistical association scores of Context 2, rescaled and potentially filtered.
    :param context1: Observed data of Context 1, potentially filtered.
    :param context2: Observed data of Context 2, potentially filtered.
    :param edge_metric: Edge metric used to construct the differential network. Defaults to 'pre-LS'.
    :param node_metric: Node metric used to construct the differential network. Defaults to 'STC'.
    :param stc_test: Statistical test to use for significance testing in STC node metric. Defaults to 'mwu'.
    :param max_path_length: Maximum length of paths to consider in the computation of integrated interaction scores. Defaults to 2.
    :param correction: Correction method for multiple testing. Defaults to 'bh'.
    :param dc_metric: Edge metric used for differential degree centrality computation. Defaults to 'pre-P'.
    :param path: Optional path to save the differential scores as CSV files. Defaults to None.
    :param format: File format to save the differential network. Options are 'csv' and 'graphml'. Defaults to 'csv'.
    :return: A tuple (edges_diff, nodes_diff) containing the computed differential edges and nodes.
    """

    # Rescaling
    if not 'pre-E' in scores1.columns or not 'pre-E' in scores2.columns:
        scores1, scores2 = pre_rescaling(scores1=scores1, scores2=scores2, metric='pre-E') 
    if not 'pre-P' in scores1.columns or not 'pre-P' in scores2.columns:
        scores1, scores2 = pre_rescaling(scores1=scores1, scores2=scores2, metric='pre-P')

    # Edges
    edges_diff = _compute_diff_edges(scores1=scores1, scores2=scores2, edge_metric=edge_metric, max_path_length=max_path_length)

    # Nodes
    nodes_diff = _compute_diff_nodes(context1=context1, context2=context2, scores1=scores1, scores2=scores2,
                                     node_metric=node_metric, correction=correction, stc_test=stc_test, dc_metric=dc_metric)

    if path is not None:
        if format == 'csv':
            file_path_edges = os.path.join(path, f'diff_edges.csv')
            file_path_nodes = os.path.join(path, f'diff_nodes.csv')
            
            edges_diff.to_csv(file_path_edges)
            nodes_diff.to_csv(file_path_nodes)

        elif format == 'graphml':
            file_path = os.path.join(path, f'diff_net.graphml')
            diff_net = nx.from_pandas_edgelist(edges_diff, 'label1', 'label2', edge_metric)
            nx.set_node_attributes(diff_net, nodes_diff[node_metric].to_dict(), node_metric)

            nx.write_graphml(diff_net, file_path)

        else:
            raise ValueError(f"Invalid format {format}. Choose from 'csv' or 'graphml'.")     

    return edges_diff, nodes_diff 


# Ranking
def compute_ranking(nodes_diff: pd.DataFrame, edges_diff: pd.DataFrame, ranking_alg: str, 
                     node_metric: Optional[str] = None, edge_metric: Optional[str] = None, 
                     path : Optional[str] = None) -> list:
    """
    Compute a ranking based on the specified ranking algorithm.
    
    :param nodes_diff: Differential node scores.
    :param edges_diff: Differential edge scores.
    :param ranking_alg: Ranking algorithm to compute. Options are 'PageRank+', 'PageRank', 'absDimontRank', 'DimontRank', 'direct_node' and 'direct_edge'.
    :param node_metric: Node metric used for ranking. Required for 'PageRank+' and 'direct_node' algorithms.
    :param edge_metric: Edge metric used for ranking. Required for 'PageRank+', 'PageRank', 'absDimontRank' and 'DimontRank' algorithms.
    :param path: Optional path to save the ranking as a CSV file.
    """

    # Personalized PageRank
    if ranking_alg == 'PageRank+':
        if node_metric is None or edge_metric is None:
            raise ValueError('Personalized PageRank requires a node and edge metric.')
        
        invert = True if node_metric == 'STC' else False
        ranking_scores = pagerank(nodes_diff=nodes_diff, edges_diff=edges_diff,
                                node_metric=node_metric, edge_metric=edge_metric,
                                invert=invert, personalization=True)
        ranks = pd.Series(ranking_scores).sort_values(ascending=False).index.tolist()

    # PageRank
    elif ranking_alg == 'PageRank':
        if edge_metric is None:
            raise ValueError('PageRank requires an edge metric.')
        ranking_scores = pagerank(edges_diff=edges_diff, edge_metric=edge_metric,
                                personalization=False)
        ranks = pd.Series(ranking_scores).sort_values(ascending=False).index.tolist()

    # DimontRank with absolute difference
    elif ranking_alg == 'absDimontRank':
        if edge_metric is None:
            raise ValueError('absDimontRank requires an edge metric.')
        ranking_scores = dimontrank(edges_diff=edges_diff, edge_metric=edge_metric, mode='abs')
        ranks = pd.Series(ranking_scores).sort_values(ascending=False).index.tolist()

    # DimontRank with signed difference
    elif ranking_alg == 'DimontRank':
        if edge_metric is None:
            raise ValueError('DimontRank requires an edge metric.')
        ranking_scores = dimontrank(edges_diff=edges_diff, edge_metric=edge_metric, mode='signed')
        ranks = pd.Series(ranking_scores).sort_values(ascending=False).index.tolist()

    # Direct ranking based on node metric
    elif ranking_alg == 'direct_node':
        if node_metric is None:
            raise ValueError('Direct ranking requires a node metric.')
        ranking_scores = nodes_diff[node_metric]

        if node_metric == 'STC':
            ranks = pd.Series(ranking_scores).sort_values(ascending=True).index.tolist()
        else:
            ranks = pd.Series(ranking_scores).sort_values(ascending=False).index.tolist()
    
    elif ranking_alg == 'direct_edge':
        if edge_metric is None:
            raise ValueError('Direct edge ranking requires an edge metric.')
        ranking_scores = edges_diff[['label1', 'label2', edge_metric]].copy()
        ranking_scores = ranking_scores.sort_values(by=edge_metric, ascending=False).reset_index(drop=True)
        ranks = ranking_scores[['label1', 'label2']].values.tolist()
    
    else:
        raise ValueError(f"Invalid ranking algorithm {ranking_alg}. "
                         "Choose from: 'PageRank+', 'PageRank', 'absDimontRank', 'DimontRank', 'direct_node' or 'direct_edge'.")
    
    if path is not None:
        ranking_df = pd.DataFrame({"node": ranks, "rank": range(1, len(ranks) + 1)})
        ranking_df = ranking_df.sort_values("node")
        ranking_df.to_csv(path, index=False)

    return ranks


# Differential edge computation
def _compute_diff_edges(scores1: pd.DataFrame, scores2: pd.DataFrame, edge_metric: str = 'pre-P', max_path_length: int = 2) -> pd.DataFrame:
    """
    Compute differential edge scores based on the specified edge metric.

    :param scores1: Statistical association scores of Context 1, rescaled and potentially filtered.
    :param scores2: Statistical association scores of Context 2, rescaled and potentially filtered.
    :param edge_metric: Edge metric to compute the differential edge scores.
    :param max_path_length: Maximum length of paths to consider in the computation of integrated interaction scores. Defaults to 2.
    :return: A DataFrame containing the computed differential edge scores.
    """

    edges_diff = None

    if edge_metric is None:
        # TODO: rethink this warning, maybe moDiNA should also run without computing an edge metric
        logging.warning('No edge_metric was specified. This setting should only be applied for direct node rankings. pre-P will now be computed to enable other rankings.')
        # Take pre-P per default
        edges_diff = subtract_edges(scores1, scores2,
                                            metrics=['pre-P'], included_cols=('test_type',))

    # Pre-rescaled effect size (pre-E) or rescaled multiple-testing adjusted p-value (pre-P)
    if edge_metric == 'pre-P' or edge_metric == 'pre-E':
        pass
    
    # Post-rescaled p-value (post-P)
    elif edge_metric == 'post-P':
        # Compute differences in edge metrics first
        edges_diff = subtract_edges(scores1, scores2, metrics=['raw-P'], included_cols=['test_type'])
        # Min-Max rescaling
        edges_diff, _ = post_rescaling(diff_scores=edges_diff, metric=edge_metric)

    # Post-rescaled effect size (post-E)
    elif edge_metric == 'post-E':
        # Compute differences in edge metrics first
        edges_diff = subtract_edges(scores1, scores2, metrics=['raw-E'], included_cols=['test_type'])
        # Min-Max rescaling
        edges_diff, _ = post_rescaling(diff_scores=edges_diff, metric=edge_metric)

    # Pre-rescaled combined score (pre-CS)
    elif edge_metric == 'pre-CS':
        # Compute combined score from rescaled effect size and p-value
        scores1[edge_metric] = scores1['pre-E'] - scores1['pre-P']
        scores2[edge_metric] = scores2['pre-E'] - scores2['pre-P']

    # Post-rescaled combined score (post-CS)
    elif edge_metric == 'post-CS':
        # Compute differences in edge metrics first
        edges_diff = subtract_edges(scores1, scores2, metrics=['raw-P', 'raw-E'], included_cols=['test_type'])
        # Rescale difference in effect sizes
        edges_diff, _ = post_rescaling(diff_scores=edges_diff, metric='post-E')
        edges_diff, _ = post_rescaling(diff_scores=edges_diff, metric='post-P')
        # Compute combined score
        edges_diff[edge_metric] = (edges_diff['post-E'] + edges_diff['post-P'])

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
        edges_diff = subtract_edges(scores1, scores2,
                                    metrics=['raw-LS'], included_cols=['test_type'])
        # Min-Max rescaling
        edges_diff, _ = post_rescaling(diff_scores=edges_diff, metric=edge_metric)

    else:
        raise ValueError(f"Invalid edge metric '{edge_metric}'. Choose from: 'pre-P', 'pre-E', 'post-E', 'int-IS', 'pre-CS', 'post-CS', 'pre-LS' or 'post-LS'.")

    if edges_diff is None:
        # Compute difference in edge scores
        edges_diff = subtract_edges(scores1, scores2, metrics=[edge_metric], included_cols=('test_type',))
    
    return edges_diff


# Differential node computation
def _compute_diff_nodes(scores1: pd.DataFrame, scores2: pd.DataFrame, context1: pd.DataFrame, context2: pd.DataFrame,
                        node_metric: str, correction: str = 'bh', stc_test: str = 'mwu', dc_metric: str = 'pre-P') -> pd.DataFrame:
    """
    Compute differential node scores based on the specified node metric.

    :param scores1: Statistical association scores of Context 1, rescaled and potentially filtered.
    :param scores2: Statistical association scores of Context 2, rescaled and potentially filtered.
    :param context1: Observed data of Context 1, potentially filtered.
    :param context2: Observed data of Context 2, potentially filtered.
    :param node_metric: Node metric to compute the differential node scores.
    :param correction: Correction method for multiple testing. Only needed if node_metric is 'STC'. Defaults to 'bh'.
    :param stc_test: Statistical test to use for significance testing in STC node metric. Defaults to 'mwu'.
    :param dc_metric: Edge metric used for differential degree centrality computation. Defaults to 'pre-P'.
    :return: A DataFrame containing the computed differential node scores.
    """

    assert context1.columns.equals(context2.columns), 'Context a and b need to have the same structure.'

    nodes_diff = None

    if node_metric is None:
        logging.warning('No node_metric was specified. This setting will only work for the PageRank and DimontRank algorithm.')
        nodes_diff = subtract_nodes(context1=context1, context2=context2, test=False)

    # Statistical test centrality (STC)
    if node_metric == 'STC':
        nodes_diff = subtract_nodes(context1=context1, context2=context2, test=True, test_type=stc_test, correction=correction)

    # Degree centrality (DC)
    elif node_metric == 'DC':
        nodes_diff = subtract_nodes(context1=context1, context2=context2, test=False)
        nodes_diff = calculate_degree_centrality(nodes_diff=nodes_diff, metric=dc_metric, weighted=False,
                                                        scores1=scores1, scores2=scores2)
        
    # Degree centrality and statistical test centrality combined (DC-STC)
    elif node_metric == 'DC-STC':
        nodes_diff = subtract_nodes(context1=context1, context2=context2, test=True, test_type=stc_test, correction=correction)
        nodes_diff = calculate_degree_centrality(nodes_diff=nodes_diff, metric=dc_metric, weighted=False,
                                                        scores1=scores1, scores2=scores2)
        
        nodes_diff['DC-STC'] = nodes_diff['DC'] - nodes_diff['STC'] + 1.0        
    
    # Weighted degree centrality based on pre-P (WDC-P)
    elif node_metric == 'WDC-P':
        nodes_diff = subtract_nodes(context1=context1, context2=context2, test=False)
        nodes_diff = calculate_degree_centrality(nodes_diff=nodes_diff, metric='pre-P', weighted=True,
                                                        scores1=scores1, scores2=scores2)

    # Weighted degree centrality based on pre-E (WDC-E)
    elif node_metric == 'WDC-E':
        nodes_diff = subtract_nodes(context1=context1, context2=context2, test=False)
        nodes_diff = calculate_degree_centrality(nodes_diff=nodes_diff, metric='pre-E', weighted=True,
                                                        scores1=scores1, scores2=scores2)

    # PageRank centrality (PRC)
    elif node_metric == 'PRC':
        nodes_diff = subtract_nodes(context1=context1, context2=context2, test=False)

        # TODO: allow 'pre-P' metric for PRC (don't forget to invert)

        nodes_diff = calculate_pagerank_centrality(nodes_diff=nodes_diff, metric='pre-E', invert=False,
                                                            scores1=scores1, scores2=scores2)

    else:
        raise ValueError(f"Invalid node metric '{node_metric}'. Choose from: 'DC', 'STC', 'DC-STC', 'WDC-P', 'WDC-E' or 'PRC'.")

    return nodes_diff


# Check input format of context data
def _check_input_data(context: pd.DataFrame, meta_file: pd.DataFrame) -> bool:
    """
    Check if the input data is in the expected format.
    
    :param context: The context data to check.
    :param meta_file: Metadata file containing one row per variable in the context data.
    :return: The checked context data.
    """
    # Check if context is a DataFrame
    if not isinstance(context, pd.DataFrame):
        raise ValueError('The context data should be provided as a pandas DataFrame.')

    # Check if meta_file is a DataFrame
    if not isinstance(meta_file, pd.DataFrame):
        raise ValueError('The meta_file should be provided as a pandas DataFrame.')

    # Check if meta_file contains required columns
    required_columns = {'label', 'type'}
    if not required_columns.issubset(set(meta_file.columns)):
        raise ValueError(f'The meta_file should contain the following columns: {required_columns}.')

    # Check if all variables in context are present in meta_file
    context_vars = set(context.columns)
    meta_vars = set(meta_file['label'].values)
    if not context_vars.issubset(meta_vars):
        missing_vars = context_vars - meta_vars
        raise ValueError(f'The following variables are missing in the meta_file: {missing_vars}.')

    return True
    