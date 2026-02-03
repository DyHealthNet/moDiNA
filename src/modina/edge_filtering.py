from modina.statistics_utils import pre_rescaling

import logging
import math
import os
from typing import Tuple, Optional
import pandas as pd
import numpy as np


# Edge filtering
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

    # Compute number of edges according to the specified method
    threshold1 = None
    threshold2 = None
    n_nodes = context1.shape[1]
    n_edges_before = scores1.shape[0]

    if filter_method == 'quantile':
        if filter_param <= 0.0 or filter_param > 1.0:
            raise ValueError("For 'quantile' filtering, 'filter_param' must be between 0 and 1.")
        
        n_filtered_edges = math.ceil(filter_param * n_edges_before)

    elif filter_method == 'degree':
        degree = filter_param

        if degree < 1 or degree >= n_nodes:
            raise ValueError(f"For 'degree' filtering, 'filter_param' must be between 1 and {n_nodes - 1}.")
        
        n_filtered_edges = math.ceil(degree * n_nodes / 2)

    elif filter_method == 'density':
        density = filter_param

        if density <= 0.0 or density > 1.0:
            raise ValueError("For 'density' filtering, 'filter_param' must be between 0 and 1.")
        
        possible_edges = n_nodes * (n_nodes - 1) / 2
        n_filtered_edges = math.ceil(density * possible_edges)

    else:
        raise ValueError(f"Invalid filtering method '{filter_method}'. Choose from: 'quantile', 'degree' or 'density'")

    # Set threshold
    threshold1 = scores1[filter_metric].sort_values(ascending=ascending).iloc[n_filtered_edges - 1]
    threshold2 = scores2[filter_metric].sort_values(ascending=ascending).iloc[n_filtered_edges - 1]

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