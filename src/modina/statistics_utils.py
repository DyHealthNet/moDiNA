import numpy as np
import pandas as pd
from typing import Tuple

# Pre-rescaling (Z-score normalization)
# TODO: implement a filtering version (abs-E)?
def pre_rescaling(scores1, scores2, metric='rescaled-E'):
    scores1 = scores1.copy()
    scores2 = scores2.copy()
    
    if metric == 'rescaled-E':
        metric_raw = 'raw-E'

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

            # Z-score normalization
            mean = np.mean(values)
            std = np.std(values)

            rescaled1 = None
            rescaled2 = None

            if std == 0:
                rescaled1 = 0
                rescaled2 = 0
            else:
                rescaled1 = (scores1_filtered[metric_raw] - mean) / std
                rescaled2 = (scores2_filtered[metric_raw] - mean) / std

            scores1.loc[scores1['test_type'] == test, metric] = rescaled1
            scores2.loc[scores2['test_type'] == test, metric] = rescaled2
    
    else:
        raise ValueError(f"Invalid metric '{metric}'. Only 'rescaled-E' is supported.")
    
    return scores1, scores2


# Post-rescaling (Min-Max normalization)
# TODO: instead of taking the whole dataframe, only create a new column maybe
def post_rescaling(diff_scores, metric):
    diff_scores = diff_scores.copy()

    if metric == 'post-LS':
        metric_raw = 'diff-LS'
    elif metric == 'post-E':
        metric_raw = 'diff-E'

    else:
        raise ValueError(f"Invalid metric '{metric}'. Only 'post-E' and 'post-LS' are supported.")
    
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


def _separate_types(all_data, meta_file) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Separating the data into ordinal, nominal, continuous and binary variables.
    :param all_data: DataFrame with all data
    :param meta_file: DataFrame with metadata of the variables
    :return: tuple with the ordinal, nominal, continuous and binary variables
    """

    # Check if meta_file has an invalid type
    if not meta_file['type'].str.lower().isin(['ordinal', 'nominal', 'binary', 'continuous']).all():
        raise ValueError("Invalid type found in meta_file. Allowed types are 'ordinal', 'nominal', 'binary', and 'continuous'.")

    # Extract ordinal phenotypes
    ord_data = all_data.iloc[:, all_data.columns.isin(meta_file[meta_file.type.str.lower() == 'ordinal'].label)].copy()
    
    # Extract nominal phenotypes
    nom_data = all_data.iloc[:, all_data.columns.isin(meta_file[meta_file.type.str.lower() == 'nominal'].label)].copy()

    # Extract binary phenotypes
    bi_data = all_data.iloc[:, all_data.columns.isin(meta_file[meta_file.type.str.lower() == 'binary'].label)].copy()

    # Extract continuous phenotypes
    cont_data = all_data.iloc[:, all_data.columns.isin(meta_file[meta_file.type.str.lower() == 'continuous'].label)].copy()

    return ord_data, nom_data, cont_data, bi_data


def _df_to_numpy(df: pd.DataFrame):
    cols = df.columns
    df_np = df.to_numpy(dtype=np.float64).copy()
    return df_np, cols