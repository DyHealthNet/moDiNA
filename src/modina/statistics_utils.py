import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple


# Convert Cohen's d to point-biserial r for ttest edges.
# Uses context sample sizes to account for unequal groups:
# equal sizes (n1==n2): r = d / sqrt(d² + 4)
# unequal sizes:        r = d / sqrt(d² + (n1+n2)² / (n1*n2))
def cohens_d_to_r(scores1, scores2, n1: int, n2: int):
    scores1 = scores1.copy()
    scores2 = scores2.copy()

    correction = (n1 + n2) ** 2 / (n1 * n2)

    for scores in [scores1, scores2]:
        mask = scores['test_type'] == 'ttest'
        if mask.any():
            d = scores.loc[mask, 'raw-E'].to_numpy()
            scores.loc[mask, 'raw-E'] = d / np.sqrt(d ** 2 + correction)

    return scores1, scores2



# Materialize p-value transforms used by the differential metrics.
# Computes (idempotently) two columns derived from the adjusted p-value 'raw-P':
#   log-P = -log10(p)   (significance strength; higher = more significant)
#   inv-P = 1 - p       (linear significance strength in [0, 1])
# Zero p-values are floored with an epsilon (min non-zero p / 10) so -log10 is finite;
# if every p-value is zero, a fixed fallback epsilon is used (avoids min() on an empty array).
def add_pval_transforms(scores):
    p = scores['raw-P'].to_numpy(dtype=float)
    nonzero = p[p > 0]
    epsilon = nonzero.min() / 10.0 if nonzero.size else 1e-10
    scores['log-P'] = -np.log10(np.where(p == 0, epsilon, p))
    scores['inv-P'] = 1.0 - scores['raw-P']
    return scores


# Probit rescaling (rank-based normalization)
def probit_rescaling(scores1, scores2, metric='rescaled-E'):
    scores1 = scores1.copy()
    scores2 = scores2.copy()

    if metric != 'rescaled-E':
        raise ValueError(f"Invalid metric '{metric}'. Only 'rescaled-E' is supported.")

    metric_raw = 'raw-E'
    scores1[metric] = np.nan

    if scores2 is not None:
            scores2 = scores2.copy()
            scores2[metric] = np.nan
            if not scores1['test_type'].equals(scores2['test_type']):
                raise ValueError("scores1 and scores2 must have identical 'test_type' columns.")

    for test in np.unique(scores1['test_type']):
        idx1 = scores1['test_type'] == test
        v1 = scores1.loc[idx1, metric_raw].to_numpy()

        if scores2 is not None:
            idx2 = scores2['test_type'] == test
            v2 = scores2.loc[idx2, metric_raw].to_numpy()
            combined = np.concatenate([v1, v2])
            n = len(combined)

        # Folded probit: rank |raw-E| so association strength (not sign) determines rank.
        # Percentile mapped to (0.5, 1) so norm.ppf gives values in (0, +inf).
        # Sign is restored afterward, so strong negative associations rank equally to
        # strong positive ones. For non-negative test types sign=+1 always (no-op).
        if n == 1:
            scores1.loc[idx1, metric] = 0.0
            scores2.loc[idx2, metric] = 0.0
            continue
        signs = np.sign(combined)
        ranks = stats.rankdata(np.abs(combined))
        # Map rank 1 → percentile 0.5 → probit 0.0; rank n → percentile 0.975 → probit 1.96.
        # Factor 0.475 gives max = norm.ppf(0.975) = 1.96 (z-score for p=0.05 two-sided).
        # Formula (rank-1)/(n-1) eliminates n-dependency at both endpoints.
        percentiles = 0.5 + (ranks - 1) / (n - 1) * 0.475
        probit_magnitude = stats.norm.ppf(percentiles)
        probit_vals = signs * probit_magnitude

        scores1.loc[idx1, metric] = probit_vals[:len(v1)]
        scores2.loc[idx2, metric] = probit_vals[len(v1):]

        else:
            n = len(v1)
            signs = np.sign(v1)
            ranks = stats.rankdata(np.abs(v1))
            percentiles = 0.5 + ranks / (n + 1) * 0.5
            probit_magnitude = stats.norm.ppf(percentiles)
            probit_vals = signs * probit_magnitude
            scores1.loc[idx1, metric] = probit_vals

    return (scores1, scores2) if scores2 is not None else scores1


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