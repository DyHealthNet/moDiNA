import logging
import os
import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional, Tuple


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



# Single multiple-testing correction across a flat family of p-values.
# Applies one Benjamini-Hochberg ('bh') or Benjamini-Yekutieli ('by') FDR pass over all
# supplied p-values at once (as opposed to napy's per-test-call correction). NaN entries are
# untested pairs/nodes and are excluded from the correction family (kept as NaN in the output),
# so they do not inflate the FDR denominator.
def fdr_correction(pvalues, method='bh'):
    if method not in ('bh', 'by'):
        raise ValueError(f"Invalid correction method '{method}'. Choose from: 'bh' or 'by'.")
    p = np.asarray(pvalues, dtype=float)
    valid = ~np.isnan(p)
    out = p.copy()
    if valid.any():
        out[valid] = np.clip(stats.false_discovery_control(p[valid], method=method), 0.0, 1.0)
    return out


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


# Test types whose raw effect size (see TEST_EFFECT_SIZES in context_net_inference.py) is
# already a variance-explained (r²-equivalent) quantity: ANOVA's partial η² and Kruskal-Wallis's
# η². These are used as-is by effect_size_to_r2 below, rather than being squared.
_R2_SCALE_TEST_TYPES = {'anova', 'kruskal'}


# Transform raw effect sizes onto a common, absolute r² (variance-explained) scale.
# Unlike probit_rescaling's rank-based normalization (which only makes effect sizes comparable
# relative to the observed sample), this uses known statistical conversions so that a given
# transformed-E value has the same interpretation ("proportion of variance explained")
# regardless of which test produced it:
#   - pearson (r), spearman (rho), mwu (rank-biserial rb), ttest (already converted to
#     point-biserial r upstream) and chi2 (Cramer's V, which equals phi for 2x2 tables and is
#     treated as r-equivalent for larger tables) are all correlation-like coefficients on an
#     r scale -> squared to obtain r².
#   - anova (partial eta²) and kruskal (eta²) are already r²-equivalent quantities -> used as-is.
# Sign is preserved for the squared test types (sign(r) * r²); the r²-scale test types have no
# natural direction and their raw effect size is already non-negative, so this is a no-op there.
# Operates on each context independently (no cross-context pooling is needed, unlike
# probit_rescaling), but takes a (scores1, scores2) pair for calling-convention parity.
def effect_size_to_r2(scores1, scores2, metric='transformed-E'):
    scores1 = scores1.copy()
    scores2 = scores2.copy()

    for scores in (scores1, scores2):
        raw = scores['raw-E'].to_numpy(dtype=float)
        r2_scale = scores['test_type'].isin(_R2_SCALE_TEST_TYPES).to_numpy()
        scores[metric] = np.where(r2_scale, raw, np.sign(raw) * raw ** 2)

    return scores1, scores2


# Probit rescaling (rank-based normalization)
def probit_rescaling(scores1, scores2, metric='rescaled-E'):
    # Remove variables flagged (single observed category, or entirely missing) in only one of
    # the two contexts, so scores1/scores2 are row-aligned below. This is often the first point
    # in a pipeline where both contexts' scores meet (e.g. the Nextflow pipeline's
    # rescaling_networks.py, which reads scores1/scores2 straight from CSV before any filtering
    # or differential-network step has a chance to reconcile them).
    scores1, scores2, _, _, removed_variables = reconcile_flagged_variables(scores1, scores2)

    scores1 = scores1.copy()
    scores2 = scores2.copy()

    # Always attach the (possibly empty) removed-variables list, so a direct Python caller can
    # inspect it via scores1.attrs['removed_variables'] even without going through a Nextflow bin
    # script or passing a 'path' anywhere.
    scores1.attrs['removed_variables'] = removed_variables
    scores2.attrs['removed_variables'] = removed_variables

    if metric != 'rescaled-E':
        raise ValueError(f"Invalid metric '{metric}'. Only 'rescaled-E' is supported.")

    metric_raw = 'raw-E'
    scores1[metric] = np.nan
    scores2[metric] = np.nan

    if not scores1['test_type'].equals(scores2['test_type']):
        raise ValueError("scores1 and scores2 must have identical 'test_type' columns.")

    for test in np.unique(scores1['test_type']):
        idx1 = scores1['test_type'] == test
        idx2 = scores2['test_type'] == test
        v1 = scores1.loc[idx1, metric_raw].to_numpy()
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

    return scores1, scores2


# Find variables that are flagged (single observed category, or entirely missing) in only one
# of the two contexts. A flagged variable is dropped from that context's network before any
# tests are run (see context_net_inference._drop_single_category_variables), so it produces no
# rows at all in that context's 'scores' -- while it may still be a perfectly valid, tested
# variable in the other context. This asymmetry is exactly the symmetric difference of the two
# contexts' node sets (the labels appearing in 'label1'/'label2'), so no extra bookkeeping is
# needed to find it.
def find_flagged_variables(scores1: pd.DataFrame, scores2: pd.DataFrame) -> list:
    """
    Return the sorted list of variables that appear as an edge endpoint in only one of
    scores1/scores2 -- i.e. variables flagged (single observed category, or entirely missing)
    in exactly one context and therefore absent from that context's association scores.

    :param scores1: Association scores of Context 1.
    :param scores2: Association scores of Context 2.
    :return: Sorted list of flagged variable names.
    """
    nodes1 = set(scores1['label1']) | set(scores1['label2'])
    nodes2 = set(scores2['label1']) | set(scores2['label2'])
    return sorted(nodes1 ^ nodes2)


# Reconcile two context-specific networks so they reference exactly the same set of edges.
# Two independent things can make a pair (label1, label2) present in only one context's scores:
# (1) a variable flagged as single-category/entirely-missing in one context (dropped from that
#     context's network entirely, see context_net_inference._drop_single_category_variables), or
# (2) NApy returning NaN for that specific pair in one context (dropped individually, see
#     compute_context_scores' scores_na handling) even though the same pair tested fine in the
#     other context.
# Both leave scores1/scores2 with a mismatched, un-aligned set of rows, which breaks every
# downstream step that assumes they're row-aligned (e.g. _subtract_edges, probit_rescaling). An
# inner merge on (label1, label2, test_type) fixes both at once: any pair missing from either
# side -- for either reason -- is naturally dropped from both, and the two DataFrames
# reconstructed from the merge are guaranteed row-count-equal and identically ordered, so nothing
# downstream needs to change how it compares scores1 to scores2.
def reconcile_flagged_variables(scores1: pd.DataFrame, scores2: pd.DataFrame,
                                context1: Optional[pd.DataFrame] = None, context2: Optional[pd.DataFrame] = None,
                                path: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame], list]:
    """
    Align two context-specific networks onto their common set of edges, so scores1 and scores2
    stay row-aligned, and report which variables ended up with no edges left in either context.

    :param scores1: Association scores of Context 1.
    :param scores2: Association scores of Context 2.
    :param context1: Optional observed data of Context 1; if given, variables with no edges left are also dropped as columns.
    :param context2: Optional observed data of Context 2; if given, variables with no edges left are also dropped as columns.
    :param path: Optional path to save the list of removed variables as a CSV file. Defaults to None.
    :return: A tuple (scores1, scores2, context1, context2, removed_variables), with scores1/scores2
             restricted to their common edges (row-aligned) and context1/context2 returned unchanged
             (None) if not provided. 'removed_variables' is the sorted list of variables that no
             longer appear as an edge endpoint in either context (empty if none) -- callers should
             attach it as `.attrs['removed_variables']` on whatever they return, so it stays
             discoverable to a caller even when no 'path' is given to write it to disk (e.g. a
             direct Python/API call that never passes through a Nextflow bin script).
    """
    merge_keys = ['label1', 'label2', 'test_type']
    value_cols = [c for c in scores1.columns if c in scores2.columns and c not in merge_keys]

    nodes_before = (set(scores1['label1']) | set(scores1['label2'])
                    | set(scores2['label1']) | set(scores2['label2']))
    n1_before, n2_before = len(scores1), len(scores2)

    merged = scores1.merge(scores2, on=merge_keys, how='inner', suffixes=('_1', '_2'))

    # Reconstruct scores1/scores2 from the merge, preserving each one's own original column order.
    original_cols1, original_cols2 = list(scores1.columns), list(scores2.columns)
    scores1 = merged[merge_keys + [f'{c}_1' for c in value_cols]].rename(columns={f'{c}_1': c for c in value_cols})
    scores2 = merged[merge_keys + [f'{c}_2' for c in value_cols]].rename(columns={f'{c}_2': c for c in value_cols})
    scores1 = scores1[original_cols1].sort_values(by=merge_keys).reset_index(drop=True)
    scores2 = scores2[original_cols2].sort_values(by=merge_keys).reset_index(drop=True)

    nodes_after = set(scores1['label1']) | set(scores1['label2'])
    flagged = sorted(nodes_before - nodes_after)
    n_pairs_dropped = (n1_before - len(merged)) + (n2_before - len(merged))

    if flagged or n_pairs_dropped:
        logging.warning(f'Reconciling the two contexts\' networks: {n_pairs_dropped} pair(s) present in only '
                        f'one context (either a variable flagged there, or NApy returning NaN for that specific '
                        f'pair) were dropped from both. {len(flagged)} variable(s) now have no edges left in '
                        f'either context: {flagged}.')
        if path is not None:
            pd.DataFrame({'label': flagged}).to_csv(os.path.join(path, 'flagged_variables_removed.csv'), index=False)

    if flagged and context1 is not None and context2 is not None:
        context1 = context1.drop(columns=flagged, errors='ignore')
        context2 = context2.drop(columns=flagged, errors='ignore')

    return scores1, scores2, context1, context2, flagged


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