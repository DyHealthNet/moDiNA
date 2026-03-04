# Adapted from https://github.com/DyHealthNet/DHN-backend.git

from modina.statistics_utils import _df_to_numpy, _separate_types

import numpy as np
import pandas as pd
import napypi as napy
import logging
from typing import Optional, Tuple

EXCLUDED_EFFECTS = {'chi2', 't', 'F', 'U', 'H'}


def calculate_association_scores(ord_data, nom_data, cont_data, bi_data, test_type='nonparametric', num_workers=1, nan_value=-89) -> pd.DataFrame:
    cont_data = cont_data.copy()
    if not cont_data.select_dtypes(include=[np.number]).shape[1] == cont_data.shape[1]:
        raise ValueError('Continuous data contains non-numeric columns.')

    # Remap categories to start at 0 and be consecutive integers
    nom_data = _order_categories(nom_data)
    bi_data = _order_categories(bi_data)

    cont_nom_results = napy_nom_cont(cont_data, nom_data, test=test_type, num_workers=num_workers, nan_value=nan_value)
    logging.info("Finished continuous-nominal score calculation.")
    
    cont_ord_results = napy_ord_cont(cont_data, ord_data, num_workers=num_workers, nan_value=nan_value)
    logging.info("Finished continuous-ordinal score calculation.")

    bi_cont_results = napy_bi_cont(cont_data, bi_data, test=test_type, num_workers=num_workers, nan_value=nan_value)
    logging.info("Finished continuous-binary score calculation.")

    bi_nom_results = napy_bi_nom(nom_data, bi_data, num_workers=num_workers, nan_value=nan_value)
    logging.info("Finished binary-nominal score calculation.")

    cont_cont_results = napy_cont_cont(cont_data, test=test_type, num_workers=num_workers, nan_value=nan_value)
    logging.info("Finished continuous-continuous score calculation")

    bi_ord_results = napy_bi_ord(ord_data, bi_data, num_workers=num_workers, nan_value=nan_value)
    logging.info("Finished binary-ordinal score calculation.")

    ord_nom_results = napy_ord_nom(ord_data, nom_data, num_workers=num_workers, nan_value=nan_value)
    logging.info("Finished ordinal-nominal score calculation.")

    scores = _combine_tests(cont_nom_results, cont_ord_results, bi_cont_results, bi_nom_results,
                            cont_cont_results, bi_ord_results, ord_nom_results)

    return scores


def compute_context_scores(context_data: pd.DataFrame, meta_file: pd.DataFrame,
                           test_type: str = 'nonparametric',
                           correction: str = 'bh', num_workers: int = 1,
                           path: Optional[str] = None) -> pd.DataFrame:
    """
    Compute association scores for a given context.

    :param context_data: The raw context data (rows: samples, columns: variables).
    :param meta_file: Metadata file containing a 'label' and 'type' column to specify the data type of each variable.
    :param test_type: Type of tests to use for network inference. Defaults to 'nonparametric'.
    :param correction: Correction method for multiple testing. Defaults to 'bh'.
    :param num_workers: Number of workers for parallel processing. Defaults to 1.
    :param path: Optional path to save the computed scores as a CSV file. Defaults to None.
    :return: A pd.DataFrame containing the computed association scores.
    """
    # Check nan values and input format
    context_data, nan_value = _check_input_data(context=context_data, meta_file=meta_file)

    # Separate the data into categorical and continuous data
    ord, nom, cont, bi = _separate_types(context_data, meta_file)

    # Calculate scores
    scores = calculate_association_scores(ord_data=ord, nom_data=nom, cont_data=cont, bi_data=bi, 
                                          test_type=test_type, num_workers=num_workers, nan_value=nan_value)

    # Take the adjusted p-value and the corresponding effect size
    column_names = scores.iloc[:, 2:].columns
    if correction == 'bh':
        correction = 'benjamini_hb'
    elif correction == 'by':
        correction = 'benjamini_yek'
    else:
        raise ValueError(f"Invalid correction method '{correction}'. Choose from: 'bh' or 'yek'.")

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


def napy_bi_nom(nom_phenotypes: pd.DataFrame, bi_phenotypes: pd.DataFrame, num_workers=8, nan_value=-89):
    # Combine nominal and binary phenotypes for chi-squared test
    discrete_phenotypes = pd.concat([nom_phenotypes, bi_phenotypes], axis=1)
    discrete_phenotypes, cols = _df_to_numpy(discrete_phenotypes)
    if discrete_phenotypes.shape[1] < 2:
        return [None]

    output = napy.chi_squared(discrete_phenotypes, axis=1, threads=num_workers, nan_value=nan_value, use_numba=False)
    results = _napy_formatting(output, [cols], 'chi2')
    assert results is not None, "Results should not be None here."

    for col in results.columns:
        if "_e_" in col:
            results[col] = results[col].fillna(0.0)
        elif "_p_" in col:
            results[col] = results[col].fillna(1.0)

    return [results]


def napy_nom_cont(cont_phenotypes: pd.DataFrame, nom_phenotypes: pd.DataFrame, test: str='nonparametric', num_workers=8, nan_value=-89):
    if nom_phenotypes.shape[1] < 1 or cont_phenotypes.shape[1] < 1:
        return [None]

    cont_phenotypes, cont_cols = _df_to_numpy(cont_phenotypes)
    nom_phenotypes, nom_cols = _df_to_numpy(nom_phenotypes)

    result = None
    done_test = None

    if test == 'parametric':
        result = napy.anova(cat_data=nom_phenotypes, cont_data=cont_phenotypes, axis=1,
                                      threads=num_workers, nan_value=nan_value)
        done_test = "anova"

    elif test == 'nonparametric':
        result = napy.kruskal_wallis(cat_data=nom_phenotypes, cont_data=cont_phenotypes, axis=1,
                                               threads=num_workers, nan_value=nan_value)
        done_test = "kruskal"

    else:
        raise ValueError(f"Invalid test type '{test}'. Specify 'parametric' or 'nonparametric' for nominal-continuous association testing.")

    return [_napy_formatting(result, [nom_cols, cont_cols], done_test)]


def napy_ord_nom(ord_phenotypes: pd.DataFrame, nom_phenotypes: pd.DataFrame, num_workers=8, nan_value=-89):
    if nom_phenotypes.shape[1] < 1 or ord_phenotypes.shape[1] < 1:
        return [None]

    ord_phenotypes, ord_cols = _df_to_numpy(ord_phenotypes)
    nom_phenotypes, nom_cols = _df_to_numpy(nom_phenotypes)

    result = napy.kruskal_wallis(cat_data=nom_phenotypes, cont_data=ord_phenotypes, axis=1,
                                            threads=num_workers, nan_value=nan_value)
    done_test = "kruskal"

    return [_napy_formatting(result, [nom_cols, ord_cols], done_test)]


def napy_bi_cont(cont_phenotypes: pd.DataFrame, bi_phenotypes: pd.DataFrame, test: str='nonparametric', num_workers=8, nan_value=-89):
    if bi_phenotypes.shape[1] < 1 or cont_phenotypes.shape[1] < 1:
        return [None]

    if (bi_phenotypes.nunique() > 2).any():
        raise ValueError("All binary variables must not have more than two unique values.")

    cont_phenotypes, cont_cols = _df_to_numpy(cont_phenotypes)
    bi_phenotypes_two, bi_cols = _df_to_numpy(bi_phenotypes)

    result = None
    done_test = None

    if test == 'parametric':
        result = napy.ttest(bin_data=bi_phenotypes_two, cont_data=cont_phenotypes, axis=1,
                                     threads=num_workers, nan_value=nan_value)
        done_test = "ttest"

    elif test == 'nonparametric':
        result = napy.mwu(bin_data=bi_phenotypes_two, cont_data=cont_phenotypes, axis=1, threads=num_workers,
                                   nan_value=nan_value)
        done_test = "mwu"

    else:
        raise ValueError(f"Invalid test type '{test}'. Specify 'parametric' or 'nonparametric' for binary-continuous association testing.")

    results = [_napy_formatting(result, [bi_cols, cont_cols], done_test)]

    # Special case: variables with only one category
    bi_phenotypes_one = bi_phenotypes.loc[:, bi_phenotypes.nunique() <= 1].copy()
    bi_phenotypes_one, bi_cols_one = _df_to_numpy(bi_phenotypes_one)

    if bi_phenotypes_one.shape[1] > 0:
        logging.warning(
            f'There were binary variables found with only one category: {bi_cols_one}. '
            'These will be added as dummy rows with p-value 1.0 and effect size 0.0.')

        # For each combination of bi_cols_one and cont_cols, add a row with p-value 1.0 and effect size 0.0
        for i, df in enumerate(results):
            if df is None:
                continue

            p_cols = [col for col in df.columns if "_p_" in col]
            e_cols = [col for col in df.columns if "_e_" in col]

            special_rows = []
            for bi_var in bi_cols_one:
                for cont_var in cont_cols:
                    row = {
                        'label1': bi_var,
                        'label2': cont_var
                    }
                    for col in p_cols:
                        row[col] = 1.0
                    for col in e_cols:
                        row[col] = 0.0
                    special_rows.append(row)

            # Add special case rows to results
            results[i] = pd.concat([df, pd.DataFrame(special_rows)], ignore_index=True)

    # Replace NaNs
    results_final = []
    for df in results:
        if df is None:
            results_final.append(None)
            continue

        df = df.copy()
        p_cols = [col for col in df.columns if "_p_" in col]
        e_cols = [col for col in df.columns if "_e_" in col]

        if p_cols:
            df[p_cols] = df[p_cols].fillna(1.0)
        if e_cols:
            df[e_cols] = df[e_cols].fillna(0.0)

        results_final.append(df)

    return results_final


def napy_bi_ord(ord_phenotypes: pd.DataFrame, bi_phenotypes: pd.DataFrame, num_workers=8, nan_value=-89):
    if bi_phenotypes.shape[1] < 1 or ord_phenotypes.shape[1] < 1:
        return [None]

    if (bi_phenotypes.nunique() > 2).any():
        raise ValueError("All binary variables must not have more than two unique values.")

    ord_phenotypes, ord_cols = _df_to_numpy(ord_phenotypes)
    bi_phenotypes_two, bi_cols = _df_to_numpy(bi_phenotypes)

    result = napy.mwu(bin_data=bi_phenotypes_two, cont_data=ord_phenotypes, axis=1, threads=num_workers,
                                nan_value=nan_value)
    done_test = "mwu"

    results = [_napy_formatting(result, [bi_cols, ord_cols], done_test)]

    # Special case: variables with only one category
    bi_phenotypes_one = bi_phenotypes.loc[:, bi_phenotypes.nunique() <= 1].copy()
    bi_phenotypes_one, bi_cols_one = _df_to_numpy(bi_phenotypes_one)

    if bi_phenotypes_one.shape[1] > 0:
        logging.warning(
            f'There were binary variables found with only one category: {bi_cols_one}. '
            'These will be added as dummy rows with p-value 1.0 and effect size 0.0.')

        # For each combination of bi_cols_one and ord_cols, add a row with p-value 1.0 and effect size 0.0
        for i, df in enumerate(results):
            if df is None:
                continue

            p_cols = [col for col in df.columns if "_p_" in col]
            e_cols = [col for col in df.columns if "_e_" in col]

            special_rows = []
            for bi_var in bi_cols_one:
                for ord_var in ord_cols:
                    row = {
                        'label1': bi_var,
                        'label2': ord_var
                    }
                    for col in p_cols:
                        row[col] = 1.0
                    for col in e_cols:
                        row[col] = 0.0
                    special_rows.append(row)

            # Add special case rows to results
            results[i] = pd.concat([df, pd.DataFrame(special_rows)], ignore_index=True)

    # Replace NaNs
    results_final = []
    for df in results:
        if df is None:
            results_final.append(None)
            continue

        df = df.copy()
        p_cols = [col for col in df.columns if "_p_" in col]
        e_cols = [col for col in df.columns if "_e_" in col]

        if p_cols:
            df[p_cols] = df[p_cols].fillna(1.0)
        if e_cols:
            df[e_cols] = df[e_cols].fillna(0.0)

        results_final.append(df)

    return results_final


def napy_cont_cont(cont_phenotypes: pd.DataFrame, test: str='nonparametric', num_workers=8, nan_value=-89):
    if cont_phenotypes.shape[1] < 2:
        return [None]
    
    cont_phenotypes, cont_cols = _df_to_numpy(cont_phenotypes)
    result = None
    done_test = None

    if test == 'parametric':
        result = napy.pearsonr(cont_phenotypes, nan_value=nan_value, threads=num_workers,
                                    axis=1)
        done_test = "pearson"

    elif test == 'nonparametric':
        result = napy.spearmanr(cont_phenotypes, threads=num_workers, nan_value=nan_value,
                                     axis=1)
        done_test = "spearman"

    else:
        raise ValueError(f"Invalid test type '{test}'. Specify 'parametric' or 'nonparametric' for continuous-continuous association testing.")

    return [_napy_formatting(result, [cont_cols], done_test)]


def napy_ord_cont(cont_phenotypes: pd.DataFrame, ord_phenotypes: pd.DataFrame, num_workers=8, nan_value=-89):
    if cont_phenotypes.shape[1] < 1 or ord_phenotypes.shape[1] < 1:
        return [None]
    
    combined_phenotypes = pd.concat([cont_phenotypes, ord_phenotypes], axis=1)

    combined_phenotypes, combined_cols = _df_to_numpy(combined_phenotypes)
    cont_phenotypes, cont_cols = _df_to_numpy(cont_phenotypes)
    ord_phenotypes, ord_cols = _df_to_numpy(ord_phenotypes)

    result = napy.spearmanr(combined_phenotypes, threads=num_workers, nan_value=nan_value, axis=1)
    done_test = "spearman"

    return [_napy_formatting(result, [combined_cols], done_test, ord_cols=ord_cols.tolist(), cont_cols=cont_cols.tolist())]


# Check input format of context data
def _check_input_data(context: pd.DataFrame, meta_file: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Check if the input data is in the expected format and identify nan values.

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

    # Search for non-numeric and NaN values
    if context.apply(lambda col: pd.to_numeric(col, errors="coerce").isna()).values.any() > 0:
        logging.warning('The context data contains non-numeric or NaN values.')

    # Replace non-numeric values with a numeric value that does not exist in the data
    existing = set(pd.to_numeric(context.stack(), errors="coerce").dropna().values)
    while True:
        nan_value = np.random.randint(-10**5, -10**3)
        if nan_value not in existing:
            break

    context = context.fillna(nan_value)

    return context, nan_value


def _combine_tests(*result_groups) -> pd.DataFrame:
    all_results = []

    for results in result_groups:
        merged = None
        for test in results:
            if test is None:
                continue
            # Convert all columns except 'label1' and 'label2' to float32
            cols_to_convert = test.columns.difference(['label1', 'label2'])
            test[cols_to_convert] = test[cols_to_convert].astype('float32')

            if merged is None:
                merged = test
                continue
            merged = pd.merge(merged, test, on=['label1', 'label2'], how='outer')
        all_results.append(merged)

    out = pd.concat(all_results, ignore_index=True)
    out = out.sort_values(by=['label1', 'label2'], kind='mergesort').reset_index(drop=True)

    return out


def _napy_formatting(assoc_out: dict[np.array], labels: list, test: str, 
                     ord_cols: Optional[list] = None, cont_cols: Optional[list] = None, 
                     file_name: Optional[str] = None) -> Optional[pd.DataFrame]:
    if not assoc_out:
        return None

    if len(labels) == 1:
        rows_idx, cols_idx = np.tril_indices(assoc_out['p_unadjusted'].shape[0], k=-1)
        # Pre-format labels and values
        label1 = np.array(labels[0])[rows_idx]
        label2 = np.array(labels[0])[cols_idx]
    else:
        rows_idx, cols_idx = np.indices(assoc_out['p_unadjusted'].shape)
        label1 = np.array(labels[0])[rows_idx.ravel()]
        label2 = np.array(labels[1])[cols_idx.ravel()]

    p_values_raw = {key: assoc_out[key][rows_idx, cols_idx].ravel() for key in assoc_out if key.startswith('p_')}
    effects_raw = {key: assoc_out[key][rows_idx, cols_idx].ravel() for key in assoc_out if not key.startswith('p_') and
                   key not in EXCLUDED_EFFECTS}

    p_columns = [f"{test}_{key}" for key in p_values_raw.keys()]
    e_columns = [f"{test}_e_{key}" for key in effects_raw.keys()]

    df = pd.DataFrame({
        'label1': label1,
        'label2': label2,
        **{p_columns[i]: p_values_raw[key] for i, key in enumerate(p_values_raw)},
        **{e_columns[i]: effects_raw[key] for i, key in enumerate(effects_raw)},
    })

    if ord_cols is not None and cont_cols is not None:
        mask = (
            (df["label1"].isin(ord_cols) & df["label2"].isin(cont_cols)) |
            (df["label1"].isin(cont_cols) & df["label2"].isin(ord_cols)) |
            (df["label1"].isin(ord_cols) & df["label2"].isin(ord_cols))
        )
        
        df = df[mask]

    if file_name:
        df.to_csv(file_name, sep=',', index=True, header=False, lineterminator='\n')

    return df


def _order_categories(data: pd.DataFrame):
    """
    Order categories in a dataframe such that they start at 0 and are consecutive integers.
    :param data: the dataframe to order
    :return: the ordered dataframe
    """
    data = data.copy()
    order_table = {col: {o: n for n, o in enumerate(sorted(data[col].dropna().unique()))} for col in data.columns}
    for col, mapping in order_table.items():
        data[col] = data[col].map(mapping).astype(pd.Int64Dtype())
    return data
