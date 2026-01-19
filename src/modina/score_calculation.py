# Copied and modified from https://github.com/DyHealthNet/DHN-backend.git

import numpy as np
import pandas as pd
import napy as nanpy
import logging

EXCLUDED_EFFECTS = {'chi2', 't', 'F', 'U', 'H'}


def _df_to_numpy(df: pd.DataFrame, nan_value=-89):
    cols = df.columns
    df = df.fillna(nan_value)
    df_np = df.to_numpy(dtype=np.float64).copy()
    return df_np, cols


def _nanpy_formatting(assoc_out: dict[np.array], labels: list, test: str, file_name: str = None):
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

    if file_name:
        df.to_csv(file_name, sep=',', index=True, header=False, lineterminator='\n')

    return df


def _combine_tests(cat_cat, cont_cont, bi_cont, cont_cat) -> pd.DataFrame:
    """
    Combine the non-parametric tests with the parametric tests, giving the non-parametric tests the suffix '_np'.
    If no non-parametric results are given, empty columns 'pval_np', 'effsize_np', 'test_np' are created.
    :param np_results: the non-parametric results
    :param p_results: the parametric results
    :return: results with both tests combined
    """
    all_results = []

    for results in [cat_cat, cont_cont, bi_cont, cont_cat]:
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

    return out


def nanpy_cat_cat(cat_phenotypes: pd.DataFrame, num_workers=8, nan_value=-89):
    cat_phenotypes, cols = _df_to_numpy(cat_phenotypes)
    if cat_phenotypes.shape[1] < 2:
        return [None]
    output = nanpy.chi_squared(cat_phenotypes, axis=1, threads=num_workers, nan_value=nan_value, use_numba=False)
    results = _nanpy_formatting(output, [cols], 'chi2')

    for col in results.columns:
        if "_e_" in col:
            results[col] = results[col].fillna(0.0)
        elif "_p_" in col:
            results[col] = results[col].fillna(1.0)
    
    return [results]


def nanpy_cat_cont(cont_phenotypes: pd.DataFrame, cat_phenotypes: pd.DataFrame, tests: str, num_workers=8, nan_value=-89):
    # split cat_phenotypes into two dataframes, one with columns that contain only two unique values and one with more
    # than two unique values
    if cat_phenotypes.shape[1] < 2:
        return [None, None]
    cat_phenotypes_more = cat_phenotypes.loc[:, cat_phenotypes.nunique() > 2].copy()

    cont_phenotypes, cont_cols = _df_to_numpy(cont_phenotypes)
    cat_phenotypes_more, cat_cols_more = _df_to_numpy(cat_phenotypes_more)
    # this is just to shut the IDE up
    more_cont_out_a, more_cont_out_k = None, None
    done_test_a, done_test_k = None, None

    if tests in ['all', 'anova']:
        more_cont_out_a = nanpy.anova(cat_phenotypes_more, cont_phenotypes, axis=1,
                                      threads=num_workers, nan_value=nan_value)
        done_test_a = "anova"

    if tests in ['all', 'kruskal-wallis']:
        more_cont_out_k = nanpy.kruskal_wallis(cat_phenotypes_more, cont_phenotypes, axis=1,
                                               threads=num_workers, nan_value=nan_value)
        done_test_k = "kruskal"

    return [_nanpy_formatting(more_cont_out_a, [cat_cols_more, cont_cols], done_test_a),
            _nanpy_formatting(more_cont_out_k, [cat_cols_more, cont_cols], done_test_k)]


def nanpy_binary_cat_cont(cont_phenotypes: pd.DataFrame, cat_phenotypes: pd.DataFrame, test: str, num_workers=8, nan_value=-89):
    """
    Do binary categorical-continuous association testing of binary categorical variables with continuous variables.
    As the binary categorical variables can be seen as a special case of the categorical variables, this function
    allows for the same tests as the categorical-continuous association testing. In addition, it also allows for
    tests specific to binary categorical variables.
    :param cont_phenotypes: DataFrame with continuous variables
    :param cat_phenotypes: DataFrame with binary categorical variables
    :param test: the test to perform
    :return: DataFrame with the results of the association testing
    """
    if cat_phenotypes.shape[1] < 2:
        return [None, None, None, None]
    cat_phenotypes_two = cat_phenotypes.loc[:, cat_phenotypes.nunique() == 2].copy()
    cont_phenotypes, cont_cols = _df_to_numpy(cont_phenotypes)
    cat_phenotypes_two, cat_cols_two = _df_to_numpy(cat_phenotypes_two)

    two_cont_out_t, two_cont_out_a, two_cont_out_m, two_cont_out_k = None, None, None, None
    done_test_t, done_test_a, done_test_m, done_test_k = None, None, None, None

    if test in ['all', 't-test']:
        two_cont_out_t = nanpy.ttest(cat_phenotypes_two, cont_phenotypes, axis=1,
                                     threads=num_workers, nan_value=nan_value)
        done_test_t = "ttest"
    if test in ['all', 'anova']:
        two_cont_out_a = nanpy.anova(cat_phenotypes_two, cont_phenotypes, axis=1,
                                     threads=num_workers, nan_value=nan_value)
        done_test_a = "anova"

    if test in ['all', 'mann-whitney u']:
        two_cont_out_m = nanpy.mwu(cat_phenotypes_two, cont_phenotypes, axis=1, threads=num_workers,
                                   nan_value=nan_value)
        done_test_m = "mwu"

    if test in ['all', 'kruskal-wallis']:
        two_cont_out_k = nanpy.kruskal_wallis(cat_phenotypes_two, cont_phenotypes, axis=1, threads=num_workers,
                                              nan_value=nan_value)
        done_test_k = "kruskal"

    results = [_nanpy_formatting(two_cont_out_t, [cat_cols_two, cont_cols], done_test_t),
               _nanpy_formatting(two_cont_out_a, [cat_cols_two, cont_cols], done_test_a),
               _nanpy_formatting(two_cont_out_m, [cat_cols_two, cont_cols], done_test_m),
               _nanpy_formatting(two_cont_out_k, [cat_cols_two, cont_cols], done_test_k)]
    
    # Special case: variables with only one category
    cat_phenotypes_one = cat_phenotypes.loc[:, cat_phenotypes.nunique() <= 1].copy()
    cat_phenotypes_one, cat_cols_one = _df_to_numpy(cat_phenotypes_one)
    
    if cat_phenotypes_one.shape[1] > 0:
        logging.warning(
            f'There were categorical variables found with only one category: {cat_cols_one}. '
            'These will be added as dummy rows with p-value 1.0 and effect size 0.0.')

        # For each combination of cat_cols_one and cont_cols, add a row with p-value 1.0 and effect size 0.0
        for i, df in enumerate(results):
            if df is None:
                continue

            p_cols = [col for col in df.columns if "_p_" in col]
            e_cols = [col for col in df.columns if "_e_" in col]

            special_rows = []
            for cat_var in cat_cols_one:
                for cont_var in cont_cols:
                    row = {
                        'label1': cat_var,
                        'label2': cont_var,
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


def nanpy_cont_cont(cont_phenotypes: pd.DataFrame, test: str, num_workers=8, nan_value=-89):
    if cont_phenotypes.shape[1] < 2:
        return [None, None]
    cont_phenotypes, cont_cols = _df_to_numpy(cont_phenotypes)
    cont_out_p, cont_out_s = None, None
    test_p, test_s = None, None

    if test in ['all', 'pearson']:
        cont_out_p = nanpy.pearsonr(cont_phenotypes, nan_value=nan_value, threads=num_workers,
                                    axis=1)
        test_p = "pearson"

    if test in ['all', 'spearman']:
        cont_out_s = nanpy.spearmanr(cont_phenotypes, threads=num_workers, nan_value=nan_value,
                                     axis=1)
        test_s = "spearman"
    return [_nanpy_formatting(cont_out_p, [cont_cols], test_p),
            _nanpy_formatting(cont_out_s, [cont_cols], test_s)]


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


def separate_cat_cont(all_data, meta_file) -> tuple[pd.DataFrame, pd.DataFrame] | tuple[None, None]:
    """
    We separate the categorical and continuous variables from the data. Any other input type is continuous.
    :param all_data: DataFrame with all data
    :param phenotypes_meta: DataFrame with metadata of the variables
    :return: tuple with the categorical and continuous phenotypes
    """
    if isinstance(all_data, type(None)):
        return None, None
    if isinstance(meta_file, type(None)):
        return pd.DataFrame(), all_data.copy()

    cat_data = all_data.iloc[:, all_data.columns.isin(meta_file[meta_file.type.str.lower()
                                                      .isin(["categorical", "boolean"])].label)].copy()

    # Extract continuous phenotypes
    cont_data = all_data.iloc[:, ~all_data.columns.isin(meta_file[meta_file.type.str.lower()
                                                        .isin(["categorical", "boolean", "time"])].label)].copy()
    return cat_data, cont_data


def calculate_association_scores(cat_data, cont_data, tests, num_workers=1, nan_value=-89) -> pd.DataFrame:
    if isinstance(tests, str):
        tests = {'cont_cont': tests,
                 'cat_cat': tests,
                 'bi_cont': tests,
                 'cont_cat': tests}

    cont_data = cont_data.copy()
    cont_data = cont_data.select_dtypes(include=[np.number])

    cat_data = _order_categories(cat_data)

    tests = {k: v.lower() for k, v in tests.items()}

    cont_cat_results = nanpy_cat_cont(cont_data, cat_data, tests.get('cont_cat'), num_workers=num_workers, nan_value=nan_value)
    logging.info("Finished continuous-categorical score creation")

    bi_cont_results = nanpy_binary_cat_cont(cont_data, cat_data, test=tests.get('bi_cont'), num_workers=num_workers, nan_value=nan_value)
    logging.info("Finished continuous-binary score creation")
    
    cat_cat_results = nanpy_cat_cat(cat_data, num_workers=num_workers, nan_value=nan_value)
    logging.info("Finished categorical-categorical score creation")
    
    cont_cont_results = nanpy_cont_cont(cont_data, test=tests.get('cont_cont'), num_workers=num_workers, nan_value=nan_value)
    logging.info("Finished continuous-continuous score creation")

    scores = _combine_tests(cat_cat_results, cont_cont_results, bi_cont_results, cont_cat_results)
    
    return scores
