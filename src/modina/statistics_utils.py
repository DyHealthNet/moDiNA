import numpy as np


# Pre-rescaling (Z-score rescaling)
def pre_rescaling(scores1, scores2, metric):
    scores1 = scores1.copy()
    scores2 = scores2.copy()

    # TODO: eventually remove p-value rescaling completely (change all code snippets where pre-P is used!)
    if metric == 'pre-P': # no rescaling
        metric_raw = 'raw-P'
        scores1[metric] = scores1[metric_raw]
        scores2[metric] = scores2[metric_raw]
    
    elif metric == 'pre-E':
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

            # Min-Max normalization
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
        raise ValueError(f"Invalid metric '{metric}'. Only 'pre-E' and 'pre-P' are supported.")
    
    return scores1, scores2


# Post-rescaling
def post_rescaling(diff_scores, metric):
    diff_scores = diff_scores.copy()

    if metric == 'post-LS':
        metric_raw = 'raw-LS'
    elif metric == 'post-E':
        metric_raw = 'raw-E'
    elif metric == 'post-P':
        metric_raw = 'raw-P'
    elif metric == 'post-CS':
        metric_raw = 'raw-CS'
    else:
        raise ValueError(f"Invalid metric '{metric}'. Only 'post-E', 'post-P', 'post-CS', and 'post-LS' are supported.")
    
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

    