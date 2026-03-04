from typing import Optional
import numpy as np
import random
import pandas as pd
import scipy as sc
import os


# Simulate mixed data using a gaussian copula 
def simulate_copula(path=None, name1='context1', name2='context2',
                    n_bi=50, n_cont=50, n_cat=50, n_samples=500, 
                    n_shift_cont=0, n_shift_bi=0, n_shift_cat=0, 
                    n_corr_cont_cont=0, n_corr_bi_bi=0, n_corr_cat_cat=0, n_corr_bi_cont=0, n_corr_bi_cat=0, n_corr_cont_cat=0, 
                    n_both_cont_cont=0, n_both_bi_bi=0, n_both_cat_cat=0, n_both_bi_cont=0, n_both_bi_cat=0, n_both_cont_cat=0,
                    shift=0.5, corr=0.7):
    """
    Simulate two contexts with binary and continuous nodes using a Gaussian copula.
    
    :param path: Path to save the simulated contexts, the meta file and the ground truth information. If None, files are not saved.
    :param name1: Name of the first context.
    :param name2: Name of the second context.
    :param n_bi: Number of binary nodes to simulate.
    :param n_cont: Number of continuous nodes to simulate.
    :param n_cat: Number of categorical nodes to simulate.
    :param n_samples: Number of samples per context.
    :param n_shift_cont: Number of continuous nodes with an artificially introduced mean shift.
    :param n_shift_bi: Number of binary nodes with an artificially introduced mean shift.
    :param n_shift_cat: Number of categorical nodes with an artificially introduced mean shift.
    :param n_corr_cont_cont: Number of continuous node pairs with an artifically introduced correlation difference.
    :param n_corr_bi_bi: Number of binary node pairs with an artificially introduced correlation difference.
    :param n_corr_cat_cat: Number of categorical node pairs with an artificially introduced correlation difference.
    :param n_corr_bi_cat: Number of binary-categorical node pairs with an artificially introduced correlation difference.
    :param n_corr_cont_cat: Number of continuous-categorical node pairs with an artificially introduced correlation difference.
    :param n_corr_bi_cont: Number of mixed node pairs with an artificially introduced correlation difference.
    :param n_both_cont_cont: Number of continuous node pairs with both an aritificially introduced mean shift and correlation difference.
    :param n_both_bi_bi: Number of binary node pairs with both an artificially introduced mean shift and correlation difference.
    :param n_both_cat_cat: Number of categorical node pairs with both an artificially introduced mean shift and correlation difference.
    :param n_both_bi_cat: Number of binary-categorical node pairs with both an artificially introduced mean shift and correlation difference.
    :param n_both_cont_cat: Number of continuous-categorical node pairs with both an artificially introduced mean shift and correlation difference.
    :param n_both_bi_cont: Number of mixed node pairs with both an artificially introduced mean shift and correlation difference.
    :param shift: Magnitude of the mean shift.
    :param corr: Magnitude of the correlation difference (measured as correlation coefficient between 0 and 1).
    :return: A tuple containing the two simulated contexts, a meta file and a list of ground truth nodes.
             - context1: pd.DataFrame of the first simulated context.
             - context2: pd.DataFrame of the second simulated context.
             - meta: pd.DataFrame containing the data type for each simulated variable.
             - ground_truth: A tuple containing three lists of ground truth nodes: (shift_nodes, corr_nodes, shift_corr_nodes).
    """
    if n_bi <= 0 and n_cont <= 0 and n_cat <= 0:
        raise ValueError('Either n_bi, n_cont, or n_cat needs to be larger than zero.') 
    if n_shift_cont + n_corr_cont_cont*2 + n_both_cont_cont*2 + n_corr_bi_cont + n_both_bi_cont + n_corr_cont_cat + n_both_cont_cat > n_cont:
        raise ValueError('The number of continuous abnormal nodes is larger than the total number of continuous nodes.')
    if n_shift_bi + n_corr_bi_bi*2 + n_both_bi_bi*2 + n_corr_bi_cont + n_both_bi_cont + n_corr_bi_cat + n_both_bi_cat > n_bi:
        raise ValueError('The number of binary abnormal nodes is larger than the total number of binary nodes.')
    if n_shift_cat + n_corr_cat_cat*2 + n_both_cat_cat*2 + n_corr_bi_cat + n_both_bi_cat + n_corr_cont_cat + n_both_cont_cat > n_cat:
        raise ValueError('The number of categorical abnormal nodes is larger than the total number of categorical nodes.')

    # Prepare dataframes
    cont_cols = [f"cont{i+1}" for i in range(n_cont)]
    context1_cont = pd.DataFrame(np.nan, index=range(n_samples), columns=cont_cols)
    context2_cont = pd.DataFrame(np.nan, index=range(n_samples), columns=cont_cols)

    bi_cols = [f"bi{i+1}" for i in range(n_bi)]
    context1_bi = pd.DataFrame(np.nan, index=range(n_samples), columns=bi_cols)
    context2_bi = pd.DataFrame(np.nan, index=range(n_samples), columns=bi_cols)

    cat_cols = [f"ord{i+1}" for i in range(n_cat)]
    context1_cat = pd.DataFrame(np.nan, index=range(n_samples), columns=cat_cols)
    context2_cat = pd.DataFrame(np.nan, index=range(n_samples), columns=cat_cols)

    # Create meta file
    all_cols = cont_cols + bi_cols + cat_cols
    meta = pd.DataFrame({
        "label": all_cols,
        "type": ["continuous"] * n_cont + ["binary"] * n_bi + ["ordinal"] * n_cat
    })

    # Initialize lists for ground truth nodes
    shift_nodes = []
    corr_nodes = []
    shift_corr_nodes = []

    normal_nodes_cont = list(context1_cont.columns) if n_cont > 0 else []
    normal_nodes_bi = list(context1_bi.columns) if n_bi > 0 else []
    normal_nodes_cat = list(context1_cat.columns) if n_cat > 0 else []
    nodes = normal_nodes_cont + normal_nodes_bi + normal_nodes_cat

    # Create correlation matrix
    n_vars = n_bi + n_cont + n_cat
    corr1 = np.eye(n_vars)
    corr2 = np.eye(n_vars)

    # Introduce fixed correlations in context 1 (leave context 2 uncorrelated)
    for _ in range(n_corr_cont_cont):
        node_pair, corr1, corr2, _, normal_nodes_cont, _ = _set_corr(nodes=nodes, corr_param=corr, corr_matrix1=corr1, corr_matrix2=corr2, normal_nodes_cont=normal_nodes_cont)
        corr_nodes.append(node_pair)

    for _ in range(n_corr_bi_bi):
        node_pair, corr1, corr2, normal_nodes_bi, _, _ = _set_corr(nodes=nodes, corr_param=corr, corr_matrix1=corr1, corr_matrix2=corr2, normal_nodes_bi=normal_nodes_bi)
        corr_nodes.append(node_pair)

    for _ in range(n_corr_cat_cat):
        node_pair, corr1, corr2, _, _, normal_nodes_cat = _set_corr(nodes=nodes, corr_param=corr, corr_matrix1=corr1, corr_matrix2=corr2, normal_nodes_cat=normal_nodes_cat)
        corr_nodes.append(node_pair)

    for _ in range(n_corr_bi_cont):
        node_pair, corr1, corr2, normal_nodes_bi, normal_nodes_cont, _ = _set_corr(nodes=nodes, corr_param=corr, corr_matrix1=corr1, corr_matrix2=corr2, normal_nodes_bi=normal_nodes_bi, normal_nodes_cont=normal_nodes_cont)
        corr_nodes.append(node_pair)

    for _ in range(n_corr_bi_cat):
        node_pair, corr1, corr2, normal_nodes_bi, _, normal_nodes_cat = _set_corr(nodes=nodes, corr_param=corr, corr_matrix1=corr1, corr_matrix2=corr2, normal_nodes_bi=normal_nodes_bi, normal_nodes_cat=normal_nodes_cat)
        corr_nodes.append(node_pair)

    for _ in range(n_corr_cont_cat):
        node_pair, corr1, corr2, _, normal_nodes_cont, normal_nodes_cat = _set_corr(nodes=nodes, corr_param=corr, corr_matrix1=corr1, corr_matrix2=corr2, normal_nodes_cont=normal_nodes_cont, normal_nodes_cat=normal_nodes_cat)
        corr_nodes.append(node_pair)

    for _ in range(n_both_cont_cont):
        node_pair, corr1, corr2, _, normal_nodes_cont, _ = _set_corr(nodes=nodes, corr_param=corr, corr_matrix1=corr1, corr_matrix2=corr2, normal_nodes_cont=normal_nodes_cont)
        shift_corr_nodes.append(node_pair)

    for _ in range(n_both_bi_bi):
        node_pair, corr1, corr2, normal_nodes_bi, _, _ = _set_corr(nodes=nodes, corr_param=corr, corr_matrix1=corr1, corr_matrix2=corr2, normal_nodes_bi=normal_nodes_bi)
        shift_corr_nodes.append(node_pair)

    for _ in range(n_both_cat_cat):
        node_pair, corr1, corr2, _, _, normal_nodes_cat = _set_corr(nodes=nodes, corr_param=corr, corr_matrix1=corr1, corr_matrix2=corr2, normal_nodes_cat=normal_nodes_cat)
        shift_corr_nodes.append(node_pair)

    for _ in range(n_both_bi_cont):
        node_pair, corr1, corr2, normal_nodes_bi, normal_nodes_cont, _ = _set_corr(nodes=nodes, corr_param=corr, corr_matrix1=corr1, corr_matrix2=corr2, normal_nodes_bi=normal_nodes_bi, normal_nodes_cont=normal_nodes_cont)
        shift_corr_nodes.append(node_pair)

    for _ in range(n_both_bi_cat):
        node_pair, corr1, corr2, normal_nodes_bi, _, normal_nodes_cat = _set_corr(nodes=nodes, corr_param=corr, corr_matrix1=corr1, corr_matrix2=corr2, normal_nodes_bi=normal_nodes_bi, normal_nodes_cat=normal_nodes_cat)
        shift_corr_nodes.append(node_pair)

    for _ in range(n_both_cont_cat):
        node_pair, corr1, corr2, _, normal_nodes_cont, normal_nodes_cat = _set_corr(nodes=nodes, corr_param=corr, corr_matrix1=corr1, corr_matrix2=corr2, normal_nodes_cont=normal_nodes_cont, normal_nodes_cat=normal_nodes_cat)
        shift_corr_nodes.append(node_pair)

    # Randomly select nodes for mean shifts
    for _ in range(n_shift_cont):
        assert normal_nodes_cont, 'Introducing correlations was unsuccessful.'
        node = random.choice(normal_nodes_cont)
        normal_nodes_cont.remove(node)
        shift_nodes.append(node)

    for _ in range(n_shift_bi):
        assert normal_nodes_bi, 'Introducing correlations was unsuccessful.'
        node = random.choice(normal_nodes_bi)
        normal_nodes_bi.remove(node)
        shift_nodes.append(node)

    for _ in range(n_shift_cat):
        assert normal_nodes_cat, 'Introducing correlations was unsuccessful.'
        node = random.choice(normal_nodes_cat)
        normal_nodes_cat.remove(node)
        shift_nodes.append(node)

    mean_vector = np.zeros(n_vars)
    for node in nodes:
        if node in shift_nodes or any(node in pair for pair in shift_corr_nodes):
            sign = random.choice([1, -1])
            mean_vector[nodes.index(node)] = sign * shift

    # Gaussian copula
    u1 = _simu_gaussian(n=n_vars, m=n_samples, corr_matrix=corr1, mean_vector=np.zeros(n_vars))
    u2 = _simu_gaussian(n=n_vars, m=n_samples, corr_matrix=corr2, mean_vector=mean_vector)

    # Transform to marginal distributions using the inverse CDF
    for i, node in enumerate(nodes):
        if i < n_cont:
            # Continuous node
            mean = 0.0
            std = 0.5
            context1_cont[node] = sc.stats.norm.ppf(u1[i, :], loc=mean, scale=std)
            context2_cont[node] = sc.stats.norm.ppf(u2[i, :], loc=mean, scale=std)

        elif n_cont <= i < n_cont + n_bi:
            # Binary node
            p = np.random.uniform(0, 1)
            context1_bi[node] = sc.stats.bernoulli.ppf(u1[i, :], p=p).astype(int)
            context2_bi[node] = sc.stats.bernoulli.ppf(u2[i, :], p=p).astype(int)
        
        else:
            # Categorical node
            #n_categories = np.random.randint(3, 10) # Randomly choose number of categories between 3 and 10
            n_categories = 5 # fixed number of categories
            p = np.random.dirichlet(np.ones(n_categories), size=1).flatten()  # Random probabilities for each category
            cdf = np.cumsum(p)
            context1_cat[node] = np.searchsorted(cdf, u1[i, :])
            context2_cat[node] = np.searchsorted(cdf, u2[i, :])

    # Combine continuous and binary data
    context1 = context1_cont.join(context1_bi)
    context2 = context2_cont.join(context2_bi)

    context1 = context1.join(context1_cat)
    context2 = context2.join(context2_cat)

    context1 = context1.sort_index(axis=1)
    context2 = context2.sort_index(axis=1)

    # Save simulated contexts and ground truth nodes
    if path:
        context1.to_csv(os.path.join(path, f'{name1}.csv'))
        context2.to_csv(os.path.join(path, f'{name2}.csv'))
        meta.to_csv(os.path.join(path, 'meta.csv'), index=False)
        save_gt((shift_nodes, corr_nodes, shift_corr_nodes), os.path.join(path, 'ground_truth_nodes.txt'), mode='node')
        save_gt((shift_nodes, corr_nodes, shift_corr_nodes), os.path.join(path, 'ground_truth_edges.txt'), mode='edge')

    return context1, context2, meta, (shift_nodes, corr_nodes, shift_corr_nodes)


# Helper function to set correlation in copula-based simulation
def _set_corr(nodes, corr_param, corr_matrix1, corr_matrix2, normal_nodes_bi=None, normal_nodes_cont=None, normal_nodes_cat=None):
    if normal_nodes_bi is not None and normal_nodes_cont is not None and normal_nodes_cat is None:
        node1 = random.choice(normal_nodes_cont)
        normal_nodes_cont.remove(node1)
        node2 = random.choice(normal_nodes_bi)
        normal_nodes_bi.remove(node2)

    elif normal_nodes_bi is not None and normal_nodes_cat is not None and normal_nodes_cont is None:
        node1 = random.choice(normal_nodes_bi)
        normal_nodes_bi.remove(node1)
        node2 = random.choice(normal_nodes_cat)
        normal_nodes_cat.remove(node2)

    elif normal_nodes_cont is not None and normal_nodes_cat is not None and normal_nodes_bi is None:
        node1 = random.choice(normal_nodes_cont)
        normal_nodes_cont.remove(node1)
        node2 = random.choice(normal_nodes_cat)
        normal_nodes_cat.remove(node2)

    elif normal_nodes_bi is not None and normal_nodes_cont is None and normal_nodes_cat is None:
        node1 = random.choice(normal_nodes_bi)
        normal_nodes_bi.remove(node1)
        node2 = random.choice(normal_nodes_bi)
        normal_nodes_bi.remove(node2)
    
    elif normal_nodes_cont is not None and normal_nodes_bi is None and normal_nodes_cat is None:
        node1 = random.choice(normal_nodes_cont)
        normal_nodes_cont.remove(node1)
        node2 = random.choice(normal_nodes_cont)
        normal_nodes_cont.remove(node2)

    elif normal_nodes_cat is not None and normal_nodes_bi is None and normal_nodes_cont is None:
        node1 = random.choice(normal_nodes_cat)
        normal_nodes_cat.remove(node1)
        node2 = random.choice(normal_nodes_cat)
        normal_nodes_cat.remove(node2)
    
    else:
        raise ValueError('At least one of normal_nodes_bi, normal_nodes_cont, or normal_nodes_cat must be provided.')

    idx1 = nodes.index(node1)
    idx2 = nodes.index(node2)

    # Choose direction of correlation change
    sign = random.choice([1, -1])
    which = random.choice([1, 2])
    
    if which == 1:
        corr_matrix1[idx1, idx2] = corr_matrix1[idx2, idx1] = corr_param * sign
    else:
        corr_matrix2[idx1, idx2] = corr_matrix2[idx2, idx1] = corr_param * sign

    return (node1, node2), corr_matrix1, corr_matrix2, normal_nodes_bi, normal_nodes_cont, normal_nodes_cat


# Adapted from pycop package
def _simu_gaussian(n: int, m: int, corr_matrix: np.ndarray, mean_vector: Optional[np.ndarray]=None):
    """ 
    # Gaussian Copula simulations with a given correlation matrix

    :param n: number of simulated variables
    :param m: sample size
    :param corr_matrix: correlation matrix
    :return: simulated samples from a Gaussian copula

    """

    if not all(isinstance(v, int) for v in [n, m]):
        raise TypeError("The 'n' and 'm' arguments must both be integer types.")
    if not isinstance(corr_matrix, np.ndarray):
        raise TypeError("The 'corr_matrix' argument must be a numpy array.")
    if not isinstance(mean_vector, np.ndarray):
        mean_vector = np.zeros(n)

    # Generate n independent standard Gaussian random variables V = (v1 ,..., vn):
    v = [np.random.normal(mean_vector[i], 1, m) for i in range(0, n)]

    # Compute the lower triangular Cholesky factorization of the correlation matrix:
    l = sc.linalg.cholesky(corr_matrix, lower=True)
    y = np.dot(l, v)
    u = sc.stats.norm.cdf(y, 0, 1)

    return u


# Save ground truth nodes to file
def save_gt(groundtruths, path, mode='node'):
    shift = groundtruths[0]
    corr = groundtruths[1]
    shift_corr = groundtruths[2]

    if mode == 'node':
        with open(path, 'w') as f:
            f.write('node, description\n')
            for node in shift:
                f.write(node + ', mean shift\n')
            for pair in corr:
                f.write(pair[0] + ', diff. corr.\n')
                f.write(pair[1] + ', diff. corr.\n')
            for pair in shift_corr:
                f.write(pair[0] + ', mean shift + diff. corr.\n')
                f.write(pair[1] + ', mean shift + diff. corr.\n')

    if mode == 'edge':
        with open(path, 'w') as f:
            f.write('edge, description\n')
            for pair in corr:
                edge = '_'.join(sorted(pair))
                f.write(edge + ', diff. corr.\n')
            for pair in shift_corr:
                edge = '_'.join(sorted(pair))
                f.write(edge + ', mean shift + diff. corr.\n')

