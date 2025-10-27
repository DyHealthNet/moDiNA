import numpy as np
import random
import pandas as pd
import scipy.stats as sc
from pycop import simulation
import os


# Simulate mixed data using a gaussian copula 
def simulate_copula(path=None, name1='context1', name2='context2',
                    n_bi=50, n_cont=50, n_samples=500, 
                    n_shift_cont=4, n_shift_bi=4, n_corr_cont=2, n_corr_bi=2, n_corr_mixed=2, n_both_cont=2, n_both_bi=2, n_both_mixed=2,
                    shift_cont=1.0, shift_bi=0.3, corr=0.7):
    """
    Simulate two contexts with binary and continuous nodes using a Gaussian copula.
    
    :param path: Path to save the simulated contexts and meta file. If None, files are not saved.
    :param name1: Name of the first context.
    :param name2: Name of the second context.
    :param n_bi: Number of binary nodes to simulate.
    :param n_cont: Number of continuous nodes to simulate.
    :param n_samples: Number of samples per context.
    :param n_shift_cont: Number of continuous nodes with an artificially introduced mean shift.
    :param n_shift_bi: Number of binary nodes with an artificially introduced mean shift.
    :param n_corr_cont: Number of continuous node pairs with an artifically introduced correlation difference.
    :param n_corr_bi: Number of binary node pairs with an artificially introduced correlation difference.
    :param n_corr_mixed: Number of mixed node pairs with an artificially introduced correlation difference.
    :param n_both_cont: Number of continuous node pairs with both an aritificially introduced mean shift and correlation difference.
    :param n_both_bi: Number of binary node pairs with both an artificially introduced mean shift and correlation difference.
    :param n_both_mixed: Number of mixed node paris with both an artificially introduced mean shift and correlation difference.
    :param shift_cont: Magnitude of the mean shift for continuous nodes (measured in standard deviations).
    :param shift_bi: Magnitude of the mean shift for binary nodes (measured as change in class probability).
    :param corr: Magnitude of the correlation difference (measured as correlation coefficient between 0 and 1).
    :return: A tuple containing the two simulated contexts, a meta file and a list of ground truth nodes.
             - context1: pd.DataFrame of the first simulated context.
             - context2: pd.DataFrame of the second simulated context.
             - meta: pd.DataFrame containing the data type for each simulated variable.
             - ground_truth: A tuple containing three lists of ground truth nodes: (shift_nodes, corr_nodes, shift_corr_nodes).
    """
    if n_bi <= 0 and n_cont <= 0:
        raise ValueError('Either n_bi or n_cont needs to be larger than zero.') 
    if n_shift_cont + n_corr_cont*2 + n_both_cont*2 + n_corr_mixed + n_both_mixed > n_cont:
        raise ValueError('The number of continuous abnormal nodes is larger than the total number of continuous nodes.')
    if n_shift_bi + n_corr_bi*2 + n_both_bi*2 + n_corr_mixed + n_both_mixed > n_bi:
        raise ValueError('The number of binary abnormal nodes is larger than the total number of binary nodes.')

    # Prepare dataframes
    cont_cols = [f"cont{i+1}" for i in range(n_cont)]
    context1_cont = pd.DataFrame(np.nan, index=range(n_samples), columns=cont_cols)
    context2_cont = pd.DataFrame(np.nan, index=range(n_samples), columns=cont_cols)

    bi_cols = [f"bi{i+1}" for i in range(n_bi)]
    context1_bi = pd.DataFrame(np.nan, index=range(n_samples), columns=bi_cols)
    context2_bi = pd.DataFrame(np.nan, index=range(n_samples), columns=bi_cols)

    # Create meta file
    all_cols = cont_cols + bi_cols
    meta = pd.DataFrame({
        "label": all_cols,
        "type": ["continuous"] * n_cont + ["boolean"] * n_bi
    })

    # Initialize lists for ground truth nodes
    shift_nodes = []
    corr_nodes = []
    shift_corr_nodes = []

    normal_nodes_cont = list(context1_cont.columns) if n_cont > 0 else []
    normal_nodes_bi = list(context1_bi.columns) if n_bi > 0 else []
    nodes = normal_nodes_cont + normal_nodes_bi

    # Create correlation matrix
    n_vars = n_bi + n_cont
    corr1 = np.eye(n_vars)
    corr2 = np.eye(n_vars)

    # Introduce fixed correlations in context 1 (leave context 2 uncorrelated)
    for _ in range(n_corr_cont):
        node_pair, corr1, _, normal_nodes_cont = _set_corr(nodes=nodes, corr_param=corr, corr_matrix=corr1, normal_nodes_cont=normal_nodes_cont)
        corr_nodes.append(node_pair)

    for _ in range(n_corr_bi):
        node_pair, corr1, normal_nodes_bi, _ = _set_corr(nodes=nodes, corr_param=corr, corr_matrix=corr1, normal_nodes_bi=normal_nodes_bi)
        corr_nodes.append(node_pair)

    for _ in range(n_corr_mixed):
        node_pair, corr1, normal_nodes_bi, normal_nodes_cont = _set_corr(nodes=nodes, corr_param=corr, corr_matrix=corr1, normal_nodes_bi=normal_nodes_bi, normal_nodes_cont=normal_nodes_cont)
        corr_nodes.append(node_pair)

    for _ in range(n_both_cont):
        node_pair, corr1, _, normal_nodes_cont = _set_corr(nodes=nodes, corr_param=corr, corr_matrix=corr1, normal_nodes_cont=normal_nodes_cont)
        shift_corr_nodes.append(node_pair)

    for _ in range(n_both_bi):
        node_pair, corr1, normal_nodes_bi, _ = _set_corr(nodes=nodes, corr_param=corr, corr_matrix=corr1, normal_nodes_bi=normal_nodes_bi)
        shift_corr_nodes.append(node_pair)
    
    for _ in range(n_both_mixed):
        node_pair, corr1, normal_nodes_bi, normal_nodes_cont = _set_corr(nodes=nodes, corr_param=corr, corr_matrix=corr1, normal_nodes_bi=normal_nodes_bi, normal_nodes_cont=normal_nodes_cont)
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

    # Gaussian copula
    u1 = simulation.simu_gaussian(n=n_vars, m=n_samples, corr_matrix=corr1)
    u2 = simulation.simu_gaussian(n=n_vars, m=n_samples, corr_matrix=corr2)

    # Transform to marginal distributions using the inverse CDF
    for i, node in enumerate(nodes):
        if i < n_cont:
            # Continuous node
            if node in shift_nodes or any(node in pair for pair in shift_corr_nodes):
                # Mean shift
                mean = np.random.uniform(-100, 100)
                std = 1.0
                sign = random.choice([1, -1])
                context1_cont[node] = sc.norm.ppf(u1[i, :], loc=mean, scale=std)
                mean += shift_cont * std * sign
                context2_cont[node] = sc.norm.ppf(u2[i, :], loc=mean, scale=std)
            
            else:
                # No mean shift
                mean = np.random.uniform(-100, 100)
                std = 1.0
                context1_cont[node] = sc.norm.ppf(u1[i, :], loc=mean, scale=std)
                context2_cont[node] = sc.norm.ppf(u2[i, :], loc=mean, scale=std)

        else:
            # Binary node
            if node in shift_nodes or any(node in pair for pair in shift_corr_nodes):
                # Mean shift
                if shift_bi > 0.5:
                    raise ValueError('Binary mean shift needs to be smaller or equal to 0.5')
                p = np.random.uniform(0, 1)
                context1_bi[node] = sc.bernoulli.ppf(u1[i, :], p=p)
                if p > 0.5:
                    p -= shift_bi
                else:
                    p += shift_bi  
                context2_bi[node] = sc.bernoulli.ppf(u2[i, :], p=p)
            else:
                # No mean shift
                p = np.random.uniform(0, 1)
                context1_bi[node] = sc.bernoulli.ppf(u1[i, :], p=p)
                context2_bi[node] = sc.bernoulli.ppf(u2[i, :], p=p)

    ground_truth = shift_nodes + corr_nodes + shift_corr_nodes

    # Combine continuous and binary data
    context1 = context1_cont.join(context1_bi)
    context2 = context2_cont.join(context2_bi)

    context1 = context1.sort_index(axis=1)
    context2 = context2.sort_index(axis=1)

    # Save simulated contexts and ground truth nodes
    if path:
        context1.to_csv(os.path.join(path, f'{name1}.csv'))
        context2.to_csv(os.path.join(path, f'{name2}.csv'))

    return context1, context2, meta, (shift_nodes, corr_nodes, shift_corr_nodes)


# Helper function to set correlation in copula-based simulation
def _set_corr(nodes, corr_param, corr_matrix, normal_nodes_bi=None, normal_nodes_cont=None):
    if normal_nodes_bi is not None and normal_nodes_cont is not None:
        node1 = random.choice(normal_nodes_cont)
        normal_nodes_cont.remove(node1)
        node2 = random.choice(normal_nodes_bi)
        normal_nodes_bi.remove(node2)

    elif normal_nodes_cont is None and normal_nodes_bi is not None:
        node1 = random.choice(normal_nodes_bi)
        normal_nodes_bi.remove(node1)
        node2 = random.choice(normal_nodes_bi)
        normal_nodes_bi.remove(node2)
    
    elif normal_nodes_bi is None and normal_nodes_cont is not None:
        node1 = random.choice(normal_nodes_cont)
        normal_nodes_cont.remove(node1)
        node2 = random.choice(normal_nodes_cont)
        normal_nodes_cont.remove(node2)
    
    else:
        raise ValueError('At least one of normal_nodes_bi or normal_nodes_cont must be provided.')

    idx1 = nodes.index(node1)
    idx2 = nodes.index(node2)
    sign = random.choice([1, -1])
    corr_matrix[idx1, idx2] = corr_matrix[idx2, idx1] = corr_param * sign

    return (node1, node2), corr_matrix, normal_nodes_bi, normal_nodes_cont
