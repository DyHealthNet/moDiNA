import os
import logging
import math
from typing import Optional, Tuple
import json

from modina.score_calculation import *
from modina.statistics_utils import *


class DiffNet:
    def __init__(self, context1: pd.DataFrame, context2: pd.DataFrame, meta_file: pd.DataFrame, edge_metric: str = 'pre-LS', node_metric: str = 'STC',
                 filter_method: Optional[str] = None, filter_param: float = 0.0, filter_metric: Optional[str] = None, filter_rule: Optional[str]=None,
                 stc_test: str = 'non-parametric', max_path_length: int=2,
                 cont_cont: str = 'spearman', cat_cat: str = 'chi2', bi_cont: str = 'mann-whitney u', cont_cat: str = 'kruskal-wallis', 
                 correction: str = 'bh', nan_value: int = -89, num_workers: int=1,
                 project_path: Optional[str] = None, name1: str = 'context1', name2: str = 'context2', save_config: bool = False):
        """
        Initialize the Differential Network class.

        :param context1: The first context for the differential network analysis.
        :param context2: The second context for the differential network analysis.
        :param meta_file: Metadata file containing a 'label' and 'type' column to specify the data type of each variable.
        :param edge_metric: Edge metric used to construct the differential network. Defaults to 'pre-LS'.
        :param node_metric: Node metric used to construct the differential network. Defaults to 'STC'.
        :param filter_method: Method used for filtering. Defaults to None.
        :param filter_param: Parameter for the specified filtering method. Defaults to 0.0.
        :param filter_metric: Edge metric used for filtering. Defaults to None.
        :param filter_rule: Rule to integrate the networks during filtering. Defaults to None.
        :param stc_test: Statistical test to use for significance testing in STC node metric. Defaults to 'non-parametric'.
        :param max_path_length: Maximum length of paths to consider in the computation of integrated interaction scores. Defaults to 2.
        :param cont_cont: Test for continuous-continuous association scores. Defaults to 'spearman'.
        :param cat_cat: Test for categorical-categorical association scores. Defaults to 'chi2'.
        :param bi_cont: Test for categorical-continuous association (binary) scores. Defaults to 'mann-whitney u'.
        :param cont_cat: Test for categorical-continuous association (multiple) scores. Defaults to 'kruskal-wallis'.
        :param correction: Correction method for multiple testing. Defaults to 'bh'.
        :param nan_value: Value to represent NaN in the data. Defaults to -89.
        :param num_workers: Number of workers for parallel processing. Defaults to 1.
        :param project_path: Path to the project directory. Defaults to None.
        :param name1: Name for the first context. Defaults to 'context1'.
        :param name2: Name for the second context. Defaults to 'context2'.
        :param save_config: Whether to save the configurations used to construct the differential network. Defaults to True.
        """
        # Check if input data is in a valid format
        self._check_input_data(context=context1, meta_file=meta_file)
        self._check_input_data(context=context2, meta_file=meta_file)

        # Project path and context names
        self._project_path = project_path
        self._name1 = name1
        self._name2 = name2

        # Raw and processed scores
        self._scores1 = None
        self._scores2 = None
        self._scores1_processed = None
        self._scores2_processed = None

        # Association score tests
        self._cont_cont = cont_cont
        self._cat_cat = cat_cat
        self._bi_cont = bi_cont
        self._cont_cat = cont_cat
        self._correction = correction

        # Differential network
        self._filter_method = filter_method
        self._filter_metric = filter_metric
        self._filter_rule = filter_rule
        self._filter_param = filter_param
        self._edge_metric = edge_metric
        self._node_metric = node_metric
        self._stc_test = stc_test
        self._max_path_length = max_path_length

        # Rankings
        self._personalized_pagerank = None
        self._pagerank = None
        self._dimontrank = None
        self._abs_dimontrank = None
        self._direct_node_rank = None
        self._direct_edge_rank = None

        # Compute differential network
        self._edges_diff, self._nodes_diff = self._compute_diff_network(context1=context1, context2=context2, meta_file=meta_file, nan_value=nan_value, num_workers=num_workers)

        # Save configuration
        if save_config:
            self.save_config()


    def _compute_diff_network(self, context1: pd.DataFrame, context2: pd.DataFrame, meta_file: pd.DataFrame,
                              nan_value: int = -89, num_workers: int = 1) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        End-to-end computation of a differential network by:
        - Computing association scores for each context,
        - Optionally filtering scores and context data according to the filtering configurations,
        - Computing differential edge and node scores based on the specified edge and node metrics.
        
        :param context1: The first context for the differential network analysis.
        :param context2: The second context for the differential network analysis.
        :param meta_file: Metadata file containing a 'label' and 'type' column to specify the data type of each variable.
        :param nan_value: Value to represent NaN in the data. Defaults to -89.
        :param num_workers: Number of workers for parallel processing. Defaults to 1.
        :return: A tuple (edges_diff, nodes_diff) containing the computed differential edges and nodes.
        """
        logging.info('Starting differential network computation...')

        # Compute context scores
        self._scores1 = self._compute_context_scores(context_data=context1, meta_file=meta_file, nan_value=nan_value, num_workers=num_workers)
        self._scores2 = self._compute_context_scores(context_data=context2, meta_file=meta_file, nan_value=nan_value, num_workers=num_workers)
        assert self._scores1 is not None and self._scores2 is not None, 'Score calculation was unsuccessful.'
        logging.info('Association score calculation was successful for both contexts.')

        # Min-Max rescaling for raw-P and raw-E
        self._scores1, self._scores2 = min_max_rescaling(scores1=self._scores1,
                                                         scores2=self._scores2,
                                                         metric='adj-P')
        self._scores1, self._scores2 = min_max_rescaling(scores1=self._scores1,
                                                         scores2=self._scores2,
                                                         metric='pre-E')
        assert self._scores2 is not None, 'Min-Max rescaling was unsuccessful.'

        # Filtering
        if any([self._filter_method, self._filter_metric, self._filter_rule, self._filter_param]):
            self._scores1_processed, self._scores2_processed, context1_filtered, context2_filtered = self._filter(context1=context1, context2=context2)
        else:
            logging.warning('The differential network will be computed based on unfiltered data. No filter parameters were specified.')
            self._scores1_processed = self._scores1.copy()
            self._scores2_processed = self._scores2.copy()
            context1_filtered = context1.copy()
            context2_filtered = context2.copy()
            
        # Edges
        edges_diff = self._compute_diff_edges()

        # Nodes
        nodes_diff = self._compute_diff_nodes(context1=context1_filtered, context2=context2_filtered)

        logging.info('Differential network computation was successfully completed. It is now possible to compute rankings using the compute_rankings() method.')

        return edges_diff, nodes_diff 


    def _compute_context_scores(self, context_data: pd.DataFrame, meta_file: pd.DataFrame, nan_value: int = -89, num_workers: int = 1) -> pd.DataFrame:
        """
        Compute association scores for a given context.
        
        :param context_data: The raw context data (rows: samples, columns: variables).
        :param meta_file: Metadata file containing a 'label' and 'type' column to specify the data type of each variable.
        :param nan_value: Value to represent NaN in the data. Defaults to -89.
        :param num_workers: Number of workers for parallel processing. Defaults to 1.
        :return: A pd.DataFrame containing the computed association scores.
        """
        # Separate the data into categorical and continuous data
        cat, cont = separate_cat_cont(context_data, meta_file)

        # Get test types
        tests = {
            "cont_cont": self._cont_cont,
            "cat_cat": self._cat_cat,
            "bi_cont": self._bi_cont,
            "cont_cat": self._cont_cat
        }

        # Calculate scores
        scores = calculate_association_scores(cat_data=cat, cont_data=cont, tests=tests, num_workers=num_workers, nan_value=nan_value)

        # Take the adjusted p-value and the corresponding effect size
        column_names = scores.iloc[:, 2:].columns
        if self._correction == 'bh':
            correction = 'benjamini_hb'
        else:
            correction = self._correction

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
        
        scores_final = scores_final.sort_values(by=['label1', 'label2']).reset_index(drop=True)

        return scores_final     
        

    def _filter(self, context1: pd.DataFrame, context2: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Filter association scores and context data based on the specified filtering configurations.

        :param context1: The first context for the differential network analysis.
        :param context2: The second context for the differential network analysis.
        :return: A tuple containing the filtered scores and context data.
        """

        assert self._scores1 is not None and self._scores2 is not None, 'Association scores have not been computed yet.'
        assert self._scores1['label1'].equals(self._scores2['label1']), 'Contexts need to have the same structure and order of edges.'
        assert self._scores1['label2'].equals(self._scores2['label2']), 'Contexts need to have the same structure and order of edges.'
        assert self._filter_metric in self._scores1.columns and self._filter_metric in self._scores2.columns, 'Min-Max rescaling was not performed yet.'
        
        # Check input parameters
        if self._filter_method is None:
            raise ValueError("Please provide a 'filter_method'.")
        
        if self._filter_metric is None:
            raise ValueError("Please provide a 'filter_metric'.")
        
        if self._filter_rule is None:
            raise ValueError("Please provide a 'filter_rule'.")
        
        if self._filter_param is None:
            raise ValueError("Please provide a 'filter_param'.")


        ascending = True if self._filter_metric == 'adj-P' else False

        # Compute filtering threshold according to the specified method
        threshold1 = None
        threshold2 = None
        n_nodes = context1.shape[1]
        n_edges_before = self._scores1.shape[0]

        if self._filter_method == 'threshold':
            threshold1 = threshold2 = self._filter_param

        elif self._filter_method == 'degree':
            degree = self._filter_param
            n_filtered_edges = math.ceil(degree * n_nodes / 2)

            threshold1 = self._scores1[self._filter_metric].sort_values(ascending=ascending).iloc[n_filtered_edges - 1]
            threshold2 = self._scores2[self._filter_metric].sort_values(ascending=ascending).iloc[n_filtered_edges - 1]

        elif self._filter_method == 'density':
            density = self._filter_param
            possible_edges = n_nodes * (n_nodes - 1) / 2
            n_filtered_edges = math.ceil(density * possible_edges)

            threshold1 = self._scores1[self._filter_metric].sort_values(ascending=ascending).iloc[n_filtered_edges - 1]
            threshold2 = self._scores2[self._filter_metric].sort_values(ascending=ascending).iloc[n_filtered_edges - 1]

        else:
            raise ValueError(f"Invalid filtering method '{self._filter_method}'. Choose from: 'threshold', 'degree' or 'density'")
        
        # Apply the filtering threshold to scores and raw data if provided
        if self._filter_rule == 'union':
            if ascending is True:
                mask = (self._scores1[self._filter_metric] <= threshold1) | (
                        self._scores2[self._filter_metric] <= threshold2)
            else:
                mask = (self._scores1[self._filter_metric] >= threshold1) | (
                        self._scores2[self._filter_metric] >= threshold2)
            
            # Apply mask
            scores1_processed = self._scores1[mask].copy()
            scores2_processed = self._scores2[mask].copy()

        elif self._filter_rule == 'zero':
            if ascending is True:
                mask1 = self._scores1[self._filter_metric] <= threshold1
                mask2 = self._scores2[self._filter_metric] <= threshold2
            else:
                mask1 = self._scores1[self._filter_metric] >= threshold1
                mask2 = self._scores2[self._filter_metric] >= threshold2
            
            # Apply mask
            filtered1 = self._scores1[mask1].copy()
            filtered2 = self._scores2[mask2].copy()
            filtered1 = filtered1.set_index(['label1', 'label2'])
            filtered2 = filtered2.set_index(['label1', 'label2'])

            # Unify indices and set missing values
            indices = filtered1.index.union(filtered2.index)
            scores1_processed = filtered1.reindex(indices)
            scores2_processed = filtered2.reindex(indices)

            fill_values = {
                'raw-P': 1.0,
                'adj-P': 1.0,
                'raw-E': 0.0,
                'pre-E': 0.0,
            }

            for metric, value in fill_values.items():
                if metric in scores1_processed.columns:
                    scores1_processed[metric] = scores1_processed[metric].fillna(value)
                if metric in scores2_processed.columns:
                    scores2_processed[metric] = scores2_processed[metric].fillna(value)
            
            if 'test_type' in scores1_processed.columns and 'test_type' in scores2_processed.columns:
                merged_test_type = scores1_processed['test_type'].combine_first(scores2_processed['test_type'])
                scores1_processed['test_type'] = merged_test_type
                scores2_processed['test_type'] = merged_test_type

            scores1_processed = scores1_processed.reset_index()
            scores2_processed = scores2_processed.reset_index()
        
        else:
            raise ValueError(f"Invalid filtering rule '{self._filter_rule}'.")

        filtered_nodes = np.concatenate((scores1_processed['label1'].values,
                                         scores1_processed['label2'].values))
        filtered_nodes = pd.unique(filtered_nodes)
        context1_filtered = context1[filtered_nodes].copy()
        context2_filtered = context2[filtered_nodes].copy()

        n_edges_after = scores1_processed.shape[0]

        logging.info(f'Successfully filtered the single networks using a {self._filter_method} of {self._filter_param}'
                     f'based on {self._filter_metric} with the {self._filter_rule} rule. Reduced the number of edges from '
                     f'{n_edges_before} to {n_edges_after}.')      
       
        return scores1_processed, scores2_processed, context1_filtered, context2_filtered


    def _compute_diff_edges(self) -> pd.DataFrame:
        """
        Compute differential edge scores based on the specified edge metric.
        
        :return: A DataFrame containing the computed differential edge scores.
        """
        assert self._scores1_processed is not None and self._scores2_processed is not None, 'Filtering was unsuccessful.'
        assert 'adj-P' in self._scores1_processed.columns and 'adj-P' in self._scores2_processed.columns, 'Min-Max rescaling for raw-P was unsuccessful.'
        assert 'pre-E' in self._scores1_processed.columns and 'pre-E' in self._scores2_processed.columns, 'Min-Max rescaling for raw-E was unsuccessful.'

        edges_diff = None

        if self._edge_metric is None:
            logging.warning('No edge_metric was specified. This setting will only work for the direct ranking algorithm.')
            # Take adj-P per default
            edges_diff = subtract_edges(self._scores1_processed, self._scores2_processed,
                                              metrics=['adj-P'], included_cols=('test_type',))

        # Pre-rescaled effect size (pre-E) or rescaled multiple-testing adjusted p-value (adj-P)
        if self._edge_metric == 'adj-P' or self._edge_metric == 'pre-E':
            pass

        # Post-rescaled effect size (post-E)
        elif self._edge_metric == 'post-E':
            # Compute differences in edge metrics first
            edges_diff = subtract_edges(self._scores1_processed, self._scores2_processed,
                                             metrics=['raw-E'], included_cols=['test_type'])
            # Min-Max rescaling
            edges_diff, _ = min_max_rescaling(scores1=edges_diff, metric=self._edge_metric)

        # Pre-rescaled combined score (pre-CS)
        elif self._edge_metric == 'pre-CS':
            # Compute combined score from rescaled effect size and p-value
            self._scores1_processed[self._edge_metric] = self._scores1_processed['pre-E'] - self._scores1_processed['adj-P']
            self._scores2_processed[self._edge_metric] = self._scores2_processed['pre-E'] - self._scores2_processed['adj-P']

        # Post-rescaled combined score (post-CS)
        elif self._edge_metric == 'post-CS':
            # Compute differences in edge metrics first
            edges_diff = subtract_edges(self._scores1_processed, self._scores2_processed,
                                             metrics=['adj-P', 'raw-E'], included_cols=['test_type'])
            # Rescale difference in effect sizes
            edges_diff, _ = min_max_rescaling(scores1=edges_diff, metric='post-E')
            # Compute combined score
            edges_diff[self._edge_metric] = (edges_diff['post-E'] + edges_diff['adj-P'])

        # Integrated Interaction Score (int-IS)
        elif self._edge_metric == 'int-IS':
            # Compute interaction score using DrDimont method
            self._scores1_processed = calculate_interaction_score(self._scores1_processed,
                                                                  metric='pre-E',
                                                                  max_path_length=self._max_path_length)
            self._scores2_processed = calculate_interaction_score(self._scores2_processed,
                                                                  metric='pre-E',
                                                                  max_path_length=self._max_path_length)
        
        # Log-transformed p-value and pre-rescaled effect size combined score
        elif self._edge_metric == 'pre-LS':
            # Replace zero values by small epsilon (1/10 of the minimum non-zero value)
            p_vals_combined = np.concatenate([self._scores1_processed[self._scores1_processed['adj-P'] > 0]['adj-P'].to_numpy(), 
                                              self._scores2_processed[self._scores2_processed['adj-P'] > 0]['adj-P'].to_numpy()])
            min_non_zero = p_vals_combined.min()
            epsilon = min_non_zero / 10.0

            p_vals1 = self._scores1_processed['adj-P'].to_numpy()
            p_vals2 = self._scores2_processed['adj-P'].to_numpy()
            p_vals1 = np.where(p_vals1 == 0, epsilon, p_vals1)
            p_vals2 = np.where(p_vals2 == 0, epsilon, p_vals2)

            # - log10(adj-P) * pre-E
            values1 = - np.log10(p_vals1) * self._scores1_processed['pre-E']
            values2 = - np.log10(p_vals2) * self._scores2_processed['pre-E']

            # Replace -0.0 with +0.0
            values1 = np.where(values1 == -0.0, 0.0, values1)
            values2 = np.where(values2 == -0.0, 0.0, values2)

            self._scores1_processed[self._edge_metric] = values1
            self._scores2_processed[self._edge_metric] = values2

        # Post rescaled absolute difference in (log-transformed raw p-value multiplied by raw effect size)
        elif self._edge_metric == 'post-LS':
            # Replace zero values by small epsilon (1/10 of the minimum non-zero value)
            p_vals_combined = np.concatenate([self._scores1_processed[self._scores1_processed['raw-P'] > 0]['raw-P'].to_numpy(), 
                                              self._scores2_processed[self._scores2_processed['raw-P'] > 0]['raw-P'].to_numpy()])
            min_non_zero = p_vals_combined.min()
            epsilon = min_non_zero / 10.0

            p_vals1 = self._scores1_processed['raw-P'].to_numpy()
            p_vals2 = self._scores2_processed['raw-P'].to_numpy()
            p_vals1 = np.where(p_vals1 == 0, epsilon, p_vals1)
            p_vals2 = np.where(p_vals2 == 0, epsilon, p_vals2)

            # - log10(raw-P) * raw-E
            values1 = - np.log10(p_vals1) * self._scores1_processed['raw-E']
            values2 = - np.log10(p_vals2) * self._scores2_processed['raw-E']

            # Replace -0.0 with +0.0
            values1 = np.where(values1 == -0.0, 0.0, values1)
            values2 = np.where(values2 == -0.0, 0.0, values2)

            self._scores1_processed['raw-LS'] = values1
            self._scores2_processed['raw-LS'] = values2

            # Compute differences in edge metrics first
            edges_diff = subtract_edges(self._scores1_processed, self._scores2_processed,
                                        metrics=['raw-LS'], included_cols=['test_type'])
            # Min-Max rescaling
            edges_diff, _ = min_max_rescaling(scores1=edges_diff, metric=self._edge_metric)

        else:
            raise ValueError(f"Invalid edge metric '{self._edge_metric}'. Choose from: 'adj-P', 'pre-E', 'post-E', 'int-IS', 'pre-CS', 'post-CS', 'pre-LS' or 'post-LS'.")

        if edges_diff is None:
            # Compute difference in edge scores
            edges_diff = subtract_edges(self._scores1_processed, self._scores2_processed,
                                              metrics=[self._edge_metric], included_cols=('test_type',))
        
        return edges_diff
    

    def _compute_diff_nodes(self, context1: pd.DataFrame, context2: pd.DataFrame) -> pd.DataFrame:
        """
        Compute differential node scores based on the specified node metric.
        
        :return: A DataFrame containing the computed differential node scores.
        """
        assert self._scores1_processed is not None and self._scores2_processed is not None, 'Filtering was unsuccessful.'
        assert context1.columns.equals(context2.columns), 'Context a and b need to have the same structure.'
        assert 'adj-P' in self._scores1_processed.columns and 'adj-P' in self._scores2_processed.columns, 'Min-Max rescaling for raw-P was unsuccessful.'
        assert 'pre-E' in self._scores1_processed.columns and 'pre-E' in self._scores2_processed.columns, 'Min-Max rescaling for pre-E was unsuccessful.'

        nodes_diff = None

        if self._node_metric is None:
            logging.warning('No node_metric was specified. This setting will only work for the PageRank and DimontRank algorithm.')
            nodes_diff = subtract_nodes(context1=context1, context2=context2, test=False)

        # Statistical test centrality (STC)
        if self._node_metric == 'STC':
            nodes_diff = subtract_nodes(context1=context1, context2=context2, test=True, test_type=self._stc_test, correction=self._correction)

        # Degree centrality (DC)
        elif self._node_metric == 'DC':
            if self._filter_metric is None:
                # Take 'adj-P' per default
                dc_metric = 'adj-P'
            else:
                dc_metric = self._filter_metric
            nodes_diff = subtract_nodes(context1=context1, context2=context2, test=False)
            nodes_diff = calculate_degree_centrality(nodes_diff=nodes_diff, metric=dc_metric, weighted=False,
                                                           scores1=self._scores1_processed, scores2=self._scores2_processed)
            
        # Degree centrality and statistical test centrality combined (DC-STC)
        elif self._node_metric == 'DC-STC':
            if self._filter_metric is None:
                # Take 'adj-P' per default
                dc_metric = 'adj-P'
            else:
                dc_metric = self._filter_metric
            nodes_diff = subtract_nodes(context1=context1, context2=context2, test=True, test_type=self._stc_test, correction=self._correction)
            nodes_diff = calculate_degree_centrality(nodes_diff=nodes_diff, metric=dc_metric, weighted=False,
                                                           scores1=self._scores1_processed, scores2=self._scores2_processed)
            
            nodes_diff['DC-STC'] = nodes_diff['DC'] - nodes_diff['STC'] + 1.0        
        
        # Weighted degree centrality based on adj-P (WDC-P)
        elif self._node_metric == 'WDC-P':
            nodes_diff = subtract_nodes(context1=context1, context2=context2, test=False)
            nodes_diff = calculate_degree_centrality(nodes_diff=nodes_diff, metric='adj-P', weighted=True,
                                                           scores1=self._scores1_processed, scores2=self._scores2_processed)

        # Weighted degree centrality based on pre-E (WDC-E)
        elif self._node_metric == 'WDC-E':
            nodes_diff = subtract_nodes(context1=context1, context2=context2, test=False)
            nodes_diff = calculate_degree_centrality(nodes_diff=nodes_diff, metric='pre-E', weighted=True,
                                                           scores1=self._scores1_processed, scores2=self._scores2_processed)

        # PageRank centrality (PRC)
        elif self._node_metric == 'PRC':
            nodes_diff = subtract_nodes(context1=context1, context2=context2, test=False)

            # TODO: allow 'adj-P' metric for PRC (don't forget to invert)

            nodes_diff = calculate_pagerank_centrality(nodes_diff=nodes_diff, metric='pre-E', invert=False,
                                                             scores1=self._scores1_processed, scores2=self._scores2_processed)

        else:
            raise ValueError(f"Invalid node metric '{self._node_metric}'. Choose from: 'DC', 'STC', 'DC-STC', 'WDC-P', 'WDC-E' or 'PRC'.")

        return nodes_diff
    

    def compute_rankings(self, ranking_algs: list[str]):
        """
        Compute multiple rankings based on the specified ranking algorithms.
        
        :param ranking_algs: List of ranking algorithms to compute. Options are 'PageRank+', 'PageRank', 'absDimontRank', 'DimontRank', 'direct_node' and 'direct_edge'.
        """
        assert self._nodes_diff is not None and self._edges_diff is not None, 'The differential network has not been computed yet.'
        
        for alg in ranking_algs:        
            ranks = None

            # Personalized PageRank
            if alg == 'PageRank+':
                if self._personalized_pagerank is not None:
                    logging.info('PageRank+ was already applied. It can be accessed directly.')

                if self._node_metric is None or self._edge_metric is None:
                    raise ValueError('Personalized PageRank requires a node and edge metric'
                                    'but the differential network was defined without one of them.')
                
                invert = True if self._node_metric == 'STC' else False
                ranking_scores = pagerank(nodes_diff=self._nodes_diff, edges_diff=self._edges_diff,
                                        node_metric=self._node_metric, edge_metric=self._edge_metric,
                                        invert=invert, personalization=True)
                ranks = pd.Series(ranking_scores).sort_values(ascending=False).index.tolist()
                self._personalized_pagerank = ranks

            # PageRank
            elif alg == 'PageRank':
                if self._pagerank is not None:
                    logging.info('PageRank was already applied. It can be accessed directly.')

                if self._edge_metric is None:
                    raise ValueError('PageRank requires an edge metric'
                                    'but the differential network was defined without one.')
                ranking_scores = pagerank(edges_diff=self._edges_diff, edge_metric=self._edge_metric,
                                        personalization=False)
                ranks = pd.Series(ranking_scores).sort_values(ascending=False).index.tolist()
                self._pagerank = ranks

            # DimontRank with absolute difference
            elif alg == 'absDimontRank':
                if self._abs_dimontrank is not None:
                    logging.info('absDimontRank was already applied. It can be accessed directly.')

                if self._edge_metric is None:
                    raise ValueError('absDimontRank requires an edge metric'
                                    'but the differential network was defined without one.')
                ranking_scores = dimontrank(edges_diff=self._edges_diff, edge_metric=self._edge_metric, mode='abs')
                ranks = pd.Series(ranking_scores).sort_values(ascending=False).index.tolist()
                self._abs_dimontrank = ranks

            # DimontRank with signed difference
            elif alg == 'DimontRank':
                if self._dimontrank is not None:
                    logging.info('DimontRank was already applied. It can be accessed directly.')

                if self._edge_metric is None:
                    raise ValueError('DimontRank requires an edge metric'
                                    'but the differential network was defined without one.')
                ranking_scores = dimontrank(edges_diff=self._edges_diff, edge_metric=self._edge_metric, mode='signed')
                ranks = pd.Series(ranking_scores).sort_values(ascending=False).index.tolist()
                self._dimontrank = ranks

            # Direct ranking based on node metric
            elif alg == 'direct_node':
                if self._direct_node_rank is not None:
                    logging.info('Direct rank was already applied. It can be accessed directly.')

                if self._node_metric is None:
                    raise ValueError('Direct ranking requires a node metric'
                                    'but the differential network was defined without one.')
                ranking_scores = self._nodes_diff[self._node_metric]

                if self._node_metric == 'STC':
                    ranks = pd.Series(ranking_scores).sort_values(ascending=True).index.tolist()
                else:
                    ranks = pd.Series(ranking_scores).sort_values(ascending=False).index.tolist()
                self._direct_node_rank = ranks
            
            elif alg == 'direct_edge':
                if self._direct_edge_rank is not None:
                    logging.info('Direct edge rank was already applied. It can be accessed directly.')

                if self._edge_metric is None:
                    raise ValueError('Direct edge ranking requires an edge metric'
                                    'but the differential network was defined without one.')
                
                ranking_scores = self._edges_diff[['label1', 'label2', self._edge_metric]].copy()
                ranking_scores = ranking_scores.sort_values(by=self._edge_metric, ascending=False).reset_index(drop=True)
                ranks = ranking_scores[['label1', 'label2']].values.tolist()

                self._direct_edge_rank = ranks
            
            else:
                raise ValueError(f"Invalid ranking algorithm {alg}. "
                                "Choose from: 'PageRank+', 'PageRank', 'absDimontRank', 'DimontRank', 'direct_node' or 'direct_edge'.")
    

    def _check_input_data(self, context: pd.DataFrame, meta_file: pd.DataFrame) -> bool:
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
    

    def save_rankings(self, ranking_algs: list[str], path: Optional[str] = None):
        """
        Save specified rankings to CSV files. If the specified rankings have not been computed yet, they will be computed first.
        
        :param ranking_algs: List of ranking algorithms to save. Options are 'PageRank+', 'PageRank', 'absDimontRank', 'DimontRank', 'direct_node' and 'direct_edge'.
        """
        for alg in ranking_algs:            
            if path is not None:
                file_path = os.path.join(path, f"ranking_{alg}.csv")
            elif self._project_path is not None:
                file_path = os.path.join(self._project_path, f"ranking_{alg}.csv")
            else:
                raise ValueError('Please provide a path where to save the rankings.')
        
            # Personalized PageRank
            if alg == 'PageRank+':
                if self._personalized_pagerank is None:
                    self.compute_rankings(ranking_algs=[alg])
                else:
                    ranking_list = self._personalized_pagerank

            # PageRank
            if alg == 'PageRank':
                if self._pagerank is None:
                    self.compute_rankings(ranking_algs=[alg])
                else:
                    ranking_list = self._pagerank

            # DimontRank with absolute difference
            if alg == 'absDimontRank':
                if self._abs_dimontrank is None:
                    self.compute_rankings(ranking_algs=[alg])
                else:
                    ranking_list = self._abs_dimontrank

            # DimontRank with signed difference
            if alg == 'DimontRank':
                if self._dimontrank is None:
                    self.compute_rankings(ranking_algs=[alg])
                else:
                    ranking_list = self._dimontrank

            # Direct ranking based on node metric
            if alg == 'direct_node':
                if self._direct_node_rank is None:
                    self.compute_rankings(ranking_algs=[alg])
                else:
                    ranking_list = self._direct_node_rank
            
            # Direct ranking based on edge metric
            if alg == 'direct_edge':
                if self._direct_edge_rank is None:
                    self.compute_rankings(ranking_algs=[alg])
                else:
                    ranking_list = self._direct_edge_rank            
            else:
                raise ValueError(f"Invalid ranking algorithm {alg}."
                                 "Choose from: 'PageRank+', 'PageRank', 'absDimontRank', 'DimontRank', 'direct_node' or 'direct_edge.")

            ranking_df = pd.DataFrame({"node": ranking_list, "rank": range(1, len(ranking_list) + 1)})
            ranking_df = ranking_df.sort_values("node")
            ranking_df.to_csv(file_path, index=False)


    # Save (processed) association scores
    def save_scores(self, path: Optional[str] = None, processed: bool = True):
        """
        Save association scores to CSV files.
        
        :param path: Directory path where to save the scores. If None, the project_path specified during initialization will be used.
        :param processed: If true, save the processed (filtered) scores instead of the raw scores. Defaults to True.
        """
        if processed:
            suffix = 'scores_processed.csv'
        else:
            suffix = 'scores.csv'

        if path is not None:
            file_path1 = os.path.join(path, f'{self._name1}_{suffix}')
            file_path2 = os.path.join(path, f'{self._name2}_{suffix}')
        elif self._project_path is not None:
            file_path1 = os.path.join(self._project_path, f'{self._name1}_{suffix}.csv')
            file_path2 = os.path.join(self._project_path, f'{self._name2}_{suffix}.csv')
        else:
            raise ValueError('Please provide a path where to save the edge scores.')
        
        if processed:
            assert self._scores1_processed is not None and self._scores2_processed is not None, 'The association scores have not been processed yet.'
            self._scores1_processed.to_csv(file_path1)
            self._scores2_processed.to_csv(file_path2)
        else:
            assert self._scores1 is not None and self._scores2 is not None, 'The association socres have not been computed yet.'
            self._scores1.to_csv(file_path1)
            self._scores2.to_csv(file_path2)

    
    # Save differential network
    def save_diffnet(self, path: Optional[str] = None, format: str = 'csv'):
        """
        Save the differential network to CSV files or as GraphML.
        
        :param path: Directory path where to save the differential network. If None, the project_path specified during initialization will be used.
        :param format: Format to save the differential network. Options are 'csv' or 'graphml'. Defaults to 'csv'.
        """
        assert self._edges_diff is not None, 'The differential edge scores have not been computed yet.'
        assert self._nodes_diff is not None, 'The differential node scores have not been computed yet.'

        if format == 'csv':
            if path is not None:
                file_path_edges = os.path.join(path, f'diff_edges.csv')
                file_path_nodes = os.path.join(path, f'diff_nodes.csv')
            elif self._project_path is not None:
                file_path_edges = os.path.join(self._project_path, f'diff_edges.csv')
                file_path_nodes = os.path.join(self._project_path, f'diff_nodes.csv')
            else:
                raise ValueError('Please provide a path where to save the differential network.')
            
            self._edges_diff.to_csv(file_path_edges)
            self._nodes_diff.to_csv(file_path_nodes)

        elif format == 'graphml':
            if path is not None:
                file_path = os.path.join(path, f'diff_net.graphml')
            elif self._project_path is not None:
                file_path = os.path.join(self._project_path, f'diff_net.graphml')
            else:
                raise ValueError('Please provide a path where to save the differential network.')
            diff_net = nx.from_pandas_edgelist(self._edges_diff, 'label1', 'label2', self._edge_metric)
            nx.set_node_attributes(diff_net, self._nodes_diff[self._node_metric].to_dict(), self._node_metric)

            nx.write_graphml(diff_net, file_path)

        else:
            raise ValueError(f"Invalid format {format}. Choose from 'csv' or 'graphml'.")          
            

    # Print summary of parameters
    def summary(self):
        """
        Return a summary of the DiffNet parameters as a dictionary.
        
        :return: A dictionary containing the DiffNet parameters.
        """
        params = {
            'name1': self._name1,
            'name2': self._name2,
            'cont_cont': self._cont_cont,
            'cat_cat': self._cat_cat,
            'bi_cont': self._bi_cont,
            'cont_cat': self._cont_cat,
            'correction':  self._correction,
            'filter_method': self._filter_method,
            'filter_metric': self._filter_metric,
            'filter_rule': self._filter_rule,
            'filter_param': self._filter_param,
            'edge_metric': self._edge_metric,
            'node_metric': self._node_metric        
        }
        
        if self._node_metric == 'STC':
            params['stc_test'] = self._stc_test
        if self._edge_metric == 'int-IS':
            params['max_path_length'] = self._max_path_length

        return params
    

    # Save summary of parameters
    def save_config(self, path: Optional[str] = None):
        """
        Save a summary of the configurations to a JSON file.
        
        :param path: File path where to save the parameters. If None, the project_path specified during initialization will be used.
        """
        if path is not None:
            file_path = path
        elif self._project_path is not None:
            file_path = os.path.join(self._project_path, 'diffnet_config.json')
        else:
            raise ValueError('Please provide a path where to save the parameters.')
        
        params = self.summary()
        with open(file_path, 'w') as f:
            json.dump(params, f, indent=4)
        

    # Getter and setter methods for public params
    @property
    def project_path(self):
        return self._project_path
    
    @project_path.setter
    def project_path(self, path: str | None):
        if path is None:
            logging.info('No project path was specified.')
        elif not os.path.exists(path):
            raise ValueError(f"The specified project path '{path}' does not exist.")
        self._project_path = path

    @property
    def name1(self):
        return self._name1
    
    @name1.setter
    def name1(self, name: str):
        if name == self._name2:
            raise ValueError('The names of the two contexts must be different.')
        self._name1 = name
    
    @property
    def name2(self):
        return self._name2
    
    @name2.setter
    def name2(self, name: str):
        if name == self._name1:
            raise ValueError('The names of the two contexts must be different.')
        self._name2 = name


    # Getter methods for private params (no setter methods to ensure immutability after initialization)
    @property
    def filter_method(self):
        return self._filter_method
    
    @property
    def filter_metric(self):
        return self._filter_metric
    
    @property
    def filter_rule(self):
        return self._filter_rule
    
    @property
    def filter_param(self):
        return self._filter_param
    
    @property
    def cont_cont(self):
        return self._cont_cont
    
    @property
    def cat_cat(self):
        return self._cat_cat
    
    @property
    def bi_cont(self):
        return self._bi_cont
    
    @property
    def cont_cat(self):
        return self._cont_cat
    
    @property
    def correction(self):
        return self._correction
    
    @property
    def edge_metric(self):
        return self._edge_metric
    
    @property
    def node_metric(self):
        return self._node_metric
    
    @property
    def stc_test(self):
        return self._stc_test
    
    @property
    def max_path_length(self):
        return self._max_path_length
    
    @property
    def scores1(self):
        assert self._scores1 is not None, 'Association scores have not been computed yet.'
        return self._scores1.copy()
    
    @property
    def scores2(self):
        assert self._scores2 is not None, 'Association scores have not been computed yet.'
        return self._scores2.copy()
    
    @property
    def scores1_processed(self):
        assert self._scores1_processed is not None, 'Association scores have not been computed yet.'
        return self._scores1_processed.copy()
    
    @property
    def scores2_processed(self):
        assert self._scores2_processed is not None, 'Association scores have not been computed yet.'
        return self._scores2_processed.copy()
    
    @property
    def edges_diff(self):
        assert self._edges_diff is not None, 'Differential edge scores have not been computed yet.'
        return self._edges_diff.copy()
    
    @property
    def nodes_diff(self):
        assert self._nodes_diff is not None, 'Differential node scores have not been computed yet.'
        return self._nodes_diff.copy()
    
    @property
    def personalized_pagerank(self):
        assert self._personalized_pagerank is not None, 'Personalized PageRank has not been applied yet.'
        return self._personalized_pagerank
    
    @property 
    def pagerank(self):
        assert self._pagerank is not None, 'PageRank has not been applied yet.'
        return self._pagerank
    
    @property
    def abs_dimontrank(self):
        assert self._abs_dimontrank is not None, 'absDimontRank has not been applied yet.'
        return self._abs_dimontrank
    
    @property
    def dimontrank(self):
        assert self._dimontrank is not None, 'DimontRank has not been applied yet.'
        return self._dimontrank
    
    @property
    def direct_node_rank(self):
        assert self._direct_node_rank is not None, 'Direct node rank has not been applied yet.'
        return self._direct_node_rank
    
    @property
    def direct_edge_rank(self):
        assert self._direct_edge_rank is not None, 'Direct edge rank has not been applied yet.'
        return self._direct_edge_rank


