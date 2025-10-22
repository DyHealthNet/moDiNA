from score_calculation import *
from statistics_utils import *
import os
import logging
import math

class DiffNet:
    def __init__(self, context1, context2, pheno_meta=None, 
                 edge_metric=None, node_metric=None,
                 filter_method=None, filter_param=0, filter_metric=None, filter_rule=None,
                 stc_test='parametric', max_path_length=2,
                 cont_cont='spearman', cat_cat='chi2', cat_cont_b='mann-whitney u', cat_cont_m='kruskal-wallis', correction='bh',
                 nan_value=-89, num_workers=1,
                 project_path=None, name1='context1', name2='context2'):
        # Project path and context names
        self.project_path = project_path
        self.name1 = name1
        self.name2 = name2

        # Association scores
        self._scores1 = None
        self._scores2 = None
        self._scores1_processed = None
        self._scores2_processed = None
        self._cont_cont = cont_cont
        self._cat_cat = cat_cat
        self._cat_cont_b = cat_cont_b
        self._cat_cont_m = cat_cont_m
        self._correction = correction
        self._nan_value = nan_value
        self._num_workers = num_workers

        # Differential network
        self._filter_method = filter_method
        self._filter_metric = filter_metric
        self._filter_rule = filter_rule
        self._filter_param = filter_param
        self._edge_metric = edge_metric
        self._node_metric = node_metric
        self._stc_test = stc_test
        self._max_path_length = max_path_length
        self._edges_diff = None
        self._nodes_diff = None

        # Rankings
        self._personalized_pagerank = None
        self._pagerank = None
        self._dimontrank = None
        self._abs_dimontrank = None
        self._direct_rank = None

        # Compute differential network
        self._compute_diff_network(context1=context1, context2=context2, pheno_meta=pheno_meta)


    def _compute_diff_network(self, context1, context2, pheno_meta):
        # Compute context scores
        self._scores1 = self._compute_context_scores(context_data=context1, pheno_meta=pheno_meta, target=1)
        self._scores2 = self._compute_context_scores(context_data=context2, pheno_meta=pheno_meta, target=2)
        assert self._scores1 is not None and self._scores2 is not None, 'Score calculation was unsuccessful.'
        
        # Min-Max rescaling for raw-P and raw-E
        self._scores1, self._scores2 = min_max_rescaling(scores1=self._scores1,
                                                         scores2=self._scores2,
                                                         metric='adj-P')
        self._scores1, self._scores2 = min_max_rescaling(scores1=self._scores1,
                                                         scores2=self._scores2,
                                                         metric='pre-E')
        assert self._scores2 is not None, 'Min-Max rescaling was unsuccessful.'

        # Filtering
        if any([self.filter_method, self.filter_metric, self.filter_rule, self.filter_param]):
            self._scores1_processed, self._scores2_processed, context1_filtered, context2_filtered = self._filter(context1=context1, context2=context2)
        else:
            logging.warning('The differential network will be computed based on unfiltered data. No filter parameters were specified.')
            self._scores1_processed = self._scores1.copy()
            self._scores2_processed = self._scores2.copy()
            context1_filtered = context1.copy()
            context2_filtered = context2.copy()
            
        # Edges
        self._edges_diff = self._compute_diff_edges()

        # Nodes
        self._nodes_diff = self._compute_diff_nodes(context1=context1_filtered, context2=context2_filtered)

        return self._edges_diff, self._nodes_diff 


    def _compute_context_scores(self, context_data, pheno_meta, target):
        # Separate the data into categorical and continuous data
        cat, cont = separate_cat_cont(context_data, pheno_meta)

        # Get test types
        tests = {
            "contCont": self._cont_cont,
            "catCat": self._cat_cat,
            "catContB": self._cat_cont_b,
            "catContM": self._cat_cont_m
        }

        # Calculate scores
        scores = calculate_association_scores(cat, cont, tests)

        # Take the adjusted p-value and the corresponding effect size
        column_names = scores.iloc[:, 2:].columns
        # TODO: this is unnecessary, remove this
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

        # Update scores
        if target == 1 or target == self.name1:
            self._scores1 = scores_final
            logging.info(f'Successfully computed association scores for context {target}.')
        elif target == 2 or target == self.name2:
            self._scores2 = scores_final
            logging.info(f'Successfully computed association scores for context {target}.')
        else:
            raise ValueError(f'Invalid target {target}. Choose context data by name or number.')

        return scores_final     
        

    def _filter(self, context1, context2):
        assert self._scores1 is not None and self._scores2 is not None, 'Association scores have not been computed yet.'
        assert self._scores1['label1'].equals(self._scores2['label1']), 'Contexts need to have the same structure and order of edges.'
        assert self._scores1['label2'].equals(self._scores2['label2']), 'Contexts need to have the same structure and order of edges.'
        assert self._filter_metric in self._scores1.columns and self._filter_metric in self._scores2.columns, 'Min-Max rescaling was not performed yet.'
        
        # Check input parameters
        if self.filter_method is None:
            raise ValueError("Please provide a 'filter_method'.")
        
        if self.filter_metric is None:
            raise ValueError("Please provide a 'filter_metric'.")
        
        if self.filter_rule is None:
            raise ValueError("Please provide a 'filter_rule'.")
        
        if self.filter_param is None:
            raise ValueError("Please provide a 'filter_param'.")


        ascending = True if self.filter_metric == 'adj-P' else False

        # Compute filtering threshold according to the specified method
        threshold1 = None
        threshold2 = None
        n_nodes = len(np.unique(self._scores1[['label1', 'label2']].values.flatten()))
        # TODO: find out if this is sufficient
        n_nodes_easy = context1.shape[1]
        print(n_nodes)
        print(n_nodes_easy)
        n_edges_before = self._scores1.shape[0]

        if filter == 'threshold':
            threshold1 = threshold2 = self.filter_param

        elif filter == 'degree':
            degree = self.filter_param
            n_filtered_edges = math.ceil(degree * n_nodes / 2)

            threshold1 = self._scores1[self.filter_metric].sort_values(ascending=ascending).iloc[n_filtered_edges - 1]
            threshold2 = self._scores2[self.filter_metric].sort_values(ascending=ascending).iloc[n_filtered_edges - 1]

        elif filter == 'density':
            density = self.filter_param
            possible_edges = n_nodes * (n_nodes - 1) / 2
            n_filtered_edges = math.ceil(density * possible_edges)

            threshold1 = self._scores1[self.filter_metric].sort_values(ascending=ascending).iloc[n_filtered_edges - 1]
            threshold2 = self._scores2[self.filter_metric].sort_values(ascending=ascending).iloc[n_filtered_edges - 1]

        else:
            raise ValueError(f"Invalid filtering method '{filter}'. Choose from: 'threshold', 'degree' or 'density'")
        
        # Apply the filtering threshold to scores and raw data if provided
        if self.filter_rule == 'union':
            if ascending is True:
                mask = (self._scores1[self.filter_metric] <= threshold1) | (
                        self._scores2[self.filter_metric] <= threshold2)
            else:
                mask = (self._scores1[self.filter_metric] >= threshold1) | (
                        self._scores2[self.filter_metric] >= threshold2)
            
            # Apply mask
            self._scores1_processed = self._scores1[mask].copy()
            self._scores2_processed = self._scores2[mask].copy()

        elif self.filter_rule == 'zero':
            if ascending is True:
                mask1 = self._scores1[self.filter_metric] <= threshold1
                mask2 = self._scores2[self.filter_metric] <= threshold2
            else:
                mask1 = self._scores1[self.filter_metric] >= threshold1
                mask2 = self._scores2[self.filter_metric] >= threshold2
            
            # Apply mask
            filtered1 = self._scores1[mask1].copy()
            filtered2 = self._scores2[mask2].copy()
            filtered1 = filtered1.set_index(['label1', 'label2'])
            filtered2 = filtered2.set_index(['label1', 'label2'])

            # Unify indices and set missing values
            indices = filtered1.index.union(filtered2.index)
            self._scores1_processed = filtered1.reindex(indices)
            self._scores2_processed = filtered2.reindex(indices)

            fill_values = {
                'raw-P': 1.0,
                'adj-P': 1.0,
                'raw-E': 0.0,
                'pre-E': 0.0,
            }

            for metric, value in fill_values.items():
                if metric in self._scores1_processed.columns:
                    self._scores1_processed[metric] = self._scores1_processed[metric].fillna(value)
                if metric in self._scores2_processed.columns:
                    self._scores2_processed[metric] = self._scores2_processed[metric].fillna(value)
            
            if 'test_type' in self._scores1_processed.columns and 'test_type' in self._scores2_processed.columns:
                merged_test_type = self._scores1_processed['test_type'].combine_first(self._scores2_processed['test_type'])
                self._scores1_processed['test_type'] = merged_test_type
                self._scores2_processed['test_type'] = merged_test_type

            self._scores1_processed = self._scores1_processed.reset_index()
            self._scores2_processed = self._scores2_processed.reset_index()
        
        else:
            raise ValueError(f"Invalid filtering rule '{self.filter_rule}'.")

        filtered_nodes = np.concatenate((self._scores1_processed['label1'].values,
                                         self._scores1_processed['label2'].values))
        filtered_nodes = pd.unique(filtered_nodes)
        context1_filtered = context1[filtered_nodes].copy()
        context2_filtered = context2[filtered_nodes].copy()

        n_edges_after = self._scores1_processed.shape[0]

        logging.info(f'Successfully filtered the single networks using a {self._filter} of {self._filter_param}'
                     f'based on {self._filter_metric} with the {self._filter_rule} rule. Reduced the number of edges from '
                     f'{n_edges_before} to {n_edges_after}.')      
       
        return self._scores1_processed, self._scores2_processed, context1_filtered, context2_filtered


    def _compute_diff_edges(self):
        assert self._scores1_processed is not None and self._scores2_processed is not None, 'Filtering was unsuccessful.'
        assert 'adj-P' in self._scores1_processed.columns and 'adj-P' in self._scores2_processed.columns, 'Min-Max rescaling for raw-P was unsuccessful.'
        assert 'pre-E' in self._scores1_processed.columns and 'pre-E' in self._scores2_processed.columns, 'Min-Max rescaling for pre-E was unsuccessful.'

        if self._edge_metric is None:
            logging.warning('No edge_metric was specified. This setting will only work for the direct ranking algorithm.')
            # Take adj-P per default
            self._edges_diff = subtract_edges(self._scores1_processed, self._scores2_processed,
                                              metrics=['adj-P'], included_cols=('test_type',))

        # Pre-rescaled effect size (pre-E) or rescaled multiple-testing adjusted p-value (adj-P)
        if self._edge_metric == 'adj-P' or self._edge_metric == 'pre-E':
            pass

        # Post-rescaled effect size (post-E)
        elif self._edge_metric == 'post-E':
            # Compute differences in edge metrics first
            self._edges_diff = subtract_edges(self._scores1_processed, self._scores2_processed,
                                             metrics=['raw-E'], included_cols=['test_type'])
            # Min-Max rescaling
            self._edges_diff, _ = min_max_rescaling(scores1=self._edges_diff, metric='post-E')

        # Pre-rescaled combined score (pre-CS)
        elif self._edge_metric == 'pre-CS':
            # Compute combined score from rescaled effect size and p-value
            self._scores1_processed[self._edge_metric] = self._scores1_processed['pre-E'] - self._scores1_processed['adj-P']
            self._scores2_processed[self._edge_metric] = self._scores2_processed['pre-E'] - self._scores2_processed['adj-P']

        # Post-rescaled combined score (post-CS)
        elif self._edge_metric == 'post-CS':
            # Compute differences in edge metrics first
            self._edges_diff = subtract_edges(self._scores1_processed, self._scores2_processed,
                                             metrics=['adj-P', 'raw-E'], included_cols=['test_type'])
            # Rescale difference in effect sizes
            self._edges_diff, _ = min_max_rescaling(scores1=self._edges_diff, metric='post-E')
            # Compute combined score
            self._edges_diff[self._edge_metric] = (self._edges_diff['post-E'] + self._edges_diff['adj-P'])

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
        elif self._edge_metric == 'log-CS':
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

        else:
            raise ValueError(f"Invalid edge metric '{self._edge_metric}'. Choose from: 'adj-P', 'pre-E', 'post-E', 'int-IS', 'pre-CS', 'post-CS' or 'log-CS'.")

        if self._edges_diff is None:
            # Compute difference in edge scores
            self._edges_diff = subtract_edges(self._scores1_processed, self._scores2_processed,
                                              metrics=[self._edge_metric], included_cols=('test_type',))
        
        return self._edges_diff
    

    def _compute_diff_nodes(self, context1, context2):
        assert self._scores1_processed is not None and self._scores2_processed is not None, 'Filtering was unsuccessful.'
        assert context1.columns.equals(context2.columns), 'Context a and b need to have the same structure.'
        assert 'adj-P' in self._scores1_processed.columns and 'adj-P' in self._scores2_processed.columns, 'Min-Max rescaling for raw-P was unsuccessful.'
        assert 'pre-E' in self._scores1_processed.columns and 'pre-E' in self._scores2_processed.columns, 'Min-Max rescaling for pre-E was unsuccessful.'

        if self._node_metric is None:
            logging.warning('No node_metric was specified. This setting will only work for the PageRank and DimontRank algorithm.')
            self._nodes_diff = subtract_nodes(context1=context1, context2=context2, test=False)

        # Statistical test centrality (STC)
        if self._node_metric == 'STC':
            self._nodes_diff = subtract_nodes(context1=context1, context2=context2, test=True, test_type=self._stc_test, correction=self._correction)

        # Degree centrality (DC)
        elif self._node_metric == 'DC':
            if self._filter_metric is None:
                # Take 'adj-P' per default
                dc_metric = 'adj-P'
            else:
                dc_metric = self._filter_metric
            self._nodes_diff = subtract_nodes(context1=context1, context2=context2, test=False)
            self._nodes_diff = calculate_degree_centrality(nodes_diff=self._nodes_diff, metric=dc_metric, weighted=False,
                                                           scores1=self._scores1_processed, scores2=self._scores2_processed)
            
        # Degree centrality and statistical test centrality combined (DC-STC)
        elif self._node_metric == 'DC-STC':
            if self._filter_metric is None:
                # Take 'adj-P' per default
                dc_metric = 'adj-P'
            else:
                dc_metric = self._filter_metric
            self._nodes_diff = subtract_nodes(context1=context1, context2=context2, test=True, test_type=self._stc_test, correction=self._correction)
            self._nodes_diff = calculate_degree_centrality(nodes_diff=self._nodes_diff, metric=dc_metric, weighted=False,
                                                           scores1=self._scores1_processed, scores2=self._scores2_processed)
            
            self._nodes_diff['DC-STC'] = self._nodes_diff['DC'] - self._nodes_diff['STC'] + 1.0        
        
        # Weighted degree centrality based on adj-P (WDC-P)
        elif self._node_metric == 'WDC-P':
            self._nodes_diff = subtract_nodes(context1=context1, context2=context2, test=False)
            self._nodes_diff = calculate_degree_centrality(nodes_diff=self._nodes_diff, metric='adj-P', weighted=True,
                                                           scores1=self._scores1_processed, scores2=self._scores2_processed)

        # Weighted degree centrality based on pre-E (WDC-E)
        elif self._node_metric == 'WDC-E':
            self._nodes_diff = subtract_nodes(context1=context1, context2=context2, test=False)
            self._nodes_diff = calculate_degree_centrality(nodes_diff=self._nodes_diff, metric='pre-E', weighted=True,
                                                           scores1=self._scores1_processed, scores2=self._scores2_processed)

        # PageRank centrality (PRC)
        elif self._node_metric == 'PRC':
            self._nodes_diff = subtract_nodes(context1=context1, context2=context2, test=False)

            # TODO: allow 'adj-P' metric for PRC (don't forget to invert)

            self._nodes_diff = calculate_pagerank_centrality(nodes_diff=self._nodes_diff, metric='pre-E', invert=False,
                                                             scores1=self._scores1_processed, scores2=self._scores2_processed)

        else:
            raise ValueError(f"Invalid node metric '{self._node_metric}'. Choose from: 'DC', 'STC', 'DC-STC', 'WDC-P', 'WDC-E' or 'PRC'.")

        return self._nodes_diff
    

    def compute_nodes_ranking(self, ranking_algs):
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
            elif alg == 'direct':
                if self._direct_rank is not None:
                    logging.info('Direct rank was already applied. It can be accessed directly.')

                if self._node_metric is None:
                    raise ValueError('Direct ranking requires a node metric'
                                    'but the differential network was defined without one.')
                ranking_scores = self._nodes_diff[self._node_metric]

                if self._node_metric == 'STC':
                    ranks = pd.Series(ranking_scores).sort_values(ascending=True).index.tolist()
                else:
                    ranks = pd.Series(ranking_scores).sort_values(ascending=False).index.tolist()
                self._direct_rank = ranks
            
            else:
                raise ValueError(f"Invalid ranking algorithm {alg}."
                                "Choose from: 'PageRank+', 'PageRank', 'absDimontRank', 'DimontRank' or 'direct'.")
    

    def save_ranking(self, ranking_algs, path=None):
        for alg in ranking_algs:            
            if path is not None:
                file_path = os.path.join(path, f"ranking_{alg}.csv")
            elif self.project_path is not None:
                file_path = os.path.join(self.project_path, f"ranking_{alg}.csv")
            else:
                raise ValueError('Please provide a path where to save the rankings.')
        
            # Personalized PageRank
            if alg == 'PageRank+':
                if self._personalized_pagerank is None:
                    raise ValueError('PageRank+ has not been applied yet.')
                else:
                    ranking_list = self._personalized_pagerank

            # PageRank
            if alg == 'PageRank':
                if self._pagerank is None:
                    raise ValueError('PageRank has not been applied yet.')
                else:
                    ranking_list = self._pagerank

            # DimontRank with absolute difference
            if alg == 'absDimontRank':
                if self._abs_dimontrank is None:
                    raise ValueError('absDimontRank has not been applied yet.')
                else:
                    ranking_list = self._abs_dimontrank

            # DimontRank with signed difference
            if alg == 'DimontRank':
                if self._dimontrank is None:
                    raise ValueError('DimontRank has not been applied yet.')
                else:
                    ranking_list = self._dimontrank

            # Direct ranking based on node metric
            if alg == 'direct':
                if self._direct_rank is None:
                    raise ValueError('Direct ranking has not been applied yet.')
                else:
                    ranking_list = self._direct_rank
            
            else:
                raise ValueError(f"Invalid ranking algorithm {alg}."
                                 "Choose from: 'PageRank+', 'PageRank', 'absDimontRank', 'DimontRank' or 'direct'.")

            ranking_df = pd.DataFrame({"node": ranking_list, "rank": range(1, len(ranking_list) + 1)})
            ranking_df = ranking_df.sort_values("node")
            ranking_df.to_csv(file_path, index=False)


    # Save (processed) association scores
    def save_scores(self, path=None, processed=True):
        if processed:
            suffix = 'scores_processed.csv'
        else:
            suffix = 'scores.csv'

        if path is not None:
            file_path1 = os.path.join(path, f'{self.name1}_{suffix}')
            file_path2 = os.path.join(path, f'{self.name2}_{suffix}')
        elif self.project_path is not None:
            file_path1 = os.path.join(self.project_path, f'{self.name1}_{suffix}.csv')
            file_path2 = os.path.join(self.project_path, f'{self.name2}_{suffix}.csv')
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
    def save_diff_net(self, path=None, nodes=True, edges=True):
        if path is not None:
            file_path_edges = os.path.join(path, 'diff_net_edges.csv')
            file_path_nodes = os.path.join(path, 'diff_net_nodes.csv')
        elif self.project_path is not None:
            file_path_edges = os.path.join(self.project_path, 'diff_net_edges.csv')
            file_path_nodes = os.path.join(self.project_path, 'diff_net_nodes.csv')
        else:
            raise ValueError('Please provide a path where to save the differential network.')
        
        if edges:
            assert self._edges_diff is not None, 'The differential edge scores have not been computed yet.'
            self._edges_diff.to_csv(file_path_edges)
        if nodes:
            assert self._nodes_diff is not None, 'The differential node scores have not been computed yet.'
            self._nodes_diff.to_csv(file_path_nodes)


    # Print summary of parameters
    def summary(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


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
    def cat_cont_b(self):
        return self._cat_cont_b
    
    @property
    def cat_cont_m(self):
        return self._cat_cont_m
    
    @property
    def correction(self):
        return self._correction
    
    @property
    def nan_value(self):
        return self._nan_value
    
    @property
    def num_workers(self):
        return self._num_workers
    
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
    def direct_rank(self):
        assert self._direct_rank is not None, 'Direct rank has not been applied yet.'
        return self._direct_rank




