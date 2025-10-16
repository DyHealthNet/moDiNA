from score_calculation import *
import os
import warnings

class DiffNet:
    def __init__(self, context1, context2, pheno_meta=None, 
                 edge_weight=None, node_weight=None,
                 filter_method=None, filter_param=0, filter_metric=None, filter_rule=None,
                 stc_test='parametric', max_path_length=2,
                 cont_cont='spearman', cat_cat='chi2', cat_cont_b='mann-whitney u', cat_cont_m='kruskal-wallis', correction='bh',
                 nan_value=-89, num_workers=1,
                 project_path=None, name1='context1', name2='context2'):
        # Context data
        # TODO: Read the context paths and check the format of the context tables
        self.context1 = context1
        self.context2 = context2
        self._context1_filtered = None
        self._context2_filtered = None
        self.project_path = project_path
        self.name1 = name1
        self.name2 = name2

        # Association scores
        self.scores1 = self.compute_context_scores(self.context1)
        self.scores2 = self.compute_context_scores(self.context2)
        self._scores1_filtered = None
        self._scores2_filtered = None
        self.cont_cont = cont_cont
        self.cat_cat = cat_cat
        self.cat_cont_b = cat_cont_b
        self.cat_cont_m = cat_cont_m
        self.correction = correction
        self.pheno_meta = pheno_meta
        self.nan_value = nan_value
        self.num_workers = num_workers

        # Differential network
        self.filter_method = filter_method
        self.filter_metric = filter_metric
        self.filter_rule = filter_rule
        self.filter_param = filter_param
        self.edge_weight = edge_weight
        self.node_weight = node_weight
        self.stc_test = stc_test
        self.max_path_length = max_path_length
        self._edges_diff = None
        self._nodes_diff = None
    

    def compute_context_scores(self, context_data):
        # Separate the data into categorical and continuous data
        cat, cont = separate_cat_cont(context_data, self.pheno_meta)

        # Get test types
        tests = {
            "contCont": self.cont_cont,
            "catCat": self.cat_cat,
            "catContB": self.cat_cont_b,
            "catContM": self.cat_cont_m
        }

        # Calculate scores
        scores = calculate_association_scores(cat, cont, tests)

        # Take the adjusted p-value and the corresponding effect size
        column_names = scores.iloc[:, 2:].columns
        # TODO: this is unnecessary, remove this
        if self.correction == 'bh':
            correction = 'benjamini_hb'
        else:
            correction = self.correction
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

        return scores_final
    

    def compute_diff_network(self):
        assert self.scores1['label1'].equals(self.scores2['label1']), 'Contexts need to have the same structure and order of edges.'
        assert self.scores1['label2'].equals(self.scores2['label2']), 'Contexts need to have the same structure and order of edges.'

        # Filtering
        if any([self.filter_method, self.filter_metric, self.filter_rule, self.filter_param]):
            self._scores1_filtered, self._scores2_filtered, self._context1_filtered, self._context2_filtered = self.filter()
        else:
            warnings.warn('The differential network will be computed based on unfiltered data. No filter parameters were specified.', UserWarning)
        
            self._scores1_filtered = self.scores1.copy()
            self._scores2_filtered = self.scores2.copy()
            self._context1_filtered = self.context1.copy()
            self._context2_filtered = self.context2.copy()
            
        # Edges
        self.edges_diff = self.compute_diff_edges()

        # Nodes
        self.nodes_diff = self.compute_diff_nodes()

        return self.edges_diff, self.nodes_diff      
        

    def filter(self, filter_method=None, filter_param=None, filter_metric=None, filter_rule=None):
        # Check input parameters
        if filter_method is None and self.filter_method is None:
            raise ValueError("Please provide a 'filter_method'.")
        elif filter_method is not None:
            if filter_method != self.filter_method:
                self.filter_method(filter_method)
        
        if filter_metric is None and self.filter_metric is None:
            raise ValueError("Please provide a 'filter_metric'.")
        elif filter_metric is not None:
            if filter_metric != self.filter_metric:
                self.filter_metric(filter_metric)
        
        if filter_rule is None and self.filter_rule is None:
            raise ValueError("Please provide a 'filter_rule'.")
        elif filter_rule is not None:
            if filter_rule != self.filter_rule:
                self.filter_rule(filter_rule)
        
        if filter_param is None and self.filter_param is None:
            raise ValueError("Please provide a 'filter_param'.")
        elif filter_param is not None:
            if filter_param != self.filter_param:
                self.filter_param(filter_param)

        # TODO: insert filter logic


        return self._scores1_filtered, self._scores2_filtered, self._context1_filtered, self._context2_filtered


    def compute_diff_edges(self):
        return self.edges_diff
    

    def compute_diff_nodes(self):
        return self.nodes_diff


    def save_scores(self, path=None, filtered=True):
        if filtered:
            suffix = 'scores_filtered.csv'
        else:
            suffix = 'scores.csv'

        if path is not None:
            file_path1 = os.path.join(path, f'{self.name1}_{suffix}')
            file_path2 = os.path.join(path, f'{self.name2}_{suffix}')
        elif self.project_path is not None:
            file_path1 = os.path.join(self.project_path, f'{self.name1}_{suffix}.csv')
            file_path2 = os.path.join(self.project_path, f'{self.name2}_{suffix}.csv')
        else:
            raise ValueError('Please provide a path where to save the computed association scores.')
        
        if filtered:
            if self._scores1_filtered is None or self._scores2_filtered is None:
                raise ValueError('Scores have not been filtered yet. Please apply filtering first or save the unfiltered scores.')
            else:
                self._scores1_filtered.to_csv(file_path1)
                self._scores2_filtered.to_csv(file_path2)
        else:
            self.scores1.to_csv(file_path1)
            self.scores2.to_csv(file_path2)


    # Print summary of parameters
    def summary(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


    @property
    def filter_method(self):
        return self.filter_method

    @filter_method.setter
    def filter_method(self, value):
        if value in ['threshold', 'degree', 'density']:
            print(f"Updating 'filter_method' to '{value}'.")
            self.filter_method = value
        else:
            raise ValueError(f"{value} is not a valid 'filter_method'.")
    
    @property
    def filter_metric(self):
        return self.filter_metric

    @filter_metric.setter
    def filter_metric(self, value):
        if value in ['adj-P', 'pre-E']:
            print(f"Updating 'filter_metric' to '{value}'.")
            self.filter_metric = value
        else:
            raise ValueError(f"{value} is not a valid 'filter_metric'.")
    
    @property
    def filter_rule(self):
        return self.filter_rule

    @filter_rule.setter
    def filter_rule(self, value):
        if value in ['union', 'intersection', 'zero']:
            print(f"Updating 'filter_rule' to '{value}'.")
            self.filter_rule = value
        else:
            raise ValueError(f"{value} is not a valid 'filter_rule'.")
    
    @property
    def filter_param(self):
        return self.filter_param

    @filter_param.setter
    def filter_param(self, value):
        if self.filter_method in ['threshold', 'density'] and (value <= 0.0 or value >= 1.0):
            raise ValueError(f"{value} is invalid. For {self.filter_method} filtering the 'filter_param' needs to be within the range ]0, 1[.")
        elif self.filter_method == 'degree' and value <= 0:
            raise ValueError(f"{value} is invalid. The average degree needs to be greater than zero.")
        else:
            print(f"Updating 'filter_param' to '{value}'.")
            self.filter_param = value





