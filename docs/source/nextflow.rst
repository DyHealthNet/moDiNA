Nextflow Pipeline
========

For users interested in systematically comparing results across multiple moDiNA configurations, 
the :doc:`Nextflow pipeline <nextflow>` is the correct choice as it enables parallel, local execution using conda, Docker, Singularity, and distributed execution on HPC clusters using SLURM. 



Installation 
--------------

.. note::
    If you are new to Nextflow, please refer to `this page <https://nf-co.re/docs/usage/installation>`_ on how to set-up Nextflow. 

Then you need to clone the moDiNA Nextflow pipeline repository:

.. code-block:: bash

    git clone https://github.com/DyHealthNet/moDiNA_nf_pipeline.git


TODO: create conda environments if this is not working automatically...

Parameters 
--------------

The Nextflow pipeline is designed to run multiple configurations in parallel, allowing users to systematically evaluate different settings and their impact on the results. 
To specify the configurations, you can either change the parameters in the nextflow.config file or create a configuration file (e.g., `params.yml`) that contains the parameters for each configuration you want to run.


Pipeline Parameters
--------------

This page documents all configurable parameters of the moDiNA Nextflow pipeline.
Parameters are set in ``nextflow.config`` or via a YAML params file (``-params-file params.yml``).


General Parameters
~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 15 15 40

   * - Parameter
     - Type
     - Default
     - Description
   * - ``name_context_1``
     - ``string``
     - ``'context1'``
     - Label assigned to the first context (e.g. ``"Female"`` or ``"Control"``).
   * - ``name_context_2``
     - ``string``
     - ``'context2'``
     - Label assigned to the second context (e.g. ``"Male"`` or ``"Disease"``).
   * - ``run_type``
     - ``string``
     - ``'file'``
     - Determines how moDiNA configurations are specified. One of ``single`` (one fixed configuration), ``all`` (all valid combinations), or ``file`` (read from a CSV samplesheet).
   * - ``data_type``
     - ``string``
     - ``'simulation'``
     - Source of input data. Either ``simulation`` (generate synthetic data via copula) or ``real`` (provide real-world data files).
   * - ``out_dir``
     - ``path``
     - ``''``
     - Absolute path to the directory where all pipeline outputs are written.
    * - ``evaluation``
     - ``boolean``
     - ``true``
     - Boolean specifying whether evaluation plots should be generated.
 
Environment Parameters
~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Parameter
     - Type
     - Description
   * - ``conda_modina_env``
     - ``path``
     - Absolute path to the conda environment used for all core moDiNA processes (network inference, ranking, …).
   * - ``conda_eval_env``
     - ``path``
     - Absolute path to the conda environment used for R-based evaluation processes.


Simulation Parameters (``simulation.*``)
~~~~~~~~~~~~~~~

Only required when ``data_type = 'simulation'``.

.. list-table::
   :header-rows: 1
   :widths: 30 15 15 40

   * - Parameter
     - Type
     - Default
     - Description
   * - ``simulation.n_simulations``
     - ``integer``
     - ``3``
     - Number of independent simulation replicates to run.
   * - ``simulation.n_bi``
     - ``integer``
     - ``50``
     - Number of binary nodes to simulate.
   * - ``simulation.n_cont``
     - ``integer``
     - ``50``
     - Number of continuous nodes to simulate.
   * - ``simulation.n_cat``
     - ``integer``
     - ``50``
     - Number of categorical nodes to simulate.
   * - ``simulation.n_samples``
     - ``integer``
     - ``500``
     - Number of samples (observations) per context.
   * - ``simulation.n_shift_cont``
     - ``integer``
     - ``4``
     - Number of continuous nodes with an artificially introduced mean shift between contexts.
   * - ``simulation.n_shift_bi``
     - ``integer``
     - ``4``
     - Number of binary nodes with an artificially introduced mean shift between contexts.
   * - ``simulation.n_shift_cat``
     - ``integer``
     - ``4``
     - Number of categorical nodes with an artificially introduced mean shift between contexts.
   * - ``simulation.n_corr_cont_cont``
     - ``integer``
     - ``2``
     - Number of continuous–continuous node pairs with an artificially introduced correlation difference between contexts.
   * - ``simulation.n_corr_bi_bi``
     - ``integer``
     - ``2``
     - Number of binary–binary node pairs with an artificially introduced correlation difference between contexts.
   * - ``simulation.n_corr_cat_cat``
     - ``integer``
     - ``2``
     - Number of categorical–categorical node pairs with an artificially introduced correlation difference between contexts.
   * - ``simulation.n_corr_bi_cont``
     - ``integer``
     - ``2``
     - Number of binary–continuous node pairs with an artificially introduced correlation difference between contexts.
   * - ``simulation.n_corr_bi_cat``
     - ``integer``
     - ``2``
     - Number of binary–categorical node pairs with an artificially introduced correlation difference between contexts.
   * - ``simulation.n_corr_cont_cat``
     - ``integer``
     - ``2``
     - Number of continuous–categorical node pairs with an artificially introduced correlation difference between contexts.
   * - ``simulation.n_both_cont_cont``
     - ``integer``
     - ``2``
     - Number of continuous–continuous node pairs with both a mean shift and a correlation difference.
   * - ``simulation.n_both_bi_bi``
     - ``integer``
     - ``2``
     - Number of binary–binary node pairs with both a mean shift and a correlation difference.
   * - ``simulation.n_both_cat_cat``
     - ``integer``
     - ``2``
     - Number of categorical–categorical node pairs with both a mean shift and a correlation difference.
   * - ``simulation.n_both_bi_cont``
     - ``integer``
     - ``2``
     - Number of binary–continuous node pairs with both a mean shift and a correlation difference.
   * - ``simulation.n_both_bi_cat``
     - ``integer``
     - ``2``
     - Number of binary–categorical node pairs with both a mean shift and a correlation difference.
   * - ``simulation.n_both_cont_cat``
     - ``integer``
     - ``2``
     - Number of continuous–categorical node pairs with both a mean shift and a correlation difference.
   * - ``simulation.shift``
     - ``float``
     - ``0.5``
     - Magnitude of the mean shift introduced in shifted nodes. Must be ≥ 0.
   * - ``simulation.corr``
     - ``float``
     - ``0.7``
     - Target correlation difference introduced in correlated node pairs. Must be in [0, 1].


Real-World Data Parameters (``real_world_data.*``)
~~~~~~~~~~~~~~~

Only required when ``data_type = 'real'``.

.. list-table::
   :header-rows: 1
   :widths: 30 15 15 40

   * - Parameter
     - Type
     - Default
     - Description
   * - ``real_world_data.path_context_1``
     - ``path``
     - ``''``
     - Absolute path to the CSV data file for context 1 (rows: samples, columns: variables).
   * - ``real_world_data.path_context_2``
     - ``path``
     - ``''``
     - Absolute path to the CSV data file for context 2 (rows: samples, columns: variables).
   * - ``real_world_data.path_meta``
     - ``path``
     - ``''``
     - Absolute path to the metadata CSV file containing variable labels and data types.


Differential Network Analysis Parameters (``diff_net_analysis.*``)
~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 15 15 40

   * - Parameter
     - Type
     - Default
     - Description
   * - ``diff_net_analysis.path_config``
     - ``path``
     - ``'input_samplesheet.csv'``
     - Path to the CSV samplesheet listing moDiNA configurations to run. Required when ``run_type = 'file'``. Columns: ``node_metric``, ``edge_metric``, ``ranking_algorithm``.
   * - ``diff_net_analysis.node_metric``
     - ``string``
     - ``'STC'``
     - Node-level differential metric. Used when ``run_type = 'single'``. One of: ``STC``, ``DC-P``, ``DC-E``, ``WDC-P``, ``WDC-E``, ``PRC-P``, ``None``.
   * - ``diff_net_analysis.edge_metric``
     - ``string``
     - ``'pre-LS'``
     - Edge-level differential metric. Used when ``run_type = 'single'``. One of: ``diff-P``, ``pre-E``, ``post-E``, ``int-IS``, ``pre-LS``, ``post-LS``, ``pre-PE``, ``post-PE``, ``None``.
   * - ``diff_net_analysis.ranking_algorithm``
     - ``string``
     - ``'PageRank+'``
     - Algorithm used to rank nodes and edges in the differential network. Used when ``run_type = 'single'``. One of: ``PageRank+``, ``PageRank``, ``absDimontRank``, ``DimontRank``, ``direct_node``, ``direct_edge``.
   * - ``diff_net_analysis.filter_method``
     - ``string``
     - ``''``
     - Optional method to filter context-specific networks before differential analysis. One of: ``degree``, ``density``, ``quantile``. Leave empty to skip filtering.
   * - ``diff_net_analysis.filter_param``
     - ``integer / float``
     - ``2``
     - Parameter for the chosen filter method. Integer for ``degree``; float in (0, 1) for ``density`` and ``quantile``.
   * - ``diff_net_analysis.filter_metric``
     - ``string``
     - ``'pre-P'``
     - Edge score used as the basis for filtering. One of: ``raw-P``, ``rescaled-E``. Required when ``filter_method`` is set.
   * - ``diff_net_analysis.filter_rule``
     - ``string``
     - ``'zero'``
     - Rule applied when filtering edges. One of: ``zero`` (set filtered edges to zero), ``union`` (keep edges present in at least one context).
   * - ``diff_net_analysis.max_path_length``
     - ``integer``
     - ``2``
     - Maximum path length used for the ``int-IS`` edge metric. Must be an integer in [0, 4]. Only relevant when ``int-IS`` is selected as an edge metric.
   * - ``diff_net_analysis.test_type``
     - ``string``
     - ``'nonparametric'``
     - Type of statistical test used during context network inference. One of: ``parametric``, ``nonparametric``.
   * - ``diff_net_analysis.nan_value``
     - ``integer``
     - ``-89``
     - Sentinel value in input data that represents missing / NA values.
   * - ``diff_net_analysis.multiple_testing``
     - ``string``
     - ``'bh'``
     - Multiple testing correction method applied to association p-values. One of: ``bh`` (Benjamini–Hochberg), ``by`` (Benjamini–Yekutieli).


Usage 
--------------

You can choose between the following profiles: conda, docker, singularity, slurm. The respective profile will be used to execute the pipeline with the appropriate containerization or job scheduling system.
You can specify the profile using the `-profile` flag when running the Nextflow pipeline. For example, to run the pipeline using conda for package management and SLURM for distributed execution, you would use the following command:

Now, you can run the Nextflow pipeline using the following command:

.. code-block:: bash

    nextflow run main.nf -profile conda,slurm

If you have stored the parameters in a params.yml file instead of the nextflow.config file, you can specify the path to this file using the `-params-file` flag:

.. code-block:: bash

    nextflow run main.nf -profile conda,slurm -params-file path/to/your/params.yml


Pipeline Output
--------------

The results are stored in the specified output directory, which is organized by differential network analysis steps and evaluation steps.

The main output of the pipeline includes:
- Simulated datasets for each simulation replicate (if ``data_type = 'simulation'``).
- Inferred context-specific networks for each context and simulation.
- Node and edge metrics files for each configuration and simulation.
- Ranking results for each configuration and simulation.
- Evaluation results

In the following, we provide some examples of evaluation results generated by the Nextflow pipeline.

For evaluating the strength of the simulation, we can check the distribution of the introduced mean shifts and correlation differences across the simulated nodes and node pairs.

.. image:: _static/mean_shift_point_plot.png
   :alt: Mean shift point plot
   :width: 600px
   :align: center

.. image:: _static/association_scores_point_plot.png
   :alt: Association scores point plot
   :width: 600px
   :align: center


The performance of the different configurations can be evaluated using for example the AUC heatmap, which shows the AUC values for each configuration averaged across all simulation replicates.

.. image:: _static/overall_heatmap_auc.png
   :alt: AUC heatmap
   :width: 600px
   :align: center
