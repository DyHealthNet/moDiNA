Tutorial
========

In most cases, moDiNA can be conveniently used as a standalone Python package. 
For users interested in systematically comparing results across multiple moDiNA configurations, 
a dedicated :doc:`Nextflow pipeline <nextflow>` is provided.


Input Data
--------------

.. _real_world:

Real-World Data
~~~~~~~~~~~~~~~

To run **moDiNA** on your own data, you need two ``pandas`` DataFrames containing
the sample data from two biological conditions (e.g. healthy and diseased). 
Each row represents a sample, while each column corresponds to a variable.

Both DataFrames must contain the same set of variables. However, they may differ
in the number of samples, as samples are generally assumed to be independent.

The data type of each variable must be specified in a metadata file containing
two columns: ``label`` and ``type``. Supported variable types are
``continuous``, ``binary``, ``nominal``, and ``ordinal``. 
Categorical variables must be encoded as numerical values.

The tables below show an example of a context DataFrame and the corresponding metadata file.

.. table:: Context Data

   +--------+-----------+-----------+--------+---------------+-----------+
   |        | Protein_A | Protein_B | Gender | Disease_Stage | Ethnicity |
   +========+===========+===========+========+===============+===========+
   | S1     | 2.31      | 5.12      | 0      | 1             | 2         |
   +--------+-----------+-----------+--------+---------------+-----------+
   | S2     | 3.04      | 4.87      | 1      | 2             | 0         |
   +--------+-----------+-----------+--------+---------------+-----------+
   | S3     | 2.76      | 5.33      | 0      | 3             | 1         |
   +--------+-----------+-----------+--------+---------------+-----------+
   | S4     | 3.18      | 4.95      | 1      | 2             | 2         |
   +--------+-----------+-----------+--------+---------------+-----------+

.. table:: Metadata

   +----------------+-----------+
   | label          | type      |
   +================+===========+
   | Protein_A      | continuous|
   +----------------+-----------+
   | Protein_B      | continuous|
   +----------------+-----------+
   | Gender         | binary    |
   +----------------+-----------+
   | Disease_Stage  | ordinal   |
   +----------------+-----------+
   | Ethnicity      | nominal   |
   +----------------+-----------+


.. _simulations:

Simulated Data
~~~~~~~~~~~~~~

Alternatively, the ``simulate_copula`` function allows you to generate two synthetic biological contexts
with continuous, binary, and ordinal categorical variables, using Gaussian copula sampling.
The occurence and the magnitude of differential effects can be controlled via the input parameters. 
This approach is particularly useful for benchmarking **moDiNA** under controlled conditions.

Parameters:

- ``name1``, ``name2``: Names of the two biological contexts.
- ``n_cont``, ``n_bi``, ``n_cat``: Number of continuous, binary and ordinal variables per context, respectively.
- ``n_samples``: Number of samples per context.
- ``n_shift_*``: Number of variables with an artificial mean shift. Replace ``*`` with the variable type, e.g., ``n_shift_cont``, ``n_shift_bi``, ``n_shift_cat``.
- ``n_corr_*``: Number of variable pairs with a correlation difference. ``*`` represents the variable type combinations, e.g., ``n_corr_cont_cont``, ``n_corr_bi_cat``, etc.
- ``n_both_*``: Number of variable pairs with both a mean shift and a correlation difference. ``*`` represents variable type combinations as above.
- ``shift``: Magnitude of the mean shift.
- ``corr``: Magnitude of the correlation difference (correlation coefficient between 0 and 1).
- ``path``: Optional path to save simulated contexts, metadata, and ground truth.

Returns:

A tuple ``(context1, context2, meta, ground_truth)``:

- ``context1``: pandas DataFrame of the first simulated context.
- ``context2``: pandas DataFrame of the second simulated context.
- ``meta``: pandas DataFrame containing the data type for each variable.
- ``ground_truth``: Tuple of three lists specifying the nodes with differential effects:
  
  - nodes with mean shifts,
  - nodes with correlation differences,
  - nodes with both mean shifts and correlation differences.

Example:

.. code-block:: python

    from modina.context_simulation import simulate_copula

    context1, context2, meta, ground_truth = simulate_copula(
        n_cont=30,
        n_bi=20,
        n_cat=10,
        n_samples=200,
        n_shift_cont=5,
        n_corr_bi_cat=2,
        n_both_cont_cat=2,
        shift=0.8,
        corr=0.6
    )


Differential Network Analysis
-----------------------------
