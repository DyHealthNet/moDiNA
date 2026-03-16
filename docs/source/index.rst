moDiNA: Differential Network Analysis of Mixed-Type Multi-Omics Data 
====================================================================

.. figure:: _static/moDiNA_Introduction.png
   :alt: Overview of the moDiNA pipeline
   :width: 800px
   :align: center

   **Figure 1:** Overview of the moDiNA pipeline. 
   Created with BioRender.com.

Overview
--------

The **moDiNA** pipeline facilitates differential network analysis of mixed-type multi-omics data.
It compares two biologically distinct contexts and constructs a ranked differential network that captures both differentially abundant variables (nodes) and differential associations (edges).

The framework supports multiple data types, including continuous, binary, nominal, and ordinal categorical variables.
All processing steps are configurable through a user-defined configuration file, allowing flexible adaptation to different datasets and analysis goals.

The **moDiNA** workflow is organized into six main steps:

1. Launch and Context Simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**moDiNA** supports both :ref:`real-world <real_world>` multi-omics datasets and :ref:`simulated <simulations>` data, enabling practical analyses as well as controlled benchmarking studies.  
For simulations, context data can be generated via Gaussian copula sampling, allowing users to control the magnitude and frequency of differential effects across contexts.

2. Context Network Inference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For each context, a network is inferred by computing pairwise statistical associations between features.
The implementation relies on NApy, which provides statistical tests for mixed data types and enhanced handling of missing values.

3. Network Filtering
~~~~~~~~~~~~~~~~~~~~

Optional edge filtering methods can be applied to the context-specific networks to remove weak or insignificant associations.

4. Differential Network Inference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Node- and edge-level information from the context-specific networks is aggregated to construct a differential network.
Multiple metrics can be used, incorporating p-values (P), effect sizes (E), or raw observations (obs) to quantify the differences across contexts.

5. Node and Edge Ranking
~~~~~~~~~~~~~~~~~~~~~~~~

Nodes and edges in the differential network are ranked using various network-based ranking algorithms.
These algorithms differ in how they integrate node and edge scores and whether they consider the direction of differential effects.

6. Evaluation
~~~~~~~~~~~~~

Using the provided :doc:`Nextflow pipeline <nextflow>`, multiple configurations can be systematically evaluated for their ranking performance and similarity across simulation scenarios.


.. note::

   This project is under active development.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   tutorial
   nextflow
   modules

