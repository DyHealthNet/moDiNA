# moDiNA
The Python package moDiNA provides a customizable end-to-end pipeline for **m**ulti-**o**mics **D**ifferential **N**etwork **A**nalysis in mixed-type data.

![Pipeline Overview](docs/figures/workflow.png)

The moDiNA pipeline takes tabular data as input, where each row represents an individual sample and each column corresponds to a feature. Context-specific networks are inferred using data-type-specific parametric or non-parametric tests. The resulting association scores (p-values and effect sizes) are then rescaled through min-max normalization, prior to an optional edge filtering step with flexible configuration of the filter parameter, metric, and integration rule. To construct the differential network, node- and edge-level information from the context-specific networks is aggregated employing various metrics. These can take p-values (P), effect sizes (E), or the raw observations (obs) into account. Finally, a ranking algorithm is applied to yield a node- or edge-based output ranking. The ranking algorithms differ in their ability to consider differential node and edge scores and their sign.

## Installation
It is strongly recommended to install moDiNA in a conda environment. Currently, the package is only available on GitHub. 

1. Install the package from the GitHub repository:
```bash
pip install git+https://github.com/DyHealthNet/moDiNA.git
```

2. Install NApy by following the instructions on https://github.com/DyHealthNet/NApy.

## Recommended Settings
Based on an extensive benchmark analysis performed on simulated data, we recommend the following pipeline configuration:

- Filtering: For reasonably small datasets, no filtering is required. For high-dimensional data, use density or degree filtering.
- Edge Metric: *log-CS*
- Node Metric: *STC*
- Ranking Algorithm: *PageRank+*

These are the default settings of moDiNA.
