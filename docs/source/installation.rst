Installation
============

Currently, the **moDiNA** package is only available on `GitHub <https://github.com/DyHealthNet/moDiNA>`_.
**moDiNA** requires Python version 3.11.

It is recommended to install **moDiNA** in a clean conda (`Miniconda <https://www.anaconda.com/docs/getting-started/miniconda/main>`_) environment.
We suggest using `mamba <https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html>`_, a faster drop-in replacement for conda that
improves dependency resolution. Mamba is automatically installed when
using `Miniforge <https://github.com/conda-forge/miniforge>`_.

First, follow the installation instructions for your operating system.
Then create and activate a new environment:

.. code-block:: bash

   mamba create -n modina_env python=3.11
   mamba activate modina_env

Next, install the package:

.. code-block:: bash

   pip install git+https://github.com/DyHealthNet/moDiNA.git

If you face any issues, feel free to open an issue on `GitHub <https://github.com/DyHealthNet/moDiNA/issues>`_. 