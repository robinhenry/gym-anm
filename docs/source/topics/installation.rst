.. gym-anm installation documentation

Installation
=============

:code:`gym-anm` requires Python 3.7+ and can run on Linux, MaxOS, and Windows. You can
install it through :ref:`from_pip`, :ref:`from_conda`, or even :ref:`from source <from_source>`.

We recommend installing :code:`gym-anm` in a Python environment (e.g., `virtualenv
<https://virtualenv.pypa.io/en/stable/index.html>`_ or `conda <https://conda.io/en/latest/#>`_).

.. _from_pip:

pip
---
Using pip (preferably after activating your virtual environment): ::

    pip install gym-anm

.. _from_conda:

conda
-----
If you would like to run :code:`gym-anm` inside a conda environment, you can:

1. `Install conda <https://docs.conda.io/projects/conda/en/latest/user-guide/install/>`_ (or simply
   `miniconda <https://docs.conda.io/en/latest/miniconda.html>`_).
2. Create a new environment :code:`my_env`: ::

    conda create --name my_env
    conda activate my_env

3. Install :code:`gym-anm`: ::

    conda install gym-anm

.. _from_source:

Building from source
--------------------
Alternatively, you can build :code:`gym-anm` directly from source: ::

    git clone https://github.com/robinhenry/gym-anm.git
    cd gym-anm
    pip install -e .
