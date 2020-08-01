.. gym-anm installation documentation

Installation
=============

:code:`gym-anm` requires Python 3.5+ and can run on Linux, MaxOS, and Windows. You can
install it through `pip`_, :ref:`conda`, or even from source.

We recommend installing :code:`gym-anm` in a Python environment (e.g., `virtualenv
<https://virtualenv.pypa.io/en/stable/index.html>`_ or `conda <https://conda.io/en/latest/#>`_).

pip
---
Using pip (preferably after activating your virtual environment): ::

    pip install gym-anm

conda
-----
If you would like to run :code:`gym-anm` inside a conda environment, you can:

1. Install `conda <https://conda.io/en/latest/#>`_.
2. Create a new environment :code:`my_env`: ::

    conda create --name my_env
    conda activate my_end

3. Install :code:`gym-anm`: ::

    conda install gym-anm

Building from Source
--------------------
Alternatively, you can build :code:`gym-anm` directly from source: ::

    git clone https://github.com/robinhenry/gym-anm.git
    cd gym-anm
    pip install -e .

